"""Weather cleaning, availability masks and normalization for daily covariates."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from src.data.interpolation import LinearInterpolator
from src.data.transforms import ArcsinhScaler


WEATHER_FEATURE_COUNT = 8
COVARIATE_COUNT = 18
LEGACY_IMPUTER_INPUT_DIM = 27
WIND, HUMIDITY, IRRADIANCE, RAINFALL, PRESSURE, AVG_TEMP, MIN_TEMP, MAX_TEMP = range(8)
TEMPERATURES = slice(AVG_TEMP, MAX_TEMP + 1)
WEATHER = slice(WEATHER_FEATURE_COUNT)
COVARIATES = slice(COVARIATE_COUNT)


@dataclass(frozen=True)
class WeatherCovariates:
    values: np.ndarray
    original_missing: np.ndarray
    invalid: np.ndarray
    available: np.ndarray

    @property
    def weather_mask(self) -> np.ndarray:
        return self.available[:, WEATHER]


@dataclass(frozen=True)
class WeatherPreprocessor:
    """Select M0 linear, M1 masking, M2 historical QC, or M3 with future clipping."""

    profile: str = "M0"
    temperature_tolerance_c: float = 1.0
    temperature_policy: str = "mask_triplet"

    def __post_init__(self):
        if self.profile not in ("M0", "M1", "M2", "M3"):
            raise ValueError(f"Unknown weather preprocessing profile: {self.profile!r}")
        if not np.isfinite(self.temperature_tolerance_c) or self.temperature_tolerance_c < 0:
            raise ValueError("temperature_tolerance_c must be finite and non-negative")
        if self.temperature_policy not in ("mask_triplet", "ignore"):
            raise ValueError("temperature_policy must be 'mask_triplet' or 'ignore'")

    @property
    def uses_masks(self) -> bool:
        return self.profile != "M0"

    @property
    def imputer_input_dim(self) -> int:
        return LEGACY_IMPUTER_INPUT_DIM + (WEATHER_FEATURE_COUNT if self.uses_masks else 0)

    def for_imputer(self) -> "WeatherPreprocessor":
        # M3 changes only future data. Reuse exactly the M2 historical preprocessing.
        if self.profile == "M3":
            return replace(self, profile="M2")
        if self.profile in ("M0", "M1"):
            return WeatherPreprocessor(profile=self.profile)
        return self

    def imputer_checkpoint_path(self, path: str | Path) -> Path:
        path = Path(path)
        profile = self.for_imputer().profile
        if profile == "M0":
            return path
        return path.with_stem(f"{path.stem}_{profile.lower()}")

    def _historical_invalid(
        self, weather: np.ndarray, original_missing: np.ndarray, available: np.ndarray
    ) -> np.ndarray:
        invalid = np.isinf(weather) & ~original_missing
        invalid[:, [WIND, IRRADIANCE, RAINFALL]] |= weather[:, [WIND, IRRADIANCE, RAINFALL]] < 0
        humidity = weather[:, HUMIDITY]
        invalid[:, HUMIDITY] |= (humidity < 0) | (humidity > 100)
        invalid[:, PRESSURE] |= weather[:, PRESSURE] <= 0
        invalid &= ~original_missing

        if self.temperature_policy == "mask_triplet":
            inconsistent = np.zeros(len(weather), dtype=bool)
            for lower, upper in ((MIN_TEMP, AVG_TEMP), (AVG_TEMP, MAX_TEMP), (MIN_TEMP, MAX_TEMP)):
                inconsistent |= (
                    available[:, lower] & available[:, upper]
                    & (weather[:, lower] > weather[:, upper] + self.temperature_tolerance_c)
                )
            invalid[:, TEMPERATURES] |= inconsistent[:, None] & available[:, TEMPERATURES]
        return invalid

    def transform(
        self,
        raw: np.ndarray,
        missing_mask: np.ndarray,
        scaler: ArcsinhScaler,
        *,
        future: bool = False,
        interpolate_history: bool = False,
    ) -> WeatherCovariates:
        """Return 18 normalized channels and masks for one sequence of raw covariates.

        Original and invalid masks cover the eight weather variables. Availability
        covers all 18 covariates and refers to original usable values.
        """
        raw = np.asarray(raw, dtype=np.float32)
        missing_mask = np.asarray(missing_mask, dtype=bool)
        if raw.ndim != 2 or raw.shape[1] not in (18, 21) or raw.shape != missing_mask.shape:
            raise ValueError("Expected matching (L, 18) or (L, 21) values and masks")

        values = raw[:, COVARIATES].copy()
        finite = np.isfinite(values)
        source_mask = missing_mask[:, COVARIATES]
        original_missing = source_mask[:, WEATHER] | np.isnan(values[:, WEATHER])
        available = finite & ~source_mask
        invalid = np.zeros_like(original_missing)
        if not future and self.profile in ("M2", "M3"):
            invalid = self._historical_invalid(values[:, WEATHER], original_missing, available[:, WEATHER])
            available[:, WEATHER] &= ~invalid

        use_linear = not future and interpolate_history and not self.uses_masks
        # For M0, the historical neural inputs and future covariates used only
        # numerical missingness. The historical forecasting loader also used source masks.
        unusable = ~available if self.uses_masks or use_linear else ~finite
        values[unusable] = np.nan
        if use_linear:
            missing_columns = np.flatnonzero(unusable.any(axis=0))
            if missing_columns.size:
                values = LinearInterpolator().interpolate(values, mask=unusable, columns=missing_columns)

        if future and self.profile == "M3":
            values[:, [WIND, IRRADIANCE, RAINFALL]] = np.maximum(
                values[:, [WIND, IRRADIANCE, RAINFALL]], 0.0
            )
            values[:, HUMIDITY] = np.clip(values[:, HUMIDITY], 0.0, 100.0)
            values[:, 8:11] = np.clip(values[:, 8:11], 0.0, 100.0)

        normalized = scaler.transform_features(values, feature_slice=COVARIATES)
        np.nan_to_num(normalized, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return WeatherCovariates(normalized, original_missing, invalid, available)


def configure_imputer_weather(cfg: dict, model_key: str) -> WeatherPreprocessor:
    """Record the effective historical preprocessing and width in the training config."""
    weather = WeatherPreprocessor(**cfg.get("weather_preprocessing", {})).for_imputer()
    cfg["weather_preprocessing"] = asdict(weather)
    cfg[model_key]["input_dim"] = weather.imputer_input_dim
    return weather


def checkpoint_weather_preprocessor(
    config: dict, input_dim: int, expected: WeatherPreprocessor | None = None
) -> WeatherPreprocessor:
    """Recover the historical preprocessing used at training time."""
    options = config.get("weather_preprocessing")
    if options is None:
        # Old checkpoints lack weather metadata. The earlier mask-only variant
        # can be identified by its appended eight channels.
        profile = "M1" if input_dim == LEGACY_IMPUTER_INPUT_DIM + WEATHER_FEATURE_COUNT else "M0"
        options = {"profile": profile}
    weather = WeatherPreprocessor(**options).for_imputer()
    if input_dim != weather.imputer_input_dim:
        raise ValueError(f"Weather profile {weather.profile} is incompatible with {input_dim} input channels")
    if expected is not None and weather != expected.for_imputer():
        raise ValueError(
            f"Checkpoint weather preprocessing {weather} differs from the configured {expected.for_imputer()}"
        )
    return weather


def model_weather_preprocessor(model) -> WeatherPreprocessor:
    """Use the checkpoint metadata, or the layout of a directly constructed model."""
    weather = getattr(model, "weather_preprocessor", None)
    if weather is not None:
        return weather
    return checkpoint_weather_preprocessor({}, getattr(model, "input_dim", LEGACY_IMPUTER_INPUT_DIM))
