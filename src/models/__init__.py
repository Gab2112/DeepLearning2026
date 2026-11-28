"""
Forecasting and imputation models package.
"""

from src.models.history_imputer import BiGRUHistoryImputer
from src.models.mae_imputer import MAEHistoryImputer
from src.models.frozen import (
    freeze_model,
    load_frozen_model,
    save_frozen_checkpoint,
)
from src.models.baselines import (
    PersistenceBaseline,
    ProbabilisticPersistenceBaseline,
    extract_clear_sky_history,
    extract_last_clear_sky,
)

__all__ = [
    "BiGRUHistoryImputer",
    "MAEHistoryImputer",
    "freeze_model",
    "save_frozen_checkpoint",
    "load_frozen_model",
    "PersistenceBaseline",
    "ProbabilisticPersistenceBaseline",
    "extract_clear_sky_history",
    "extract_last_clear_sky",
]
