"""
Data loading, processing, and transformation modules.
"""

from src.data.transforms import ArcsinhScaler
from src.data.climate import encode_climate, CLIMATE_TO_ID, ID_TO_CLIMATE, NUM_CLIMATES
from src.data.splits import get_tile_dominant_climate, get_train_val_tile_split
from src.data.history_dataset import HistoryImputationDataset, collate_history_batch
from src.data.vegetation_dataset import VegetationDataset, collate_vegetation_batch
from src.models.bigru_imputer import (
    BiGRUHistoryImputer,
    compute_orbital_dynamics,
    build_imputer_features,
    load_imputer_checkpoint,
    impute_sequence,
    impute_batch_sequences,
    impute_npz_file,
    compute_imputation_metrics,
    evaluate_imputer_on_sample,
)
from src.data.mae_imputer import (
    MAEHistoryImputer,
    load_mae_imputer_checkpoint,
    impute_sequence_mae,
    impute_batch_sequences_mae,
    evaluate_mae_imputer_on_sample,
)
from src.data.interpolation import (
    BaseInterpolator,
    InterpolationFactory,
    build_interpolator_from_config,
    LinearInterpolator,
    ForwardFillInterpolator,
    BackwardFillInterpolator,
    NearestInterpolator,
    SplineInterpolator,
    MeanInterpolator,
    MedianInterpolator,
    ConstantInterpolator,
    NoneInterpolator,
    ModelInterpolator,
    MAEInterpolator,
)

__all__ = [
    "ArcsinhScaler",
    "encode_climate",
    "CLIMATE_TO_ID",
    "ID_TO_CLIMATE",
    "NUM_CLIMATES",
    "get_tile_dominant_climate",
    "get_train_val_tile_split",
    "HistoryImputationDataset",
    "collate_history_batch",
    "VegetationDataset",
    "collate_vegetation_batch",
    "BiGRUHistoryImputer",
    "compute_orbital_dynamics",
    "build_imputer_features",
    "load_imputer_checkpoint",
    "impute_sequence",
    "impute_batch_sequences",
    "impute_npz_file",
    "compute_imputation_metrics",
    "evaluate_imputer_on_sample",
    "MAEHistoryImputer",
    "load_mae_imputer_checkpoint",
    "impute_sequence_mae",
    "impute_batch_sequences_mae",
    "evaluate_mae_imputer_on_sample",
    "BaseInterpolator",
    "InterpolationFactory",
    "build_interpolator_from_config",
    "LinearInterpolator",
    "ForwardFillInterpolator",
    "BackwardFillInterpolator",
    "NearestInterpolator",
    "SplineInterpolator",
    "MeanInterpolator",
    "MedianInterpolator",
    "ConstantInterpolator",
    "NoneInterpolator",
    "ModelInterpolator",
    "MAEInterpolator",
]
