"""
Bidirectional Recurrent Neural Imputer for the Past (History) Time Series.
Integrates Sentinel-2 5-day orbital mechanics and robust Climate Zone Embeddings
with native support for '<UNKNOWN>' climate tokens (crucial for OOD-S generalization).
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

# NOTE: src.data.climate is imported lazily inside __init__ to avoid a circular
# import (src.data -> bigru_imputer -> src.models.history_imputer -> src.data).


class BiGRUHistoryImputer(nn.Module):
    """
    Bidirectional GRU for imputing and densifying sparse satellite target time series.
    Input dims:
        - 18 weather & Fourier calendar features
        - 3 sparse normalized target values
        - 3 observation mask indicators (1 = observed, 0 = missing)
        - 3 orbital dynamics features (delta_t, orbital_phase (t % 5)/5, missed_5d_revisits)
        - 8 optional weather availability flags
        Total input_dim = 27 (M0) or 35 (M1/M2/M3).
    Climate Embedding:
        - 9 classes (0 = <UNKNOWN>, 1..8 = Köppen classes) -> mapped to 8-dim embedding
    Output dims:
        - 3 reconstructed targets (NDVI, GNDVI, EVI) per timestep
    """

    def __init__(
        self,
        input_dim: int = 27,
        climate_dim: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        from src.data.climate import NUM_CLIMATES

        self.input_dim = input_dim
        self.climate_embedding = nn.Embedding(NUM_CLIMATES, climate_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim + climate_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.bigru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        climate_id: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (Batch, Seq_len, input_dim)
        climate_id: (Batch,) integer tensor in [0, 8]
        padding_mask: (Batch, Seq_len) boolean tensor (True where real, False where padding)
        Returns:
            pred_targets: (Batch, Seq_len, 3) daily reconstructed normalized targets
        """
        B, L, input_dim = x.shape
        if input_dim != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} history channels, received {input_dim}")

        if climate_id is not None:
            # (B, climate_dim) -> expand to (B, L, climate_dim)
            c_emb = self.climate_embedding(climate_id).unsqueeze(1).expand(B, L, -1)
            x_combined = torch.cat([x, c_emb], dim=-1)
        else:
            # Default to index 0 (<UNKNOWN>)
            unknown_ids = torch.zeros(B, dtype=torch.long, device=x.device)
            c_emb = self.climate_embedding(unknown_ids).unsqueeze(1).expand(B, L, -1)
            x_combined = torch.cat([x, c_emb], dim=-1)

        h = self.input_proj(x_combined)  # (B, L, hidden_dim)
        gru_out, _ = self.bigru(h)       # (B, L, 2 * hidden_dim)
        pred_targets = self.head(gru_out)  # (B, L, 3)

        if padding_mask is not None:
            pred_targets = pred_targets * padding_mask.unsqueeze(-1).float()

        return pred_targets

    def impute_and_blend(
        self,
        x: torch.Tensor,
        raw_sparse_targets: torch.Tensor,
        obs_mask: torch.Tensor,
        climate_id: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Inference helper: predicts dense targets, but preserves exact ground truth
        values at the observed clear-sky timestamps (obs_mask == True).
        """
        self.eval()
        with torch.no_grad():
            preds = self.forward(x, climate_id=climate_id, padding_mask=padding_mask)
            dense_targets = torch.where(obs_mask.unsqueeze(-1), raw_sparse_targets, preds)
            if padding_mask is not None:
                dense_targets = dense_targets * padding_mask.unsqueeze(-1).float()
            return dense_targets
