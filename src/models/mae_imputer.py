"""
Masked Autoencoder (MAE) History Imputer for irregular satellite vegetation time series.
Uses a Transformer encoder-decoder over daily multivariate sequences (weather, sparse
clear-sky indices, observation masks, Sentinel-2 orbital dynamics, and Köppen climate
embeddings) with learnable mask tokens for missing target observations and optional
self-supervised weather masking during training.
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard non-learned sinusoidal positional encoding supporting variable sequence lengths.
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        pe = self._build_pe(max_len, d_model, device=torch.device("cpu"), dtype=torch.float32)
        self.register_buffer("pe", pe, persistent=False)

    @staticmethod
    def _build_pe(
        length: int,
        d_model: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, length, d_model, device=device, dtype=torch.float32)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            cos_term = torch.cos(position * div_term[: d_model // 2])
            pe[0, :, 1::2] = cos_term
        return pe.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        if L > self.pe.size(1):
            pe = self._build_pe(L, self.d_model, device=x.device, dtype=x.dtype)
            self.pe = pe
        return x + self.pe[:, :L, :].to(device=x.device, dtype=x.dtype)


class MAEHistoryImputer(nn.Module):
    """
    Transformer-based Masked Autoencoder for imputing sparse clear-sky vegetation indices.

    Expected input x of shape (B, L, 27):
        - [:, :, 0:11]  : normalized weather & cloud covariates (masked during MAE training)
        - [:, :, 11:18] : deterministic day_of_year & Fourier calendar covariates
        - [:, :, 18:21] : sparse normalized target values (NDVI, GNDVI, EVI)
        - [:, :, 21:24] : observation mask flags (1.0 if observed, 0.0 if missing)
        - [:, :, 24:27] : Sentinel-2 orbital dynamics (delta_t, orbital_phase, missed_revisits)
    """

    def __init__(
        self,
        input_dim: int = 27,
        climate_dim: int = 8,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 128,
        output_dim: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Lazy import avoids circular import with src.data
        from src.data.climate import NUM_CLIMATES

        self.input_dim = input_dim
        self.climate_dim = climate_dim
        self.d_model = d_model
        self.output_dim = output_dim

        self.climate_embedding = nn.Embedding(NUM_CLIMATES, climate_dim)

        self.weather_proj = nn.Sequential(
            nn.Linear(11, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.target_proj = nn.Sequential(
            nn.Linear(6, d_model),  # 3 sparse normalized targets + 3 obs mask flags
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(7 + 3 + climate_dim, d_model),  # 7 calendar + 3 orbital + climate embedding
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.target_mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.weather_mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.target_mask_token, std=0.02)
        nn.init.normal_(self.weather_mask_token, std=0.02)

        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model, max_len=512)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            enable_nested_tensor=False,
        )

        self.decoder_proj = nn.Linear(d_model, d_model)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            enable_nested_tensor=False,
        )
        self.decoder_norm = nn.LayerNorm(d_model)

        self.target_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )
        self.weather_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 11),
        )

    def _build_key_padding_mask(
        self,
        padding_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if padding_mask is None:
            return None
        # TransformerEncoder expects True at padded (ignored) positions
        key_pad = ~padding_mask.bool()
        # Safeguard against any all-padded row producing NaNs in attention softmax
        if key_pad.all(dim=-1).any():
            key_pad = key_pad.clone()
            key_pad[key_pad.all(dim=-1), 0] = False
        return key_pad

    def forward(
        self,
        x: torch.Tensor,
        climate_id: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        weather_mask: Optional[torch.Tensor] = None,
        return_weather: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (B, L, 27)
            climate_id: (B,) integer tensor in [0, NUM_CLIMATES-1]
            padding_mask: (B, L) boolean tensor (True where valid, False where padding)
            weather_mask: optional (B, L) boolean tensor (True where weather tokens are masked)
            return_weather: if True, returns (pred_targets, pred_weather)
        Returns:
            pred_targets: (B, L, 3) or tuple ((B, L, 3), (B, L, 18))
        """
        B, L, _ = x.shape

        x_weather = x[:, :, :11]
        x_calendar = x[:, :, 11:18]
        x_targets = x[:, :, 18:21]
        x_obs_flags = x[:, :, 21:24]
        x_orbital = x[:, :, 24:27]

        if climate_id is None:
            climate_id = torch.zeros(B, dtype=torch.long, device=x.device)
        c_emb = self.climate_embedding(climate_id).unsqueeze(1).expand(B, L, -1)

        weather_tok = self.weather_proj(x_weather)
        if weather_mask is not None:
            w_mask_exp = weather_mask.bool().unsqueeze(-1)
            weather_tok = torch.where(
                w_mask_exp,
                self.weather_mask_token.expand(B, L, -1),
                weather_tok,
            )

        target_obs_mask = x_obs_flags[:, :, 0] > 0.5
        target_tok = self.target_proj(torch.cat([x_targets, x_obs_flags], dim=-1))
        target_tok = torch.where(
            target_obs_mask.unsqueeze(-1),
            target_tok,
            self.target_mask_token.expand(B, L, -1),
        )

        context_tok = self.context_proj(torch.cat([x_calendar, x_orbital, c_emb], dim=-1))

        h = self.input_norm(weather_tok + target_tok + context_tok)
        h = self.input_dropout(self.pos_enc(h))

        src_key_padding_mask = self._build_key_padding_mask(padding_mask)
        latent = self.encoder(h, src_key_padding_mask=src_key_padding_mask)

        dec_in = self.pos_enc(self.decoder_proj(latent))
        dec_out = self.decoder_norm(
            self.decoder(dec_in, src_key_padding_mask=src_key_padding_mask)
        )

        pred_targets = self.target_head(dec_out)
        if padding_mask is not None:
            pred_targets = pred_targets * padding_mask.unsqueeze(-1).float()

        if return_weather:
            pred_weather = self.weather_head(dec_out)
            if padding_mask is not None:
                pred_weather = pred_weather * padding_mask.unsqueeze(-1).float()
            return pred_targets, pred_weather

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
        Inference helper: predicts dense targets, preserving exact ground-truth
        values at observed clear-sky timestamps (obs_mask == True).
        """
        self.eval()
        with torch.no_grad():
            preds = self.forward(x, climate_id=climate_id, padding_mask=padding_mask)
            dense_targets = torch.where(obs_mask.unsqueeze(-1), raw_sparse_targets, preds)
            if padding_mask is not None:
                dense_targets = dense_targets * padding_mask.unsqueeze(-1).float()
            return dense_targets
