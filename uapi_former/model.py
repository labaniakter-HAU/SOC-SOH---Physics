import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

from .evidential import EvidentialNIGHead


class PositionalEncoding(nn.Module):
    pe: torch.Tensor

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)].to(x.device)


class EITE(nn.Module):
    """Electrochemistry-Informed Token Embedding (EITE).

    Accepts raw [V, I, T] sequences plus the precomputed SPMe baseline voltage
    V_spme and its residual (V - V_spme), giving a 5-channel input.  The
    projection layer lifts this to d_model.

    If the caller only provides the 3-channel raw signal (e.g. for baselines or
    when the SPMe cache is unavailable) the module accepts an optional
    spme_baseline tensor; when absent it forwards zeros for the two augmented
    channels so ablation is trivial.
    """

    def __init__(self, in_channels: int = 5, d_model: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.proj = nn.Linear(in_channels, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, v_spme: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, T, C) where C <= in_channels.

        v_spme: (B, T) precomputed SPMe voltage; if supplied and x has only 3
        channels the method augments to 5 channels before zero-padding to in_channels.
        """
        if x.size(-1) == 3 and v_spme is not None:
            residual = x[..., 0] - v_spme
            x = torch.cat([x, v_spme.unsqueeze(-1), residual.unsqueeze(-1)], dim=-1)
        # Zero-pad any x that has fewer channels than the projection expects
        if x.size(-1) < self.in_channels:
            B, T, _ = x.shape
            pad = torch.zeros(B, T, self.in_channels - x.size(-1),
                              device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)
        return self.norm(self.proj(x))


class PRAPPooling(nn.Module):
    """Physics-Residual Attention Pooling (PRAP).

    Replaces uniform mean pooling with a softmax-weighted sum over the time
    dimension, where weights are proportional to |V − V_spme| at each timestep.

    Timesteps where the physics model disagrees most with the measured voltage
    are information-rich (fast-charge events, SOH-degradation signatures) and
    receive stronger pooling weight.  Falls back to uniform mean pooling when
    no residual is available, keeping the no-EITE ablation path unchanged.
    """

    def forward(
        self,
        x: torch.Tensor,
        residual_mag: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: (B, T, d_model); residual_mag: (B, T) or None."""
        if residual_mag is None:
            return x.mean(dim=1)
        weights = torch.softmax(residual_mag, dim=1)   # (B, T)
        return (x * weights.unsqueeze(-1)).sum(dim=1)  # (B, d_model)


class DynamicChemistryToken(nn.Module):
    """Dynamic Chemistry Token (DCT).

    Extracts a d_model-dimensional chemistry fingerprint from the first n_ocv
    timesteps of the raw [V, I, T, V_spme, V−V_spme] input, corresponding to
    the OCV relaxation window, and lifts it to d_model via a two-layer MLP.

    The fingerprint is prepended as a [CLS]-style position-0 token before the
    Transformer encoder so every sequence position can attend to chemistry
    identity.  This enables zero-shot cross-chemistry transfer with NO labeled
    target-domain data: the encoder conditions its representations on a
    chemistry fingerprint derived solely from the OCV window.

    Input tolerance: if x has fewer channels than in_channels (e.g. 3-channel
    real data vs 5-channel full EITE), the forward method zero-pads so the MLP
    always sees in_channels * n_ocv inputs.
    """

    def __init__(
        self,
        in_channels: int = 5,
        chem_dim: int = 32,
        d_model: int = 128,
        n_ocv: int = 10,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_ocv = n_ocv
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * n_ocv, chem_dim),
            nn.ReLU(),
            nn.Linear(chem_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) raw input before EITE projection.  C may be < in_channels.
        Returns (B, d_model) chemistry token."""
        if x.size(-1) < self.in_channels:
            pad = torch.zeros(
                x.size(0), x.size(1), self.in_channels - x.size(-1),
                device=x.device, dtype=x.dtype,
            )
            x = torch.cat([x, pad], dim=-1)
        B = x.size(0)
        ocv_window = x[:, : self.n_ocv, :].reshape(B, -1)  # (B, n_ocv * in_channels)
        return self.mlp(ocv_window)


class ICFeatureExtractor(nn.Module):
    """Incremental Capacity (IC) Feature Extractor.

    Computes a differentiable dQ/dV soft-histogram from the raw [V, I]
    channels, then projects the histogram to a compact d_out vector that is
    concatenated to the SOH task representation before the NIG head.

    IC curves are the most widely cited SOH-correlated feature in the
    battery estimation literature (2024-2025).  Computing them inside the
    network makes the IC branch end-to-end differentiable so gradients from
    the NIG loss can tune the histogram projection weights.
    """

    def __init__(self, n_bins: int = 20, d_out: int = 32):
        super().__init__()
        self.n_bins = n_bins
        self.proj = nn.Sequential(
            nn.Linear(n_bins, 64),
            nn.GELU(),
            nn.Linear(64, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, >=2) where channel 0 = V, channel 1 = I.
        Returns (B, d_out) IC feature vector."""
        V = x[..., 0]   # (B, T)
        I = x[..., 1]   # (B, T)

        # Finite-difference approximation of dQ and dV
        dV = V[:, 1:] - V[:, :-1]                              # (B, T-1)
        dQ = -I[:, :-1]                                         # sign: discharge = negative I

        # IC = dQ/dV; clamp denominator to avoid division by zero
        ic = dQ / (dV.abs().clamp(min=1e-6)) * dV.sign()       # (B, T-1)
        ic = ic.clamp(-50.0, 50.0)                              # winsorise extremes

        # Soft histogram over voltage bins via Gaussian kernel weighting
        V_mid = V[:, :-1]                                        # (B, T-1)
        v_min = V_mid.min(dim=-1, keepdim=True).values           # (B, 1)
        v_max = V_mid.max(dim=-1, keepdim=True).values           # (B, 1)
        v_norm = (V_mid - v_min) / (v_max - v_min + 1e-6) * (self.n_bins - 1)  # (B, T-1)

        bins = torch.arange(self.n_bins, device=x.device, dtype=x.dtype)  # (n_bins,)
        kernel = torch.exp(-0.5 * (v_norm.unsqueeze(-1) - bins) ** 2)     # (B, T-1, n_bins)
        ic_hist = (kernel * ic.unsqueeze(-1)).sum(dim=1)                   # (B, n_bins)
        ic_hist = ic_hist / (kernel.sum(dim=1) + 1e-6)                    # normalise

        return self.proj(ic_hist)                                           # (B, d_out)


class CrossTaskAttentionBlock(nn.Module):
    """Cross-task bidirectional adapter (CTBA; class name kept for checkpoint compatibility).

    After DTAG produces the task vectors soc_rep and soh_rep, each direction applies a
    multi-head-attention sublayer and a feed-forward sublayer: soc_rep (query) uses soh_rep
    as key/value, then soh_rep uses the updated soc_rep. Both inputs are single vectors
    (length-one sequences), so the softmax is over one position and equals 1 at inference:
    the block is a residual cross-task projection, not attention over positions.

    A parameter-matched control (scripts/ablation.py::SelfProjectionBlock, each stream
    projecting its own vector) matches this block in accuracy, so its measured benefit comes
    from added capacity, not from cross-task information flow (paper Sec. 4.11, R2.7).
    """

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.soc_attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.soh_attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.soc_norm1 = nn.LayerNorm(d_model)
        self.soh_norm1 = nn.LayerNorm(d_model)
        self.soc_ff    = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(),
                                       nn.Linear(d_model * 2, d_model), nn.Dropout(dropout))
        self.soh_ff    = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(),
                                       nn.Linear(d_model * 2, d_model), nn.Dropout(dropout))
        self.soc_norm2 = nn.LayerNorm(d_model)
        self.soh_norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        soc_rep: torch.Tensor,
        soh_rep: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """soc_rep, soh_rep: (B, d_model).  Returns updated (soc_rep, soh_rep)."""
        # Expand to (B, 1, d_model) for MultiheadAttention
        s = soc_rep.unsqueeze(1)
        h = soh_rep.unsqueeze(1)

        # SOC attends to SOH
        s_x, _ = self.soc_attn(s, h, h)
        s = self.soc_norm1(s + s_x)
        s = self.soc_norm2(s + self.soc_ff(s))

        # SOH attends to (updated) SOC
        h_x, _ = self.soh_attn(h, s, s)
        h = self.soh_norm1(h + h_x)
        h = self.soh_norm2(h + self.soh_ff(h))

        return s.squeeze(1), h.squeeze(1)


class DTAGBlock(nn.Module):
    """Dual-Task Adaptive Gating (DTAG).

    After the shared encoder produces a pooled representation `rep` of shape
    (B, d_model), DTAG routes it through two learned soft gates — one per
    task — before the separate NIG heads see it.  Each gate is a small
    two-layer MLP that outputs a sigmoid mask over d_model dimensions,
    effectively insulating SOC and SOH gradient paths from each other.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.soc_fc1 = nn.Linear(d_model, d_model // 2)
        self.soc_fc2 = nn.Linear(d_model // 2, d_model)
        self.soh_fc1 = nn.Linear(d_model, d_model // 2)
        self.soh_fc2 = nn.Linear(d_model // 2, d_model)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, rep: torch.Tensor):
        """rep: (B, d_model)  ->  (soc_rep, soh_rep) each (B, d_model)."""
        soc_gate = self.sigmoid(self.soc_fc2(self.relu(self.soc_fc1(rep))))
        soh_gate = self.sigmoid(self.soh_fc2(self.relu(self.soh_fc1(rep))))
        return rep * soc_gate, rep * soh_gate


class UAPIFormer(nn.Module):
    """Uncertainty-Aware Physics-Informed Transformer (UAPI-Former).

    Architecture (seven components; the paper's multi-seed ablation reports which ones
    measurably change accuracy):
        raw x  →  IC  →  IC feature vector  (B, d_model//4)   [IC]
        raw x  →  DCT  →  chemistry token   (B, 1, d_model)   [DCT]
        raw x  →  EITE  →  Positional Encoding                 [EITE]
               →  [DCT token ‖ sequence]   (B, T+1, d_model)
               →  4-layer Transformer Encoder
               →  PRAP (positions 1..T, physics-residual weighted)  [PRAP]
               →  DTAG  →  soc_rep, soh_rep  (B, d_model each)     [DTAG]
               →  CTBA  →  cross-task-updated soc_rep, soh_rep      [CTBA]
               →  SOC NIG head (soc_rep)                            [NIG]
               →  SOH NIG head (cat[soh_rep, IC features])          [IC+NIG]

    Parameter count: 1,103,112 trainable (d_model=128, 4 layers).

    in_channels=6: [V_norm, I, T_norm, V_ocv_norm, soc_from_ocv, Q_cum_norm]
    EITE zero-pads if fewer channels are provided (e.g. CALCE/Oxford with 3 channels).
    """

    def __init__(
        self,
        in_channels: int = 6,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        seq_len: int = 200,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dct          = DynamicChemistryToken(in_channels=in_channels, d_model=d_model)
        self.eite         = EITE(in_channels=in_channels, d_model=d_model)
        self.pos_enc      = PositionalEncoding(d_model, max_len=seq_len)
        encoder_layer     = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder      = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.prap         = PRAPPooling()
        self.dtag         = DTAGBlock(d_model)
        self.ic_extractor = ICFeatureExtractor(n_bins=20, d_out=d_model // 4)
        self.ctba         = CrossTaskAttentionBlock(d_model, nhead=nhead, dropout=dropout)
        self.soc_head     = EvidentialNIGHead(d_model, out_features=1)
        self.soh_head     = EvidentialNIGHead(d_model + d_model // 4, out_features=1)

    def forward(
        self,
        x: torch.Tensor,
        v_spme: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """x: (B, T, in_channels).

        Returns ((soc_gamma, soc_v, soc_alpha, soc_beta),
                 (soh_gamma, soh_v, soh_alpha, soh_beta))
        """
        # IC features from raw x (V, I channels) before any projection
        ic_feat = self.ic_extractor(x)                      # (B, d_model//4)

        # Physics residual magnitudes for PRAP (raw V channel, before EITE)
        residual_mag: Optional[torch.Tensor] = None
        if v_spme is not None:
            residual_mag = torch.abs(x[..., 0] - v_spme)   # (B, T)

        # Chemistry fingerprint from OCV relaxation window (raw x, before EITE)
        chem_token = self.dct(x).unsqueeze(1)               # (B, 1, d_model)

        x = self.eite(x, v_spme)                            # (B, T, d_model)
        x = self.pos_enc(x)

        # Prepend chemistry token as [CLS] at position 0
        x = torch.cat([chem_token, x], dim=1)              # (B, T+1, d_model)
        x = self.encoder(x)                                  # (B, T+1, d_model)

        # PRAP over measurement timesteps (positions 1..T; DCT token excluded)
        rep = self.prap(x[:, 1:, :], residual_mag)          # (B, d_model)

        # DTAG: route shared rep to task-specific gated representations
        soc_rep, soh_rep = self.dtag(rep)                   # each (B, d_model)

        # CTBA: bidirectional cross-task attention between SOC and SOH paths
        soc_rep, soh_rep = self.ctba(soc_rep, soh_rep)     # each (B, d_model)

        # IC concatenation: augment SOH representation with IC physics features
        soh_rep_full = torch.cat([soh_rep, ic_feat], dim=-1)  # (B, d_model + d_model//4)

        return self.soc_head(soc_rep), self.soh_head(soh_rep_full)
