"""Baseline SOTA models reproduced LOCALLY for apples-to-apples comparison
against UAPI-Former, trained/evaluated on the exact same NASA data pipeline
and intra_cell_random split (see uapi_former/dataset.py, checkpoints/nasa_v12).

This file is NEW and does not modify uapi_former/model.py, train.py,
checkpoints/nasa_v12/best.pt, or results/ablation_v12.csv (Rule: never
degrade the floor).

=============================================================================
ARCHITECTURAL ASSUMPTIONS (CNN-BiLSTM) — read before citing any number
=============================================================================
The cited paper (Chen, Zhao, Zhang, Shu, Shen, Liu — IEEE TII 2023,
"State of charge and state of health estimation of lithium-ion batteries
using convolutional neural network and bidirectional long short-term
memory") reports architecture at a level (CNN front-end -> BiLSTM ->
FC heads) that does not fully specify: exact number of conv layers,
filter counts, kernel size, LSTM hidden size, or number of LSTM layers.
Absent those specifics, this implementation makes the following EXPLICIT,
DOCUMENTED assumptions (NOT to be stated as literature fact in the paper —
only ever describe the reproduced number as "our local reproduction under
matched split/pipeline", not as "the exact architecture of Chen et al."):

  1. Input channels: 3 (V_norm, I, T_norm) — the same physically-meaningful
     raw channels UAPI-Former uses BEFORE its SPMe/physics-augmentation
     (channels 3-5 of the 6-channel UAPI-Former tensor: V_ocv_norm,
     soc_from_ocv, Q_cum_norm are physics-derived features specific to
     UAPI-Former's own EITE design and are explicitly NOT given to this
     baseline, since a plain CNN-BiLSTM in the cited paper has no physics
     front-end to consume them, and soc_from_ocv in particular would leak
     a near-ground-truth SOC estimate — an unfair advantage unrelated to
     architecture).
  2. CNN front-end: 3 x (Conv1d -> BatchNorm1d -> ReLU) blocks over the
     temporal axis, channels 3 -> 32 -> 64 -> 128, kernel_size=5 (padding=2,
     "same"), with MaxPool1d(kernel=2) after the first two blocks only
     (temporal downsampling 200 -> 100 -> 50, a standard CNN-BiLSTM design
     pattern for this literature), plus Dropout(0.1) after each block.
  3. BiLSTM: single-layer, bidirectional, hidden_size=256 (-> 512-dim output
     per timestep after concatenating both directions). Single layer chosen
     (rather than stacking) to keep total parameter count in the same order
     of magnitude as UAPI-Former (~1.1M) rather than an arbitrarily larger
     network, which would make any RMSE gap uninterpretable (capacity vs.
     architecture confound).
  4. Read-out: last-timestep BiLSTM hidden state (standard many-to-one
     sequence regression), matching the label convention already used by
     NASABatteryDataset (SOC/SOH are defined at the LAST timestep of each
     200-sample window) — consistent with how the cited paper frames the
     task (predict SOC/SOH from a fixed-length input window).
  5. Heads: one shared FC trunk (512 -> 128, ReLU, Dropout) feeding two
     independent small MLP heads (128 -> 64 -> 1) for SOC and SOH
     respectively — i.e. co-estimation with a shared body and separate
     task heads, matching the cited paper's task framing (SOC and SOH
     estimated jointly from the same network).
  6. Parameter count: ~0.95-1.0M with the defaults below (printed at
     construction time by scripts/train_baseline.py) — same order of
     magnitude as UAPI-Former's 1,103,112 trainable parameters, chosen
     deliberately so neither model has a raw-capacity advantage.
  7. Loss / optimization: trained with the SAME weighted MSE objective
     UAPI-Former's canonical v12-MSE run uses (10 * MSE_SOC + 2 * MSE_SOH),
     same AdamW (lr=1e-3, weight_decay=1e-4) + CosineAnnealingLR schedule,
     same batch size (256) and epoch budget (400) — see train.py and
     scripts/train_baseline.py. This keeps the comparison about
     architecture, not about who got a better-tuned optimizer.

None of the above is claimed as "the" CNN-BiLSTM architecture from the
paper — it is our best-effort, standard-choices reconstruction used ONLY
to get a locally-reproduced, apples-to-apples number under the identical
NASA intra_cell_random split/pipeline that UAPI-Former uses. Report results
as "CNN-BiLSTM (reproduced, this work)" vs. "CNN-BiLSTM (reported, Chen et
al. 2023)" — never conflate the two numbers.
=============================================================================
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple


class CNN1DFeatureExtractor(nn.Module):
    """1D-CNN front-end: temporal feature extraction over (B, T, C_in).

    Three Conv1d blocks (channels C_in -> 32 -> 64 -> 128, kernel_size=5,
    "same" padding), BatchNorm1d + ReLU + Dropout after each block, with
    MaxPool1d(2) after the first two blocks (temporal downsampling only,
    channel count unaffected by pooling).
    """

    def __init__(self, in_channels: int = 3,
                 channels: Tuple[int, int, int] = (32, 64, 128),
                 kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2
        c1, c2, c3 = channels

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size, padding=pad),
            nn.BatchNorm1d(c1), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(c1, c2, kernel_size, padding=pad),
            nn.BatchNorm1d(c2), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(c2, c3, kernel_size, padding=pad),
            nn.BatchNorm1d(c3), nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out_channels = c3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C_in)  ->  (B, T', out_channels)."""
        x = x.transpose(1, 2)          # (B, C_in, T)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x.transpose(1, 2)       # (B, T', out_channels)


class CNNBiLSTM(nn.Module):
    """CNN-BiLSTM baseline for joint SOC/SOH estimation (Chen et al. 2023,
    IEEE TII, architecture reconstructed — see module docstring for the
    explicit, documented assumptions made where the citation under-specifies
    hyperparameters).

    Forward contract matches a plain (non-evidential) regression model:
        forward(x) -> (soc_pred, soh_pred), each shape (B,)
    where x: (B, T, in_channels), in_channels=3 by default
    ([V_norm, I, T_norm] — see dataset.py channel ordering).
    """

    def __init__(self, in_channels: int = 3, seq_len: int = 200,
                 cnn_channels: Tuple[int, int, int] = (32, 64, 128),
                 kernel_size: int = 5,
                 lstm_hidden: int = 256, lstm_layers: int = 1,
                 fc_hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.cnn = CNN1DFeatureExtractor(in_channels=in_channels, channels=cnn_channels,
                                          kernel_size=kernel_size, dropout=dropout)
        self.lstm = nn.LSTM(
            input_size=self.cnn.out_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=(dropout if lstm_layers > 1 else 0.0),
        )
        lstm_out_dim = lstm_hidden * 2  # bidirectional concat
        self.shared_fc = nn.Sequential(
            nn.Linear(lstm_out_dim, fc_hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.soc_head = nn.Sequential(
            nn.Linear(fc_hidden, fc_hidden // 2), nn.ReLU(),
            nn.Linear(fc_hidden // 2, 1),
        )
        self.soh_head = nn.Sequential(
            nn.Linear(fc_hidden, fc_hidden // 2), nn.ReLU(),
            nn.Linear(fc_hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.cnn(x)                 # (B, T', 128)
        out, _ = self.lstm(feat)           # (B, T', 2*lstm_hidden)
        last = out[:, -1, :]               # many-to-one: last timestep (matches SOC/SOH label convention)
        rep = self.shared_fc(last)         # (B, fc_hidden)
        soc = self.soc_head(rep).squeeze(-1)
        soh = self.soh_head(rep).squeeze(-1)
        return soc, soh


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


"""
=============================================================================
ARCHITECTURAL ASSUMPTIONS (PI-Transformer) — read before citing any number
=============================================================================
The cited paper (Gu, See, Li, Shan — Energy, vol 282, p.128860, 2023,
"Physics-informed transformer for state-of-charge and state-of-health
co-estimation of lithium-ion batteries", see docs/refs.bib:ref_pi_transformer)
reports a Transformer-encoder architecture with a physics-informed input
representation, but — like the CNN-BiLSTM citation above — does not fully
specify every hyperparameter needed for an exact reproduction: exact
encoder depth, d_model, nhead, feedforward dimension, positional encoding
scheme, or the precise construction of its "physics-informed" input
features. Absent those specifics, this implementation makes the following
EXPLICIT, DOCUMENTED assumptions (NOT to be stated as literature fact —
only ever describe the reproduced number as "PI-Transformer (reproduced,
this work, matched split)", NEVER as "PI-Transformer (reported, Gu et al.
2023, their own split)" — these two numbers must never be conflated):

  1. Physics-informed input construction (4 channels): [V_norm, I, T_norm,
     V_ocv_norm] — the same 3 raw physical channels given to CNN-BiLSTM,
     PLUS one additional physics-derived channel: V_ocv_norm (channel 4 of
     the 6-channel NASABatteryDataset tensor, an IR-drop-corrected
     per-timestep open-circuit-voltage estimate, V + R0*|I|, R0=0.15 Ohm —
     see uapi_former/dataset.py). This is deliberately chosen over the
     remaining two physics-derived channels UAPI-Former also has access to:
       - soc_from_ocv (channel 5): an OCV->SOC lookup-table evaluation.
         EXCLUDED because it is a direct table-based SOC estimate — feeding
         it to a SOC-estimation model is close to handing the model the
         label itself (a leakage risk far more severe than architecture
         differences), which would make any RMSE comparison meaningless.
       - Q_cum_norm (channel 6): windowed cumulative Coulomb-counted charge
         normalised by capacity. EXCLUDED for the same reason: coulomb
         counting is itself a competing SOC-estimation method, and its
         cumulative-sum construction is monotonic within a window and
         highly correlated with the SOC label by construction — an
         architecture-independent shortcut, not a fair physics prior.
     V_ocv_norm is kept precisely because it is one step further removed
     from the label: it is a raw IR-corrected voltage value (still in volts,
     not in SOC units), requiring the network to learn its own OCV->SOC
     relationship exactly as a real physics-informed Transformer would have
     to, rather than being handed a value already in SOC units or a
     monotonic partial-integral of the label's defining quantity. This is a
     defensible, literature-consistent reading of "physics-informed input"
     for this baseline: an OCV-family voltage feature is exactly the kind of
     input a physics-informed transformer for SOC estimation is expected to
     exploit (see e.g. equivalent-circuit-model OCV terms used across this
     literature), without crossing into label leakage. If a reviewer judges
     even V_ocv_norm too informative, the ablation in results/ablation_v12.csv
     already reports UAPI-Former's own performance with these channels
     removed, so the sensitivity is documented elsewhere.
  2. Input embedding: a single Linear(in_channels -> d_model) projection
     followed by LayerNorm(d_model) — the standard "token embedding" used
     by every Transformer-encoder time-series model in this literature
     (structurally analogous to UAPI-Former's own EITE projection in
     uapi_former/model.py, but REIMPLEMENTED LOCALLY in this file rather
     than imported, so this baseline has zero code dependency on the frozen
     canonical uapi_former/model.py).
  3. Positional encoding: standard fixed sinusoidal positional encoding
     (Vaswani et al. 2017), reimplemented locally in this file (same
     formula as uapi_former/model.py's PositionalEncoding, not imported).
     Chosen because the cited paper does not propose or describe a custom
     positional scheme, and sinusoidal PE remains the default, parameter-
     free choice for time-series Transformers in this literature (it also
     avoids adding a learned positional-embedding parameter block, which
     would otherwise inflate parameter count beyond the matched budget
     below for no architectural reason).
  4. Encoder: nn.TransformerEncoder, d_model=128, nhead=4,
     dim_feedforward=512 (4x d_model, the standard Vaswani et al. ratio),
     dropout=0.1, num_layers=5. num_layers is set ONE LAYER DEEPER than
     UAPI-Former's own 4-layer encoder (uapi_former/model.py) specifically
     to compensate for the fact that PI-Transformer, as reproduced here,
     has none of UAPI-Former's auxiliary physics/attention modules (no
     DCT chemistry token, no PRAP residual-weighted pooling, no DTAG
     gating, no CTBA cross-task attention, no IC dQ/dV branch). Without
     this compensation the reproduced PI-Transformer would sit at a much
     LOWER parameter count than UAPI-Former, which would bias any RMSE gap
     in UAPI-Former's favor for a capacity reason rather than an
     architecture reason — the opposite failure mode from over-inflating a
     baseline, but equally a violation of fair-comparison discipline.
  5. Read-out: last-timestep encoder output (standard many-to-one sequence
     regression), matching the SAME convention already used for the
     CNN-BiLSTM baseline in this project (see CNNBiLSTM docstring, point 4)
     and consistent with NASABatteryDataset's label convention (SOC/SOH
     defined at the LAST timestep of each 200-sample window). A CLS-token
     or mean-pooling read-out would be an equally defensible choice in the
     abstract, but last-timestep is used here to hold the read-out
     mechanism CONSTANT across all locally-reproduced baselines, isolating
     the encoder architecture (CNN-BiLSTM vs. Transformer) as the only
     thing that differs between the two comparisons.
  6. Heads: one shared FC trunk (d_model -> 128, ReLU, Dropout) feeding two
     independent MLP heads (128 -> 64 -> 1) for SOC and SOH respectively —
     IDENTICAL head design to CNNBiLSTM, again to hold everything but the
     sequence-encoder architecture constant.
  7. Parameter count: ~1.0-1.05M with the defaults below (printed at
     construction time by scripts/train_baseline_pi.py) — same order of
     magnitude as UAPI-Former's 1,103,112 trainable parameters and
     CNN-BiLSTM's ~0.95-1.0M, chosen deliberately so no model in the
     comparison has a raw-capacity advantage.
  8. Loss / optimization: identical to CNN-BiLSTM and UAPI-Former's
     canonical v12-MSE run — 10*MSE(SOC) + 2*MSE(SOH), AdamW (lr=1e-3,
     weight_decay=1e-4) + CosineAnnealingLR, batch size 256, epoch budget
     400 (see train.py, scripts/train_baseline.py, scripts/train_baseline_pi.py).

None of the above is claimed as "the" PI-Transformer architecture from Gu
et al. — it is our best-effort, standard-choices reconstruction used ONLY
to get a locally-reproduced, apples-to-apples number under the identical
NASA intra_cell_random split/pipeline that UAPI-Former and CNN-BiLSTM use.
Report results as "PI-Transformer (reproduced, this work, matched split)"
vs. "PI-Transformer (reported, Gu et al. 2023, their own split, SOC RMSE
~1.95%)" — never conflate the two numbers.
=============================================================================
"""


class SinusoidalPositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positional encoding (Vaswani et al. 2017).

    Reimplemented locally (same formula as uapi_former/model.py's
    PositionalEncoding) so this baseline module has zero import dependency
    on the frozen canonical uapi_former/model.py file.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)].to(x.device)


class PhysicsInformedInputEmbedding(nn.Module):
    """Token embedding: Linear(in_channels -> d_model) + LayerNorm.

    Standard Transformer time-series token embedding, structurally
    analogous to UAPI-Former's own EITE projection but reimplemented
    locally (no import from uapi_former/model.py). See PITransformer
    docstring, assumption #2.
    """

    def __init__(self, in_channels: int = 4, d_model: int = 128):
        super().__init__()
        self.proj = nn.Linear(in_channels, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class PITransformer(nn.Module):
    """PI-Transformer baseline for joint SOC/SOH estimation (Gu, See, Li,
    Shan — Energy, vol 282, p.128860, 2023, architecture reconstructed —
    see the module-level ARCHITECTURAL ASSUMPTIONS (PI-Transformer) block
    above for the explicit, documented assumptions made where the citation
    under-specifies hyperparameters).

    Forward contract matches a plain (non-evidential) regression model:
        forward(x) -> (soc_pred, soh_pred), each shape (B,)
    where x: (B, T, in_channels), in_channels=4 by default
    ([V_norm, I, T_norm, V_ocv_norm] — physics-informed, non-leaking input;
    see dataset.py channel ordering and assumption #1 above for why
    soc_from_ocv and Q_cum_norm are excluded).
    """

    def __init__(self, in_channels: int = 4, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 5, dim_feedforward: Optional[int] = None,
                 seq_len: int = 200, fc_hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        ff = dim_feedforward if dim_feedforward is not None else d_model * 4

        self.embedding = PhysicsInformedInputEmbedding(in_channels=in_channels, d_model=d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.shared_fc = nn.Sequential(
            nn.Linear(d_model, fc_hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.soc_head = nn.Sequential(
            nn.Linear(fc_hidden, fc_hidden // 2), nn.ReLU(),
            nn.Linear(fc_hidden // 2, 1),
        )
        self.soh_head = nn.Sequential(
            nn.Linear(fc_hidden, fc_hidden // 2), nn.ReLU(),
            nn.Linear(fc_hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.embedding(x)              # (B, T, d_model)
        h = self.pos_enc(h)                # (B, T, d_model)
        h = self.encoder(h)                # (B, T, d_model)
        last = h[:, -1, :]                 # many-to-one: last timestep (matches SOC/SOH label convention)
        rep = self.shared_fc(last)         # (B, fc_hidden)
        soc = self.soc_head(rep).squeeze(-1)
        soh = self.soh_head(rep).squeeze(-1)
        return soc, soh


if __name__ == "__main__":
    # Quick param-count sanity check against UAPI-Former's ~1.1M budget.
    m = CNNBiLSTM()
    n = count_params(m)
    print(f"CNNBiLSTM parameters: {n:,}")
    x = torch.randn(4, 200, 3)
    soc, soh = m(x)
    print("soc:", soc.shape, "soh:", soh.shape)

    m2 = PITransformer()
    n2 = count_params(m2)
    print(f"PITransformer parameters: {n2:,}")
    x2 = torch.randn(4, 200, 4)
    soc2, soh2 = m2(x2)
    print("soc:", soc2.shape, "soh:", soh2.shape)
