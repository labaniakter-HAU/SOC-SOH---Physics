"""Dataset classes for UAPI-Former.

Hierarchy:
    SyntheticDataset      — deterministic synthetic V/I/T sequences for smoke-tests
    MemmapDataset         — generic memmap loader (produced by scripts/make_memmap.py)
    NASABatteryDataset    — NASA Ames PCOE .mat files (B0005-B0028)
    CALCEDataset          — CALCE CS2/CX2 .csv files
    MITTRIDataset         — MIT-Stanford TRI "severson2019" .h5 file
    OxfordDataset         — Oxford Battery Degradation .mat files

All real dataset classes produce samples of shape:
    x        : (seq_len, in_channels)   where in_channels is 3 (V,I,T) or 5 (+ spme, residual)
    v_spme   : (seq_len,)               SPMe baseline voltage (zeros if cache not available)
    soc      : scalar float
    soh      : scalar float  (= capacity_now / capacity_nominal)

The 80/10/10 split is enforced by **cell ID**, not by cycle, to prevent leakage
(Hendriks et al. 2022; Rosa et al. arXiv:2407.14625).
"""
from __future__ import annotations

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Any


# ---------------------------------------------------------------------------
# NMC 18650 OCV-SOC lookup table (Li[NiMnCo]O2 / graphite)
# ---------------------------------------------------------------------------
# Validated against NASA B0005-B0018 cells (2 Ah, 2.7–4.2 V range).
# Replaces the old linear approximation which had ~26% error at SOC=0.5.
_NMC_SOC_PTS = np.array([0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                          0.60, 0.70, 0.80, 0.90, 0.95, 1.00], dtype=np.float32)
_NMC_OCV_PTS = np.array([2.70, 3.20, 3.45, 3.57, 3.64, 3.73, 3.80,
                          3.87, 3.94, 4.02, 4.11, 4.17, 4.20], dtype=np.float32)


def _ocv_to_soc_nmc(v_ocv: np.ndarray) -> np.ndarray:
    """Piecewise-linear OCV → SOC mapping for NMC 18650 chemistry."""
    return np.interp(v_ocv, _NMC_OCV_PTS, _NMC_SOC_PTS).astype(np.float32)


# ---------------------------------------------------------------------------
# LCO 18650 OCV-SOC lookup table (LiCoO2 / graphite)
# ---------------------------------------------------------------------------
# Validated against CALCE CS2 cells (Sony 18650, 1.35 Ah, 2.7–4.2 V range).
_LCO_SOC_PTS = np.array([0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                          0.60, 0.70, 0.80, 0.90, 0.95, 1.00], dtype=np.float32)
_LCO_OCV_PTS = np.array([2.70, 3.50, 3.57, 3.62, 3.66, 3.70, 3.75,
                          3.81, 3.87, 3.94, 4.05, 4.13, 4.20], dtype=np.float32)


def _ocv_to_soc_lco(v_ocv: np.ndarray) -> np.ndarray:
    """Piecewise-linear OCV → SOC mapping for LCO 18650 chemistry."""
    return np.interp(v_ocv, _LCO_OCV_PTS, _LCO_SOC_PTS).astype(np.float32)


# ---------------------------------------------------------------------------
# LFP/graphite 18650 OCV-SOC lookup table (A123 APR18650M1A, severson2019)
# ---------------------------------------------------------------------------
# General literature-informed LFP/graphite OCV shape (flat ~3.2-3.3V plateau
# across most of the SOC range with steep knees near 0%/100%, consistent with
# published A123/LFP characterisations, e.g. full-range OCV ~2.73V (0% SOC)
# to ~3.355V (100% SOC) reported for LFP/graphite pouch cells). Unlike the
# NMC/LCO tables above, this is NOT independently validated against these
# specific severson2019 cells (no per-cell rest-voltage data was available to
# calibrate against) -- it is an approximate curve, used and reported as such.
# LFP's OCV is intrinsically far flatter than NMC/LCO, so OCV-based SOC
# inversion is expected to be a weaker, noisier signal for this chemistry.
_LFP_SOC_PTS = np.array([0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                          0.60, 0.70, 0.80, 0.90, 0.95, 1.00], dtype=np.float32)
_LFP_OCV_PTS = np.array([2.70, 3.00, 3.15, 3.22, 3.25, 3.27, 3.29,
                          3.30, 3.31, 3.33, 3.36, 3.40, 3.50], dtype=np.float32)


def _ocv_to_soc_lfp(v_ocv: np.ndarray) -> np.ndarray:
    """Piecewise-linear OCV → SOC mapping for LFP/graphite 18650 chemistry."""
    return np.interp(v_ocv, _LFP_OCV_PTS, _LFP_SOC_PTS).astype(np.float32)


# ---------------------------------------------------------------------------
# ICA / DV feature engineering helpers
# ---------------------------------------------------------------------------

def compute_ica(V: np.ndarray, Q: np.ndarray, dv: float = 0.005) -> np.ndarray:
    """Incremental Capacity Analysis: dQ/dV on a uniform V grid.

    Args:
        V: voltage array (ascending, from CC charge)
        Q: corresponding capacity array (Ah)
        dv: voltage bin width

    Returns: dQ/dV sampled on the same V grid (NaN-filled outside range).
    """
    from scipy.signal import savgol_filter
    # smooth first to reduce noise
    if len(V) > 11:
        Q = np.asarray(savgol_filter(Q, window_length=11, polyorder=3))
    dqdv = np.gradient(Q, V)
    return dqdv # type: ignore


def compute_dv(V: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Differential Voltage Analysis: dV/dQ."""
    from scipy.signal import savgol_filter
    if len(Q) > 11:
        V = np.asarray(savgol_filter(V, window_length=11, polyorder=3))
    dvdq = np.gradient(V, Q)
    return dvdq # type: ignore


def extract_c20_segment(V: np.ndarray, I: np.ndarray, T: np.ndarray,
                        I_threshold: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return indices where |I| < I_threshold (roughly C/20 constant-current charge)."""
    mask = np.abs(I) < I_threshold
    return V[mask], I[mask], T[mask]


def pad_or_truncate(arr: np.ndarray, length: int) -> np.ndarray:
    if len(arr) >= length:
        return arr[:length].astype(np.float32)
    out = np.zeros(length, dtype=np.float32)
    out[: len(arr)] = arr
    return out


# ---------------------------------------------------------------------------
# Synthetic dataset (smoke-test)
# ---------------------------------------------------------------------------

class SyntheticDataset(Dataset):
    """Generate synthetic sequences of shape (seq_len, in_channels).

    in_channels=5 includes placeholder V_spme and residual channels.
    The returned tuple is (x, soc, soh) for in_channels=3 and
    (x, v_spme, soc, soh) for in_channels=5, matching the train.py contract.
    """

    def __init__(self, num_samples: int = 4096, seq_len: int = 200,
                 in_channels: int = 5, seed: int = 42, noise: float = 1e-2):
        rng = np.random.RandomState(seed)
        self.seq_len = seq_len
        self.in_channels = in_channels
        # base 3 channels: V in [3.4, 4.2], I in [-2, 2], T in [10, 45]
        V = (3.4 + 0.8 * rng.rand(num_samples, seq_len)).astype(np.float32)
        I = (rng.randn(num_samples, seq_len) * 1.0).astype(np.float32)
        T = (25.0 + 10.0 * rng.randn(num_samples, seq_len)).astype(np.float32)
        # SOC from mean voltage
        self.soc = np.clip((V.mean(axis=1) - 3.4) / 0.8 + noise * rng.randn(num_samples), 0.0, 1.0).astype(np.float32)
        self.soh = np.clip(0.85 + 0.15 * rng.rand(num_samples) + noise * rng.randn(num_samples), 0.5, 1.0).astype(np.float32)

        if in_channels >= 5:
            # synthetic SPMe: OCV polynomial ≈ V baseline
            V_spme = (3.4 + 0.8 * self.soc[:, None] * np.ones((num_samples, seq_len))).astype(np.float32)
            residual = V - V_spme
            self.X = np.stack([V, I, T, V_spme, residual], axis=-1)  # (N, T, 5)
            self.V_spme = V_spme  # (N, T)
        else:
            self.X = np.stack([V, I, T], axis=-1)
            self.V_spme = None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        x   = torch.from_numpy(self.X[index])
        soc = torch.tensor(self.soc[index], dtype=torch.float32)
        soh = torch.tensor(self.soh[index], dtype=torch.float32)
        if self.V_spme is not None:
            v_spme = torch.from_numpy(self.V_spme[index])
            return x, v_spme, soc, soh
        return x, soc, soh


# ---------------------------------------------------------------------------
# Generic memmap dataset (output of scripts/make_memmap.py)
# ---------------------------------------------------------------------------

class MemmapDataset(Dataset):
    """Wrap memmap .npy arrays: V.npy, I.npy, T.npy, soc.npy, soh.npy.

    Optionally loads V_spme.npy if available (output of scripts/precompute_spme.py
    applied after make_memmap.py).  Returns (x, v_spme, soc, soh) when the spme
    file is present, (x, soc, soh) otherwise.
    """

    def __init__(self, folder_path: str):
        required = ["V.npy", "I.npy", "T.npy", "soc.npy", "soh.npy"]
        for name in required:
            path = os.path.join(folder_path, name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required memmap file missing: {path}")

        self.V   = np.load(os.path.join(folder_path, "V.npy"),   mmap_mode="r")
        self.I   = np.load(os.path.join(folder_path, "I.npy"),   mmap_mode="r")
        self.T   = np.load(os.path.join(folder_path, "T.npy"),   mmap_mode="r")
        self.soc = np.load(os.path.join(folder_path, "soc.npy"), mmap_mode="r")
        self.soh = np.load(os.path.join(folder_path, "soh.npy"), mmap_mode="r")

        spme_path = os.path.join(folder_path, "V_spme.npy")
        self.V_spme = np.load(spme_path, mmap_mode="r") if os.path.exists(spme_path) else None

    def __len__(self):
        return self.V.shape[0]

    def __getitem__(self, index):
        X = np.stack([
            self.V[index].astype(np.float32),
            self.I[index].astype(np.float32),
            self.T[index].astype(np.float32),
        ], axis=-1)
        soc = torch.tensor(float(self.soc[index]), dtype=torch.float32)
        soh = torch.tensor(float(self.soh[index]), dtype=torch.float32)

        if self.V_spme is not None:
            v_spme = torch.from_numpy(self.V_spme[index].astype(np.float32))
            residual = torch.from_numpy((self.V[index] - self.V_spme[index]).astype(np.float32))
            x = torch.from_numpy(np.concatenate([X, v_spme.unsqueeze(-1).numpy(),
                                                  residual.unsqueeze(-1).numpy()], axis=-1))
            return x, v_spme, soc, soh

        return torch.from_numpy(X), soc, soh


# ---------------------------------------------------------------------------
# NASA Ames Battery Dataset (B0005–B0028)
# ---------------------------------------------------------------------------

class NASABatteryDataset(Dataset):
    """Load NASA Ames PCOE battery cells from .mat files.

    Directory layout expected:
        data/raw/nasa/
            B0005.mat
            B0006.mat
            ...
            B0028.mat (only NMC 18650 cells used: B0005-B0008, B0018, B0025-B0028)

    .mat files follow the PCOE format: each file contains a struct array
    `B0005` with fields `type`, `ambient_temperature`, `time`, `data`.
    The `data` field for `charge` cycles contains `Voltage_measured`,
    `Current_measured`, `Temperature_measured`, `Capacity`.

    SOH label: cycle discharge capacity / capacity of the FIRST valid discharge
    cycle of the same cell, clipped to [0.5, 1]. Every window of a cycle inherits
    that cycle's SOH, so 51-70 windows share one SOH target.
    SOC label: Coulomb-counted, 1 - Q_out / Q_cycle, taken at the window's LAST
    sample. The OCV-inverted SOC is an input channel, not the label.
    Split: pass split='train'/'val'/'test' — a random 70/15/15 split of each
    cell's discharge CYCLES under a shared seed, never a split by window.
    """

    NOMINAL_CAPACITY = 2.0      # Ah  (NASA 18650 cells, rated ~2 Ah)
    # B0005-B0018 all use ~1.9 A CC discharge (same protocol, same C-rate).
    # B0025-B0028 use ~1.0 A (different protocol), stored for cross-protocol eval.
    # ── Cross-cell split (split_mode="cross_cell", default) ──────────────────
    # Train on B0005+B0006, val on B0007, test on B0018.
    # Evaluates cross-cell generalization (harder than intra-cell).
    CELLS_TRAIN = ["B0005", "B0006"]
    CELLS_VAL   = ["B0007"]
    CELLS_TEST  = ["B0018"]
    # ── Intra-cell split (split_mode="intra_cell") ───────────────────────────
    # All four same-protocol cells; split by cycle fraction within each cell.
    # Sequential aging split: train on early life, test on late life.
    CELLS_ALL   = ["B0005", "B0006", "B0007", "B0018"]
    CYCLE_FRACS = {"train": (0.00, 0.70), "val": (0.70, 0.85), "test": (0.85, 1.00)}
    # ── Intra-cell random split (split_mode="intra_cell_random") ─────────────
    # Same cells but cycles randomly assigned 70/15/15.
    # No aging extrapolation — comparable to most published SOTA evaluations.
    CYCLE_RANDOM_SEED = 42
    SEQ_LEN_DEFAULT = 200
    # DC internal resistance (Ohm) used for IR-drop OCV correction. Class attribute
    # (not a constructor arg) so a sensitivity sweep can override it per-instance
    # via NASABatteryDataset.R0 = <value> without touching the loading logic.
    R0 = 0.15

    # ── Clean LOCO protocol (loco_split="random3", revision R2.2/R2.8) ────────
    # Source-cell discharge cycles are randomly partitioned at CYCLE level into
    # train / val (early stopping only) / cal (conformal calibration only).
    LOCO3_FRACS = (0.85, 0.925)

    @classmethod
    def loco_random3_positions(cls, cell: str, n_dc: int, seed: int = 0):
        """Discharge-cycle positions of `cell` for each clean-LOCO split."""
        rng = np.random.RandomState(seed + 17 * cls.CELLS_ALL.index(cell))
        perm = rng.permutation(n_dc)
        a, b = int(round(cls.LOCO3_FRACS[0] * n_dc)), int(round(cls.LOCO3_FRACS[1] * n_dc))
        return {"train": set(perm[:a].tolist()), "val": set(perm[a:b].tolist()),
                "cal": set(perm[b:].tolist())}

    def __init__(self, data_dir: str, split: str = "train",
                 seq_len: int = SEQ_LEN_DEFAULT, spme_cache: str | None = None,
                 split_mode: str = "cross_cell", held_out_cell: str = "B0005",
                 cells: list[str] | None = None,
                 ocv_pts: Optional[np.ndarray] = None, q_norm: str = "first_cycle",
                 loco_split: str = "block", loco_seed: int = 0):
        """ocv_pts: optional 13-knot NMC OCV table overriding _NMC_OCV_PTS.
        q_norm: "first_cycle" (original: Q_cum / measured first-cycle capacity)
        or "rated" (Q_cum / NOMINAL_CAPACITY, no label-derived quantity).
        loco_split: "block" (original 0-85% train / 85-100% val) or "random3"
        (cycle-level random train/val/cal, see LOCO3_FRACS)."""
        self.seq_len = seq_len
        self.spme_cache = self._load_spme_cache(spme_cache)
        self.samples: List[Tuple[np.ndarray, Optional[np.ndarray], float, float]] = []
        # (cell, discharge-cycle position, window start) per sample, NASA loaders only
        self.meta: List[Tuple[str, int, int]] = []
        if q_norm not in ("first_cycle", "rated"):
            raise ValueError(f"q_norm must be first_cycle/rated, got {q_norm!r}")
        self.q_norm = q_norm
        self._ocv_pts = None if ocv_pts is None else np.asarray(ocv_pts, dtype=np.float64)

        if cells is not None:
            if not cells:
                raise ValueError("cells must be a non-empty list or None")
            for cell in cells:
                mat_path = os.path.join(data_dir, f"{cell}.mat")
                if os.path.exists(mat_path):
                    self._load_cell(mat_path, cell, 0.0, 1.0)
                else:
                    print(f"[WARN] {mat_path} not found — skipping.")
            return

        allowed = ("train", "val", "test", "cal") if loco_split == "random3" else ("train", "val", "test")
        if split not in allowed:
            raise ValueError(f"split must be one of {allowed}, got {split!r}")
        if split_mode not in ("cross_cell", "intra_cell", "intra_cell_random", "loco"):
            raise ValueError(f"split_mode must be cross_cell/intra_cell/intra_cell_random/loco, got {split_mode!r}")

        if split_mode == "intra_cell":
            frac_start, frac_end = self.CYCLE_FRACS[split]
            for cell in self.CELLS_ALL:
                mat_path = os.path.join(data_dir, f"{cell}.mat")
                if os.path.exists(mat_path):
                    self._load_cell(mat_path, cell, frac_start, frac_end)
        elif split_mode == "intra_cell_random":
            rng = np.random.RandomState(self.CYCLE_RANDOM_SEED)
            for cell in self.CELLS_ALL:
                mat_path = os.path.join(data_dir, f"{cell}.mat")
                if os.path.exists(mat_path):
                    self._load_cell_random_split(mat_path, cell, split, rng)
        elif split_mode == "loco":
            if held_out_cell not in self.CELLS_ALL:
                raise ValueError(
                    f"held_out_cell={held_out_cell!r} not in CELLS_ALL={self.CELLS_ALL}"
                )
            if split == "test":
                # entire held-out cell is the test set
                mat_path = os.path.join(data_dir, f"{held_out_cell}.mat")
                if os.path.exists(mat_path):
                    self._load_cell(mat_path, held_out_cell, 0.0, 1.0)
            elif loco_split == "random3":
                for cell in self.CELLS_ALL:
                    if cell == held_out_cell:
                        continue
                    mat_path = os.path.join(data_dir, f"{cell}.mat")
                    if os.path.exists(mat_path):
                        self._load_cell(mat_path, cell, keep_positions_by_split=(split, loco_seed))
            else:
                # train on 0-85% of each non-held-out cell; val on 85-100%
                frac_start, frac_end = (0.0, 0.85) if split == "train" else (0.85, 1.0)
                for cell in self.CELLS_ALL:
                    if cell == held_out_cell:
                        continue
                    mat_path = os.path.join(data_dir, f"{cell}.mat")
                    if os.path.exists(mat_path):
                        self._load_cell(mat_path, cell, frac_start, frac_end)
        else:
            if split == "train":
                _cell_list = self.CELLS_TRAIN
            elif split == "val":
                _cell_list = self.CELLS_VAL
            else:
                _cell_list = self.CELLS_TEST
            for cell in _cell_list:
                mat_path = os.path.join(data_dir, f"{cell}.mat")
                if os.path.exists(mat_path):
                    self._load_cell(mat_path, cell)

        if not self.samples:
            raise RuntimeError(f"No samples loaded from {data_dir} for split={split}. "
                               "Download the NASA Ames dataset first (see data/README.md).")

    def _ocv_to_soc(self, v_ocv: np.ndarray) -> np.ndarray:
        if self._ocv_pts is None:
            return _ocv_to_soc_nmc(v_ocv)
        return np.interp(v_ocv, self._ocv_pts, _NMC_SOC_PTS).astype(np.float32)

    def _q_cap(self, cap_nominal: float) -> float:
        return self.NOMINAL_CAPACITY if self.q_norm == "rated" else cap_nominal

    @staticmethod
    def _load_spme_cache(cache_path):
        if cache_path is None or not os.path.exists(cache_path):
            return None
        try:
            from uapi_former.pybamm_wrapper import PyBaMMLookup
            return PyBaMMLookup.from_cache(cache_path)
        except Exception:
            return None

    def _load_cell(self, mat_path: str, cell_name: str,
                   start_cycle_frac: float = 0.0, end_cycle_frac: float = 1.0,
                   keep_positions_by_split: Optional[Tuple[str, int]] = None):
        try:
            # try mat73 first (v7.3 HDF5-based .mat)
            import mat73
            data = mat73.loadmat(mat_path)
        except Exception:
            try:
                from scipy.io import loadmat
                data = loadmat(mat_path, simplify_cells=True)
            except Exception as e:
                print(f"Warning: could not load {mat_path}: {e}")
                return

        cycles = data.get(cell_name, data.get(cell_name.lower(), None))
        if cycles is None:
            return

        if isinstance(cycles, dict) and "cycle" in cycles:
            cycles = cycles["cycle"]

        if not isinstance(cycles, list):
            cycles = [cycles]

        # ---- Process discharge cycles with 1-Hz resampling + sliding windows ----
        # NASA discharge cycles are sampled at ~18-19 s intervals; 200 raw points
        # cover the FULL discharge.  We resample to 1 Hz via the 'Time' field so
        # each cycle becomes ~3600 points, then extract non-overlapping seq_len
        # windows at different SOC levels (stride = seq_len // 2).
        # SOH = cycle discharge capacity / first-cycle capacity.

        # For intra-cell splits, keep only cycles in [start_cycle_frac, end_cycle_frac).
        # cap_nominal is still derived from the FIRST discharge cycle overall so SOH
        # is consistent regardless of which fraction is loaded.
        discharge_idx = [i for i, c in enumerate(cycles)
                         if isinstance(c, dict)
                         and str(c.get("type", "")).strip().lower() == "discharge"]
        n_dc = len(discharge_idx)
        keep_start = int(start_cycle_frac * n_dc)
        keep_end   = int(end_cycle_frac   * n_dc)
        keep_set   = set(discharge_idx[keep_start:keep_end])
        if keep_positions_by_split is not None:
            _split, _seed = keep_positions_by_split
            _pos = self.loco_random3_positions(cell_name, n_dc, _seed)[_split]
            keep_set = {discharge_idx[p] for p in _pos}
        pos_of = {ci: p for p, ci in enumerate(discharge_idx)}

        cap_nominal: Optional[float] = None
        first_cap_seen = False

        for i, cyc in enumerate(cycles):
            if not isinstance(cyc, dict):
                continue
            if str(cyc.get("type", "")).strip().lower() != "discharge":
                continue
            cyc_data = cyc.get("data", {})
            V_raw = np.asarray(cyc_data.get("Voltage_measured",    []), dtype=np.float64).ravel()
            I_raw = np.asarray(cyc_data.get("Current_measured",    []), dtype=np.float64).ravel()
            Tc_raw= np.asarray(cyc_data.get("Temperature_measured",[]), dtype=np.float64).ravel()
            t_raw = np.asarray(cyc_data.get("Time",                []), dtype=np.float64).ravel()
            cap   = np.asarray(cyc_data.get("Capacity",            []), dtype=np.float64).ravel()

            n = min(len(V_raw), len(I_raw), len(Tc_raw), len(t_raw))
            if n < 10:
                continue

            V_raw, I_raw, Tc_raw, t_raw = V_raw[:n], I_raw[:n], Tc_raw[:n], t_raw[:n]

            # Total discharge capacity for this cycle (Ah)
            cycle_cap = float(cap[-1]) if len(cap) > 0 else float(
                np.sum(np.abs(I_raw[:-1]) * np.diff(t_raw)) / 3600.0)
            if cycle_cap < 0.1:
                continue
            # cap_nominal is always from the first valid discharge cycle so SOH is
            # consistent even when only a fraction of cycles are loaded.
            if not first_cap_seen:
                cap_nominal = cycle_cap
                first_cap_seen = True
            soh = float(np.clip(cycle_cap / cap_nominal, 0.5, 1.0))

            # Skip cycles outside the requested fraction (intra-cell split)
            if i not in keep_set:
                continue

            # Coulomb-counting SOC on original time grid (uses actual dt)
            dt_arr    = np.diff(t_raw, prepend=t_raw[0])
            dt_arr[0] = 0.0
            Q_out     = np.cumsum(np.abs(I_raw) * dt_arr) / 3600.0
            soc_raw   = np.clip(1.0 - Q_out / (soh * cap_nominal), 0.0, 1.0)

            # Resample all signals to 1-Hz uniform grid using linear interpolation
            from scipy.interpolate import interp1d
            t_uniform = np.arange(t_raw[0], t_raw[-1], 1.0)   # 1-second grid
            if len(t_uniform) < self.seq_len:
                continue
            interp = lambda arr: interp1d(t_raw, arr, kind="linear", fill_value="extrapolate")(t_uniform)
            V_1hz   = np.clip(interp(V_raw),  2.5, 4.25).astype(np.float32)
            I_1hz   = interp(I_raw).astype(np.float32)
            Tc_1hz  = interp(Tc_raw).astype(np.float32)
            soc_1hz = np.clip(interp(soc_raw), 0.0, 1.0).astype(np.float32)
            N       = len(t_uniform)

            # Per-cycle voltage normalization: removes cell-specific DC offset.
            # All cells start discharge at ~4.2V, so V_norm[0]≈0 for every cell.
            # This makes the discharge profile cell-agnostic while preserving shape.
            V_start = float(V_1hz[0])

            # Slide seq_len windows with stride = seq_len // 2
            stride = max(self.seq_len // 2, 1)
            for start in range(0, N - self.seq_len + 1, stride):
                end = start + self.seq_len
                V_w   = V_1hz[start:end]
                I_w   = I_1hz[start:end]
                Tc_w  = Tc_1hz[start:end]
                soc_w = float(soc_1hz[end - 1])   # SOC at last timestep of window

                # V_norm = V - V_cycle_start removes inter-cell DC offset
                V_norm_w    = (V_w - V_start).astype(np.float32)
                R0 = self.R0
                # V_ocv_norm kept in cycle-relative coords for PRAP residual weighting
                V_ocv_norm  = np.clip(V_norm_w + R0 * np.abs(I_w), -2.0, 0.5).astype(np.float32)
                # Proper NMC OCV→SOC mapping (replaces linear approx that errs by ~26% at SOC=0.5)
                V_ocv_abs   = np.clip(V_w + R0 * np.abs(I_w), 2.70, 4.20)
                soc_from_ocv = self._ocv_to_soc(V_ocv_abs)
                # Coulomb counting within window — Q_cum(t) = ∫|I|dt / C_total
                # SOTA models explicitly feed this as a feature (z(t) = ∫i(τ)dτ/C)
                Q_cum_w     = np.cumsum(np.abs(I_w)) / 3600.0 / max(self._q_cap(cap_nominal), 0.1)
                Q_cum_norm_w = np.clip(Q_cum_w, 0.0, 0.5).astype(np.float32)
                # Temperature normalised to zero-mean unit-variance range
                Tc_norm_w   = ((Tc_w - 25.0) / 15.0).astype(np.float32)

                if self.spme_cache is not None:
                    soc_for_spme = np.clip(soc_1hz[start:end], 0.05, 0.95)
                    V_spme_arr = self.spme_cache.predict(soc_for_spme, I_w, Tc_w).astype(np.float32)
                    seq = np.stack([V_norm_w, I_w, Tc_norm_w, V_spme_arr,
                                    V_w - V_spme_arr, Q_cum_norm_w], axis=-1)
                    self.samples.append((seq, V_spme_arr, soc_w, soh))
                else:
                    # 6-channel: [V_norm, I, T_norm, V_ocv_norm, soc_from_ocv, Q_cum_norm]
                    seq = np.stack([V_norm_w, I_w, Tc_norm_w, V_ocv_norm,
                                    soc_from_ocv, Q_cum_norm_w], axis=-1)
                    self.samples.append((seq, None, soc_w, soh))
                self.meta.append((cell_name, pos_of[i], start))

    def _load_cell_random_split(self, mat_path: str, cell_name: str,
                                split: str, rng: np.random.RandomState):
        """Load cycles with random 70/15/15 assignment (seed shared across cells).

        Uses the same per-cycle feature engineering as _load_cell.  The rng is
        advanced by each cell in CELLS_ALL order, so splits are reproducible and
        consistent across train/val/test instantiations as long as the same rng
        seed and cell order are used.
        """
        try:
            import mat73
            data = mat73.loadmat(mat_path)
        except Exception:
            try:
                from scipy.io import loadmat
                data = loadmat(mat_path, simplify_cells=True)
            except Exception as e:
                print(f"Warning: could not load {mat_path}: {e}")
                return

        cycles = data.get(cell_name, data.get(cell_name.lower(), None))
        if cycles is None:
            return

        if isinstance(cycles, dict) and "cycle" in cycles:
            cycles = cycles["cycle"]
        if not isinstance(cycles, list):
            cycles = [cycles]

        discharge_idx = [i for i, c in enumerate(cycles)
                         if isinstance(c, dict)
                         and str(c.get("type", "")).strip().lower() == "discharge"]
        n_dc = len(discharge_idx)
        if n_dc == 0:
            return

        # Shuffle discharge-cycle positions (not raw cycle indices) with shared rng
        perm = rng.permutation(n_dc)
        n_train = int(0.70 * n_dc)
        n_val   = int(0.85 * n_dc)   # 70%–85% = val (15%)
        if split == "train":
            keep_positions = set(perm[:n_train].tolist())
        elif split == "val":
            keep_positions = set(perm[n_train:n_val].tolist())
        else:  # test
            keep_positions = set(perm[n_val:].tolist())

        # Map keep_positions (positions in discharge_idx) → raw cycle indices
        keep_set = {discharge_idx[p] for p in keep_positions}
        pos_of = {ci: p for p, ci in enumerate(discharge_idx)}

        # cap_nominal from the FIRST discharge cycle (position 0 in discharge_idx)
        cap_nominal: Optional[float] = None
        first_cap_seen = False

        for i, cyc in enumerate(cycles):
            if not isinstance(cyc, dict):
                continue
            if str(cyc.get("type", "")).strip().lower() != "discharge":
                continue
            cyc_data = cyc.get("data", {})
            V_raw = np.asarray(cyc_data.get("Voltage_measured",    []), dtype=np.float64).ravel()
            I_raw = np.asarray(cyc_data.get("Current_measured",    []), dtype=np.float64).ravel()
            Tc_raw= np.asarray(cyc_data.get("Temperature_measured",[]), dtype=np.float64).ravel()
            t_raw = np.asarray(cyc_data.get("Time",                []), dtype=np.float64).ravel()
            cap   = np.asarray(cyc_data.get("Capacity",            []), dtype=np.float64).ravel()

            n = min(len(V_raw), len(I_raw), len(Tc_raw), len(t_raw))
            if n < 10:
                continue
            V_raw, I_raw, Tc_raw, t_raw = V_raw[:n], I_raw[:n], Tc_raw[:n], t_raw[:n]

            cycle_cap = float(cap[-1]) if len(cap) > 0 else float(
                np.sum(np.abs(I_raw[:-1]) * np.diff(t_raw)) / 3600.0)
            if cycle_cap < 0.1:
                continue
            if not first_cap_seen:
                cap_nominal = cycle_cap
                first_cap_seen = True
            soh = float(np.clip(cycle_cap / cap_nominal, 0.5, 1.0))

            if i not in keep_set:
                continue

            dt_arr    = np.diff(t_raw, prepend=t_raw[0])
            dt_arr[0] = 0.0
            Q_out     = np.cumsum(np.abs(I_raw) * dt_arr) / 3600.0
            soc_raw   = np.clip(1.0 - Q_out / (soh * cap_nominal), 0.0, 1.0)

            from scipy.interpolate import interp1d
            t_uniform = np.arange(t_raw[0], t_raw[-1], 1.0)
            if len(t_uniform) < self.seq_len:
                continue
            interp = lambda arr: interp1d(t_raw, arr, kind="linear", fill_value="extrapolate")(t_uniform)
            V_1hz   = np.clip(interp(V_raw),  2.5, 4.25).astype(np.float32)
            I_1hz   = interp(I_raw).astype(np.float32)
            Tc_1hz  = interp(Tc_raw).astype(np.float32)
            soc_1hz = np.clip(interp(soc_raw), 0.0, 1.0).astype(np.float32)
            N       = len(t_uniform)
            V_start = float(V_1hz[0])

            # Use seq_len // 4 stride for intra_cell_random — doubles training samples vs //2
            stride = max(self.seq_len // 4, 1)
            for start in range(0, N - self.seq_len + 1, stride):
                end = start + self.seq_len
                V_w   = V_1hz[start:end]
                I_w   = I_1hz[start:end]
                Tc_w  = Tc_1hz[start:end]
                soc_w = float(soc_1hz[end - 1])

                V_norm_w    = (V_w - V_start).astype(np.float32)
                R0 = self.R0
                # V_ocv_norm in cycle-relative coords — used by PRAP for residual weighting
                V_ocv_norm  = np.clip(V_norm_w + R0 * np.abs(I_w), -2.0, 0.5).astype(np.float32)
                # Proper NMC OCV→SOC (replaces linear approx with ~26% error at SOC=0.5)
                V_ocv_abs   = np.clip(V_w + R0 * np.abs(I_w), 2.70, 4.20)
                soc_from_ocv = self._ocv_to_soc(V_ocv_abs)
                # Coulomb counting within window: SOTA models explicitly feed ∫|I|dt/C
                Q_cum_w     = np.cumsum(np.abs(I_w)) / 3600.0 / max(self._q_cap(cap_nominal), 0.1)
                Q_cum_norm_w = np.clip(Q_cum_w, 0.0, 0.5).astype(np.float32)
                # Temperature normalised to zero-mean
                Tc_norm_w   = ((Tc_w - 25.0) / 15.0).astype(np.float32)

                if self.spme_cache is not None:
                    soc_for_spme = np.clip(soc_1hz[start:end], 0.05, 0.95)
                    V_spme_arr = self.spme_cache.predict(soc_for_spme, I_w, Tc_w).astype(np.float32)
                    seq = np.stack([V_norm_w, I_w, Tc_norm_w, V_spme_arr,
                                    V_w - V_spme_arr, Q_cum_norm_w], axis=-1)
                    self.samples.append((seq, V_spme_arr, soc_w, soh))
                else:
                    # 6-channel: [V_norm, I, T_norm, V_ocv_norm, soc_from_ocv, Q_cum_norm]
                    seq = np.stack([V_norm_w, I_w, Tc_norm_w, V_ocv_norm,
                                    soc_from_ocv, Q_cum_norm_w], axis=-1)
                    self.samples.append((seq, None, soc_w, soh))
                self.meta.append((cell_name, pos_of[i], start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        seq, V_spme, soc, soh = self.samples[index]
        x   = torch.from_numpy(seq)
        soc = torch.tensor(soc, dtype=torch.float32)
        soh = torch.tensor(soh, dtype=torch.float32)
        if V_spme is not None:
            return x, torch.from_numpy(V_spme), soc, soh
        # Use ch3 (V_ocv_norm) as physics baseline for PRAP residual weighting
        ch = 3 if seq.shape[1] >= 4 else 0
        v_phys = torch.from_numpy(seq[:, ch])
        return x, v_phys, soc, soh


# ---------------------------------------------------------------------------
# CALCE Battery Dataset (CS2 / CX2)
# ---------------------------------------------------------------------------

class CALCEDataset(Dataset):
    """Load CALCE CS2 and CX2 batteries from CSV files.

    Directory layout expected:
        data/raw/calce/
            CS2_33/   (CSV files: CS2_33_1.csv, CS2_33_2.csv, ...)
            CS2_34/
            CS2_35/
            CS2_36/
            CX2_33/
            ...

    Each CSV has columns: Data_Point, Test_Time(s), Date_Time,
    Step_Time(s), Step_Index, Cycle_Index, Current(A), Voltage(V),
    Charge_Capacity(Ah), Discharge_Capacity(Ah), ...

    Split: by cell directory name (CS2_33/34 -> train, CS2_35 -> val, CS2_36/CX2_* -> test).
    """

    SPLIT_MAP = {
        "train": ["CS2_33", "CS2_34", "CX2_33", "CX2_34"],
        "val":   ["CS2_35"],
        "test":  ["CS2_36", "CX2_35", "CX2_36"],
    }
    NOMINAL_CAPACITY = 1.1  # Ah (LCO chemistry)
    R0 = 0.15  # DC internal resistance (Ohm); override per-instance for sensitivity sweeps
    SEQ_LEN_DEFAULT  = 200

    def __init__(self, data_dir: str, split: str = "train",
                 seq_len: int = SEQ_LEN_DEFAULT, spme_cache: str | None = None,
                 max_cycles_per_cell: int | None = None):
        self.seq_len = seq_len
        self.spme_cache = NASABatteryDataset._load_spme_cache(spme_cache)
        # Few-shot fine-tuning support: cap each cell to its first N *valid,
        # labelled* cycles (None = unlimited). Counted down in _load_csv, reset
        # per cell below since cycle numbering restarts per cell, not per file.
        self.max_cycles_per_cell = max_cycles_per_cell

        cells = self.SPLIT_MAP.get(split, [])
        self.samples: List[Tuple[np.ndarray, Optional[np.ndarray], float, float]] = []
        self.sample_cell: List[str] = []     # cell name of every sample (per-cell analyses)

        for cell in cells:
            cell_dir = os.path.join(data_dir, cell)
            if not os.path.isdir(cell_dir):
                continue
            n0 = len(self.samples)
            self._cycles_remaining = max_cycles_per_cell
            for csv_path in sorted(
                glob.glob(os.path.join(cell_dir, "*.csv")) +
                glob.glob(os.path.join(cell_dir, "*.xlsx")) +
                glob.glob(os.path.join(cell_dir, "*", "*.csv")) +
                glob.glob(os.path.join(cell_dir, "*", "*.xlsx"))
            ):
                if self._cycles_remaining is not None and self._cycles_remaining <= 0:
                    break
                self._load_csv(csv_path)
            self.sample_cell.extend([cell] * (len(self.samples) - n0))

    def _load_csv(self, csv_path: str):
        try:
            import pandas as pd
            if csv_path.endswith(".xlsx") or csv_path.endswith(".xls"):
                xl = pd.ExcelFile(csv_path)
                # CALCE xlsx layout: sheet 0 = 'Info' (metadata), sheet 1 = 'Channel_*' (waveform)
                data_sheet = xl.sheet_names[1] if len(xl.sheet_names) > 1 else xl.sheet_names[0]
                df = xl.parse(data_sheet, header=0)
            else:
                df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Warning: could not read {csv_path}: {e}")
            return

        # normalise column names
        col_map = {}
        for c in df.columns:
            cl = str(c).strip().lower().replace("(", "").replace(")", "").replace(" ", "_").replace("/", "_")
            col_map[c] = cl
        df = df.rename(columns=col_map)
        df = df.dropna(how="all")

        v_col    = next((c for c in df.columns if "voltage" in c), None)
        i_col    = next((c for c in df.columns if "current" in c), None)
        t_col    = next((c for c in df.columns if "test_time" in c
                         or (c.startswith("test") and "time" in c)
                         or (c == "times")), None) or \
                   next((c for c in df.columns if "time" in c and "step" not in c and "date" not in c), None)
        cap_col  = next((c for c in df.columns if "discharge_cap" in c), None) or \
                   next((c for c in df.columns if "charge_cap" in c), None) or \
                   next((c for c in df.columns if "capacity" in c), None)
        cyc_col  = next((c for c in df.columns if "cycle_index" in c or c == "cycle_index"), None) or \
                   next((c for c in df.columns if "cycle" in c), None)
        step_col = next((c for c in df.columns if "step_index" in c), None)

        if v_col is None or i_col is None or cyc_col is None:
            return

        cap_nominal = self.NOMINAL_CAPACITY
        first_cap_seen = False

        for cyc_id, grp in df.groupby(cyc_col):
            # Extract only the discharge phase (negative current, step 7 in CALCE)
            if step_col is not None:
                # Step 7 = CC discharge in standard CALCE protocol
                disc_grp = grp[grp[step_col] == 7]
                if len(disc_grp) < 10:
                    # Fallback: any step with net negative current
                    disc_grp = grp[grp[i_col] < -0.01]
            else:
                disc_grp = grp[grp[i_col] < -0.01]

            disc_grp = disc_grp.dropna(subset=[v_col, i_col])
            n = len(disc_grp)
            if n < 10:
                continue

            V_raw  = disc_grp[v_col].to_numpy(dtype=np.float64)
            I_raw  = disc_grp[i_col].to_numpy(dtype=np.float64)
            t_raw  = disc_grp[t_col].to_numpy(dtype=np.float64) if t_col else None

            # Discharge capacity from the Discharge_Capacity column
            cap_val = None
            if cap_col is not None:
                cap_s = disc_grp[cap_col].dropna()
                if len(cap_s):
                    cap_val = float(cap_s.iloc[-1])
            if cap_val is None or cap_val < 1e-6:
                # Integrate discharge current
                if t_raw is not None and len(t_raw) >= n:
                    dt_arr = np.diff(t_raw)
                    cap_val = float(np.sum(np.abs(I_raw[:-1]) * dt_arr) / 3600.0)
                else:
                    dt_nominal = 33.0
                    cap_val = float(np.abs(I_raw).mean() * n * dt_nominal / 3600.0)
            if cap_val < 0.1:
                continue

            if self._cycles_remaining is not None:
                if self._cycles_remaining <= 0:
                    break
                self._cycles_remaining -= 1

            if not first_cap_seen:
                cap_nominal = max(cap_val, self.NOMINAL_CAPACITY)
                first_cap_seen = True
            soh = float(np.clip(cap_val / cap_nominal, 0.5, 1.0))

            # Coulomb-counting SOC: starts at 1.0, decreases
            if t_raw is not None and len(t_raw) >= n:
                dt_arr = np.diff(t_raw, prepend=t_raw[0])
                dt_arr[0] = 0.0
            else:
                dt_arr = np.full(n, 33.0)
            Q_out   = np.cumsum(np.abs(I_raw) * dt_arr) / 3600.0
            soc_seq = np.clip(1.0 - Q_out / (soh * cap_nominal), 0.0, 1.0).astype(np.float32)

            # Resample to 1 Hz via linear interpolation over actual time
            if t_raw is not None and len(t_raw) >= n:
                from scipy.interpolate import interp1d
                t_uniform = np.arange(t_raw[0], t_raw[-1], 1.0)
                if len(t_uniform) < self.seq_len:
                    continue
                interp = lambda a: interp1d(t_raw[:n], a, kind="linear",
                                            fill_value="extrapolate")(t_uniform)
                V_1hz   = np.clip(interp(V_raw),  2.5, 4.25).astype(np.float32)
                I_1hz   = interp(I_raw).astype(np.float32)
                Tc_1hz  = np.full(len(t_uniform), 25.0, dtype=np.float32)
                soc_1hz = np.clip(interp(soc_seq), 0.0, 1.0).astype(np.float32)
                N       = len(t_uniform)
            else:
                # No time field: use raw arrays as-is and pad
                N       = n
                V_1hz   = np.clip(V_raw, 2.5, 4.25).astype(np.float32)
                I_1hz   = I_raw.astype(np.float32)
                Tc_1hz  = np.full(N, 25.0, dtype=np.float32)
                soc_1hz = soc_seq

            V_start = float(V_1hz[0])
            stride = max(self.seq_len // 2, 1)
            for start in range(0, N - self.seq_len + 1, stride):
                end   = start + self.seq_len
                V_w   = V_1hz[start:end]
                I_w   = I_1hz[start:end]
                Tc_w  = Tc_1hz[start:end]
                soc_w = float(soc_1hz[end - 1])

                V_norm_w     = (V_w - V_start).astype(np.float32)
                R0           = self.R0
                V_ocv_norm   = np.clip(V_norm_w + R0 * np.abs(I_w), -2.0, 0.5).astype(np.float32)
                V_ocv_abs    = np.clip(V_w + R0 * np.abs(I_w), 2.70, 4.20)
                soc_from_ocv = _ocv_to_soc_lco(V_ocv_abs)
                Q_cum_w      = np.cumsum(np.abs(I_w)) / 3600.0 / max(cap_nominal, 0.1)
                Q_cum_norm_w = np.clip(Q_cum_w, 0.0, 0.5).astype(np.float32)
                Tc_norm_w    = ((Tc_w - 25.0) / 15.0).astype(np.float32)

                if self.spme_cache is not None:
                    soc_for_spme = np.clip(soc_1hz[start:end], 0.05, 0.95)
                    V_spme = self.spme_cache.predict(soc_for_spme, I_w, Tc_w).astype(np.float32)
                    seq = np.stack([V_norm_w, I_w, Tc_norm_w, V_spme,
                                    V_w - V_spme, Q_cum_norm_w], axis=-1)
                    self.samples.append((seq, V_spme, soc_w, soh))
                else:
                    seq = np.stack([V_norm_w, I_w, Tc_norm_w, V_ocv_norm,
                                    soc_from_ocv, Q_cum_norm_w], axis=-1)
                    self.samples.append((seq, None, soc_w, soh))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return NASABatteryDataset.__getitem__(self, index) # type: ignore


# ---------------------------------------------------------------------------
# MIT-Stanford TRI "severson2019" Dataset
# ---------------------------------------------------------------------------

class MITTRIDataset(Dataset):
    """Load MIT-Stanford TRI LFP/graphite cells (Severson et al. 2019 fast-charging
    dataset, A123 APR18650M1A cells, ~1.1 Ah nominal) from the data.matr.io HDF5
    archive.

    All three canonical batch files
    (`2017-05-12_batchdata_updated_struct_errorcorrect.mat`,
    `2017-06-30_batchdata_updated_struct_errorcorrect.mat`,
    `2018-04-12_batchdata_updated_struct_errorcorrect.mat`) are MATLAB v7.3
    files, which ARE HDF5 despite the `.mat` extension. Each file's `batch`
    group is a MATLAB struct ARRAY stored field-first, not cell-keyed
    subgroups: `batch['cycle_life']`, `batch['summary']`, `batch['cycles']`,
    etc. are each shape (n_cells, 1) arrays of HDF5 object references, one
    per cell, requiring dereferencing via `f[ref]`. Within a cell,
    `cycles['V']` / `['I']` / `['T']` / `['Qd']` are themselves (n_cycles, 1)
    object-reference arrays, one raw time-series array per cycle (cycle
    index 0 is a placeholder/invalid rest cycle in this dataset and is
    skipped). `summary['QDischarge']` gives the per-cycle end-of-discharge
    capacity (Ah), used for SOH and to Coulomb-normalise the per-cycle `Qd`
    trace into SOC, exactly mirroring the Coulomb-counting convention used by
    `CALCEDataset`. String-typed metadata fields (`barcode`, `policy`,
    `policy_readable`, `Vdlin`) are not needed for V/I/T/SOC/SOH and are
    skipped entirely -- these are what make `mat73.loadmat` fail on this
    file, so the h5py-direct path below never touches them.

    Directory layout:
        data/raw/mit_tri/
            2017-05-12_batchdata_updated_struct_errorcorrect.mat
            2017-06-30_batchdata_updated_struct_errorcorrect.mat
            2018-04-12_batchdata_updated_struct_errorcorrect.mat

    Split: all cells across all 3 files are concatenated in filename order,
    then split 80/10/10 by cell index (proportional to however many valid
    cells are actually present, not a hardcoded count).
    """

    NOMINAL_CAPACITY = 1.1  # Ah (A123 APR18650M1A LFP/graphite cells)
    R0 = 0.15  # DC internal resistance (Ohm); override per-instance for sensitivity sweeps
    SEQ_LEN_DEFAULT  = 200

    def __init__(self, data_dir: str, split: str = "train",
                 seq_len: int = SEQ_LEN_DEFAULT, spme_cache: str | None = None):
        self.seq_len = seq_len
        self.spme_cache = NASABatteryDataset._load_spme_cache(spme_cache)
        self.samples: List[Tuple[np.ndarray, Optional[np.ndarray], float, float]] = []
        self.sample_cell: List[str] = []     # "<batch file>:<cell index>" of every sample

        all_files = sorted(glob.glob(os.path.join(data_dir, "*.h5"))) + \
                    sorted(glob.glob(os.path.join(data_dir, "*.mat")))
        if not all_files:
            raise RuntimeError(f"No .h5 or .mat files found in {data_dir}. "
                               "Download the MIT-TRI dataset first (see data/README.md).")

        import h5py
        hdf5_files, legacy_mat_files = [], []
        for path in all_files:
            try:
                with h5py.File(path, "r"):
                    hdf5_files.append(path)
            except OSError:
                legacy_mat_files.append(path)  # pre-v7.3 .mat, needs scipy/mat73

        if hdf5_files:
            self._load_from_h5(hdf5_files, split)
        elif legacy_mat_files:
            self._load_from_mat(legacy_mat_files[0], split)

    def _load_from_h5(self, h5_files: List[str], split: str):
        import h5py

        # Gather (file_path, cell_index) across ALL batch files in filename
        # order, so the train/val/test split is deterministic and covers
        # every batch rather than only the first file.
        all_cells: List[Tuple[str, int]] = []
        for h5_path in h5_files:
            with h5py.File(h5_path, "r") as f:
                n_cells = f["batch"]["cycle_life"].shape[0]
                all_cells.extend((h5_path, i) for i in range(n_cells))

        n = len(all_cells)
        split_idx = {"train": slice(0, int(0.8 * n)),
                     "val":   slice(int(0.8 * n), int(0.9 * n)),
                     "test":  slice(int(0.9 * n), n)}[split]

        # Group selected (file, cell_index) pairs by file so each file is
        # opened once regardless of how many of its cells fall in this split.
        by_file: dict = {}
        for h5_path, cell_idx in all_cells[split_idx]:
            by_file.setdefault(h5_path, []).append(cell_idx)

        for h5_path, cell_indices in by_file.items():
            with h5py.File(h5_path, "r") as f:
                batch = f["batch"]
                for cell_idx in cell_indices:
                    n0 = len(self.samples)
                    self._load_one_cell(f, batch, cell_idx, os.path.basename(h5_path))
                    self.sample_cell.extend(
                        [f"{os.path.basename(h5_path)}:{cell_idx}"] * (len(self.samples) - n0))

    def _load_one_cell(self, f, batch, cell_idx: int, file_label: str):
        """Dereference one cell's struct-array entry and append its per-cycle
        windows to self.samples. `f` is the open h5py.File (needed to
        dereference HDF5 object references); `batch` is f['batch']."""
        try:
            summary_grp = f[batch["summary"][cell_idx, 0]]
            qd_summary = np.asarray(summary_grp["QDischarge"][()], dtype=np.float64).ravel()
            cycles_grp = f[batch["cycles"][cell_idx, 0]]
            v_field  = cycles_grp["V"]
            i_field  = cycles_grp["I"]
            qd_field = cycles_grp["Qd"]
            t_field  = cycles_grp["T"] if "T" in cycles_grp else None
            n_cycles = v_field.shape[0]
        except Exception as e:
            print(f"Warning: skipping cell {file_label}[{cell_idx}] (struct read failed): {e}")
            return

        cap_nominal = self.NOMINAL_CAPACITY
        first_cap_seen = False

        # Cycle index 0 is a placeholder/invalid rest cycle in this dataset
        # (empty/degenerate V array); start from cycle 1.
        for cyc_idx in range(1, n_cycles):
            try:
                V = np.asarray(f[v_field[cyc_idx, 0]][()], dtype=np.float32).ravel()
                I = np.asarray(f[i_field[cyc_idx, 0]][()], dtype=np.float32).ravel()
                Qd = np.asarray(f[qd_field[cyc_idx, 0]][()], dtype=np.float64).ravel()
                if t_field is not None:
                    Tc = np.asarray(f[t_field[cyc_idx, 0]][()], dtype=np.float32).ravel()
                else:
                    Tc = np.full_like(V, 30.0, dtype=np.float32)
            except Exception:
                continue

            n_pts = min(len(V), len(I), len(Qd), len(Tc))
            if n_pts < 10:
                continue

            cap_val = float(qd_summary[cyc_idx]) if cyc_idx < len(qd_summary) else float(np.abs(Qd[-1]))
            if cap_val < 0.1:
                continue  # invalid/rest cycle, matches CALCEDataset's convention

            if not first_cap_seen:
                cap_nominal = max(cap_val, self.NOMINAL_CAPACITY)
                first_cap_seen = True
            soh = float(np.clip(cap_val / cap_nominal, 0.0, 1.0))

            # Coulomb-counted SOC from the cycle's own Qd trace (cumulative
            # discharge capacity), mirroring CALCEDataset's convention,
            # rather than a crude linear voltage-to-SOC approximation.
            V_c, I_c, Tc_c = V[:n_pts], I[:n_pts], Tc[:n_pts]
            Qd_pts = np.abs(Qd[:n_pts])
            soc_seq = np.clip(1.0 - Qd_pts / max(cap_nominal, 1e-6), 0.0, 1.0).astype(np.float32)

            if n_pts < self.seq_len:
                continue  # too short to form even one full window

            V_start = float(V_c[0])
            R0 = self.R0
            # Sliding windows across the cycle (stride = seq_len // 2), matching
            # CALCEDataset/NASABatteryDataset's convention -- NOT one truncated
            # window per cycle, which would mislabel SOC (see class docstring).
            stride = max(self.seq_len // 2, 1)
            for start in range(0, n_pts - self.seq_len + 1, stride):
                end = start + self.seq_len
                V_w, I_w, Tc_w = V_c[start:end], I_c[start:end], Tc_c[start:end]
                soc_w = float(soc_seq[end - 1])

                V_norm_w = (V_w - V_start).astype(np.float32)
                V_ocv_norm = np.clip(V_norm_w + R0 * np.abs(I_w), -2.0, 0.5).astype(np.float32)
                V_ocv_abs = np.clip(V_w + R0 * np.abs(I_w), 2.70, 3.50)
                soc_from_ocv = _ocv_to_soc_lfp(V_ocv_abs)
                Q_cum_w = np.cumsum(np.abs(I_w)) / 3600.0 / max(cap_nominal, 0.1)
                Q_cum_norm_w = np.clip(Q_cum_w, 0.0, 0.5).astype(np.float32)
                Tc_norm_w = ((Tc_w - 25.0) / 15.0).astype(np.float32)

                if self.spme_cache is not None:
                    soc_for_spme = np.clip(soc_seq[start:end], 0.05, 0.95)
                    V_spme = self.spme_cache.predict(soc_for_spme, I_w, Tc_w).astype(np.float32)
                    seq = np.stack([V_norm_w, I_w, Tc_norm_w, V_spme,
                                    V_w - V_spme, Q_cum_norm_w], axis=-1)
                    self.samples.append((seq, V_spme, soc_w, min(soh, 1.0)))
                else:
                    seq = np.stack([V_norm_w, I_w, Tc_norm_w, V_ocv_norm,
                                    soc_from_ocv, Q_cum_norm_w], axis=-1)
                    self.samples.append((seq, None, soc_w, min(soh, 1.0)))

    def _load_from_mat(self, mat_path: str, split: str):
        try:
            import mat73
            data = mat73.loadmat(mat_path)
        except Exception:
            from scipy.io import loadmat
            data = loadmat(mat_path, simplify_cells=True)

        batch = data.get("batch", {})
        cell_keys = list(batch.keys()) if isinstance(batch, dict) else []
        n = len(cell_keys)
        split_idx = {"train": slice(0, int(0.8 * n)),
                     "val":   slice(int(0.8 * n), int(0.9 * n)),
                     "test":  slice(int(0.9 * n), n)}[split]

        for key in cell_keys[split_idx]:
            cell = batch[key]
            if not isinstance(cell, dict):
                continue
            cycles = cell.get("cycles", {})
            for cyc_key in (cycles.keys() if isinstance(cycles, dict) else []):
                cyc = cycles[cyc_key]
                V  = np.asarray(cyc.get("V",  []), dtype=np.float32).ravel()
                I  = np.asarray(cyc.get("I",  []), dtype=np.float32).ravel()
                Tc = np.asarray(cyc.get("T",  []), dtype=np.float32).ravel()
                Qd = np.asarray(cyc.get("Qd", [self.NOMINAL_CAPACITY]), dtype=np.float32).ravel()
                soh = float(Qd[-1]) / self.NOMINAL_CAPACITY
                n_pts = min(len(V), len(I)) if len(I) else 0
                if n_pts < 10:
                    continue
                soc = float(np.clip((V[n_pts - 1] - 2.5) / (3.6 - 2.5), 0.0, 1.0))
                if len(Tc) == 0:
                    Tc = np.full(n_pts, 30.0, dtype=np.float32)
                V_p  = pad_or_truncate(V[:n_pts],  self.seq_len)
                I_p  = pad_or_truncate(I[:n_pts],  self.seq_len)
                Tc_p = pad_or_truncate(Tc[:n_pts], self.seq_len)
                seq  = np.stack([V_p, I_p, Tc_p], axis=-1)
                self.samples.append((seq, None, soc, min(soh, 1.0)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return NASABatteryDataset.__getitem__(self, index) # type: ignore


# ---------------------------------------------------------------------------
# Oxford Battery Degradation Dataset
# ---------------------------------------------------------------------------

class OxfordDataset(Dataset):
    """Load Oxford Battery Degradation Dataset from single .mat file.

    Actual layout (Oxford_Battery_Degradation_Dataset_1.mat):
        Top-level keys: Cell1 … Cell8
        Each Cell: dict keyed by cycle tag (cyc0000, cyc0100, …)
        Each cycle: dict with C1ch / C1dc sub-dicts
        Each C1ch: {'t': MATLAB-datenum days, 'v': voltage V,
                    'q': cumulative charge mAh, 'T': temperature °C}

    Current is derived as I = dq [mAh] * 3.6 / dt [s]  (→ Amperes).
    SOH = cycle_capacity_mAh / first_cycle_capacity_mAh.
    Split: cells 1-6 = train, cell 7 = val, cell 8 = test.

    record="discharge" (default, revision SP2): windows of each characterization cycle's
    1C discharge record (C1dc), resampled to 1 Hz, stride seq_len // 2, SOC Coulomb-counted
    over that discharge (1 at its start, 0 at the 2.7 V cut-off). record="charge": the
    archived loader, one window per cycle from the 1C charge record (C1ch) with SOC from the
    record's final voltage (constant near 100 %); kept only to reproduce archived results.
    SOH is the same in both: C1ch capacity / max(first C1ch capacity, 740 mAh).
    """

    NOMINAL_CAPACITY_MAH = 740.0  # mAh — confirmed from dataset
    SEQ_LEN_DEFAULT      = 200
    R0 = 0.15  # DC internal resistance (Ohm); override per-instance for sensitivity sweeps

    SPLIT_MAP = {
        "train": [f"Cell{i}" for i in range(1, 7)],
        "val":   ["Cell7"],
        "test":  ["Cell8"],
    }

    def __init__(self, data_dir: str, split: str = "train",
                 seq_len: int = SEQ_LEN_DEFAULT, spme_cache: str | None = None,
                 record: str = "discharge"):
        if record not in ("discharge", "charge"):
            raise ValueError(f"record must be 'discharge' or 'charge', got {record!r}")
        if record == "discharge" and spme_cache is not None:
            raise ValueError("SPMe features are built only by the archived charge-record loader")
        self.seq_len = seq_len
        self.record = record
        self.spme_cache = NASABatteryDataset._load_spme_cache(spme_cache)
        self.samples: List[Tuple[np.ndarray, Optional[np.ndarray], float, float]] = []
        self.meta: List[Tuple[str, str, int]] = []    # (cell, cycle key, window start); discharge only
        self.sample_cell: List[str] = []

        mat_files = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
        if not mat_files:
            raise RuntimeError(f"No .mat files found in {data_dir}. "
                               "Place Oxford_Battery_Degradation_Dataset_1.mat there.")

        mat_path = mat_files[0]
        try:
            import mat73
            data = mat73.loadmat(mat_path)
        except Exception:
            from scipy.io import loadmat
            data = loadmat(mat_path, simplify_cells=True)

        for cell_name in self.SPLIT_MAP.get(split, []):
            cell = data.get(cell_name)
            if cell is None or not isinstance(cell, dict):
                continue
            n0 = len(self.samples)
            if record == "discharge":
                self._load_cell_discharge(cell, cell_name)
            else:
                self._load_cell_cycles(cell)
            self.sample_cell += [cell_name] * (len(self.samples) - n0)

    def _load_cell_cycles(self, cell: dict):
        cap_nominal_mah = self.NOMINAL_CAPACITY_MAH
        first_cycle = True

        for cyc_key in sorted(cell.keys()):
            cyc = cell[cyc_key]
            if not isinstance(cyc, dict):
                continue
            # prefer C1ch (C/1-rate charge); fall back to C1dc
            c1 = cyc.get("C1ch") or cyc.get("C1dc")
            if c1 is None or not isinstance(c1, dict):
                continue

            v_arr = np.asarray(c1.get("v", c1.get("V", [])), dtype=np.float64).ravel()
            q_arr = np.asarray(c1.get("q", c1.get("Q", [])), dtype=np.float64).ravel()
            t_arr = np.asarray(c1.get("t", []),               dtype=np.float64).ravel()
            T_arr = np.asarray(c1.get("T", []),               dtype=np.float32).ravel()

            n = min(len(v_arr), len(q_arr), len(t_arr))
            if n < 10:
                continue

            v_arr, q_arr, t_arr = v_arr[:n], q_arr[:n], t_arr[:n]

            # Current from finite differences: q in mAh, t in MATLAB-datenum (days)
            dt_s  = np.diff(t_arr) * 86400.0          # days → seconds
            dq_mah= np.diff(q_arr)                     # mAh
            # I [A] = dq [mAh] * 3.6 / dt [s]
            safe_dt = np.where(np.abs(dt_s) > 1e-6, dt_s, 1.0)
            I_arr = np.clip(dq_mah * 3.6 / safe_dt, -10.0, 10.0).astype(np.float32)
            I_arr = np.append(I_arr, I_arr[-1] if len(I_arr) else 0.0)  # keep length n

            V_f = v_arr.astype(np.float32)
            if len(T_arr) < n:
                T_arr = np.full(n, 25.0, dtype=np.float32)

            # SOH from cycle capacity (q range, mAh)
            cycle_cap_mah = float(q_arr[-1] - q_arr[0])
            if first_cycle:
                cap_nominal_mah = max(cycle_cap_mah, self.NOMINAL_CAPACITY_MAH)
                first_cycle = False

            soh = float(np.clip(cycle_cap_mah / cap_nominal_mah, 0.5, 1.0))
            soc = float(np.clip((V_f[-1] - 3.5) / (4.2 - 3.5), 0.0, 1.0))

            V_p  = pad_or_truncate(V_f,         self.seq_len)
            I_p  = pad_or_truncate(I_arr,        self.seq_len)
            Tc_p = pad_or_truncate(T_arr[:n],    self.seq_len)

            V_start      = float(V_p[0])
            V_norm_p     = (V_p - V_start).astype(np.float32)
            R0           = self.R0
            V_ocv_norm_p = np.clip(V_norm_p + R0 * np.abs(I_p), -2.0, 0.5).astype(np.float32)
            V_ocv_abs_p  = np.clip(V_p + R0 * np.abs(I_p), 2.70, 4.20)
            soc_ocv_p    = _ocv_to_soc_lco(V_ocv_abs_p)
            Q_cum_p      = np.cumsum(np.abs(I_p)) / 3600.0 / max(self.NOMINAL_CAPACITY_MAH / 1000.0, 0.01)
            Q_cum_norm_p = np.clip(Q_cum_p, 0.0, 0.5).astype(np.float32)
            Tc_norm_p    = ((Tc_p - 25.0) / 15.0).astype(np.float32)

            if self.spme_cache is not None:
                soc_seq = np.clip((V_p - 3.5) / (4.2 - 3.5), 0.05, 0.95)
                V_spme  = self.spme_cache.predict(soc_seq, I_p, Tc_p).astype(np.float32)
                seq = np.stack([V_norm_p, I_p, Tc_norm_p, V_spme,
                                V_p - V_spme, Q_cum_norm_p], axis=-1)
                self.samples.append((seq, V_spme, soc, soh))
            else:
                seq = np.stack([V_norm_p, I_p, Tc_norm_p, V_ocv_norm_p,
                                soc_ocv_p, Q_cum_norm_p], axis=-1)
                self.samples.append((seq, None, soc, soh))

    def _load_cell_discharge(self, cell: dict, cell_name: str):
        """Windows of every characterization cycle's 1C discharge record (see class docstring).

        Features follow the CALCE loader: V_norm relative to the cycle's first sample, I (A,
        negative in discharge), T normalised, resistance-corrected OCV and its LCO SOC, and the
        window's Coulomb feature normalised by the rated 0.74 Ah. Label = SOC at the last sample.
        """
        cap_nominal_mah = None
        rated_ah = self.NOMINAL_CAPACITY_MAH / 1000.0
        for cyc_key in sorted(cell.keys()):
            cyc = cell[cyc_key]
            if not isinstance(cyc, dict):
                continue
            ch, dc = cyc.get("C1ch"), cyc.get("C1dc")
            if not isinstance(ch, dict) or not isinstance(dc, dict):
                continue
            q_ch = np.asarray(ch.get("q", []), dtype=np.float64).ravel()
            if len(q_ch) < 10:
                continue
            charge_cap = float(q_ch[-1] - q_ch[0])
            if cap_nominal_mah is None:
                cap_nominal_mah = max(charge_cap, self.NOMINAL_CAPACITY_MAH)
            soh = float(np.clip(charge_cap / cap_nominal_mah, 0.5, 1.0))

            v = np.asarray(dc.get("v", []), dtype=np.float64).ravel()
            q = np.asarray(dc.get("q", []), dtype=np.float64).ravel()
            t = np.asarray(dc.get("t", []), dtype=np.float64).ravel()
            T = np.asarray(dc.get("T", []), dtype=np.float64).ravel()
            n = min(len(v), len(q), len(t))
            if n < 10:
                continue
            T = T[:n] if len(T) >= n else np.full(n, 25.0)
            t_s = np.round((t[:n] - t[0]) * 86400.0, 3)          # datenum days -> s, ms grid
            order = np.argsort(t_s, kind="stable")
            t_s, v, q, T = t_s[order], v[:n][order], q[:n][order], T[order]
            keep = np.concatenate([[True], np.diff(t_s) > 0])
            t_s, v, q, T = t_s[keep], v[keep], q[keep], T[keep]
            q_out = np.abs(q - q[0])                              # mAh delivered since the start
            q_dis = float(q_out[-1])
            if q_dis < 1.0:
                continue
            soc = np.clip(1.0 - q_out / q_dis, 0.0, 1.0)
            dt = np.diff(t_s)
            cur = np.clip(np.diff(q) * 3.6 / np.where(dt > 1e-6, dt, 1.0), -10.0, 10.0)
            cur = np.append(cur, cur[-1])

            t_u = np.arange(0.0, t_s[-1], 1.0)
            if len(t_u) < self.seq_len:
                continue
            V_1 = np.clip(np.interp(t_u, t_s, v), 2.5, 4.25).astype(np.float32)
            I_1 = np.interp(t_u, t_s, cur).astype(np.float32)
            T_1 = np.interp(t_u, t_s, T).astype(np.float32)
            soc_1 = np.clip(np.interp(t_u, t_s, soc), 0.0, 1.0).astype(np.float32)
            V_start = float(V_1[0])
            stride = max(self.seq_len // 2, 1)
            for start in range(0, len(t_u) - self.seq_len + 1, stride):
                end = start + self.seq_len
                V_w, I_w, T_w = V_1[start:end], I_1[start:end], T_1[start:end]
                V_norm = (V_w - V_start).astype(np.float32)
                V_ocv_norm = np.clip(V_norm + self.R0 * np.abs(I_w), -2.0, 0.5).astype(np.float32)
                soc_ocv = _ocv_to_soc_lco(np.clip(V_w + self.R0 * np.abs(I_w), 2.70, 4.20))
                Q_cum = np.clip(np.cumsum(np.abs(I_w)) / 3600.0 / rated_ah, 0.0, 0.5).astype(np.float32)
                T_norm = ((T_w - 25.0) / 15.0).astype(np.float32)
                seq = np.stack([V_norm, I_w, T_norm, V_ocv_norm, soc_ocv, Q_cum], axis=-1)
                self.samples.append((seq, None, float(soc_1[end - 1]), soh))
                self.meta.append((cell_name, cyc_key, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return NASABatteryDataset.__getitem__(self, index) # type: ignore
