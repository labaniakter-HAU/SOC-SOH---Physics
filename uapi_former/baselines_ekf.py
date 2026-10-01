"""Classical control baseline: 2RC equivalent-circuit model + dual EKF.

State filter x=[SOC, V1, V2]; slow parameter filter for capacity (-> SOH =
C_est / C_nominal). Native uncertainty = EKF covariance; the paper reports its
3-sigma empirical coverage on the same shift hierarchy as UAPI-Former, as a
control showing classical covariance-based UQ also degrades under shift.
OCV curve reuses the 13-point NMC table from uapi_former.dataset.
"""
from __future__ import annotations
import numpy as np

from uapi_former.dataset import _NMC_OCV_PTS, _NMC_SOC_PTS

C_NOMINAL_AH = 2.0   # NASA 18650 rated capacity (matches dataset convention)


def ocv_nmc(soc):
    return np.interp(soc, _NMC_SOC_PTS, _NMC_OCV_PTS)


def docv_dsoc(soc, eps=1e-3):
    return (ocv_nmc(np.asarray(soc) + eps) - ocv_nmc(np.asarray(soc) - eps)) / (2 * eps)


class ECM2RC:
    def __init__(self, r0=0.10, r1=0.03, c1=2000.0, r2=0.05, c2=20000.0):
        self.r0, self.r1, self.c1, self.r2, self.c2 = r0, r1, c1, r2, c2

    def rc_step(self, v1, v2, i_a, dt):
        a1, a2 = np.exp(-dt / (self.r1 * self.c1)), np.exp(-dt / (self.r2 * self.c2))
        return (a1 * v1 + self.r1 * (1 - a1) * i_a,
                a2 * v2 + self.r2 * (1 - a2) * i_a)


class DualEKF:
    """Joint SOC (fast EKF) + capacity (slow scalar EKF) estimator.

    Sign convention: discharge current i_a > 0 (PCOE discharge cycles report
    positive current magnitude; verify against the raw file in eval script).
    """

    def __init__(self, capacity_ah=C_NOMINAL_AH, soc0=1.0, ecm=None,
                 q_soc=1e-7, q_rc=1e-6, r_meas=1e-3, q_cap=1e-8):
        self.ecm = ecm or ECM2RC()
        self.x = np.array([soc0, 0.0, 0.0])
        self.P = np.diag([0.05, 1e-4, 1e-4])
        self.Q = np.diag([q_soc, q_rc, q_rc])
        self.R = r_meas
        self.cap = capacity_ah
        self.p_cap = 0.01
        self.q_cap = q_cap

    def step(self, v_meas, i_a, dt):
        soc, v1, v2 = self.x
        # predict
        soc_p = np.clip(soc - i_a * dt / (self.cap * 3600.0), 0.0, 1.05)
        v1_p, v2_p = self.ecm.rc_step(v1, v2, i_a, dt)
        a1 = np.exp(-dt / (self.ecm.r1 * self.ecm.c1))
        a2 = np.exp(-dt / (self.ecm.r2 * self.ecm.c2))
        F = np.diag([1.0, a1, a2])
        P = F @ self.P @ F.T + self.Q
        # update
        v_pred = ocv_nmc(soc_p) - i_a * self.ecm.r0 - v1_p - v2_p
        H = np.array([docv_dsoc(soc_p), -1.0, -1.0])
        S = H @ P @ H + self.R
        K = P @ H / S
        innov = v_meas - v_pred
        self.x = np.array([soc_p, v1_p, v2_p]) + K * innov
        self.x[0] = np.clip(self.x[0], 0.0, 1.05)
        self.P = (np.eye(3) - np.outer(K, H)) @ P
        # slow capacity filter driven by the same innovation through dV/dC
        dv_dcap = docv_dsoc(soc_p) * (i_a * dt / (self.cap ** 2 * 3600.0))
        p_cap = self.p_cap + self.q_cap
        s_cap = dv_dcap * p_cap * dv_dcap + self.R
        k_cap = p_cap * dv_dcap / s_cap
        self.cap = float(np.clip(self.cap + k_cap * innov, 0.5, 2.5))
        self.p_cap = (1 - k_cap * dv_dcap) * p_cap
        return float(self.x[0]), float(np.sqrt(max(self.P[0, 0], 1e-12)))

    @property
    def soh(self):
        return self.cap / C_NOMINAL_AH

    @property
    def soh_std(self):
        return float(np.sqrt(max(self.p_cap, 1e-12)) / C_NOMINAL_AH)


def load_discharge_cycles(mat_path, cell_name):
    """Parse PCOE .mat discharge cycles -> list of dicts with V/I/T/t arrays and
    per-cycle capacity. Mirrors NASABatteryDataset._load_cell's parsing
    (mat73 first, scipy fallback)."""
    try:
        import mat73
        data = mat73.loadmat(mat_path)
    except Exception:
        from scipy.io import loadmat
        data = loadmat(mat_path, simplify_cells=True)
    cyc = data[cell_name]["cycle"]
    out = []
    for c in cyc:
        if str(c.get("type", "")) != "discharge":
            continue
        d = c["data"]
        keys = {k.lower(): k for k in d.keys()}
        def get(name):
            return d[keys[name.lower()]]
        out.append({
            "V": np.asarray(get("Voltage_measured"), dtype=float).ravel(),
            "I": np.asarray(get("Current_measured"), dtype=float).ravel(),
            "T": np.asarray(get("Temperature_measured"), dtype=float).ravel(),
            "t": np.asarray(get("Time"), dtype=float).ravel(),
            "capacity": float(np.ravel(d.get(keys.get("capacity", "Capacity"),
                                             np.array([np.nan])))[0]),
        })
    return out
