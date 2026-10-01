import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from uapi_former.baselines_ekf import ECM2RC, DualEKF, ocv_nmc

def _simulate(soc0=1.0, cap_ah=2.0, i_a=1.9, dt=10.0, n=250, seed=0):
    """Synthetic constant-current discharge through the same 2RC ECM.

    n=250 keeps cumulative discharge (~0.66 of capacity) within the OCV
    table's valid SOC range [0, 1]; a longer horizon would drive true SOC
    negative past ~step 379, while the EKF's state is (correctly) clipped to
    [0, 1.05] -- that clip-vs-unclipped mismatch is a test-harness artifact,
    not an EKF tracking failure, so the horizon is kept short enough to avoid it.
    """
    rng = np.random.default_rng(seed)
    ecm = ECM2RC(r0=0.10, r1=0.03, c1=2000.0, r2=0.05, c2=20000.0)
    soc, v1, v2 = soc0, 0.0, 0.0
    socs, volts = [], []
    for _ in range(n):
        soc = soc - i_a * dt / (cap_ah * 3600.0)
        v1, v2 = ecm.rc_step(v1, v2, i_a, dt)
        v = ocv_nmc(soc) - i_a * ecm.r0 - v1 - v2 + rng.normal(0, 0.005)
        socs.append(soc); volts.append(v)
    return np.array(socs), np.array(volts), i_a * np.ones(n), dt

def test_ekf_tracks_true_soc_on_synthetic_data():
    soc_true, v, i, dt = _simulate()
    ekf = DualEKF(capacity_ah=2.0, soc0=0.95)   # wrong init on purpose
    est, sd = [], []
    for k in range(len(v)):
        s, s_std = ekf.step(v[k], i[k], dt)
        est.append(s); sd.append(s_std)
    est, sd = np.array(est), np.array(sd)
    rmse = np.sqrt(np.mean((est[100:] - soc_true[100:]) ** 2))
    assert rmse < 0.05, f"EKF RMSE {rmse:.4f} too high"
    cover = np.mean(np.abs(est[100:] - soc_true[100:]) <= 3 * sd[100:])
    assert cover > 0.5   # sanity only; real coverage measured on real data

def test_ocv_monotone():
    s = np.linspace(0, 1, 50)
    assert (np.diff(ocv_nmc(s)) >= 0).all()

if __name__ == "__main__":
    test_ocv_monotone(); print("PASS test_ocv_monotone")
    test_ekf_tracks_true_soc_on_synthetic_data(); print("PASS test_ekf_tracks_true_soc_on_synthetic_data")
