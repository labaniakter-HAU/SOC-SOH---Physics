"""Revision R2.9: runtime of the post-hoc calibration stages on CPU.

Measures, on the same CPU used for the backbone latency (batch size 1):
  - S0 interval issue (scalar quantile lookup)           per window
  - S1 one-off discriminator fit (2000 full-batch Adam)  per deployment batch
  - S1 per-window weight evaluation + weighted quantile  per window
  - S2 one-off OT weight computation                     per deployment batch
using the clean LOCO B0005 fold of seed 0 (the independent-calibration protocol
every LOCO result in the paper uses): 1,061 calibration x 5,008 target latents,
read from results/revision/cache/loco_clean_s0_B0005.pt. An earlier version used the
submitted-protocol checkpoint checkpoints/loco_B0005 and its 1,926-window
calibration set, mixing protocols. Run on an otherwise idle machine.
Writes results/revision/runtime_s1.json
"""
import json, os, sys, time, platform
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from uapi_former import shift_conformal as sc
from uapi_former.dataset import NASABatteryDataset
from scripts.shift_conformal_study import load_model, collect


def timeit(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return np.array(ts)


def main():
    torch.set_num_threads(6)  # match results/inference_benchmark.json
    dev = torch.device("cpu")
    c = torch.load("results/revision/cache/loco_clean_s0_B0005.pt", map_location="cpu", weights_only=False)
    cal, tgt = c["cal"], c["tgt"]
    s_cal, _ = sc.nonconformity(cal["sg"], cal["sv"], cal["sa"], cal["sb"], cal["soc"])
    zc, zt = cal["rep"].float(), tgt["rep"].float()

    fit = timeit(lambda: sc._fit_discriminator(zc, zt), 5)
    lin, std = sc._fit_discriminator(zc, zt)
    with torch.no_grad():
        w_cal = torch.clamp(torch.exp(lin(std(zc)).squeeze(-1)) * len(zc) / len(zt), max=50.0)
    order = torch.argsort(s_cal); s_sorted = s_cal[order].double(); cw = torch.cumsum(w_cal[order].double(), 0)
    total = float(w_cal.sum())

    def per_window(i=[0]):
        z = zt[i[0] % len(zt)].unsqueeze(0); i[0] += 1
        with torch.no_grad():
            w_star = float(torch.clamp(torch.exp(lin(std(z)).squeeze()) * len(zc) / len(zt), max=50.0))
        idx = int(torch.searchsorted(cw, torch.tensor(0.9 * (total + w_star), dtype=torch.float64)))
        return float("inf") if idx >= len(s_sorted) else float(s_sorted[idx])

    per = timeit(per_window, 2000)
    q0 = sc.vanilla_q(s_cal, 0.9)
    s0 = timeit(lambda: q0 * 1.0, 2000)
    ot = timeit(lambda: sc.semirelaxed_transport_weights(zt, zc), 5)
    res = {"cpu": platform.processor(), "threads": torch.get_num_threads(),
           "fold": "clean LOCO, seed 0, B0005 (results/revision/cache/loco_clean_s0_B0005.pt)",
           "n_cal": len(zc), "n_target_batch": len(zt),
           "s1_discriminator_fit_s": {"mean": float(fit.mean()), "max": float(fit.max())},
           "s1_per_window_ms": {"mean": float(1e3 * per.mean()), "p95": float(1e3 * np.percentile(per, 95)), "max": float(1e3 * per.max())},
           "s0_per_window_ms": {"mean": float(1e3 * s0.mean())},
           "s2_batch_weights_s": {"mean": float(ot.mean())}}
    os.makedirs("results/revision", exist_ok=True)
    json.dump(res, open("results/revision/runtime_s1.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
