"""SP4: fixed-R0 sensitivity (Tables S13, S14, Figure S3) for the five v12-MSE backbones and the
+-10% check for the five v12-NIG checkpoints.

Same evaluation as results/r0_sensitivity.json and results/r0_per_cell.json: R0 patched on the
dataset class (scripts/fit_r0_per_cell.eval_with_r0_classpatch), frozen model, whole NASA
random-split test set, on CPU. Before any seed is evaluated, the function is re-run on the former
canonical v12-MSE checkpoint and must reproduce both result files to 1e-4, so the per-seed values
come from the identical procedure. The per-cell R0 values are fitted from raw data
(results/r0_per_cell.json) and do not depend on the model.

Writes results/revision/seeds/r0_sensitivity_s{k}.json
"""
import json
import os
import sys

import torch

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import fit_r0_per_cell as F  # noqa: E402
from uapi_former.model import UAPIFormer  # noqa: E402

DEV = torch.device("cpu")
SWEEP = json.load(open("results/r0_sensitivity.json"))
PER_CELL = json.load(open("results/r0_per_cell.json"))


def load(path):
    c = torch.load(path, map_location="cpu", weights_only=False)
    a = c.get("args", {})
    a = vars(a) if hasattr(a, "__dict__") else a
    m = UAPIFormer(in_channels=a.get("in_channels", 6), d_model=a.get("d_model", 128), nhead=a.get("nhead", 4),
                   num_layers=a.get("num_layers", 4), seq_len=a.get("seq_len", 200))
    m.load_state_dict(c["model_state"])
    return m.eval()


def run(model, r0_values):
    return {f"{r:.6f}": F.eval_with_r0_classpatch(model, DEV, r) for r in r0_values}


def main():
    fitted = {c: v["r0_hat"] for c, v in PER_CELL["per_cell_fit"].items()}
    sweep_r0 = [v["r0"] for v in SWEEP.values()]
    # the procedure must reproduce the former canonical files before it is trusted per seed
    canon = run(load("checkpoints/nasa_v12/best.pt"), sweep_r0 + list(fitted.values()))
    for v in SWEEP.values():
        e = canon[f"{v['r0']:.6f}"]
        assert abs(e["soc_rmse"] - v["soc_rmse_pct"]) < 1e-4 and abs(e["soh_rmse"] - v["soh_rmse_pct"]) < 1e-4, v
    for c, r in fitted.items():
        e, ref = canon[f"{r:.6f}"], PER_CELL["eval_at_fitted_r0"][c]
        assert abs(e["soc_rmse"] - ref["soc_rmse"]) < 1e-4 and abs(e["soh_rmse"] - ref["soh_rmse"]) < 1e-4, c
    print("canonical files reproduced", flush=True)

    os.makedirs("results/revision/seeds", exist_ok=True)
    for k in range(5):
        path = f"results/revision/seeds/r0_sensitivity_s{k}.json"
        if os.path.exists(path):
            print("skip", path)
            continue
        mse = run(load(f"checkpoints/nasa_v12_seed{k}/best.pt"), sweep_r0 + list(fitted.values()))
        nig = run(load(f"checkpoints/nasa_v12_nig_seed{k}/best.pt"), [0.135, 0.15, 0.165])
        out = {"v12_mse": {"checkpoint": f"checkpoints/nasa_v12_seed{k}/best.pt",
                           "sweep": {name: {"r0": v["r0"], **mse[f"{v['r0']:.6f}"]} for name, v in SWEEP.items()},
                           "fitted": {c: {"r0": r, **mse[f"{r:.6f}"]} for c, r in fitted.items()}},
               "v12_nig": {"checkpoint": f"checkpoints/nasa_v12_nig_seed{k}/best.pt",
                           "eval": {f"{r:.3f}": nig[f"{r:.6f}"] for r in (0.135, 0.15, 0.165)}}}
        json.dump(out, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("wrote", path, {n: round(v["soc_rmse"], 3) for n, v in out["v12_mse"]["sweep"].items()}, flush=True)


if __name__ == "__main__":
    main()
