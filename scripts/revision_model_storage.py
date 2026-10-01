"""Revision R2.9: serialized model size, FP32 versus INT8 dynamic quantization.

The paper's storage figures (FP32 versus INT8) had no result file behind them. This
script serializes the canonical checkpoint's state_dict in both forms with torch.save
into memory and records the byte counts. Size does not depend on machine load, unlike
the latency measurements in scripts/benchmark_inference.py and revision_runtime_s1.py.

INT8: torch.ao.quantization.quantize_dynamic on every nn.Linear (weights int8,
activations quantized on the fly); other layers stay FP32.

Writes results/revision/model_storage.json
"""
import io, json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.shift_conformal_study import load_model

CKPTS = ["checkpoints/nasa_v12/best.pt", "checkpoints/nasa_v12_nig/best_calibrated.pt"]


def nbytes(sd):
    buf = io.BytesIO()
    torch.save(sd, buf)
    return buf.getbuffer().nbytes


def main():
    torch.manual_seed(0)
    out = {"torch": torch.__version__, "method": "torch.save(state_dict) into memory; INT8 = quantize_dynamic({nn.Linear}, qint8)"}
    for ck in CKPTS:
        model, _ = load_model(ck, torch.device("cpu"))
        q = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        fp32, int8 = nbytes(model.state_dict()), nbytes(q.state_dict())
        out[ck] = {"n_params": int(sum(p.numel() for p in model.parameters())),
                   "fp32_bytes": fp32, "int8_bytes": int8,
                   "fp32_mb": fp32 / 1e6, "int8_mb": int8 / 1e6, "ratio": int8 / fp32}
        print(ck, out[ck], flush=True)
    os.makedirs("results/revision", exist_ok=True)
    json.dump(out, open("results/revision/model_storage.json", "w"), indent=2)
    print("wrote results/revision/model_storage.json")


if __name__ == "__main__":
    main()
