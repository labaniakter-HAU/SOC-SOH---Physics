"""Read-only CPU inference-latency benchmark for the canonical v12-MSE checkpoint.

Loads checkpoints/nasa_v12/best.pt (FP32 UAPI-Former), runs a single
200-timestep, 6-channel window through it on CPU, and times forward-pass
latency after a short warmup. Does not modify the checkpoint or any other
tracked artifact; writes a fresh results/inference_benchmark.json.

Usage:
    python scripts/benchmark_inference.py \
        --checkpoint checkpoints/nasa_v12/best.pt \
        --n-warmup 10 --n-runs 100
"""
import argparse
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from uapi_former.model import UAPIFormer


def get_cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            import subprocess
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                stderr=subprocess.DEVNULL, text=True, timeout=15,
            ).strip()
            if out:
                return out
        except Exception:
            pass
    return platform.processor() or platform.uname().processor or "unknown"


@torch.no_grad()
def benchmark(checkpoint_path: str, n_warmup: int, n_runs: int, batch_size: int = 1):
    device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    saved = ckpt.get("args", {})

    model = UAPIFormer(
        in_channels=saved.get("in_channels", 6),
        d_model=saved.get("d_model", 128),
        nhead=saved.get("nhead", 4),
        num_layers=saved.get("num_layers", 4),
        seq_len=saved.get("seq_len", 200),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    seq_len = saved.get("seq_len", 200)
    in_channels = saved.get("in_channels", 6)
    x = torch.randn(batch_size, seq_len, in_channels, dtype=torch.float32, device=device)

    # Warmup (not timed) — lets any lazy CPU kernel init / thread-pool spin-up settle.
    for _ in range(n_warmup):
        model(x, None)

    times_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model(x, None)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    times_ms = np.asarray(times_ms)
    return {
        "checkpoint": checkpoint_path,
        "precision": "FP32",
        "device": "cpu",
        "cpu_name": get_cpu_name(),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "in_channels": in_channels,
        "n_warmup": n_warmup,
        "n_runs": n_runs,
        "mean_ms": float(times_ms.mean()),
        "std_ms": float(times_ms.std()),
        "min_ms": float(times_ms.min()),
        "max_ms": float(times_ms.max()),
        "median_ms": float(np.median(times_ms)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "all_runs_ms": times_ms.tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/nasa_v12/best.pt")
    p.add_argument("--n-warmup", type=int, default=10)
    p.add_argument("--n-runs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--out", type=str, default="results/inference_benchmark.json")
    args = p.parse_args()

    result = benchmark(args.checkpoint, args.n_warmup, args.n_runs, args.batch_size)

    print(f"Checkpoint: {result['checkpoint']} ({result['precision']})")
    print(f"CPU: {result['cpu_name']}  (torch threads={result['torch_num_threads']})")
    print(f"Mean latency over {result['n_runs']} runs "
          f"(after {result['n_warmup']} warmup): "
          f"{result['mean_ms']:.3f} ms  (std {result['std_ms']:.3f} ms, "
          f"median {result['median_ms']:.3f} ms)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if os.path.exists(args.out):
        raise RuntimeError(f"{args.out} already exists — refusing to overwrite.")
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
