"""Held-out-cell (LOCO) seeds that the manuscript reports.

The table and figure generators read per-seed result files. Without this pin, a finished
seed-3 or seed-4 run would enter every table as soon as its file appears, while the text
still says three seeds. SP4 of the weakness-removal plan switched SEEDS to (0, 1, 2, 3, 4)
together with the text (2026-10-01).
"""
import os

SEEDS = (0, 1, 2, 3, 4)


def seed_files(pattern, root=""):
    """Paths of the per-seed files of SEEDS; `pattern` contains '{s}'. A missing file fails."""
    paths = [os.path.join(root, pattern.format(s=s)) for s in SEEDS]
    missing = [p for p in paths if not os.path.exists(p)]
    assert not missing, missing
    return paths


def seed_keys(d, prefix):
    """Keys '<prefix><seed>' of SEEDS in d, in seed order. A missing key fails."""
    keys = [f"{prefix}{s}" for s in SEEDS]
    missing = [k for k in keys if k not in d]
    assert not missing, missing
    return keys
