"""Tests for the Oxford discharge-record loader (spec 2026-09-24 §5.3).

Run: python tests/test_oxford_relabel.py          (the real-data tests load the 266 MB .mat)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from uapi_former.dataset import OxfordDataset

N = 2520                      # samples of the synthetic discharge record, 1 s apart


def _cell():
    t0 = 738000.0
    def rec(q_end, v0, v1, n):
        return {"t": t0 + np.arange(n) / 86400.0, "q": np.linspace(0.0, q_end, n),
                "v": np.linspace(v0, v1, n), "T": np.full(n, 40.0)}
    return {"cyc0000": {"C1ch": rec(690.0, 2.72, 4.2, 2400), "C1dc": rec(-700.0, 4.19, 2.7, N)},
            "cyc0100": {"C1ch": rec(650.0, 2.72, 4.2, 2300), "C1dc": rec(-660.0, 4.19, 2.7, N)}}


def _empty():
    ds = OxfordDataset.__new__(OxfordDataset)
    ds.seq_len, ds.spme_cache, ds.samples, ds.meta = 200, None, [], []
    return ds


def test_synthetic_windows_labels_and_soh():
    ds = _empty()
    ds._load_cell_discharge(_cell(), "CellX")
    per = (N - 1 - 200) // 100 + 1                   # 1 Hz grid 0..N-2 s, stride 100
    assert len(ds.samples) == 2 * per, len(ds.samples)
    starts = [m[2] for m in ds.meta[:per]]
    soc = [s[2] for s in ds.samples[:per]]
    for st, y in zip(starts, soc):                    # SOC = 1 - Q_out / Q_dis at the window end
        assert abs(y - (1 - (st + 199) / (N - 1))) < 1e-5, (st, y)
    assert all(a > b for a, b in zip(soc, soc[1:]))   # decreasing within the discharge
    assert abs(ds.samples[0][3] - 690.0 / 740.0) < 1e-9          # SOH: charge capacity / max(first, 740)
    assert abs(ds.samples[per][3] - 650.0 / 740.0) < 1e-9
    cur = ds.samples[0][0][:, 1]
    assert abs(float(np.median(cur)) + 700.0 * 3.6 / (N - 1)) < 1e-3   # discharge current is negative
    assert ds.samples[0][0].shape == (200, 6) and np.isfinite(ds.samples[0][0]).all()


def test_unknown_record_rejected():
    try:
        OxfordDataset("data/raw/oxford", split="test", record="both")
    except ValueError:
        return
    raise AssertionError("record='both' was accepted")


def test_real_charge_record_reproduces_archived_loader():
    ds = OxfordDataset("data/raw/oxford", split="test", record="charge")
    soc = np.array([s[2] for s in ds.samples])
    assert len(ds) == 76 and 100 * soc.std(ddof=1) < 0.1, (len(ds), soc.std())


def test_real_discharge_record():
    ds = OxfordDataset("data/raw/oxford", split="test")
    soc = np.array([s[2] for s in ds.samples])
    soh = np.array([s[3] for s in ds.samples])
    cyc = [m[1] for m in ds.meta]
    assert ds.record == "discharge" and len(set(ds.sample_cell)) == 1
    assert len(set(cyc)) == 76, len(set(cyc))
    assert 0.0 <= soc.min() and soc.max() <= 1.0 and 100 * soc.std() > 20, (soc.min(), soc.max(), soc.std())
    for k in sorted(set(cyc)):
        s = soc[[c == k for c in cyc]]
        assert s[0] > 0.9 and s[-1] < 0.1 and np.all(np.diff(s) <= 0), (k, s[0], s[-1])
    charge = OxfordDataset("data/raw/oxford", split="test", record="charge")
    assert set(np.unique(soh)) <= {s[3] for s in charge.samples}   # SOH labels unchanged
    print("  test split:", len(ds), "windows,", len(set(cyc)), "cycles")


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for n in names:
        globals()[n]()
        print("PASS", n)
    print(f"all {len(names)} tests passed")
