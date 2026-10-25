"""Processes round 1 (R2.1): fetch the Stanford INR21700-M50T EV-profile subset and its capacities.

Source: Pozzato, Allam & Onori, "Lithium-ion battery aging dataset based on electric vehicle
real-driving profiles", Data in Brief 41 (2022) 107995; OSF project qsabn (view-only link given in
the article). Ten cells were aged with a UDDS-based discharge from 80 % to 20 % SOC at 23 C, with
C/20 capacity tests every ~25 cycles.

Subset: four cells with different charge rates, each with an early, a middle and a late cycling
block (about 25 aging cycles each), and the capacity tests that bracket every block plus the first.
Cycles completed at each capacity test are those of the README (sheet Capacity_test); the counts
agree with the block lengths of sheet Cycling_test.

Writes data/raw/stanford_m50t/{cycling/Cycling_<k>/<cell>.mat, capacity/Diag_<k>/<cell>.xlsx,
capacity.json, SHA256SUMS.txt}. Re-running skips files that are already present.
"""
import hashlib
import json
import os
import subprocess

import numpy as np

KEY = "view_only=2a03b6c78ef14922a3e244f3d549de78"
API = "https://api.osf.io/v2/nodes/qsabn/files/dropbox"
WB = "https://files.osf.io/v1/resources/qsabn/providers/dropbox"
OUT = "data/raw/stanford_m50t"

BLOCKS = {"W5": [1, 9, 14], "W8": [1, 10, 14], "G1": [1, 6, 10], "V4": [1, 6, 10]}
# cycles completed at capacity test Diag_k (README.xlsx, sheet Capacity_test)
DIAG_CYCLES = {
    "W5": {1: 0, 2: 25, 9: 219, 10: 244, 14: 344, 15: 369},
    "W8": {1: 0, 2: 25, 10: 222, 11: 247, 14: 322, 15: 347},
    "G1": {1: 0, 2: 25, 6: 87, 7: 112, 10: 187, 11: 212},
    "V4": {1: 0, 2: 20, 6: 120, 7: 145, 10: 219, 11: 244},
}


def curl(url, dst=None):
    args = ["curl", "-s", "-L", "--fail", "--retry", "5", url]
    if dst:
        subprocess.run(args + ["-o", dst], check=True)
        return None
    return json.loads(subprocess.run(args, check=True, capture_output=True, text=True).stdout)


def fetch(path, dst):
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        curl(f"{WB}{path}?{KEY}", dst + ".part")
        os.replace(dst + ".part", dst)


def c20_capacity(xlsx):
    """Discharged charge (Ah) of the longest C/20 (0.24 A) discharge step of a capacity test."""
    import openpyxl
    ws = next(w for w in openpyxl.load_workbook(xlsx, read_only=True, data_only=True).worksheets
              if w.title.startswith("Channel"))
    rows = ws.iter_rows(values_only=True)
    head = list(next(rows))
    ci, si, ti, qi = (head.index(k) for k in ("Current(A)", "Step_Index", "Test_Time(s)", "Discharge_Capacity(Ah)"))
    steps = {}
    for r in rows:
        if r[ci] is not None and -0.30 < r[ci] < -0.20:
            s = steps.setdefault(r[si], [r[ti], r[ti], r[qi], r[qi]])
            s[1], s[2], s[3] = r[ti], min(s[2], r[qi]), max(s[3], r[qi])
    t0, t1, q0, q1 = max(steps.values(), key=lambda s: s[1] - s[0])
    assert t1 - t0 > 15 * 3600, (xlsx, t1 - t0)   # a C/20 discharge of a 4.85 Ah cell lasts about 20 h
    return float(q1 - q0)


def main():
    cap = {}
    for cell, blocks in BLOCKS.items():
        for k in blocks:
            fetch(f"/cycling_tests/Cycling_{k}/_processed_mat/{cell}.mat", f"{OUT}/cycling/Cycling_{k}/{cell}.mat")
        caps = []
        for d in sorted(DIAG_CYCLES[cell]):
            dst = f"{OUT}/capacity/Diag_{d}/{cell}.xlsx"
            if not os.path.exists(dst):
                ls = curl(f"{API}/diagnostic_tests/Diag_{d}/Capacity_test/?{KEY}")["data"]
                name = [e["attributes"]["name"] for e in ls if f"_{cell}_Channel_" in e["attributes"]["name"]]
                assert len(name) == 1, (cell, d, name)
                fetch(f"/diagnostic_tests/Diag_{d}/Capacity_test/{name[0]}", dst)
            caps.append(c20_capacity(dst))
            print(cell, f"Diag_{d}", DIAG_CYCLES[cell][d], "cycles", round(caps[-1], 4), "Ah", flush=True)
        cyc = [DIAG_CYCLES[cell][d] for d in sorted(DIAG_CYCLES[cell])]
        assert np.all(np.diff(cyc) > 0)
        cap[cell] = {"cycles": cyc, "capacity_ah": caps,
                     "blocks": {f"Cycling_{k}": DIAG_CYCLES[cell][k] for k in blocks}}
    json.dump(cap, open(f"{OUT}/capacity.json", "w"), indent=1)
    with open(f"{OUT}/SHA256SUMS.txt", "w") as f:
        for root, _, files in os.walk(OUT):
            for n in sorted(files):
                if n.endswith((".mat", ".xlsx")):
                    p = os.path.join(root, n)
                    f.write(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {os.path.relpath(p, OUT)}\n")
    print("wrote", f"{OUT}/capacity.json")


if __name__ == "__main__":
    main()
