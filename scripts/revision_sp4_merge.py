"""SP4: merge the seed-3/4 ablation runs (results/revision/ablation_s34/) into the five-seed files.

Refuses to merge unless the re-run of Full seed 3 in the current environment reproduces the stored
Full seed-3 result (results/_multiseed_raw/full.json), because the variants' seeds 3-4 are
compared with those stored Full seeds by paired tests. The seed 0-2 entries are copied unchanged;
the three-seed files are archived to results/_archive_stale/sp4_three_seed/.
"""
import json
import os
import shutil

NEW = "results/revision/ablation_s34"
ARCH = "results/_archive_stale/sp4_three_seed"
TOL = 1e-3   # percentage points of RMSE

VARIANTS = {"no_eite": "results/_multiseed_raw/no_eite.json", "no_nig": "results/_multiseed_raw/no_nig.json",
            "no_dtag_ctba": "results/_multiseed_raw/no_dtag_ctba.json", "no_prap": "results/_multiseed_raw/no_prap.json",
            "no_dct": "results/_multiseed_raw/no_dct.json", "no_ctba": "results/_multiseed_raw/no_ctba.json",
            "no_ic": "results/_multiseed_raw/no_ic.json", "ctba_self": "results/revision/ctba_self.json"}


def main():
    full = {e["seed"]: e for e in json.load(open("results/_multiseed_raw/full.json"))}
    (chk,) = json.load(open(f"{NEW}/full_check_s3.json"))
    d = {k: abs(chk[k] - full[3][k]) for k in ("soc_rmse", "soh_rmse")}
    print("Full seed 3 re-run vs stored:", {k: f"{v:.2e}" for k, v in d.items()})
    assert max(d.values()) < TOL, "current environment does not reproduce the stored Full seed 3; do not pair"

    os.makedirs(ARCH, exist_ok=True)
    for v, path in VARIANTS.items():
        old = json.load(open(path))
        new = json.load(open(f"{NEW}/{v}.json"))
        assert sorted(e["seed"] for e in old) == [0, 1, 2], (v, [e["seed"] for e in old])
        assert sorted(e["seed"] for e in new) == [3, 4], (v, [e["seed"] for e in new])
        assert all(e["variant"] == v for e in old + new), v
        shutil.copy2(path, os.path.join(ARCH, os.path.basename(path)))
        merged = sorted(old + new, key=lambda e: e["seed"])
        json.dump(merged, open(path + ".tmp", "w"), indent=2)
        os.replace(path + ".tmp", path)
        print(f"{v}: seeds {[e['seed'] for e in merged]} -> {path}")


if __name__ == "__main__":
    main()
