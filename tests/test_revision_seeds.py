"""Unit tests for scripts/revision_seeds.py.

Run: python tests/test_revision_seeds.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import revision_seeds as rs


def test_reported_seeds_are_five():
    assert rs.SEEDS == (0, 1, 2, 3, 4)


def test_seed_keys_ignores_extra_seeds():
    d = {f"loco_clean_s{s}": s for s in range(max(rs.SEEDS) + 3)}
    assert rs.seed_keys(d, "loco_clean_s") == [f"loco_clean_s{s}" for s in rs.SEEDS]


def test_seed_keys_missing_seed_fails():
    try:
        rs.seed_keys({"loco_clean_s0": 0}, "loco_clean_s")
    except AssertionError:
        return
    raise AssertionError("a missing seed was not detected")


def test_seed_files_ignores_extra_files():
    with tempfile.TemporaryDirectory() as t:
        for s in range(max(rs.SEEDS) + 3):
            open(os.path.join(t, f"x_s{s}.json"), "w").write("{}")
        got = rs.seed_files("x_s{s}.json", root=t)
        assert [os.path.basename(p) for p in got] == [f"x_s{s}.json" for s in rs.SEEDS]


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for n in names:
        globals()[n]()
        print("PASS", n)
    print(f"all {len(names)} tests passed")
