"""Unit tests for uapi_former/conformal_extras.py.

Run: python tests/test_conformal_extras.py   (pytest is not installed; the runner below calls every test_*)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from uapi_former.conformal_extras import (
    active_mask, subset, restrict_active, scores, summarize, cycle_order, reference_cycles,
    rolling_refs, cycle_median, label_gate,
)


def test_active_mask_uses_rest_window_threshold():
    m = active_mask(torch.tensor([0.0, 1e-7, 2e-6, 0.4]))
    assert m.tolist() == [False, False, True, True]


def test_subset_applies_mask_to_every_field():
    d = {"a": torch.arange(4), "b": torch.arange(4) * 10}
    s = subset(d, np.array([True, False, True, False]))
    assert s["a"].tolist() == [0, 2] and s["b"].tolist() == [0, 20]


def test_scores_absolute_is_residual_with_unit_sigma():
    d = {"sg": torch.tensor([0.5, 0.2]), "sv": torch.ones(2), "sa": torch.full((2,), 2.0),
         "sb": torch.ones(2), "soc": torch.tensor([0.4, 0.5])}
    s, sigma, y, g = scores(d, "soc", absolute=True)
    assert torch.allclose(s, torch.tensor([0.1, 0.3])) and torch.all(sigma == 1)


def test_summarize_counts_refusals_separately():
    y = torch.zeros(4); g = torch.zeros(4); sigma = torch.ones(4)
    q = torch.tensor([float("inf"), 1.0, 1.0, float("inf")])
    m = summarize(y, g, sigma, q)
    assert m["refusal"] == 0.5 and m["coverage"] == 1.0
    assert m["finite_and_correct"] == 0.5 and m["accepted_coverage"] == 1.0
    assert abs(m["mean_width_pp"] - 200.0) < 1e-9


def test_summarize_scalar_quantile_has_no_refusal():
    y = torch.tensor([0.0, 0.0, 0.3]); g = torch.zeros(3); sigma = torch.ones(3)
    m = summarize(y, g, sigma, 0.2)
    assert m["refusal"] == 0.0 and abs(m["coverage"] - 2 / 3) < 1e-6    # interval_metrics averages in float32


def test_cycle_order_is_chronological_by_cycle_number():
    idx, n = cycle_order(np.array(["B0005:10", "B0005:2", "B0005:10"], dtype=object))
    assert idx.tolist() == [1, 0, 1] and n == 2


def test_reference_schedule_and_rolling_window():
    refs = reference_cycles(60, 25)
    assert refs == [0, 25, 50]
    assert rolling_refs(30, refs, 2) == [0, 25]
    assert rolling_refs(25, refs, 5) == [0]          # strictly before the evaluated cycle
    assert rolling_refs(0, refs, 1) == []
    assert rolling_refs(55, refs, 1) == [50]


def test_cycle_median_groups_by_cycle():
    ids, med = cycle_median(np.array([1.0, 3.0, 10.0, 2.0]), np.array(["c:1", "c:1", "c:2", "c:1"], dtype=object))
    assert ids == ["c:1", "c:2"] and med.tolist() == [2.0, 10.0]


def test_label_gate_flags_intervals_reaching_below_floor():
    assert label_gate(np.array([0.60, 0.70, 0.69]), 0.693).tolist() == [True, False, True]


def test_restrict_active_filters_cal_tgt_and_aligned_arrays():
    cal = {"soc": torch.tensor([0.0, 0.5, 1e-7, 0.2]), "sg": torch.arange(4.0)}
    tgt = {"soc": torch.tensor([0.3, 0.0, 0.9]), "sg": torch.arange(3.0)}
    cyc = np.array(["B0005:1", "B0005:1", "B0005:2"])
    c, t, cy = restrict_active(cal, tgt, cyc)
    assert c["sg"].tolist() == [1.0, 3.0]
    assert t["sg"].tolist() == [0.0, 2.0]
    assert cy.tolist() == ["B0005:1", "B0005:2"]


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for n in names:
        globals()[n]()
        print("PASS", n)
    print(f"all {len(names)} tests passed")
