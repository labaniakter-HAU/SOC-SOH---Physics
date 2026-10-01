"""Model evaluation utilities: compute RMSE and ECE for saved checkpoints.

The functions here are write-once helpers used by the `scripts/calibrate.py`
and `scripts/evaluate.py` CLIs (both are added under `scripts/`). None of the
helpers execute training or heavy solvers by themselves.
"""
from typing import Dict, Any
import torch
from torch.utils.data import DataLoader

from .metrics import rmse, ece_from_nig
from .evidential import nig_predictive_variance


def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, device: torch.device = torch.device("cpu")) -> Dict[str, Any]:
    model.to(device)
    model.eval()
    soc_preds = []
    soc_vars = []
    soc_true = []
    soh_preds = []
    soh_vars = []
    soh_true = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                x, v_spme, soc, soh = batch
                v_spme = v_spme.to(device)
            else:
                x, soc, soh = batch
                v_spme = None
            x = x.to(device)
            (sg, sv, sa, sb), (hg, hv, ha, hb) = model(x, v_spme)
            # convert to CPU tensors for accumulation
            soc_preds.append(sg.detach().cpu())
            soc_vars.append(nig_predictive_variance(sv, sa, sb).detach().cpu())
            soc_true.append(soc)

            soh_preds.append(hg.detach().cpu())
            soh_vars.append(nig_predictive_variance(hv, ha, hb).detach().cpu())
            soh_true.append(soh)

    soc_preds = torch.cat([t.view(-1) for t in soc_preds])
    soc_vars = torch.cat([t.view(-1) for t in soc_vars])
    soc_true = torch.cat([t.view(-1) for t in soc_true])

    soh_preds = torch.cat([t.view(-1) for t in soh_preds])
    soh_vars = torch.cat([t.view(-1) for t in soh_vars])
    soh_true = torch.cat([t.view(-1) for t in soh_true])

    results = {}
    results["soc_rmse"] = rmse(soc_preds, soc_true)
    results["soh_rmse"] = rmse(soh_preds, soh_true)
    results["soc_mae"] = float(torch.mean(torch.abs(soc_preds - soc_true)).item())
    results["soh_mae"] = float(torch.mean(torch.abs(soh_preds - soh_true)).item())

    from .metrics import ece_from_mean_var
    ece_soc, covs_soc = ece_from_mean_var(soc_preds, soc_vars, soc_true)
    results["soc_ece"] = ece_soc
    results["soc_coverage"] = covs_soc

    ece_soh, covs_soh = ece_from_mean_var(soh_preds, soh_vars, soh_true)
    results["soh_ece"] = ece_soh
    results["soh_coverage"] = covs_soh

    # Store raw predictions for downstream plotting and conformal coverage
    results["soc_gamma"] = soc_preds
    results["soc_var"]   = soc_vars
    results["soc_true"]  = soc_true
    results["soh_gamma"] = soh_preds
    results["soh_var"]   = soh_vars
    results["soh_true"]  = soh_true

    return results
