# Checkpoints

The trained models are not in Git. They are in the release asset `checkpoints.zip`; unpack it from the
repository root with `python scripts/release_assets.py unpack --from <folder with the assets>`.

Each `best.pt` is a PyTorch dictionary whose `model_state` holds the weights, saved with its training
settings; load it with `torch.load(path, map_location="cpu", weights_only=False)`. `{k}` is the seed
(0-4) and `{cell}` the held-out NASA cell (B0005, B0006, B0007, B0018).
`docs/revision_r1/FIGURE_SOURCES.md` lists the checkpoints the table script opens.

| Checkpoint | Model | Result files |
|---|---|---|
| `nasa_v12_seed{k}/` | UAPI-Former v12-MSE, NASA random split, five seeds; the backbones of the cross-chemistry analyses | `results/revision/seeds/chemistry_coverage_s{k}.json`, `eot_transfer_s{k}.json`, `r0_sensitivity_s{k}.json` |
| `nasa_v12_nig_seed{k}/` | UAPI-Former v12-NIG, NASA random split, five seeds | `results/revision/conformal_clean.json`, `results/revision/seeds/cross_protocol_acc.json`, `results/revision/seeds/r0_sensitivity_s{k}.json` |
| `loco_clean_s{k}_{cell}/` | UAPI-Former, leave-one-cell-out folds under the independent calibration protocol | `results/revision/loco_clean_s{k}.json`, `results/revision/conformal_clean.json` |
| `baseline_{cnn_bilstm,pi_transformer}_seed{k}_random_fixed/` | baselines, NASA random split | `results/revision/baselines_random_fixed.json` |
| `baseline_{cnn_bilstm,pi_transformer}_s{k}_{cell}/` | baselines, leave-one-cell-out folds | `results/revision/baselines_loco.json`, `results/revision/baseline_conformal_s{k}.json` |
| `baseline_{cnn_bilstm,pi_transformer}_s{k}_{cell}_fixed100/` | baselines trained for a fixed 100 epochs (training-rule sensitivity) | `results/revision/sensitivity/` |
| `sp4/{calce,oxford,mit_tri}_ft_s{k}/` | backbone `nasa_v12_seed{k}` fine-tuned on each target chemistry | `results/revision/seeds/chemistry_coverage_s{k}.json`, logs in `logs/revision/sp4/chem_s{k}/` |
| `sp4/calce_ft_fewshot{1,5,10,20}_s{k}/` | backbone `nasa_v12_seed{k}` fine-tuned on 1-20 labelled CALCE cycles per training cell | logs in `logs/revision/sp4/chem_s{k}/` |
| `sp4/eot_transducer_s{k}_{calce,oxford,mit_tri}.pt` | EOT alignment of the frozen backbone `nasa_v12_seed{k}` | `results/revision/seeds/eot_transfer_s{k}.json` |
| `nasa_v12/`, `nasa_v12_nig/` | single v12-MSE and v12-NIG models used for the runtime and storage measurements | `results/revision/inference_benchmark_idle*.json`, `results/revision/model_storage.json` |
