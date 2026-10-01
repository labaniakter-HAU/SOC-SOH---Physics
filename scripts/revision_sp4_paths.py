"""File locations of the cross-chemistry chain for each NASA backbone (SP4, five seeds).

seed=None is the former single-backbone lineage (checkpoints/nasa_v12/best.pt), kept only so
the scripts still run on it; the manuscript reads the five seed lineages, whose backbones are
the five-seed v12-MSE campaign (checkpoints/nasa_v12_seed{s}, 200 epochs, final weights). The
NASA random split is fixed (NASABatteryDataset.CYCLE_RANDOM_SEED), so every backbone shares the
same training, validation and test cycles and the same clean calibration half.
"""
DATASETS = ("calce", "oxford", "mit_tri")
FEWSHOT_CYCLES = (1, 5, 10, 20)

_CANON_ZS = {"calce": "logs/_leakfix_campaign/12_calce_zeroshot_eval.txt",
             "oxford": "logs/revision/sp2/ox_zeroshot.log",
             "mit_tri": "logs/_mit_tri/01_zeroshot_eval.txt"}
_CANON_FT = {"calce": "logs/_leakfix_campaign/14_calce_finetuned_eval.txt",
             "oxford": "logs/revision/sp2/ox_ft_eval.log",
             "mit_tri": "logs/_mit_tri/03_finetuned_eval.txt"}


def backbone(s):
    return "checkpoints/nasa_v12/best.pt" if s is None else f"checkpoints/nasa_v12_seed{s}/best.pt"


def ft_dir(ds, s):
    return f"checkpoints/{ds}_ft" if s is None else f"checkpoints/sp4/{ds}_ft_s{s}"


def fewshot_dir(c, s):
    return f"checkpoints/calce_ft_fewshot{c}" if s is None else f"checkpoints/sp4/calce_ft_fewshot{c}_s{s}"


def eot_transducer(ds, s):
    return f"checkpoints/eot_transducer_{ds}.pt" if s is None else f"checkpoints/sp4/eot_transducer_s{s}_{ds}.pt"


def eot_json(s):
    return "results/eot_transfer.json" if s is None else f"results/revision/seeds/eot_transfer_s{s}.json"


def log_dir(s):
    return f"logs/revision/sp4/chem_s{s}"


def zeroshot_log(ds, s):
    return _CANON_ZS[ds] if s is None else f"{log_dir(s)}/zeroshot_{ds}.log"


def ft_log(ds, s):
    return _CANON_FT[ds] if s is None else f"{log_dir(s)}/ft_eval_{ds}.log"


def fewshot_log(c, s):
    return f"logs/_fewshot_eval/eval{c}.txt" if s is None else f"{log_dir(s)}/fewshot_eval_{c}.log"


def chem_cov_json(s):
    return "results/revision/chemistry_coverage.json" if s is None else f"results/revision/seeds/chemistry_coverage_s{s}.json"


def oxford_stats_json(s):
    return "results/revision/oxford_ft_stats.json" if s is None else f"results/revision/seeds/oxford_ft_stats_s{s}.json"


def mit_diag_json(s):
    return "results/mit_tri_cell37_diagnostic.json" if s is None else f"results/revision/seeds/mit_tri_percell_s{s}.json"
