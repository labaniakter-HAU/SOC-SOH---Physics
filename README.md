# Conformal coverage under battery deployment shift

Code, result files, logs and figure/table scripts for the article

> **Diagnosing Conformal Coverage Failure under Battery Deployment Shift for Joint State-of-Charge and
> State-of-Health Estimation** (submitted to *Processes*)

The article measures the coverage of split-conformal SOC and SOH intervals when an estimator trained on
NASA cells is deployed on a held-out cell, under a different discharge protocol, or on another chemistry
(CALCE, Oxford, MIT-TRI). The common estimator is UAPI-Former (`uapi_former/`); two baseline families
and an extended Kalman filter are evaluated alongside it.

## Contents

| Path | What it holds |
|---|---|
| `uapi_former/` | model, dataset loaders, calibration and EKF code |
| `train.py`, `scripts/` | training, evaluation, figure and table scripts |
| `tests/` | unit tests (`python -m pytest tests`; `tests/test_oxford_relabel.py` reads the Oxford data file) |
| `results/` | the result files (JSON) the figure and table scripts read; `FIGURE_SOURCES.md` names the ones each figure and table uses |
| `logs/revision/sp4/` | logs of the five-seed runs; some figure and table scripts read values from them |
| `figs/` | the figures of the article and its Supplementary Materials |
| `docs/revision_r1/FIGURE_SOURCES.md` | for every figure and results table: the command, its output, and every input file it reads |
| `docs/revision_r1/tables/` | the results tables as generated (LaTeX fragments) and the in-text numbers (`numbers.tex`) |
| `checkpoints/README.md` | the trained models (release asset `checkpoints.zip`) |
| `data/README.md`, `data/raw_files_sha256.txt` | the raw data: sources, release assets, and the SHA-256 of every raw file used |

## Setup

The reported results were produced with Python 3.14.3, PyTorch 2.11.0, NumPy 2.4.6, SciPy 1.17.1,
scikit-learn 1.9.0 and Matplotlib 3.11.0.

```bash
pip install -r requirements.txt
```

## Large files (release assets)

Checkpoints and raw data are too large for Git. The release
[`v1.0`](https://github.com/labaniakter-HAU/SOC-SOH---Physics/releases/tag/v1.0) carries:

| Asset | Contents |
|---|---|
| `checkpoints.zip` | `checkpoints/`: the trained models behind the article (see `checkpoints/README.md`) |
| `data_nasa.zip` | `data/raw/nasa/*.mat`: NASA Ames PCoE battery data |
| `data_oxford.zip` | `data/raw/oxford/`: Oxford Battery Degradation Dataset 1 |
| `SHA256SUMS.txt` | checksum of each asset |

Download the assets into one folder, then, from the repository root:

```bash
python scripts/release_assets.py unpack --from <folder>   # checks the checksums, extracts into place
```

The CALCE and MIT-TRI files are not redistributed here; `data/README.md` says where to get them and where
to put them. `python scripts/release_assets.py verify` then checks every raw file against the SHA-256 of
the copy used for the article.

## Reproducing the figures and tables

`docs/revision_r1/FIGURE_SOURCES.md` lists each figure and results table with its command and inputs. Run
every command from the repository root.

- The figure scripts read only `results/` and `logs/`; they need neither checkpoints nor raw data.
- `python scripts/revision_make_tables.py` writes every results table and in-text number to
  `docs/revision_r1/tables/` and asserts the qualitative statements of the article against the result
  files. It reads training epochs from the checkpoints, so unpack `checkpoints.zip` first;
  `git diff docs/revision_r1/tables` then shows any difference from the published tables.

Splits are deterministic: NASA random split from `NASABatteryDataset.CYCLE_RANDOM_SEED = 42`, NASA
leave-one-cell-out folds by held-out cell, and CALCE, Oxford and MIT-TRI by cell
(`uapi_former/dataset.py`).

## Citation

See `CITATION.cff`.
