# Datasets

The loaders in `uapi_former/dataset.py` read the raw files from `data/raw/<dataset>/`.
`data/raw_files_sha256.txt` lists every raw file they read, with its size and SHA-256; after placing the
files, check them with

```bash
python scripts/release_assets.py verify
```

| Dataset | Where the files come from | Put them in |
|---|---|---|
| NASA Ames PCoE battery data (Saha and Goebel, 2007) | release asset `data_nasa.zip` (copy of the NASA files), or the [NASA PCoE data repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) ("Battery Data Set") | `data/raw/nasa/B0005.mat`, ... |
| CALCE CS2/CX2 (University of Maryland) | [CALCE battery data](https://calce.umd.edu/battery-data); not redistributed here | `data/raw/calce/<cell>/` for CS2_33-CS2_36 and CX2_33-CX2_36 (the loader also reads one sub-folder level, as the CALCE archives unpack) |
| Oxford Battery Degradation Dataset 1 (Howey and Birkl, 2017) | release asset `data_oxford.zip` (copy under the ODC Open Database License 1.0), or [ORA](https://doi.org/10.5287/bodleian:KO2kdmYGg) | `data/raw/oxford/Oxford_Battery_Degradation_Dataset_1.mat` |
| MIT-TRI (Severson et al., 2019) | [data.matr.io](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204); not redistributed here | `data/raw/mit_tri/` (the three `*_batchdata_updated_struct_errorcorrect.mat` files) |

`python scripts/release_assets.py unpack --from <folder>` extracts the release assets into place.

Please cite the original sources when using the data: the CALCE terms ask for the articles that describe
the experiments (for CS2/CX2: He et al., J. Power Sources 196 (2011) 10314-10321; Xing et al.,
Microelectron. Reliab. 53 (2013) 811-820), and the Oxford licence requires attribution.

Splits are fixed in the loaders, not stored as index files: NASA by cycle (random split, seed
`NASABatteryDataset.CYCLE_RANDOM_SEED`) or by held-out cell; CALCE, Oxford and MIT-TRI by cell. See the
class docstrings in `uapi_former/dataset.py`.
