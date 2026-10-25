"""
Build, unpack and verify the large files of the code release.

Git holds the code, the result files, the evaluation logs and the figures. The trained checkpoints are
too large for Git; they are attached to the GitHub release as an asset, below GitHub's limit of 2 GiB
per file:

  checkpoints.zip     the trained models behind the article (checkpoints/README.md)
  SHA256SUMS.txt      SHA-256 and size of the asset

The raw data files are not redistributed; download them from their sources (data/README.md).
data/raw_files_sha256.txt lists every raw file the loaders read, with its SHA-256, so any copy can be
checked against the files used for the article.

  python scripts/release_assets.py unpack --from DIR   # verify the downloaded asset and extract it
  python scripts/release_assets.py verify              # check data/raw/ against data/raw_files_sha256.txt
  python scripts/release_assets.py build --out DIR     # maintainers: make the asset and SHA256SUMS.txt
  python scripts/release_assets.py list                # maintainers: rewrite data/raw_files_sha256.txt
"""
import argparse
import glob
import hashlib
import os
import re
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIST = "data/raw_files_sha256.txt"
SUMS = "SHA256SUMS.txt"
LIMIT = 2 * 1024 ** 3                   # GitHub release asset limit
CALCE_CELLS = ("CS2_33", "CS2_34", "CS2_35", "CS2_36", "CX2_33", "CX2_34", "CX2_35", "CX2_36")
CELL = r"B00(05|06|07|18)"
CHECKPOINT_DIRS = (r"nasa_v12_seed[0-4]", r"nasa_v12_nig_seed[0-4]", r"nasa_v12", r"nasa_v12_nig",
                   rf"loco_clean_s[0-4]_{CELL}", r"baseline_(cnn_bilstm|pi_transformer)_seed[0-4]_random_fixed",
                   rf"baseline_(cnn_bilstm|pi_transformer)_s[0-4]_{CELL}(_fixed100)?",
                   rf"baseline_(tcn|pinn_l1|pinn_l10)_s[0-4]_{CELL}", r"sp4")
CHECKPOINT_FILES = r"best\.pt|best_calibrated\.pt|eot_transducer_s[0-4]_(calce|oxford|mit_tri)\.pt"
SOURCES = {"nasa": "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/",
           "oxford": "https://doi.org/10.5287/bodleian:KO2kdmYGg",
           "calce": "https://calce.umd.edu/battery-data",
           "mit_tri": "https://data.matr.io/1/projects/5c48dd2bc625d700019f3204",
           "stanford_m50t": "https://osf.io/qsabn/ (scripts/revision_stanford_prep.py downloads the subset)"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def rel(path, root):
    return os.path.relpath(path, root).replace("\\", "/")


def dataset(path):
    return path.split("/")[2]                                    # data/raw/<dataset>/...


def raw_files(root):
    """Every raw file the loaders in uapi_former/dataset.py read, as repository-relative paths."""
    def g(*p):
        return sorted(rel(f, root) for f in glob.glob(os.path.join(root, "data", "raw", *p)))
    files = g("nasa", "*.mat")
    for cell in CALCE_CELLS:                                     # the patterns of CALCEDataset
        for pat in (("*.csv",), ("*.xlsx",), ("*", "*.csv"), ("*", "*.xlsx")):
            files += g("calce", cell, *pat)
    stanford = g("stanford_m50t", "cycling", "Cycling_*", "*.mat") + g("stanford_m50t", "capacity", "Diag_*", "*.xlsx") \
        + g("stanford_m50t", "capacity.json")                   # StanfordM50TDataset and scripts/revision_stanford_prep.py
    return files + g("oxford", "*.mat") + g("mit_tri", "*.mat") + stanford


def checkpoint_files():
    """The model files behind the article (checkpoints/README.md): best.pt / best_calibrated.pt of each model
    and the EOT transducers. Training-resume files (last.pt) and earlier, superseded runs are not released."""
    out = []
    for d in sorted(os.listdir(os.path.join(ROOT, "checkpoints"))):
        if not any(re.fullmatch(p, d) for p in CHECKPOINT_DIRS):
            continue
        for dp, _, fs in os.walk(os.path.join(ROOT, "checkpoints", d)):
            out += [rel(os.path.join(dp, f), ROOT) for f in fs if re.fullmatch(CHECKPOINT_FILES, f)]
    assert out
    return sorted(out)


def read_list():
    """data/raw_files_sha256.txt -> {path: (sha256, size)}."""
    out = {}
    for line in open(os.path.join(ROOT, LIST), encoding="utf-8"):
        if line.strip() and not line.startswith("#"):
            h, n, p = line.split(None, 2)
            out[p.strip()] = (h, int(n))
    return out


def cmd_list(_):
    files = raw_files(ROOT)
    for d in ("nasa", "calce", "oxford", "mit_tri", "stanford_m50t"):
        assert any(dataset(f) == d for f in files), f"no {d} files under data/raw/"
    lines = ["# Raw data files read by the loaders in uapi_former/dataset.py: SHA-256, size in bytes, path.",
             "# Check a copy with: python scripts/release_assets.py verify"]
    for f in files:
        p = os.path.join(ROOT, f)
        lines.append(f"{sha256(p)}  {os.path.getsize(p):>12d}  {f}")
    open(os.path.join(ROOT, LIST), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    print(f"{LIST}: {len(files)} files")


def zip_files(dst, paths):
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for p in paths:
            z.write(os.path.join(ROOT, p), p)


def cmd_build(a):
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    assert not os.listdir(out), f"{out} is not empty"
    ck = checkpoint_files()
    zip_files(os.path.join(out, "checkpoints.zip"), ck)
    lines = []
    for name in sorted(os.listdir(out)):
        n = os.path.getsize(os.path.join(out, name))
        assert n < LIMIT, (name, n)
        lines.append(f"{sha256(os.path.join(out, name))}  {n:>12d}  {name}")
    open(os.path.join(out, SUMS), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    print(f"{len(lines)} assets ({len(ck)} checkpoint files) -> {out}")


def cmd_unpack(a):
    src, root = os.path.abspath(a.src), os.path.abspath(a.root)
    for line in open(os.path.join(src, SUMS), encoding="utf-8"):
        if not line.strip():
            continue
        h, n, name = line.split(None, 2)
        p = os.path.join(src, name.strip())
        if not os.path.exists(p):
            print(f"  not downloaded: {name.strip()}")
            continue
        assert os.path.getsize(p) == int(n) and sha256(p) == h, f"{name.strip()}: checksum mismatch, download it again"
        with zipfile.ZipFile(p) as z:
            z.extractall(root)                                   # member paths are repository-relative
        print(f"  extracted {name.strip()}")
    return cmd_verify(a)


def cmd_verify(a):
    root = os.path.abspath(a.root)
    ok, missing, bad = 0, [], []
    for p, (h, n) in read_list().items():
        q = os.path.join(root, p)
        if not os.path.exists(q):
            missing.append(p)
        elif os.path.getsize(q) != n or sha256(q) != h:
            bad.append(p)
        else:
            ok += 1
    print(f"raw data files: {ok} match, {len(missing)} missing, {len(bad)} differ")
    for d in sorted({dataset(p) for p in missing}):
        print(f"  missing from data/raw/{d}/: {sum(dataset(p) == d for p in missing)} files"
              + (f" (download from {SOURCES[d]})" if d in SOURCES else ""))
    for p in bad:
        print(f"  differs: {p}")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    for name in ("unpack", "verify"):
        s = sub.add_parser(name)
        s.add_argument("--root", default=ROOT, help="repository root to extract into or check (default: this one)")
        if name == "unpack":
            s.add_argument("--from", dest="src", required=True, help="folder holding the downloaded assets")
    a = ap.parse_args()
    return {"list": cmd_list, "build": cmd_build, "unpack": cmd_unpack, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
