r"""Generate LaTeX table fragments for the revision from results JSON.

Every number written into docs/revision_r1/tables/*.tex comes from a full
production run recorded under results/. Re-run this script after any new seed
finishes; the manuscript picks the fragments up through \input.
"""
import glob, json, math, os, re, statistics as st

from revision_seeds import SEEDS, seed_files, seed_keys

OUT = "docs/revision_r1/tables"
os.makedirs(OUT, exist_ok=True)
CELLS = ["B0005", "B0006", "B0007", "B0018"]


def shown(x):
    """A value as a one-decimal table cell displays it, for differences quoted in the text."""
    return float(f"{x:.1f}")


# ---- SP4: every multi-run result over five seeds ------------------------------------------------
# Non-LOCO uncertainty results come from the five v12-NIG seed checkpoints, the cross-chemistry
# lineage from the five v12-MSE seed backbones. A seed-varying source is loaded as the list of
# its per-seed objects and as their element-wise mean (mean_view), which keeps the structure the
# former single-checkpoint files had, so every guard below is evaluated on seed means; tables
# print mean +- SD from the per-seed lists.
import revision_sp4_paths as sp

NIG_LEVELS = ("random", "cross_protocol", "chemistry_calce", "chemistry_oxford")


def mean_view(objs):
    """Element-wise mean of structurally identical JSON objects (one per seed). Numbers: mean of the
    finite values (all non-finite -> that value); equal strings/bools kept, differing ones -> list;
    equal-length lists element-wise; integers that agree across seeds stay integers."""
    first = objs[0]
    if isinstance(first, dict):
        return {k: mean_view([o[k] for o in objs]) for k in first if all(isinstance(o, dict) and k in o for o in objs)}
    if isinstance(first, bool) or first is None or isinstance(first, str):
        return first if all(o == first for o in objs) else list(objs)
    if isinstance(first, (int, float)):
        if all(isinstance(o, int) and not isinstance(o, bool) for o in objs) and len(set(objs)) == 1:
            return first
        vals = [float(o) for o in objs]
        fin = [v for v in vals if math.isfinite(v)]
        return sum(fin) / len(fin) if fin else vals[0]
    if isinstance(first, list):
        if all(isinstance(o, list) and len(o) == len(first) for o in objs):
            return [mean_view([o[i] for o in objs]) for i in range(len(first))]
        return list(objs)
    return first


def load_cc():
    """conformal_clean.json; the non-LOCO levels as five v12-NIG seeds (<level>_seeds) and their mean."""
    d = json.load(open("results/revision/conformal_clean.json"))
    for lv in NIG_LEVELS:
        d[lv + "_seeds"] = [d[f"{lv}_s{k}"] for k in SEEDS]
        d[lv] = mean_view(d[lv + "_seeds"])
    return d


def load_protocol_acc():
    """Cross-protocol accuracy of the five v12-NIG seeds (scripts/revision_sp4_protocol_acc.py):
    {"cells": mean over seeds, "seeds": per seed}, the structure of the former cross_protocol_results.json."""
    r = json.load(open("results/revision/seeds/cross_protocol_acc.json"))
    seeds = [r[f"s{k}"] for k in SEEDS]
    return {"cells": mean_view(seeds), "seeds": seeds}


def load_seed_json(path_of):
    """Per-seed JSON files of the cross-chemistry lineages -> (mean view, per-seed list)."""
    runs = [json.load(open(path_of(s))) for s in SEEDS]
    return mean_view(runs), runs


def ms(vals, fmt="{:.2f}"):
    vals = [v for v in vals if v == v and abs(v) != float("inf")]   # drop NaN/inf
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return fmt.format(vals[0])
    return fmt.format(st.mean(vals)) + r" $\pm$ " + fmt.format(st.stdev(vals))


def loco_runs():
    return {f.split("_s")[-1].split(".")[0]: json.load(open(f))
            for f in seed_files("results/revision/loco_clean_s{s}.json")}


def seq_runs():
    return {f.split("_s")[-1].split(".")[0]: json.load(open(f))
            for f in seed_files("results/revision/sequential_s1_s{s}.json")}


def measured_soc_rmse(log):
    """SOC RMSE (%) as printed by scripts/evaluate.py ("SOC  RMSE : 0.2453")."""
    import re as _re
    m = _re.search(r"SOC\s+RMSE\s*:\s*([0-9.]+)", open(log, encoding="utf-8", errors="replace").read())
    return 100 * float(m.group(1))


def cross_chemistry_rmse_seeds():
    """Per backbone seed (SEEDS order): zero-shot SOC RMSE from that backbone's evaluation log and
    EOT SOC RMSE from its eot_transfer JSON; keys (dataset, 'zero' | 'eot')."""
    out = []
    for s in SEEDS:
        eot = json.load(open(sp.eot_json(s)))
        r = {}
        for k in sp.DATASETS:
            r[(k, "zero")] = measured_soc_rmse(sp.zeroshot_log(k, s))
            r[(k, "eot")] = eot[k]["test"]["soc_rmse"]
        out.append(r)
    return out


def cross_chemistry_rmse():
    """Mean over the five backbone lineages of cross_chemistry_rmse_seeds()."""
    runs = cross_chemistry_rmse_seeds()
    return {key: st.mean(r[key] for r in runs) for key in runs[0]}


def pct_txt(vals, f="{:.1f}"):
    """mean +- SD of a percentage that never prints 100.0 for a value below 100 (99.998)."""
    if st.mean(vals) < 100 and f.format(st.mean(vals)) == "100.0":
        f = "{:.3f}"
    return ms(vals, f)


def write(name, lines):
    with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", name)


def table_loco_accuracy():
    runs = loco_runs()
    old = json.load(open("results/loco_cv_results.json"))["folds"]
    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"Held-out cell & SOC RMSE (\%) & SOH RMSE (\%) & SOC $r$ & SOH $r$ \\", r"\midrule"]
    agg = {k: [] for k in ("soc_rmse", "soh_rmse", "soc_r", "soh_r")}
    for c in CELLS:
        v = {k: [r[c][k] for r in runs.values() if c in r] for k in agg}
        for k in agg:
            agg[k].append(st.mean(v[k]))
        L.append(f"{c} & {ms(v['soc_rmse'])} & {ms(v['soh_rmse'])} & "
                 f"{ms(v['soc_r'], '{:.4f}')} & {ms(v['soh_r'], '{:.4f}')} " + r"\\")
    L += [r"\midrule",
          f"Fold mean & {st.mean(agg['soc_rmse']):.2f} & {st.mean(agg['soh_rmse']):.2f} & "
          f"{st.mean(agg['soc_r']):.4f} & {st.mean(agg['soh_r']):.4f} " + r"\\",
          f"Non-independent protocol & {st.mean([old[c]['soc_rmse'] for c in CELLS]):.2f} & "
          f"{st.mean([old[c]['soh_rmse'] for c in CELLS]):.2f} & "
          f"{st.mean([old[c]['soc_r'] for c in CELLS]):.4f} & "
          f"{st.mean([old[c]['soh_r'] for c in CELLS]):.4f} " + r"\\",
          r"\bottomrule", r"\end{tabular}"]
    write("loco_accuracy.tex", L)
    return len(runs)


def table_sequential():
    runs = seq_runs()
    keys = [("S0", "S0 (no correction)"), ("seq_k1", "Sequential S1, $k=1$ cycle"),
            ("seq_k5", "Sequential S1, $k=5$ cycles"), ("seq_k20", "Sequential S1, $k=20$ cycles"),
            ("transductive", "Transductive S1 (full target batch)")]
    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         "Scheme & " + " & ".join(CELLS) + r" \\", r"\midrule"]
    for k, lab in keys:
        cells = []
        for c in CELLS:
            vals = [100 * r[c][k]["accepted_coverage"] for r in runs.values()
                    if c in r and k in r[c]]
            cells.append(ms(vals, "{:.1f}"))
        L.append(lab + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("sequential.tex", L)
    return len(runs)


SHIFTS = [("random", "Random intra-cell", ("random",)),
          ("loco_B0005", "LOCO B0005", ("loco", "B0005")), ("loco_B0006", "LOCO B0006", ("loco", "B0006")),
          ("loco_B0007", "LOCO B0007", ("loco", "B0007")), ("loco_B0018", "LOCO B0018", ("loco", "B0018")),
          ("p_B0025", "Protocol B0025", ("cross_protocol", "B0025")), ("p_B0026", "Protocol B0026", ("cross_protocol", "B0026")),
          ("p_B0027", "Protocol B0027", ("cross_protocol", "B0027")), ("p_B0028", "Protocol B0028", ("cross_protocol", "B0028")),
          ("calce", "CALCE zero-shot", ("chemistry_calce", "zero_shot")),
          ("oxford", "Oxford zero-shot", ("chemistry_oxford", "zero_shot"))]


def _entries(path_spec, d, loco_key):
    """Per-seed entries of one shift level: LOCO models of each seed, v12-NIG seed checkpoints otherwise."""
    if path_spec[0] == "random":
        return d["random_seeds"]
    if path_spec[0] == "loco":
        return [d[k][path_spec[1]] for k in seed_keys(d, "loco_clean_s")]
    return [e[path_spec[1]] for e in d[path_spec[0] + "_seeds"]]


def table_coverage(task):
    """Coverage/refusal/useful table for one task across every shift level."""
    d = load_cc()
    L = [r"\begin{tabular}{lccccc}", r"\toprule",
         r"Shift level & S0 (\%) & S1 accepted (\%) & S1 refusal (\%) & S1 finite \& correct (\%) & S1 width \\",
         r"\midrule"]
    for _, lab, spec in SHIFTS:
        try:
            es = _entries(spec, d, None)
        except KeyError:
            continue
        g = lambda k, f: ms([100 * e[task]["cov90"][k][f] for e in es], "{:.1f}")
        w = ms([e[task]["cov90"]["S1"]["mean_width"] * 100 for e in es
                if e[task]["cov90"]["S1"]["mean_width"] != float("inf")], "{:.2f}")
        L.append(f"{lab} & {g('S0','coverage')} & {g('S1','accepted_coverage')} & "
                 f"{g('S1','refusal')} & {g('S1','finite_and_correct')} & {w if w != '--' else 'n/a'} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(f"coverage_{task}.tex", L)
    return len(L)


def table_cp_baselines():
    runs = {f.split("_s")[-1].split(".")[0]: json.load(open(f))
            for f in seed_files("results/revision/cp_baselines_s{s}.json")}
    if not runs:
        return 0
    schemes = [("S0", "S0 split conformal"), ("Mondrian", "Mondrian (predicted-SOH terciles)"),
               ("CQR", "CQR (linear quantile heads)"), ("CQR_mlp", r"\quad CQR, one-hidden-layer heads"),
               ("ACI", "ACI (online, cycle-delayed labels)")]
    L = [r"\begin{tabular}{lcccccccc}", r"\toprule",
         r"& \multicolumn{2}{c}{B0005} & \multicolumn{2}{c}{B0006} & \multicolumn{2}{c}{B0007} & \multicolumn{2}{c}{B0018} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
         r"Scheme & Cov. & Width & Cov. & Width & Cov. & Width & Cov. & Width \\", r"\midrule"]
    for k, lab in schemes:
        cells = []
        for c in CELLS:
            cov = [100 * r[c][k]["accepted_coverage"] for r in runs.values() if c in r]
            wid = [r[c][k]["mean_width_pp"] for r in runs.values() if c in r]
            cells += [ms(cov, "{:.1f}"), ms(wid, "{:.1f}")]
        L.append(lab + " & " + " & ".join(cells) + r" \\")
        if k == "ACI":
            ref = [ms([100 * r[c]["ACI"]["refusal"] for r in runs.values() if c in r], "{:.1f}") + " & --" for c in CELLS]
            L.append(r"\quad ACI refusal (\%) & " + " & ".join(ref) + r" \\")
    for k in ("S0", "Mondrian", "CQR", "CQR_mlp"):   # caption: only ACI refuses
        assert all(r[c][k]["refusal"] == 0 for r in runs.values() for c in CELLS), k
    L += [r"\bottomrule", r"\end{tabular}"]
    write("cp_baselines.tex", L)
    return len(runs)


def table_cross_chemistry():
    """Cross-chemistry: point accuracy AND coverage in one table (R1.9). Every row is mean +- SD over
    the five v12-MSE backbone lineages (zero-shot, EOT and fine-tuned from each backbone), with the
    constant-width absolute-residual score (results/revision/seeds/chemistry_coverage_s{s}.json)."""
    _, runs = load_seed_json(sp.chem_cov_json)
    rs = cross_chemistry_rmse_seeds()
    L = [r"\begin{tabular}{lllcccc}", r"\toprule",
         r"Dataset & Setting & Calibration & SOC RMSE (\%) & S0 cov.\ (\%) & S0 width (SOC pp) & S1 refusal (\%) \\",
         r"\midrule"]
    name = {"calce": "CALCE LCO", "oxford": "Oxford LCO", "mit_tri": "MIT-TRI LFP"}
    for k in sp.DATASETS:
        for st_, lab, rk in (("zero_shot", "Zero-shot", "zero"), ("eot", "EOT, frozen backbone", "eot")):
            es = [r["source"][k][st_] for r in runs]
            for e, r in zip(es, rs):   # the RMSE column is the logged one; coverage comes from the same model
                assert abs(e["soc_rmse_pct"] - r[(k, rk)]) < 0.01, (k, st_, e["soc_rmse_pct"], r[(k, rk)])
            L.append(f"{name[k]} & {lab} & NASA (source) & {ms([r[(k, rk)] for r in rs])} & "
                     f"{ms([100 * e['soc']['S0']['coverage'] for e in es])} & "
                     f"{ms([e['soc']['S0']['mean_width_pp'] for e in es])} & "
                     f"{pct_txt([100 * e['soc']['S1']['refusal'] for e in es])} " + r"\\")
    L.append(r"\midrule")
    for k in sp.DATASETS:
        es = [r["fine_tuned"][k] for r in runs]
        L.append(f"{name[k]} & Fine-tuned & target & {ms([e['soc_rmse_pct'] for e in es])} & "
                 f"{ms([100 * e['soc']['cov90']['coverage'] for e in es])} & "
                 f"{ms([e['soc']['cov90']['mean_width_pp'] for e in es])} & -- " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("cross_chemistry.tex", L)
    return len(L)


def table_eot_variants():
    """Supplement EOT variant selection: validation SOC RMSE per regularizer, mean +- SD over the five
    backbones, and how often each variant was selected (the selection rule is asserted per backbone)."""
    runs = [json.load(open(sp.eot_json(s))) for s in SEEDS]
    cols = {"none": "None", "grad": "Gradient", "peak": "Peak", "dist": "Distribution"}
    word = {"none": "none", "grad": "gradient", "peak": "peak", "dist": "distribution"}
    L = [r"\begin{tabular}{lccccl}", r"\toprule",
         "Target & " + " & ".join(cols.values()) + r" & Selected \\", r"\midrule"]
    for k, lab in (("calce", "CALCE LCO"), ("oxford", "Oxford LCO"), ("mit_tri", "MIT-TRI LFP")):
        for e in runs:
            v = e[k]["validation_variants"]
            assert min(cols, key=lambda n: v[n]["soc_rmse"]) == e[k]["selected_variant"], k   # selection rule
        sel = [e[k]["selected_variant"] for e in runs]
        cnt = sorted(((n, sel.count(n)) for n in set(sel)), key=lambda x: (-x[1], x[0]))
        L.append(f"{lab} & " + " & ".join(ms([e[k]["validation_variants"][n]["soc_rmse"] for e in runs]) for n in cols)
                 + " & " + ", ".join(f"{word[n]} ({c}/{len(sel)})" for n, c in cnt) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("eot_variants.tex", L)
    return len(L)


def table_coverage_merged():
    """Main-text coverage table (SP3a): SOC and SOH at 90% nominal in one table. The S1
    refusal decision is shared by construction (one density ratio in the shared latent), so it
    is one column; asserted equal for the two tasks in every entry."""
    d = load_cc()
    L = [r"\begin{tabular}{lccccccccc}", r"\toprule",
         r"& \multicolumn{4}{c}{SOC} & & \multicolumn{4}{c}{SOH} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){7-10}",
         r"Shift level & S0 & S1 acc. & F\&C & Width & S1 refusal & S0 & S1 acc. & F\&C & Width \\", r"\midrule"]
    for _, lab, spec in SHIFTS:
        try:
            es = _entries(spec, d, None)
        except KeyError:
            continue
        assert all(e["soc"]["cov90"]["S1"]["refusal"] == e["soh"]["cov90"]["S1"]["refusal"] for e in es), lab
        cells = []
        for task in ("soc", "soh"):
            g = lambda k, f: ms([100 * e[task]["cov90"][k][f] for e in es], "{:.1f}")
            w = ms([100 * e[task]["cov90"]["S1"]["mean_width"] for e in es], "{:.2f}")
            cells.append([g("S0", "coverage"), g("S1", "accepted_coverage"), g("S1", "finite_and_correct"), w])
        ref = ms([100 * e["soc"]["cov90"]["S1"]["refusal"] for e in es], "{:.1f}")
        L.append(f"{lab} & " + " & ".join(cells[0]) + f" & {ref} & " + " & ".join(cells[1]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("coverage_merged.tex", L)
    return len(L)


def table_accuracy_merged():
    """Main-text SOC accuracy of the three estimators at every shift level (SP3a). Random split
    and protocol cells: five seeds (UAPI-Former = v12-NIG seeds; baselines = 400-epoch runs);
    held-out cells: the pinned LOCO seeds, same training rule for all three; chemistry: the
    canonical v12-MSE checkpoint only (baselines were not transferred)."""
    fx = json.load(open("results/revision/baselines_random_fixed.json"))
    bl = json.load(open("results/revision/baselines_loco.json"))
    bp, get = baselines_protocol_summary()
    runs = loco_runs()
    rm = cross_chemistry_rmse()
    est = ("uapi_former", "cnn_bilstm", "pi_transformer")
    rnd = {"uapi_former": uapi_seeds("nig")[0]}
    for k in est[1:]:
        rnd[k] = [v["soc_rmse"] for n, v in fx.items() if n.startswith(k)]
    assert all(len(v) == 5 for v in rnd.values())
    # Sec. 4.5 "UAPI-Former, the most accurate of the three [on the random split]"
    assert st.mean(rnd["uapi_former"]) < min(st.mean(rnd[k]) for k in est[1:])
    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"Setting & UAPI-Former & CNN--BiLSTM & Physics-guided Transformer & Seeds \\", r"\midrule",
         "Random cycle split & " + " & ".join(ms(rnd[k]) for k in est) + r" & 5 \\", r"\addlinespace"]
    fold = {}
    for c in CELLS:
        v = {"uapi_former": [runs[s][c]["soc_rmse"] for s in sorted(runs)]}
        for k in est[1:]:
            v[k] = [bl[f"{k}_s{s}_{c}"]["soc_rmse"] for s in sorted(runs)]
        fold[c] = st.mean(v["uapi_former"])
        L.append(f"Held-out {c} & " + " & ".join(ms(v[k]) for k in est) + f" & {len(runs)} " + r"\\")
    L.append(r"\addlinespace")
    prot = {}
    for c in PROT_CELLS:
        v = {k: get(k, c, lambda x: x["rmse_pct"]) for k in est}
        prot[c] = st.mean(v["uapi_former"])
        L.append(f"Protocol {c} & " + " & ".join(ms(v[k]) for k in est) + f" & {len(v['uapi_former'])} " + r"\\")
    # Discussion: "SOC RMSE rises from the random split to the held-out cells to the protocol cells"
    assert st.mean(rnd["uapi_former"]) < min(fold.values()) and max(fold.values()) < min(prot.values())
    L.append(r"\addlinespace")
    rs = cross_chemistry_rmse_seeds()
    for k, lab in (("calce", "CALCE"), ("oxford", "Oxford"), ("mit_tri", "MIT-TRI")):
        L.append(f"{lab} zero-shot & {ms([r[(k, 'zero')] for r in rs])} & -- & -- & {len(rs)} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("accuracy_merged.tex", L)
    return len(L)


def table_comparators():
    """Main-text alternative calibration schemes (SP3a merge of the former cp-baselines and
    sequential tables). Panel A: every window of the held-out cell; S0/S1/S2 from
    conformal_clean.json, Mondrian/CQR/ACI from cp_baselines_s*.json (same estimator, folds and
    calibration partitions). Panel B: windows of cycles after the 20th (sequential_s1_s*.json)."""
    d = load_cc()
    cp = {f.split("_s")[-1].split(".")[0]: json.load(open(f))
          for f in seed_files("results/revision/cp_baselines_s{s}.json")}
    sq = seq_runs()
    seeds = sorted(cp)
    assert seeds == sorted(sq) == [str(s) for s in SEEDS], (seeds, sorted(sq))
    cc = lambda s, c: d[f"loco_clean_s{s}"][c]["soc"]["cov90"]
    for s in seeds:                                   # the two S0 computations are one computation
        for c in CELLS:
            assert abs(cp[s][c]["S0"]["accepted_coverage"] - cc(s, c)["S0"]["coverage"]) < 1e-6, (s, c)
    head = [r"\begin{tabular}{lcccccccc}", r"\toprule",
            r"& \multicolumn{2}{c}{B0005} & \multicolumn{2}{c}{B0006} & \multicolumn{2}{c}{B0007} & \multicolumn{2}{c}{B0018} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
            r"Scheme & Cov. & Width & Cov. & Width & Cov. & Width & Cov. & Width \\", r"\midrule",
            r"\multicolumn{9}{l}{\textit{A. All windows of the held-out cell}} \\"]
    L = list(head)

    def row(lab, cov, wid):
        cells = []
        for c in CELLS:
            cells += [ms([cov(s, c) for s in seeds], "{:.1f}"), ms([wid(s, c) for s in seeds], "{:.1f}") if wid else "--"]
        L.append(lab + " & " + " & ".join(cells) + r" \\")

    row("S0 split conformal", lambda s, c: 100 * cp[s][c]["S0"]["accepted_coverage"],
        lambda s, c: cp[s][c]["S0"]["mean_width_pp"])
    row("S1 density-ratio weighting", lambda s, c: 100 * cc(s, c)["S1"]["accepted_coverage"],
        lambda s, c: 100 * cc(s, c)["S1"]["mean_width"])
    row("S2 optimal-transport weighting", lambda s, c: 100 * cc(s, c)["S2"]["accepted_coverage"],
        lambda s, c: 100 * cc(s, c)["S2"]["mean_width"])
    for k, lab in (("Mondrian", "Mondrian (predicted-SOH terciles)"), ("CQR", "CQR (linear quantile heads)"),
                   ("CQR_mlp", r"\quad CQR, one-hidden-layer heads"), ("ACI", "ACI (online, cycle-delayed labels)")):
        row(lab, lambda s, c, k=k: 100 * cp[s][c][k]["accepted_coverage"], lambda s, c, k=k: cp[s][c][k]["mean_width_pp"])
    L.append(r"\quad ACI refusal (\%) & " + " & ".join(
        ms([100 * cp[s][c]["ACI"]["refusal"] for s in seeds], "{:.1f}") + " & --" for c in CELLS) + r" \\")
    L += [r"\midrule", r"\multicolumn{9}{l}{\textit{B. Windows of cycles after the 20th}} \\"]
    for k, lab in (("S0", "S0 (no correction)"), ("seq_k1", "Sequential S1, $k=1$ cycle"),
                   ("seq_k5", "Sequential S1, $k=5$ cycles"), ("seq_k20", "Sequential S1, $k=20$ cycles"),
                   ("transductive", "Transductive S1 (full target batch)")):
        row(lab, lambda s, c, k=k: 100 * sq[s][c][k]["accepted_coverage"], None)
    L += [r"\bottomrule", r"\end{tabular}"]
    write("comparators.tex", L)
    return len(L)


def chemistry_macros():
    """Cross-chemistry section / abstract / Table 6 caption. Values are means over the five backbone
    lineages unless the name says Run (over all 5 x rows); every qualitative guard holds per backbone."""
    cc, runs = load_seed_json(sp.chem_cov_json)
    rs = cross_chemistry_rmse_seeds()
    rmse = cross_chemistry_rmse()
    M = {}
    for k, t in (("calce", "Calce"), ("oxford", "Ox"), ("mit_tri", "Mit")):
        M[f"{t}ZeroRmse"] = f"{rmse[(k, 'zero')]:.2f}"
        M[f"{t}EotRmse"] = f"{rmse[(k, 'eot')]:.2f}"
        f = cc["fine_tuned"][k]
        M[f"FtRmse{t}"] = f"{f['soc_rmse_pct']:.2f}"
        M[f"FtCov{t}"] = f"{100 * f['soc']['cov90']['coverage']:.1f}"
        M[f"FtWidth{t}"] = f"{f['soc']['cov90']['mean_width_pp']:.1f}"
        M[f"FtSohCov{t}"] = f"{100 * f['soh']['cov90']['coverage']:.1f}"
    assert len({r["n_cal_windows"] for r in runs}) == 1                 # one NASA calibration half
    M["ChemCalWindows"] = f"{runs[0]['n_cal_windows']:,}".replace(",", "{,}")
    src = [e for r in runs for k in r["source"] for e in r["source"][k].values()]   # every row, every backbone
    cov = [100 * e["soc"]["S0"]["coverage"] for e in src]
    ref = [100 * e["soc"]["S1"]["refusal"] for e in src]
    fc1 = [100 * e["soc"]["S1"]["finite_and_correct"] for e in src]
    M["ChemSrcCovMax"], M["ChemSrcRefMin"] = f"{max(cov):.2f}", f"{min(ref):.1f}"
    M["ChemSrcFcMax"] = f"{max(fc1):.2f}"
    ftm = [100 * cc["fine_tuned"][k]["soc"]["cov90"]["coverage"] for k in sp.DATASETS]
    ft = [100 * r["fine_tuned"][k]["soc"]["cov90"]["coverage"] for r in runs for k in sp.DATASETS]
    M["FtCovMin"], M["FtCovMax"] = f"{min(ftm):.1f}", f"{max(ftm):.1f}"
    M["FtCovRunMin"], M["FtCovRunMax"] = f"{min(ft):.1f}", f"{max(ft):.1f}"
    # "only fine-tuning with target calibration restores coverage": mean over backbones above 85% on every
    # chemistry, every single run above 80% (printed as a range); the source-side bound is ChemSrcAllMax below
    assert min(ftm) > 85 and min(ft) > 80, (min(ftm), min(ft))
    M["CalceSohZeroRmse"] = f"{cc['source']['calce']['zero_shot']['soh_rmse_pct']:.2f}"
    M["CalceSohFtRmse"] = f"{cc['fine_tuned']['calce']['soh_rmse_pct']:.2f}"
    # MIT-TRI per-cell breakdown of the fine-tuned aggregate, per backbone (scripts/revision_sp4_percell.py)
    mits = [json.load(open(sp.mit_diag_json(s))) for s in SEEDS]
    for m, r in zip(mits, runs):
        assert abs(m["aggregate_soc_rmse_pct"] - r["fine_tuned"]["mit_tri"]["soc_rmse_pct"]) < 0.01
    worst = {max(m["per_cell"], key=lambda c: m["per_cell"][c]["soc_rmse_pct"]) for m in mits}
    assert len(worst) == 1, worst                                        # the same cell for every backbone
    w = worst.pop()
    n = {m["n_cells"] for m in mits}
    assert len(n) == 1
    n = n.pop()
    rest = [v["soc_rmse_pct"] for m in mits for c, v in m["per_cell"].items() if c != w]
    words = {13: "thirteen", 14: "14"}            # mid-sentence ("Across backbones, thirteen of 14 cells")
    M["MitCellsGood"], M["MitCells"] = words[n - 1], str(n)
    M["MitCellMin"], M["MitCellMax"] = f"{min(rest):.2f}", f"{max(rest):.2f}"
    ow = [m["per_cell"][w]["soc_rmse_pct"] for m in mits]
    M["MitOutlierMin"], M["MitOutlierMax"] = f"{min(ow):.2f}", f"{max(ow):.2f}"
    M["MitOutlierShare"] = f"{100 * mits[0]['per_cell'][w]['share_of_windows']:.1f}"
    se = [100 * m["per_cell"][w]["share_of_squared_error"] for m in mits]
    M["MitOutlierSeShare"] = f"{math.floor(min(se)):.0f}"               # "at least X% in every backbone"
    assert min(se) > 50, se
    eots = [json.load(open(sp.eot_json(s))) for s in SEEDS]
    # the EOT SOC correlation depends on the backbone: reported as a range and a count of backbones
    # (r < 0 on MIT-TRI, r > 0.99 on Oxford), never as an "every backbone" property
    word = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    sgn = lambda x: f"{x:.3f}".replace("-", "$-$")
    mr = [e["mit_tri"]["test"]["soc_r"] for e in eots]
    M["MitEotR"], M["MitEotRMin"], M["MitEotRMax"] = sgn(st.mean(mr)), sgn(min(mr)), sgn(max(mr))
    M["MitEotRNegN"] = word[sum(x < 0 for x in mr)]
    assert st.mean(mr) < 0, mr                                           # "mean SOC correlation is negative"
    orr = [e["oxford"]["test"]["soc_r"] for e in eots]
    M["OxEotR"], M["OxEotRMin"], M["OxEotRMax"] = sgn(st.mean(orr)), sgn(min(orr)), sgn(max(orr))
    M["OxEotRHighN"] = word[sum(x > 0.99 for x in orr)]
    assert max(orr) > 0.99, orr                                          # "even then" needs such a backbone
    cr = [e["calce"]["test"]["soc_r"] for e in eots]
    assert st.mean(cr) >= 0 and st.mean(orr) >= 0, (cr, orr)            # Figure 3 flags only the MIT-TRI EOT bar
    ox = cc["source"]["oxford"]["eot"]["soc"]
    M["ChemOxEotCov"], M["ChemOxEotRef"] = f"{100 * ox['S0']['coverage']:.2f}", f"{100 * ox['S1']['refusal']:.1f}"
    # the S0 width is the NASA residual quantile of each backbone: identical in its source-calibrated rows
    wid = []
    for r in runs:
        w_ = [e["soc"]["S0"]["mean_width_pp"] for k in r["source"] for e in r["source"][k].values()]
        assert max(w_) - min(w_) < 1e-6, w_
        wid.append(w_[0])
    M["ChemSrcWidthMin"], M["ChemSrcWidthMax"] = f"{min(wid):.2f}", f"{max(wid):.2f}"
    # Figure 3 caption / Discussion: fine-tuning gives the lowest SOC RMSE for every backbone and chemistry;
    # the means fall zero-shot > EOT > fine-tuned on every chemistry, but EOT is not below zero-shot for
    # every backbone: the pairs where it is are counted, with the largest increase in the others
    for r, x in zip(runs, rs):
        for k in sp.DATASETS:
            assert r["fine_tuned"][k]["soc_rmse_pct"] < min(x[(k, "zero")], x[(k, "eot")]), k
    for k in sp.DATASETS:
        assert rmse[(k, "zero")] > rmse[(k, "eot")] > cc["fine_tuned"][k]["soc_rmse_pct"], k
    d = [x[(k, "eot")] - x[(k, "zero")] for x in rs for k in sp.DATASETS]
    M["EotLowerN"], M["EotPairs"] = str(sum(v < 0 for v in d)), str(len(d))
    M["EotWorseMax"] = f"{max(d):.2f}"                                   # SOC points above zero-shot
    assert max(d) > 0, d                                                 # the text names exceptions
    fred = [100 * (1 - cc["fine_tuned"][k]["soc_rmse_pct"] / rmse[(k, "zero")]) for k in sp.DATASETS]
    M["FtRedMin"], M["FtRedMax"] = f"{min(fred):.0f}", f"{max(fred):.0f}"
    # "SOC coverage above X% is reached only by fine-tuning with target-domain calibration" and "the
    # post-hoc corrections of source calibration stay at or below X%": X bounds every source-calibrated
    # SOC row of every seed (v12-MSE S0 and S1 finite-and-correct above; v12-NIG zero-shot S0, S1
    # finite-and-correct and S2), rounded up so that the printed bound holds; every fine-tuned run is above it
    nig = load_cc()
    src = cov + fc1 + [100 * e["zero_shot"]["soc"]["cov90"][m][f] for k in ("chemistry_calce", "chemistry_oxford")
                       for e in nig[k + "_seeds"]
                       for m, f in (("S0", "coverage"), ("S1", "finite_and_correct"), ("S2", "coverage"))]
    assert len(src) == 18 * len(SEEDS), len(src)        # per seed: 6 MSE rows x (S0, S1) + 2 NIG rows x 3
    M["ChemSrcAllMax"] = f"{math.ceil(10 * max(src)) / 10:.1f}"
    assert max(src) <= float(M["ChemSrcAllMax"]) < min(ft) / 2, (max(src), min(ft))
    return M


def measured_rmse(log, task):
    """SOC or SOH RMSE (%) as printed by scripts/evaluate.py ("SOH  RMSE : 0.0563")."""
    import re as _re
    m = _re.search(task.upper() + r"\s+RMSE\s*:\s*([0-9.]+)", open(log, encoding="utf-8", errors="replace").read())
    return 100 * float(m.group(1))


def fewshot_logs(s):
    """CALCE labelled-data sensitivity logs of backbone s: zero-shot, 1/5/10/20 cycles, full set."""
    return [("0", sp.zeroshot_log("calce", s))] + [(str(c), sp.fewshot_log(c, s)) for c in sp.FEWSHOT_CYCLES] + \
           [("Full set", sp.ft_log("calce", s))]


def table_calce_fewshot():
    """Supplement CALCE labelled-data sensitivity, mean +- SD over the five backbones (every run
    fine-tunes that backbone; logs of scripts/evaluate.py)."""
    _, runs = load_seed_json(sp.chem_cov_json)
    rs = cross_chemistry_rmse_seeds()
    L = [r"\begin{tabular}{lcc}", r"\toprule",
         r"Labelled cycles per training cell & SOC RMSE (\%) & SOH RMSE (\%) \\", r"\midrule"]
    for i, (lab, _) in enumerate(fewshot_logs(SEEDS[0])):
        logs = [fewshot_logs(s)[i][1] for s in SEEDS]
        if lab == "Full set":   # the full-precision values of Table 6 (the logs print four decimals)
            soc = [r["fine_tuned"]["calce"]["soc_rmse_pct"] for r in runs]
            soh = [r["fine_tuned"]["calce"]["soh_rmse_pct"] for r in runs]
        else:
            soc, soh = [measured_rmse(g, "soc") for g in logs], [measured_rmse(g, "soh") for g in logs]
        L.append(f"{lab} & {ms(soc)} & {ms(soh)} " + r"\\")
    # the end rows are the zero-shot and fine-tuned CALCE numbers quoted everywhere else
    for s, r, x in zip(SEEDS, runs, rs):
        g = fewshot_logs(s)
        assert abs(measured_rmse(g[0][1], "soc") - x[("calce", "zero")]) < 1e-9
        assert abs(measured_rmse(g[-1][1], "soc") - r["fine_tuned"]["calce"]["soc_rmse_pct"]) < 0.01
    L += [r"\bottomrule", r"\end{tabular}"]
    write("calce_fewshot.tex", L)
    return len(L)


def claim_macros(d):
    """Claim audit (2026-09-29): every qualitative word kept in the prose is asserted here;
    wording without a guard was removed from the text."""
    import torch
    M = {}
    fmt_n = lambda n: f"{n:,}".replace(",", "{,}")
    loco = {c: [d[k][c] for k in seed_keys(d, "loco_clean_s")] for c in CELLS}
    runs = loco_runs()
    # Sec. 4.2: LOCO fold-mean SOC RMSE is several times the random split; B0006 and B0018
    # are the hardest cells; the spread across cells exceeds the mean spread across seeds
    rm = {c: [r[c]["soc_rmse"] for r in runs.values() if c in r] for c in CELLS}
    fm = {c: st.mean(v) for c, v in rm.items()}
    rnd = max(st.mean(uapi_seeds("mse")[0]), st.mean(uapi_seeds("nig")[0]))
    assert st.mean(fm.values()) > 3 * rnd, (fm, rnd)                                # "several times"
    assert set(sorted(CELLS, key=fm.get)[-2:]) == {"B0006", "B0018"}, fm             # "hardest cells"
    M["LocoCellSd"] = f"{st.stdev(fm.values()):.2f}"
    M["LocoSeedSdMean"] = f"{st.mean(st.stdev(v) for v in rm.values()):.2f}"
    assert float(M["LocoCellSd"]) > float(M["LocoSeedSdMean"])
    # Sec. 4.2 protocol: B0028 has the largest error and a negative SOC r; SOH r is near zero
    pc = load_protocol_acc()["cells"]
    P = ("B0025", "B0026", "B0027", "B0028")
    # "B0028 has the largest SOC error and the weakest SOC correlation" (seed means; SP4: the negative
    # correlation of the former single checkpoint is not a five-seed result)
    assert max(P, key=lambda k: pc[k]["soc_rmse"]) == "B0028" and min(P, key=lambda k: pc[k]["soc_r"]) == "B0028"
    M["ProtRBtwentyeight"] = f"{pc['B0028']['soc_r']:.2f}"
    M["ProtSohRAbsMax"] = f"{max(abs(pc[k]['soh_r']) for k in P):.2f}"
    assert max(abs(pc[k]["soh_r"]) for k in P) < 0.05
    # Discussion / Conclusions: SOC RMSE rises random split -> LOCO -> protocol shift; Introduction:
    # "a hierarchy of increasingly difficult conditions", with zero-shot chemistry transfer the hardest
    # (SP4: by error, not by pooled coverage -- the SOC = 0 rest windows lift pooled protocol coverage
    # above the lowest LOCO fold means)
    cz = cross_chemistry_rmse()
    assert rnd < st.mean(fm.values()) < st.mean(pc[k]["soc_rmse"] for k in P) < min(cz[(k, "zero")] for k in sp.DATASETS)
    # Sec. 4.3 / abstract: every shift level lies below the random split and below nominal, SOC and SOH
    S0 = lambda e, t: 100 * e[t]["cov90"]["S0"]["coverage"]
    for t in ("soc", "soh"):
        lo = [st.mean(S0(e, t) for e in loco[c]) for c in CELLS]
        pr = [S0(d["cross_protocol"][k], t) for k in P]
        ch = [S0(d[k]["zero_shot"], t) for k in ("chemistry_calce", "chemistry_oxford")]
        assert S0(d["random"], t) > max(lo + pr + ch), t
        if t == "soh":   # Sec. 3.1 "SOH coverage follows the same hierarchy": strict, level by level
            assert max(lo) < S0(d["random"], t) and min(lo) > max(pr) and min(pr) > max(ch), (lo, pr, ch)
        assert max(lo + pr + ch) < 90, t
        # text: "for both, S0 coverage and the finite-and-correct fraction stay below the nominal level
        # at every shift level"; abstract: "SOH coverage is below nominal at every shift level", which
        # for SOH also covers the accepted-point S1 coverage (for SOC it reaches 90% on B0027)
        shift = [e for c in CELLS for e in loco[c]] + [d["cross_protocol"][k] for k in P] + \
                [d[k]["zero_shot"] for k in ("chemistry_calce", "chemistry_oxford")]
        fc_lo = [st.mean(100 * e[t]["cov90"]["S1"]["finite_and_correct"] for e in loco[c]) for c in CELLS]
        fc_rest = [100 * e[t]["cov90"]["S1"]["finite_and_correct"] for e in shift[len(CELLS) * len(loco[CELLS[0]]):]]
        assert max(fc_lo + fc_rest) < 90, t
        if t == "soh":
            acc = [st.mean(100 * e[t]["cov90"]["S1"]["accepted_coverage"] for e in loco[c]) for c in CELLS] + \
                  [100 * e[t]["cov90"]["S1"]["accepted_coverage"] for e in shift[len(CELLS) * len(loco[CELLS[0]]):]
                   if e[t]["cov90"]["S1"]["refusal"] < 1]
            assert max(acc) < 90, acc
    # Sec. 3.2: ESS does not predict refusal -- two fully refused targets have a higher ESS
    # than a LOCO fold on which S1 refuses almost nothing
    le = {c: st.mean(e["ess"]["S1"] for e in loco[c]) for c in CELLS}
    lmin = min(le, key=le.get)
    cal, b28 = d["chemistry_calce"]["zero_shot"], d["cross_protocol"]["B0028"]
    assert cal["ess"]["S1"] > le[lmin] and b28["ess"]["S1"] > le[lmin]
    assert b28["soc"]["cov90"]["S1"]["refusal"] == 1.0
    # "... yet S1 refuses at least X% of the CALCE windows and every B0028 window, against at most
    # Y% on the LOCO folds": every LOCO run refuses less than any seed refuses on CALCE
    assert max(e["soc"]["cov90"]["S1"]["refusal"] for c in CELLS for e in loco[c]) < \
        min(e["zero_shot"]["soc"]["cov90"]["S1"]["refusal"] for e in d["chemistry_calce_seeds"])
    M["EssCalce"], M["EssBtwentyeight"] = f"{cal['ess']['S1']:.1f}", f"{b28['ess']['S1']:.1f}"
    M["EssLocoMin"], M["EssLocoMinCell"] = f"{le[lmin]:.1f}", lmin
    M["EssOx"] = f"{d['chemistry_oxford']['zero_shot']['ess']['S1']:.1f}"
    # Discussion: the cross-cell shift is detectable too (fold-mean discriminator AUC)
    au = [st.mean(e["discriminator_auc"] for e in loco[c]) for c in CELLS]
    M["LocoAucMin"], M["LocoAucMax"] = f"{min(au):.3f}", f"{max(au):.3f}"
    # Sec. 4.6: no label-free scheme reaches nominal mean coverage on any held-out cell
    fold = lambda f: [st.mean(f(e) for e in loco[c]) for c in CELLS]
    lf = fold(lambda e: e["soc"]["cov90"]["S0"]["coverage"]) + \
         fold(lambda e: e["soc"]["cov90"]["S1"]["accepted_coverage"]) + \
         fold(lambda e: e["soc"]["cov90"]["S2"]["accepted_coverage"])
    cp = [json.load(open(f)) for f in seed_files("results/revision/cp_baselines_s{s}.json")]
    for k in ("S0", "Mondrian", "CQR", "CQR_mlp"):
        lf += [st.mean(r[c][k]["accepted_coverage"] for r in cp) for c in CELLS]
    gate = [json.load(open(f)) for f in seed_files("results/revision/support_gate_s{s}.json")]
    lf += [st.mean(x[c]["p99.0"]["accepted_coverage"] for x in gate) for c in CELLS]
    assert max(lf) < 0.90, max(lf)
    # best-validation epochs, read from the checkpoints: fine-tuning per backbone, v12-NIG seeds;
    # the v12-MSE seed backbones are final-epoch weights of a 200-epoch campaign
    ep = lambda p: torch.load(p, map_location="cpu", weights_only=False)["epoch"]
    for k, t_ in (("calce", "Calce"), ("oxford", "Ox"), ("mit_tri", "Mit")):
        e = [ep(f"{sp.ft_dir(k, s)}/best.pt") for s in SEEDS]
        M[f"FtEpoch{t_}Min"], M[f"FtEpoch{t_}Max"] = str(min(e)), str(max(e))
    ne = [ep(f"checkpoints/nasa_v12_nig_seed{s}/best.pt") for s in SEEDS]
    M["NigSeedEpochMin"], M["NigSeedEpochMax"] = str(min(ne)), str(max(ne))
    assert all(ep(sp.backbone(s)) == 200 for s in SEEDS)
    # Sec. 3.5: CALCE with 1-20 labelled cycles per cell stays far from the full-set result (every backbone)
    few_m = []
    for i in range(1, 1 + len(sp.FEWSHOT_CYCLES)):
        few_m.append(st.mean(measured_rmse(fewshot_logs(s)[i][1], "soc") for s in SEEDS))
    few = [measured_rmse(x[1], "soc") for s in SEEDS for x in fewshot_logs(s)[1:-1]]
    full = [measured_rmse(fewshot_logs(s)[-1][1], "soc") for s in SEEDS]
    assert min(few) > max(full), (min(few), max(full))       # no few-shot run reaches any full-set run
    M["FewSocMin"], M["FewSocMax"] = f"{min(few_m):.2f}", f"{max(few_m):.2f}"
    M["FewSocRunMin"], M["FewSocRunMax"] = f"{min(few):.2f}", f"{max(few):.2f}"
    M["FtCalceRunMin"], M["FtCalceRunMax"] = f"{min(full):.2f}", f"{max(full):.2f}"
    # Introduction, contribution 2: "S1 refuses none of [B0006's] windows"
    assert all(d[k]["B0006"]["soc"]["cov90"]["S1"]["refusal"] == 0 for k in seed_keys(d, "loco_clean_s"))
    return M


def software_macros():
    """Sec. 2.3 (Processes: name and version of the software used). The versions are read from
    the environment that runs this script, i.e. the one that regenerates every figure and results table."""
    import importlib.metadata as md
    import platform
    v = lambda p: md.version(p).split("+")[0]
    return {"VerPython": platform.python_version(), "VerTorch": v("torch"), "VerNumpy": v("numpy"),
            "VerScipy": v("scipy"), "VerSklearn": v("scikit-learn"), "VerMatplotlib": v("matplotlib")}


def loco_objective_macros():
    """Supplement S3.4: the LOCO models are trained with the v12-NIG objective (NIG loss plus the
    10/2 MSE anchor) and the evidence-regularizer weight set in scripts/loco_cv.py, whose
    train_epoch scripts/loco_clean.py imports without overriding the weight."""
    cv = open("scripts/loco_cv.py", encoding="utf-8").read()
    clean = open("scripts/loco_clean.py", encoding="utf-8").read()
    (coeff,) = re.findall(r"^NIG_COEFF = ([0-9.]+)$", cv, re.M)
    assert "from scripts.loco_cv import train_epoch" in clean and "NIG_COEFF" not in clean
    assert "coeff=NIG_COEFF" in cv and "10.0 * F.mse_loss(sg, soc)" in cv and "2.0  * F.mse_loss(hg, soh)" in cv
    return {"LocoNigCoeff": coeff}


def supplement_list():
    """Back matter (Processes): "Figure S1: title; Table S1: title; ..." from the captions of
    docs/revision_r1/supplement.tex. Floats are numbered in source order, as LaTeX numbers them;
    the title is the first sentence of the caption. Cross-checked against supplement.aux."""
    import re
    s = open("docs/revision_r1/supplement.tex", encoding="utf-8").read().replace("\r\n", "\n")
    aux = open("docs/revision_r1/supplement.aux", encoding="utf-8").read()
    num = dict(re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", aux))
    items = {"figure": [], "table": []}
    for m in re.finditer(r"\\begin\{(table|figure)\}.*?\\end\{\1\}", s, re.S):
        body = m.group(0)
        i = body.index("\\caption{") + len("\\caption{") - 1
        depth = 0
        for j in range(i, len(body)):
            depth += (body[j] == "{") - (body[j] == "}")
            if depth == 0:
                break
        cap = body[i + 1:j]
        n = len(items[m.group(1)]) + 1
        lab = re.search(r"\\label\{([^}]+)\}", body).group(1)
        assert num[lab] == f"S{n}", (lab, num[lab], n)
        first = re.split(r"\.\s+(?=[A-Z\\])", cap.strip(), maxsplit=1)[0]
        short, depth = "", 0                          # a short title: no parenthetical detail
        for ch in first:
            if ch == "(":
                depth += 1
                short = short.rstrip()
            elif ch == ")":
                depth -= 1
            elif depth == 0:
                short += ch
        title = re.sub(r"\s+,", ",", short.split(";")[0]).strip().rstrip(".")
        # in the supplement, bare labels are its own and M- labels are the article's
        title = re.sub(r"\\ref\{(?!M-)", r"\\ref{S-", title).replace("\\ref{M-", "\\ref{")
        assert "\\label" not in title and title and depth == 0, cap[:60]
        items[m.group(1)].append(f"{m.group(1).capitalize()} S{n}: {title}")
    L = ["; ".join(items["figure"] + items["table"]) + "."]
    write("supp_list.tex", L)
    return len(items["figure"]), len(items["table"])


def macros():
    """In-text numbers as LaTeX macros so prose can never drift from the tables."""
    d = load_cc()
    runs = loco_runs()
    old = json.load(open("results/loco_cv_results.json"))["folds"]
    loco = {c: [d[k][c] for k in seed_keys(d, "loco_clean_s")] for c in CELLS}
    s0 = {c: st.mean([100 * e["soc"]["cov90"]["S0"]["coverage"] for e in loco[c]]) for c in CELLS}
    acc = {c: st.mean([100 * e["soc"]["cov90"]["S1"]["accepted_coverage"] for e in loco[c]]) for c in CELLS}
    ref = max(100 * e["soc"]["cov90"]["S1"]["refusal"] for c in CELLS for e in loco[c])
    # SP4: refusal on the LOCO folds per fold-seed run ("no window in N of M runs, up to X% of <cell>")
    lref = [(100 * e["soc"]["cov90"]["S1"]["refusal"], c) for c in CELLS for e in loco[c]]
    assert sum(r == 0 for r, _ in lref) >= 0.75 * len(lref), lref          # "rarely fires"
    r7 = [100 * e["soc"]["cov90"]["S1"]["refusal"] for e in loco["B0007"]]
    fold_mean = st.mean([st.mean([r[c]["soc_rmse"] for r in runs.values() if c in r]) for c in CELLS])
    sub_mean = st.mean([old[c]["soc_rmse"] for c in CELLS])
    M = {
        "LocoCovMin": f"{min(s0.values()):.1f}", "LocoCovMax": f"{max(s0.values()):.1f}",
        "AccBfive": f"{acc['B0005']:.1f}", "AccBsix": f"{acc['B0006']:.1f}",
        "AccBseven": f"{acc['B0007']:.1f}", "AccBeighteen": f"{acc['B0018']:.1f}",
        "MaxLocoRefusal": f"{ref:.1f}", "MaxLocoRefusalCell": max(lref)[1],
        "LocoRefZeroRuns": str(sum(r == 0 for r, _ in lref)), "LocoRefRuns": str(len(lref)),
        "RefBsevenMean": f"{st.mean(r7):.1f}", "RefBsevenMax": f"{max(r7):.1f}",
        "AccBsevenRunMax": f"{max(100 * e['soc']['cov90']['S1']['accepted_coverage'] for e in loco['B0007']):.1f}",
        "LocoRmseMean": f"{fold_mean:.2f}",
        "LocoRmseSubmitted": f"{sub_mean:.2f}", "NumLocoSeeds": str(len(runs)),
        "RandomCovClean": f"{100*d['random']['soc']['cov90']['S0']['coverage']:.1f}",
    }
    # Random-split coverage for both tasks and both nominal levels, the per-fold
    # SOH hierarchy, the finite-and-correct values quoted in the prose, and the
    # LOCO width ranges. These were hand-typed once and drifted; never again.
    for t in ("soc", "soh"):
        suf = "" if t == "soc" else "Soh"
        for lv, tag in (("cov90", ""), ("cov95", "NinetyFive")):  # no digits: TeX names are letters only
            e = d["random"][t].get(lv)
            if e:
                M[f"RandomCov{tag}{suf}"] = f"{100*e['S0']['coverage']:.1f}"
        cov = {c: st.mean([100 * x[t]["cov90"]["S0"]["coverage"] for x in loco[c]]) for c in CELLS}
        wid = {c: st.mean([100 * x[t]["cov90"]["S1"]["mean_width"] for x in loco[c]]) for c in CELLS}
        fc = {c: st.mean([100 * x[t]["cov90"]["S1"]["finite_and_correct"] for x in loco[c]]) for c in CELLS}
        M[f"LocoCov{suf}Min"] = f"{min(cov.values()):.1f}"
        M[f"LocoCov{suf}Max"] = f"{max(cov.values()):.1f}"
        M[f"LocoWidth{suf}Min"] = f"{min(wid.values()):.1f}"
        M[f"LocoWidth{suf}Max"] = f"{max(wid.values()):.1f}"
        M[f"FC{suf}Bseven"] = f"{fc['B0007']:.1f}"

    # B0006 label-support diagnosis: quoted in the abstract, so it gets macros too.
    diag = [json.load(open(f)) for f in seed_files("results/revision/b0006_diagnosis_s{s}.json")]
    if diag:
        b6 = [d["B0006"] for d in diag]
        below = [100 * x["below"]["S0"]["coverage"] for x in b6]
        M["BsixFracBelow"] = f"{100*b6[0]['frac_target_below_source_range']:.1f}"
        M["BsixBelowCovMin"] = f"{min(below):.1f}"
        M["BsixBelowCovMax"] = f"{max(below):.1f}"
        M["BsixSohFloor"] = f"{diag[0]['B0005']['target_soh_min']:.3f}"      # B0005's minimum (b0006_macros)
        kr = [x["below"]["mean_knn_dist"] / x["inside"]["mean_knn_dist"] for x in b6]
        M["BsixKnnRatioMin"], M["BsixKnnRatioMax"] = f"{min(kr):.1f}", f"{max(kr):.1f}"
        M.update(b0006_macros(diag))
    M.update(baselines_shift_macros())
    # Sec. 3.5 / Table 2 (R1.5): baseline protocol and the early-stopping sensitivity
    fx = json.load(open("results/revision/baselines_random_fixed.json"))
    es = json.load(open("results/revision/sensitivity/baselines_random_earlystop.json"))
    M["BenchBestEpochMin"] = str(min(v["best_epoch"] for v in fx.values()))
    M["BenchBestEpochMax"] = str(max(v["best_epoch"] for v in fx.values()))
    uapi_soc = st.mean(uapi_seeds("mse")[0])
    for k, kt in (("cnn_bilstm", "Cnn"), ("pi_transformer", "Pi")):
        f = [v["soc_rmse"] for n, v in fx.items() if n.startswith(k)]
        e = [v["soc_rmse"] for n, v in es.items() if n.startswith(k)]
        assert len(f) == len(e) == 5
        M[f"Bench{kt}FixedSoc"] = f"{st.mean(f):.2f}"
        M[f"Bench{kt}EarlySoc"] = f"{st.mean(e):.2f}"
        assert st.mean(f) < st.mean(e)                       # text: early stopping gave slightly worse baselines
        assert uapi_soc < min(st.mean(f), st.mean(e)) / 3     # text: conclusion identical under both protocols
    # Discussion: main-text (sigma-normalized) transductive S1 minus S0, fold means over seeds
    dl = {c: st.mean(100 * (e["soc"]["cov90"]["S1"]["accepted_coverage"] - e["soc"]["cov90"]["S0"]["coverage"])
                     for e in loco[c]) for c in CELLS}
    # quoted as the difference of the one-decimal fold means Table 2 displays
    dshown = {c: shown(st.mean(100 * e["soc"]["cov90"]["S1"]["accepted_coverage"] for e in loco[c]))
                 - shown(st.mean(100 * e["soc"]["cov90"]["S0"]["coverage"] for e in loco[c])) for c in CELLS}
    M["LocoSoneDeltaMin"] = f"{min(dshown.values()):+.1f}".replace("-", "$-$")
    M["LocoSoneDeltaMax"] = f"{max(dshown.values()):+.1f}".replace("-", "$-$")
    M["LocoSoneDeltaMaxCell"] = max(dl, key=dl.get)
    worst = [100 * (e["soc"]["cov90"]["S1"]["accepted_coverage"] - e["soc"]["cov90"]["S0"]["coverage"])
             for e in loco[M["LocoSoneDeltaMaxCell"]]]
    assert min(worst) < 0 < max(worst), worst          # text: "not stable across seeds"
    # text: "on N of the four folds its mean effect is smaller than the seed-to-seed SD of S1 coverage"
    small = [c for c in CELLS
             if abs(dl[c]) < st.stdev(100 * e["soc"]["cov90"]["S1"]["accepted_coverage"] for e in loco[c])]
    assert len(small) >= 3, small
    words4 = {4: "every fold", 3: "three of the four folds"}
    M["LocoSoneSmallFolds"] = words4[len(small)]
    M["LocoSoneLargeFold"] = "" if len(small) == 4 else [c for c in CELLS if c not in small][0]
    helps = [c for c in CELLS if dl[c] > 0]
    w4 = {0: "no", 1: "one", 2: "two", 3: "three", 4: "all four"}
    M["LocoSoneHelps"], M["LocoSoneHurts"] = w4[len(helps)], w4[len(CELLS) - len(helps)]
    pc = d["cross_protocol"]
    pv = [100 * pc[c]["soc"]["cov90"]["S0"]["coverage"] for c in ("B0025", "B0026", "B0027")]
    M["ProtSzeroMin"], M["ProtSzeroMax"] = f"{min(pv):.1f}", f"{max(pv):.1f}"
    M["ProtSzeroBtwentyeight"] = f"{100 * pc['B0028']['soc']['cov90']['S0']['coverage']:.2f}"
    M.update(accuracy_macros())
    M.update(baselines_sensitivity_macros())
    M.update(cp_and_chemistry_macros())
    M.update(chemistry_macros())
    M.update(runtime_macros())
    M.update(sequential_macros())
    M.update(label_macros(d))
    M.update(ocv_macros())
    M.update(r0_tables_and_macros())
    M.update(ekf_macros())
    M.update(fc_macros(d))
    M.update(soh_coverage_macros(d))
    M.update(score_dependence_macros(d))
    M.update(baselines_protocol_macros())
    # Sec. 3.2 calibration-independence paragraph: partition sizes (R2.8)
    lc = [d[k][c]["n_calib"] for k in seed_keys(d, "loco_clean_s") for c in CELLS]
    assert len(lc) == 4 * len([k for k in seed_keys(d, "loco_clean_s")])
    M.update({"LocoCalMin": f"{min(lc):,}".replace(",", "{,}"), "LocoCalMax": f"{max(lc):,}".replace(",", "{,}"),
              "CalWindows": f"{d['random']['n_calib']:,}".replace(",", "{,}"),
              "EvalWindows": f"{d['random']['n_test']:,}".replace(",", "{,}"),
              "CalCycles": str(d["random"]["n_cal_cycles"]), "EvalCycles": str(d["random"]["n_eval_cycles"])})
    fmt_n = lambda n: f"{n:,}".replace(",", "{,}")
    lt = {c: {d[k][c]["n_test"] for k in seed_keys(d, "loco_clean_s")} for c in CELLS}
    assert all(len(v) == 1 for v in lt.values()) and lt["B0005"] == lt["B0006"] == lt["B0007"], lt
    pt = {d["cross_protocol"][c]["n_test"] for c in ("B0025", "B0026", "B0027", "B0028")}
    assert len(pt) == 1, pt                                     # text: "1,746 windows per cell"
    M.update({"LocoTestN": fmt_n(lt["B0005"].pop()), "LocoTestNeighteen": fmt_n(lt["B0018"].pop()),
              "ProtTestN": fmt_n(pt.pop()),
              "CalceTestN": fmt_n(d["chemistry_calce"]["zero_shot"]["n_test"]),
              "OxTestN": fmt_n(d["chemistry_oxford"]["zero_shot"]["n_test"])})
    # Oxford relabelled from the 1C discharge record (SP2, spec §5.3)
    ox, oxr = load_seed_json(sp.oxford_stats_json)
    for o in oxr:
        assert o["record"] == "discharge" and o["n_test_cells"] == 1            # text: "a single held-out cell"
        assert o["n_test"] == d["chemistry_oxford"]["zero_shot"]["n_test"]      # one test set everywhere
    M["OxSocRmse"], M["OxSohR"] = f"{ox['soc_rmse_pct']:.2f}", f"{ox['soh_r']:.3f}"
    M["OxSocR"] = f"{ox['soc_r']:.4f}".replace("-", "$-$")        # 0.9997 must not print as 1.000
    M["OxTrainN"], M["OxTestCycles"] = fmt_n(ox["n_train"]), str(ox["n_test_cycles"])
    assert float(M["ChemCovMax"]) <= float(M["ProtSzeroMax"]), (M["ChemCovMax"], M["ProtSzeroMax"])
    M.update(claim_macros(d))
    M.update(software_macros())
    M.update(loco_objective_macros())
    # Sec. 4.5: longer training makes the PI-Transformer "both more accurate and better covered
    # than UAPI-Former" on B0018; Sec. 4.6: ACI "does not reach [nominal] either"
    assert float(M["SensPiRmseBeighteenFixed"]) < float(M["BshiftUapiRmseBeighteen"])
    assert float(M["SensPiCovBeighteenFixed"]) > float(M["BshiftUapiCovBeighteen"])
    cpr = [json.load(open(f)) for f in seed_files("results/revision/cp_baselines_s{s}.json")]
    assert max(st.mean(r[c]["ACI"]["accepted_coverage"] for r in cpr) for c in CELLS) < 0.90   # fold means
    write("numbers.tex", [r"\newcommand{\Num" + k + "}{" + v + "}" for k, v in M.items()])
    return M


def label_macros(d):
    """Sec. 3.1 label construction and its SOC = 0 consequence (R1.4).
    label_audit.json: scripts/revision_label_audit.py (raw .mat, loader's exact formula).
    rest_windows.json: scripts/revision_rest_windows.py (cached predictions, pooled vs SOC > 0).
    soh_label_analysis.json: scripts/revision_soh_labels.py (window- vs cycle-level SOH)."""
    la = json.load(open("results/revision/label_audit.json"))
    rw = json.load(open("results/revision/rest_windows.json"))
    rw["random"] = mean_view([rw[f"random_s{k}"] for k in SEEDS])
    for c in ("B0025", "B0026", "B0027", "B0028"):
        rw[f"protocol_{c}"] = mean_view([rw[f"protocol_s{k}_{c}"] for k in SEEDS])
    sl, sls = load_seed_json(lambda s: f"results/revision/seeds/soh_label_analysis_s{s}.json")
    assert all(x["coverage"]["cov90"]["cycles_fully_missed"] == 0 for x in sls)   # per seed
    over = [100 * (x - 1) for a in la.values() for x in (a["q_out_end_over_q_cycle_min_med_max"][0],
                                                         a["q_out_end_over_q_cycle_min_med_max"][2])]
    n_clip = sum(a["n_soh_clipped_high"] + a["n_soh_clipped_low"] for a in la.values())
    clip_mag = max(100 * (a["max_ratio"] - 1) for a in la.values())
    assert all(a["n_soh_clipped_low"] == 0 for a in la.values()) and clip_mag < 0.5
    # pooled values must be the paper's values (same caches, same score, same quantile)
    for c in ("B0025", "B0026", "B0027", "B0028"):
        assert abs(rw[f"protocol_{c}"]["all"]["s0_cov90_pct"]
                   - 100 * d["cross_protocol"][c]["soc"]["cov90"]["S0"]["coverage"]) < 1e-4, c
    assert abs(rw["random"]["all"]["s0_cov90_pct"] - 100 * d["random"]["soc"]["cov90"]["S0"]["coverage"]) < 1e-4
    for s in SEEDS:
        for c in CELLS:
            assert abs(rw[f"loco_s{s}_{c}"]["all"]["s0_cov90_pct"]
                       - 100 * d[f"loco_clean_s{s}"][c]["soc"]["cov90"]["S0"]["coverage"]) < 1e-4, (s, c)
    loco_keys = [f"loco_s{s}_{c}" for s in SEEDS for c in CELLS]
    drop = {c: st.mean([rw[f"loco_s{s}_{c}"]["all"]["s0_cov90_pct"] - rw[f"loco_s{s}_{c}"]["label_positive"]["s0_cov90_pct"]
                        for s in SEEDS]) for c in CELLS}
    assert min(drop.values()) > 0                               # text: pooled coverage is optimistic on every fold
    prot = [rw[f"protocol_{c}"] for c in ("B0025", "B0026", "B0027")]
    assert all(p["label_positive"]["soc_rmse_pct"] > p["all"]["soc_rmse_pct"] for p in prot)
    assert all(p["label_positive"]["s0_cov90_pct"] < p["all"]["s0_cov90_pct"] for p in prot)
    zero_rmse = [rw[k]["label_zero"]["soc_rmse_pct"] for k in loco_keys + ["random"]] + \
                [p["label_zero"]["soc_rmse_pct"] for p in prot]
    rng = lambda xs, f="{:.1f}": (f.format(min(xs)), f.format(max(xs)))
    M = {}
    M["LabelOverMin"], M["LabelOverMax"] = rng(over)
    M["LabelClipCycles"] = str(n_clip)
    M["LabelCycles"] = str(sum(a["n_cycles"] for a in la.values()))
    M["LabelClipMag"] = f"{clip_mag:.1f}"
    M["ZeroShareRandom"] = f"{rw['random']['share_label_zero_pct']:.1f}"
    M["ZeroShareLocoMin"], M["ZeroShareLocoMax"] = rng([rw[k]["share_label_zero_pct"] for k in loco_keys])
    M["ZeroShareProtMin"], M["ZeroShareProtMax"] = rng([p["share_label_zero_pct"] for p in prot])
    M["ZeroRmseMax"] = f"{max(zero_rmse):.1f}"
    M["ZeroLocoDropMin"], M["ZeroLocoDropMax"] = rng(list(drop.values()))
    M["ZeroProtRmseAllMin"], M["ZeroProtRmseAllMax"] = rng([p["all"]["soc_rmse_pct"] for p in prot])
    M["ZeroProtRmseActMin"], M["ZeroProtRmseActMax"] = rng([p["label_positive"]["soc_rmse_pct"] for p in prot])
    M["ZeroProtCovActMin"], M["ZeroProtCovActMax"] = rng([p["label_positive"]["s0_cov90_pct"] for p in prot])
    # abstract / Discussion / Conclusions: protocol-shift coverage on the windows with SOC > 0, all four
    # cells (B0028 has no SOC = 0 windows, so its pooled value is its SOC > 0 value)
    assert rw["protocol_B0028"]["share_label_zero_pct"] == 0
    M["ProtPosCovMax"] = f"{max(rw[f'protocol_{c}']['label_positive']['s0_cov90_pct'] for c in ('B0025', 'B0026', 'B0027', 'B0028')):.1f}"
    # Sec. 4.2 cross-protocol correlation, pooled (= Table 4) and on SOC > 0 windows
    t4 = load_protocol_acc()["cells"]
    for c in ("B0025", "B0026", "B0027"):
        assert abs(t4[c]["soc_r"] - rw[f"protocol_{c}"]["all"]["soc_r"]) < 1e-3, c   # same checkpoint as Table 4
    M["ProtRAllMin"], M["ProtRAllMax"] = rng([t4[c]["soc_r"] for c in ("B0025", "B0026", "B0027")], "{:.2f}")
    M["ProtRActMin"], M["ProtRActMax"] = rng([p["label_positive"]["soc_r"] for p in prot], "{:.2f}")
    M["ProtQfirstMin"], M["ProtQfirstMax"] = rng([la[c]["q_first_ah"] for c in ("B0025", "B0026", "B0027", "B0028")], "{:.2f}")
    # SOH window/cycle granularity paragraph (Sec. 4.3)
    pos = {r["relpos"]: r["soh_rmse_pct"] for r in sl["soh_error_vs_position"]}
    c90 = sl["coverage"]["cov90"]
    wpc = sl["windows_per_cycle"]
    M.update({
        "SohCycles": str(sl["n_cycles_test"]), "SohWindows": f"{sl['n_windows_test']:,}".replace(",", "{,}"),
        "SohWpcMin": str(wpc["min"]), "SohWpcMax": str(wpc["max"]), "SohWpcMean": f"{wpc['mean']:.0f}",
        "SohPosFirst": f"{pos['0.0-0.2']:.2f}",
        "SohPosMidMin": f"{min(pos['0.2-0.4'], pos['0.4-0.6']):.2f}", "SohPosMidMax": f"{max(pos['0.2-0.4'], pos['0.4-0.6']):.2f}",
        "SohPosLastMin": f"{min(pos['0.6-0.8'], pos['0.8-1.0']):.2f}", "SohPosLastMax": f"{max(pos['0.6-0.8'], pos['0.8-1.0']):.2f}",
        "SohCovWindow": f"{100 * c90['window_level']:.1f}",
        "SohEvalCycles": str(c90["n_eval_cycles"]),
        "SohCovCycle": f"{100 * c90['cycle_level_last_window']:.1f}",
        "SohPten": f"{100 * c90['per_cycle_coverage_p10_p50_p90'][0]:.1f}",
        "SohPfifty": f"{100 * c90['per_cycle_coverage_p10_p50_p90'][1]:.1f}",
        "SohPninety": f"{100 * c90['per_cycle_coverage_p10_p50_p90'][2]:.1f}",
        "SohFullyCovered": f"{100 * c90['cycles_fully_covered']:.0f}",
    })
    assert c90["cycles_fully_missed"] == 0                      # text: "none is missed in every window"
    assert max(pos["0.6-0.8"], pos["0.8-1.0"]) > max(pos.values()) - 1e-9 and pos["0.0-0.2"] > max(pos["0.2-0.4"], pos["0.4-0.6"])
    assert abs(100 * d["random"]["soh"]["cov90"]["S0"]["coverage"] - 100 * c90["window_level"]) < 0.05
    return M


def ocv_macros():
    """Sec. 3.4 / Table A.2 (R2.2): agreement of the per-fold fitted NMC OCV tables with each
    other and with the submitted hand-constructed table, over every fold and seed."""
    import sys
    sys.path.insert(0, ".")
    import numpy as np
    from uapi_former.dataset import _NMC_OCV_PTS
    hand = np.asarray(_NMC_OCV_PTS, dtype=float)
    tabs = np.array([r[c]["ocv_pts"] for r in loco_runs().values() for c in CELLS])
    assert tabs.shape == (4 * len(loco_runs()), 13)
    spread = 1000 * (tabs.max(0) - tabs.min(0))
    dev = 1000 * np.abs(tabs - hand).max(0)
    assert dev[1:3].max() == dev.max()                            # text: largest deviation lies below 20 % SOC (knots 0.05, 0.10)
    assert spread.max() < dev.max() / 3                           # text: fold identity matters far less than hand setting
    runs = loco_runs()
    r_fold = [st.mean([r[c]["soc_r"] for r in runs.values()]) for c in CELLS]
    return {"OcvSpreadMax": f"{spread.max():.0f}", "OcvSpreadHigh": f"{spread[3:].max():.0f}",
            "OcvHandMax": f"{dev.max():.0f}", "LocoSocRMin": f"{math.floor(1000 * min(r_fold)) / 1000:.3f}"}   # a ">=" bound: floor


def fc_macros(d):
    """Finite-and-correct paragraph (R1.8) and the protocol/chemistry refusal paragraph, over the five
    v12-NIG seed checkpoints (per-seed values for "in every seed" statements, seed means otherwise)."""
    three = ("B0025", "B0026", "B0027")
    P = d["cross_protocol_seeds"]
    S1 = lambda e, c: e[c]["soc"]["cov90"]["S1"]
    acc = [100 * S1(e, c)["accepted_coverage"] for e in P for c in three if S1(e, c)["refusal"] < 1]
    fc_mean = [100 * d["cross_protocol"][c]["soc"]["cov90"]["S1"]["finite_and_correct"] for c in three]
    ref = [100 * S1(e, c)["refusal"] for e in P for c in three]
    n_acc = [round(e[c]["n_test"] * (1 - S1(e, c)["refusal"])) for e in P for c in three]
    # "S1 refuses every B0028 window in every seed, so none receives a finite and correct interval"
    assert all(S1(e, "B0028")["refusal"] == 1.0 and S1(e, "B0028")["finite_and_correct"] == 0.0 for e in P)
    # "fold-to-fold spread is far larger than the seed-to-seed spread"
    fold = {c: [100 * d[k][c]["soc"]["cov90"]["S0"]["coverage"] for k in seed_keys(d, "loco_clean_s")] for c in CELLS}
    means = [st.mean(v) for v in fold.values()]
    assert max(means) - min(means) > 3 * max(st.stdev(v) for v in fold.values())
    chem = lambda k: 100 * d[k]["zero_shot"]["soc"]["cov90"]["S0"]["coverage"]
    rng = lambda xs, f="{:.1f}": (f.format(min(xs)), f.format(max(xs)))
    M = {}
    M["FcProtAccMin"], M["FcProtAccMax"] = rng(acc, "{:.0f}")
    M["FcProtMin"], M["FcProtMax"] = rng(fc_mean)
    M["FcProtRefMin"], M["FcProtRefMax"] = rng(ref)
    M["FcProtNaccMin"], M["FcProtNaccMax"] = str(min(n_acc)), str(max(n_acc))
    M["ChemCalceSzero"], M["ChemOxSzero"] = f"{chem('chemistry_calce'):.2f}", f"{chem('chemistry_oxford'):.2f}"
    # S2 under protocol and chemistry shift (zero-shot entries)
    zs = {k: [e["zero_shot"] for e in d[k + "_seeds"]] for k in ("chemistry_calce", "chemistry_oxford")}
    s2p = [100 * d["cross_protocol"][c]["soc"]["cov90"]["S2"]["coverage"] for c in ("B0025", "B0026", "B0027", "B0028")]
    assert all(e[c]["soc"]["cov90"]["S2"]["refusal"] == 0 for e in P for c in e if c.startswith("B"))  # "never refuses"
    # chemistry refusal: the smallest share of refused windows in any seed
    M["ChemRefCalceMin"] = f"{math.floor(1000 * min(100 * e['soc']['cov90']['S1']['refusal'] for e in zs['chemistry_calce'])) / 1000:.1f}"
    M["ChemRefOxMin"] = f"{math.floor(1000 * min(100 * e['soc']['cov90']['S1']['refusal'] for e in zs['chemistry_oxford'])) / 1000:.1f}"
    assert min(float(M["ChemRefCalceMin"]), float(M["ChemRefOxMin"])) > 95                  # "refuses almost every window"
    M["STwoProtMin"], M["STwoProtMax"] = rng(s2p)
    M["STwoChemMax"] = f"{max(100 * d[k]['zero_shot']['soc']['cov90']['S2']['coverage'] for k in zs):.2f}"
    auc = [e[c]["discriminator_auc"] for e in P for c in ("B0025", "B0026", "B0027", "B0028")] + \
          [e["discriminator_auc"] for k in zs for e in zs[k]]
    # "AUC at least X": X is the minimum floored to four decimals (0.99997 must not print as 1.000)
    M["AucShiftMin"] = f"{math.floor(min(auc) * 1e4) / 1e4:.4f}"
    assert min(auc) >= float(M["AucShiftMin"])
    return M


def soh_coverage_macros(d):
    """Sec. 4.3 'SOH coverage follows the same hierarchy' (R2.1), from the Table 6 entries.
    SOH range: lowest SOH label among the four LOCO cells (label_audit.json) up to 1.0."""
    pc = d["cross_protocol"]
    S0h = lambda e: 100 * e["soh"]["cov90"]["S0"]["coverage"]
    la = json.load(open("results/revision/label_audit.json"))
    soh_min = min(la[c]["min_ratio"] for c in CELLS)
    soh_range = 100 * (1 - soh_min)
    # refusal is shared by construction: identical for SOC and SOH in every entry
    ents = [d["random"]] + [pc[c] for c in ("B0025", "B0026", "B0027", "B0028")] + \
           [d[k][c] for k in seed_keys(d, "loco_clean_s") for c in CELLS] + \
           [d[k]["zero_shot"] for k in ("chemistry_calce", "chemistry_oxford")]
    assert all(e["soc"]["cov90"]["S1"]["refusal"] == e["soh"]["cov90"]["S1"]["refusal"] for e in ents)
    wid = lambda t, c: st.mean(100 * d[k][c][t]["cov90"]["S1"]["mean_width"] for k in seed_keys(d, "loco_clean_s"))
    rel = {c: (wid("soh", c) / soh_range) / (wid("soc", c) / 100) for c in CELLS}   # SOH vs SOC width, each relative to its range
    assert min(rel.values()) > 2                                  # text: "at least twice as wide relative to range"
    rng = lambda xs, f="{:.1f}": (f.format(min(xs)), f.format(max(xs)))
    M = {}
    M["SohProtMin"], M["SohProtMax"] = rng([S0h(pc[c]) for c in ("B0025", "B0026", "B0027")])
    M["SohBtwentyeight"] = f"{S0h(pc['B0028']):.1f}"
    M["SohCalce"] = f"{S0h(d['chemistry_calce']['zero_shot']):.1f}"
    M["SohOx"] = f"{S0h(d['chemistry_oxford']['zero_shot']):.1f}"
    M["SohRangePts"], M["SohMinLabel"] = f"{soh_range:.0f}", f"{soh_min:.3f}"
    M["SohWidthRelMin"], M["SohWidthRelMax"] = rng([100 * wid("soh", c) / soh_range for c in CELLS], "{:.0f}")
    M["SohSocRelRatioMin"], M["SohSocRelRatioMax"] = rng(list(rel.values()))
    return M


PROT_CELLS = ("B0025", "B0026", "B0027", "B0028")


def baselines_protocol_summary():
    """Cross-protocol baselines (R1.6): results/revision/baseline_protocol.json, five seeds,
    constant-width absolute-residual score for all three estimators, one calibration mask."""
    r = json.load(open("results/revision/baseline_protocol.json"))
    assert sorted(r) == ["0", "1", "2", "3", "4"], sorted(r)
    get = lambda e, c, f: [f(r[s][c][e]) for s in r]
    return r, get


def table_baselines_protocol():
    r, get = baselines_protocol_summary()
    fmt1 = lambda xs: ms(xs, "{:.1f}")
    rows = [("SOC RMSE (\\%)", lambda x: x["rmse_pct"]),
            ("S0 coverage, all windows (\\%)", lambda x: 100 * x["S0"]["coverage"]),
            ("S0 coverage, SOC $>0$ (\\%)", lambda x: 100 * x["S0_soc_pos"]["coverage"]),
            ("S1 refusal (\\%)", lambda x: 100 * x["S1"]["refusal"]),
            ("S1 finite \\& correct, SOC $>0$ (\\%)", lambda x: 100 * x["S1_soc_pos"]["finite_and_correct"])]
    L = [r"\begin{tabular}{llcccc}", r"\toprule", "Estimator & Quantity & " + " & ".join(PROT_CELLS) + r" \\", r"\midrule"]
    for e, lab in BSHIFT_ESTS:
        for i, (q, f) in enumerate(rows):
            L.append(f"{lab if i == 0 else ''} & {q} & " + " & ".join(fmt1(get(e, c, f)) for c in PROT_CELLS) + r" \\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("baselines_protocol.tex", L)


def baselines_protocol_macros():
    """Sec. 4.5 cross-protocol paragraph, with guards for every qualitative sentence."""
    r, get = baselines_protocol_summary()
    E = [e for e, _ in BSHIFT_ESTS]
    mean = lambda e, c, f: st.mean(get(e, c, f))
    covp = lambda x: 100 * x["S0_soc_pos"]["coverage"]
    rmse = lambda x: x["rmse_pct"]
    ref = lambda x: 100 * x["S1"]["refusal"]
    fcp = lambda x: 100 * x["S1_soc_pos"]["finite_and_correct"]
    width = lambda x: x["S0"]["mean_width_pp"]
    base = [e for e in E if e != "uapi_former"]
    for c in PROT_CELLS:
        assert all(mean("uapi_former", c, rmse) > mean(b, c, rmse) for b in base), c    # "highest RMSE on all four"
        assert all(mean("uapi_former", c, covp) < mean(b, c, covp) for b in base), c    # "lowest SOC>0 coverage on all four"
    seed_max = max(v for e in E for c in PROT_CELLS for v in get(e, c, covp))
    assert seed_max < 90                                                                # "no seed reaches nominal"
    pi_zero = [s for s in r if all(r[s][c]["pi_transformer"]["S1"]["refusal"] < 0.01 for c in PROT_CELLS[:3])]
    assert len(pi_zero) == 1                                                            # "one seed refuses none of B0025-B0027"
    rng = lambda xs, f="{:.1f}": (f.format(min(xs)), f.format(max(xs)))
    M = {}
    M["BprotCovPosMin"], M["BprotCovPosMax"] = rng([mean(e, c, covp) for e in E for c in PROT_CELLS])
    M["BprotCovPosSeedMax"] = f"{seed_max:.1f}"
    M["BprotUapiCovPosMin"], M["BprotUapiCovPosMax"] = rng([mean("uapi_former", c, covp) for c in PROT_CELLS])
    M["BprotUapiRmseMin"], M["BprotUapiRmseMax"] = rng([mean("uapi_former", c, rmse) for c in PROT_CELLS])
    M["BprotBaseRmseMin"], M["BprotBaseRmseMax"] = rng([mean(b, c, rmse) for b in base for c in PROT_CELLS])
    M["BprotUapiWidth"] = f"{st.mean(get('uapi_former', 'B0025', width)):.1f}"
    M["BprotBaseWidthMin"], M["BprotBaseWidthMax"] = rng([st.mean(get(b, "B0025", width)) for b in base])
    M["BprotUapiRefMin"] = f"{min(mean('uapi_former', c, ref) for c in PROT_CELLS):.1f}"
    M["BprotFcPosMax"] = f"{max(mean(e, c, fcp) for e in E for c in PROT_CELLS):.1f}"
    M["BprotSeeds"] = {5: "five"}[len(r)]
    return M


def score_dependence_macros(d):
    """Sec. 4.5 (R1.6): sigma-normalized score (Table 5) vs constant-width absolute-residual
    score (Table 7) on the SAME checkpoints and calibration partitions
    (results/revision/score_dependence.json, scripts/revision_score_dependence.py)."""
    sd = json.load(open("results/revision/score_dependence.json"))
    for c in ("B0025", "B0026", "B0027", "B0028"):
        sd[f"protocol_{c}"] = mean_view([sd[f"protocol_s{k}_{c}"] for k in SEEDS])
    loco = [f"loco_s{s}_{c}" for s in SEEDS for c in CELLS]
    prot = [f"protocol_{c}" for c in ("B0025", "B0026", "B0027", "B0028")]
    # the sigma-normalized values must be the Table 5 values
    for c in ("B0025", "B0026", "B0027", "B0028"):
        assert abs(sd[f"protocol_{c}"]["sigma_normalized"]["s0_cov90_pct"]
                   - 100 * d["cross_protocol"][c]["soc"]["cov90"]["S0"]["coverage"]) < 1e-4
    gap = [sd[k]["absolute_residual"]["s0_cov90_pct"] - sd[k]["sigma_normalized"]["s0_cov90_pct"] for k in loco]
    # text: "covers more in N of the M fold-seed runs (up to X points) and at most Y points less in the others"
    pos = [g for g in gap if g > 0]
    assert len(gap) == 4 * len(SEEDS) and len(pos) >= 0.75 * len(gap) and min(gap) > -1, gap
    sig = [sd[k]["sigma_normalized"]["s0_cov90_pct_soc_pos"] for k in prot]
    ab = [sd[k]["absolute_residual"]["s0_cov90_pct_soc_pos"] for k in prot]
    ratio = [sd[k]["sigma_ratio_tgt_over_cal_median_soc_pos"] for k in prot]
    assert max(sig + ab) < 5 and max(ratio) < 1              # "almost nothing" (< 5% at 90% nominal, as for chemistry), "shrinks"
    rng = lambda xs, f="{:.1f}": (f.format(min(xs)), f.format(max(xs)))
    M = {}
    M["ScoreGapMin"], M["ScoreGapMax"] = rng(gap)
    M["ScoreGapPos"], M["ScoreGapN"] = str(len(pos)), str(len(gap))
    M["ScoreGapNegMax"] = f"{-min(gap):.1f}" if min(gap) < 0 else "0.0"
    M["ScoreProtSigMin"], M["ScoreProtSigMax"] = rng(sig)
    M["ScoreProtAbsMin"], M["ScoreProtAbsMax"] = rng(ab)
    M["ScoreProtSigmaMin"], M["ScoreProtSigmaMax"] = rng(ratio, "{:.2f}")
    return M


def ekf_macros():
    """Sec. 4.9 (R2.5a) from results/revision/ekf_revision.json (scripts/revision_ekf.py).
    A = published tuning (B0005+B0007); B = conformalized EKF (tuned B0005, calibrated B0007)."""
    d = json.load(open("results/revision/ekf_revision.json"))
    A, B = d["A_published_tuning"]["cells"], d["B_clean_conformal"]["cells"]
    assert d["A_published_tuning"]["tuned_on"] == ["B0005", "B0007"]
    assert d["B_clean_conformal"]["tuned_on"] == ["B0005"] and d["B_clean_conformal"]["calibrated_on"] == ["B0007"]
    pub = json.load(open("results/ekf_baseline.json"))                      # published run reproduced exactly
    assert all(abs(pub[c]["soc_cov3sigma"] - A[c]["soc_cov3sigma"]) < 1e-12 for c in A)
    same, prot, tune = CELLS, ["B0025", "B0026", "B0027", "B0028"], ["B0005", "B0007"]
    p1 = lambda x: f"{x:.1f}"
    rng = lambda xs, f=p1: (f(min(xs)), f(max(xs)))
    pc = lambda cells, k: [100 * A[c][k] for c in cells]
    M = {}
    M["EkfRmseSameMin"], M["EkfRmseSameMax"] = rng([A[c]["soc_rmse"] for c in same])
    M["EkfRmseProtMin"], M["EkfRmseProtMax"] = rng([A[c]["soc_rmse"] for c in prot])
    M["EkfCovTuneMin"], M["EkfCovTuneMax"] = rng(pc(tune, "soc_cov90"))
    M["EkfCovBsix"], M["EkfCovBeighteen"] = p1(100 * A["B0006"]["soc_cov90"]), p1(100 * A["B0018"]["soc_cov90"])
    M["EkfCovProtMin"], M["EkfCovProtMax"] = rng(pc(prot, "soc_cov90"))
    M["EkfThreeSameMin"], M["EkfThreeSameMax"] = rng(pc(same, "soc_cov3sigma"))
    M["EkfThreeProtMin"], M["EkfThreeProtMax"] = rng(pc(prot, "soc_cov3sigma"))
    M["EkfSohRatedMin"], M["EkfSohRatedMax"] = rng([A[c]["soh_rmse_vs_rated"] for c in same])
    M["EkfSohFirstMin"], M["EkfSohFirstMax"] = rng([A[c]["soh_rmse_vs_first_cycle"] for c in same])
    M["EkfSohCovMin"], M["EkfSohCovMax"] = rng(pc(same, "soh_cov90"))
    M["EkfSohProtRmseMin"], M["EkfSohProtRmseMax"] = rng([A[c]["soh_rmse_vs_rated"] for c in prot])
    M["EkfSohProtCovMin"], M["EkfSohProtCovMax"] = rng(pc(prot, "soh_cov90"), lambda x: f"{x:.0f}")
    M["EkfConfBsix"], M["EkfConfBeighteen"] = p1(100 * B["B0006"]["soc_conformal90"]), p1(100 * B["B0018"]["soc_conformal90"])
    M["EkfConfWBsix"], M["EkfConfWBeighteen"] = p1(B["B0006"]["soc_conformal90_mean_width_pp"]), p1(B["B0018"]["soc_conformal90_mean_width_pp"])
    M["EkfConfProtMin"], M["EkfConfProtMax"] = rng([100 * B[c]["soc_conformal90"] for c in ("B0025", "B0026", "B0027")])
    M["EkfConfBtwentyeight"], M["EkfConfWBtwentyeight"] = p1(100 * B["B0028"]["soc_conformal90"]), p1(B["B0028"]["soc_conformal90_mean_width_pp"])
    # claim guards for the prose
    assert max(pc(same + prot, "soc_cov3sigma")) < 99.7 and max(pc(same + prot, "soc_cov90")) < 50   # "under-cover badly"
    assert all(A[c]["soc_cov90"] < A[c]["soc_cov3sigma"] for c in same + prot)
    assert all(85 < 100 * B[c]["soc_conformal90"] and B[c]["soc_conformal90_mean_width_pp"] > 40 for c in ("B0006", "B0018"))  # "about half the range"
    assert max(100 * B[c]["soc_conformal90"] for c in ("B0025", "B0026", "B0027")) < 60
    return M


def r0_tables_and_macros():
    """Tables S13/S14 and the fixed-R0 sentences (R2.6), mean +- SD over the five v12-MSE backbones
    (results/revision/seeds/r0_sensitivity_s{k}.json, scripts/revision_sp4_r0.py, which first reproduces
    the former single-checkpoint files). Each fitted value is applied to the WHOLE random-split test set."""
    runs = [json.load(open(f"results/revision/seeds/r0_sensitivity_s{k}.json")) for k in SEEDS]
    pc = json.load(open("results/r0_per_cell.json"))                  # fitted from raw data, model-free
    fit = pc["per_cell_fit"]
    names = sorted(runs[0]["v12_mse"]["sweep"], key=lambda n: runs[0]["v12_mse"]["sweep"][n]["r0"])
    L = [r"\begin{tabular}{ccc}", r"\toprule", r"$R_0$ perturbation (\%) & SOC RMSE (\%) & SOH RMSE (\%) \\", r"\midrule"]
    by = {}
    for n in names:
        rows = [r["v12_mse"]["sweep"][n] for r in runs]
        pct = round(100 * (rows[0]["r0"] / 0.15 - 1))
        by[pct] = rows
        lab = "0" if pct == 0 else (f"$+{pct}$" if pct > 0 else f"$-{abs(pct)}$")
        L.append(f"{lab} & {ms([x['soc_rmse'] for x in rows])} & {ms([x['soh_rmse'] for x in rows])} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("r0_sensitivity.tex", L)
    L = [r"\begin{tabular}{lccc}", r"\toprule", r"Cell & Fitted $R_0$ ($\Omega$) & SOC RMSE (\%) & SOH RMSE (\%) \\", r"\midrule"]
    ev = {c: [r["v12_mse"]["fitted"][c] for r in runs] for c in CELLS}
    for c in CELLS:
        L.append(f"{c} & {fit[c]['r0_hat']:.4f} & {ms([x['soc_rmse'] for x in ev[c]])} & {ms([x['soh_rmse'] for x in ev[c]])} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("r0_per_cell.tex", L)
    low = [c for c in CELLS if abs(fit[c]["pct_diff_from_fixed"]) > 5]
    assert low == ["B0005", "B0007", "B0018"], low                  # text: three lower values, B0006 the exception
    for i in range(len(SEEDS)):                                      # guards per backbone
        base = by[0][i]["soc_rmse"]
        assert min(ev[c][i]["soc_rmse"] for c in low) > 5 * base     # "worsens accuracy sharply"
        assert ev["B0006"][i]["soc_rmse"] < 1.2 * base               # "almost unchanged"
        assert min(by[-10][i]["soc_rmse"], by[10][i]["soc_rmse"]) > 3 * base   # "highly sensitive"
    mean = lambda rows: st.mean(x["soc_rmse"] for x in rows)
    rng = lambda xs, f="{:.2f}": (f.format(min(xs)), f.format(max(xs)))
    M = {}
    M["RzeroBase"] = f"{mean(by[0]):.2f}"
    M["RzeroPmTenMin"], M["RzeroPmTenMax"] = rng([mean(by[-10]), mean(by[10])])
    M["RzeroPmThirtyMin"], M["RzeroPmThirtyMax"] = rng([mean(by[-30]), mean(by[30])])
    nig = [r["v12_nig"]["eval"] for r in runs]
    M["RzeroNigTenMin"], M["RzeroNigTenMax"] = rng([st.mean(x["0.135"]["soc_rmse"] for x in nig),
                                                   st.mean(x["0.165"]["soc_rmse"] for x in nig)])
    M["RzeroFitLowMin"], M["RzeroFitLowMax"] = rng([fit[c]["r0_hat"] for c in low], "{:.3f}")
    M["RzeroFitBsix"] = f"{fit['B0006']['r0_hat']:.3f}"
    M["RzeroFitRmseMin"], M["RzeroFitRmseMax"] = rng([mean(ev[c]) for c in low])
    M["RzeroFitRmseBsix"] = f"{mean(ev['B0006']):.2f}"
    return M


BASE_NAMES = {"cnn_bilstm": "CNN--BiLSTM family, reproduced",
              "pi_transformer": "Physics-guided Transformer family, reproduced"}


def uapi_seeds(variant):
    """Per-seed SOC and SOH RMSE (%) at full precision, seeds 0-4, in seed order."""
    if variant == "mse":
        r = {e["seed"]: e for e in json.load(open("results/_multiseed_raw/full.json"))}
        return [100 * r[s]["soc_rmse"] for s in range(5)], [100 * r[s]["soh_rmse"] for s in range(5)]
    r = {e["seed"]: e for e in json.load(open("results/revision/nig_multiseed_fullprecision.json"))["seeds"]}
    return [r[s]["soc_rmse"] for s in range(5)], [r[s]["soh_rmse"] for s in range(5)]


ABL = [("full", "Full"), ("no_eite", "No EITE"), ("no_nig", "Scalar MSE head"),
       ("no_dtag_ctba", "Shared representation"), ("no_prap", "No PRAP"), ("no_dct", "No DCT"),
       ("no_ctba", "No CTBA"), ("ctba_self", "CTBA self-projection (parameter-matched)"),
       ("no_ic", "No IC-inspired proxy")]


def ablation_summary():
    """Table 10 from the unrounded per-seed files; paired tests on shared seeds."""
    from scipy import stats as sps
    def raw(k):
        f = "results/revision/ctba_self.json" if k == "ctba_self" else f"results/_multiseed_raw/{k}.json"
        return {e["seed"]: (100 * e["soc_rmse"], 100 * e["soh_rmse"]) for e in json.load(open(f))}
    R = {k: raw(k) for k, _ in ABL}
    S = {}
    for k, _ in ABL:
        e = {"soc": [v[0] for v in R[k].values()], "soh": [v[1] for v in R[k].values()], "n": len(R[k])}
        if k != "full":
            sh = sorted(set(R[k]) & set(R["full"]))
            for i, t in enumerate(("soc", "soh")):
                e["p_" + t] = float(sps.ttest_rel([R[k][x][i] for x in sh], [R["full"][x][i] for x in sh]).pvalue)
        S[k] = e
    return R, S


def fmt_p(p):
    return r"$<$0.001" if p < 0.001 else f"{p:.3f}"


def table_ablation():
    R, S = ablation_summary()
    # SP4: every variant is paired with Full on the same five seeds
    assert all(sorted(R[k]) == [0, 1, 2, 3, 4] for k, _ in ABL), {k: sorted(R[k]) for k, _ in ABL}
    L = [r"\begin{tabular}{lcccccc}", r"\toprule",
         r"Variant & SOC RMSE (\%) & SOH RMSE (\%) & Seeds ($n$) & $p$ (SOC) & $p$ (SOH) \\", r"\midrule"]
    for k, lab in ABL:
        e = S[k]
        p = "-- & --" if k == "full" else f"{fmt_p(e['p_soc'])} & {fmt_p(e['p_soh'])}"
        L.append(f"{lab} & {ms(e['soc'], '{:.3f}')} & {ms(e['soh'], '{:.3f}')} & {e['n']} & {p} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("ablation.tex", L)


def table_cross_protocol():
    """Table S9: cross-protocol accuracy of the five v12-NIG seed checkpoints, mean +- SD over seeds
    per cell; the last row is the mean and the sample SD across the four cells of the seed means."""
    pa = load_protocol_acc()
    c, seeds = pa["cells"], pa["seeds"]
    cells = ["B0025", "B0026", "B0027", "B0028"]
    neg = lambda s: s.replace("-", "$-$")
    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"Cell & SOC RMSE (\%) & SOH RMSE (\%) & SOC $r$ & SOH $r$ \\", r"\midrule"]
    for k in cells:
        v = lambda f: [s[k][f] for s in seeds]
        L.append(f"{k} & {ms(v('soc_rmse'))} & {ms(v('soh_rmse'))} & {neg(ms(v('soc_r'), '{:.3f}'))} & "
                 f"{neg(ms(v('soh_r'), '{:.3f}'))} " + r"\\")
    L.append(r"\midrule")
    col = lambda f: [c[k][f] for k in cells]
    L.append(r"Across cells & " + " & ".join([ms(col("soc_rmse")), ms(col("soh_rmse")),
                                                neg(ms(col("soc_r"), "{:.3f}")), neg(ms(col("soh_r"), "{:.3f}"))]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("cross_protocol.tex", L)


def table_benchmark():
    """Table 2: random-split benchmark, multi-seed for baselines too (R1.5)."""
    # Baselines: fixed 400-epoch budget with best-validation checkpoint, the v12-NIG
    # protocol (R1.5). The earlier early-stopped runs are kept only for the sensitivity
    # sentence (results/revision/sensitivity/) and are never pooled with these.
    path = "results/revision/baselines_random_fixed.json"
    runs = json.load(open(path))
    for n, v in runs.items():
        assert v["patience"] == 0 and v["epochs_trained"] == v["epochs"] == 400, (n, v)
    assert sorted(runs) == sorted(f"{k}_seed{s}" for k in BASE_NAMES for s in range(5)), sorted(runs)
    # only runs produced by scripts/revision_baselines.py are pooled; the single-seed
    # numbers of the submitted version came from a different training protocol.
    L = [r"\begin{tabular}{lccc}", r"\toprule",
         r"Method & SOC RMSE (\%) & SOH RMSE (\%) & Seeds \\", r"\midrule"]
    for k, lab in BASE_NAMES.items():
        soc = [v["soc_rmse"] for n, v in runs.items() if n.startswith(k)]
        soh = [v["soh_rmse"] for n, v in runs.items() if n.startswith(k)]
        L.append(f"{lab} & {ms(soc)} & {ms(soh)} & {len(soc)} " + r"\\")
    # UAPI-Former rows from unrounded per-seed values. multiseed_v12_postfix.json and
    # multiseed_v12_nig.json hold seeds rounded to 3 and 2 decimals; computing SDs and
    # tests from them shifted the printed digits, so neither is read here.
    fmt3 = "{:.3f}"
    for lab, (soc, soh) in (("UAPI-Former v12-MSE", uapi_seeds("mse")), ("UAPI-Former v12-NIG", uapi_seeds("nig"))):
        # three decimals: v12-MSE and v12-NIG differ only in the third
        L.append(f"{lab} & {ms(soc, fmt3)} & {ms(soh, fmt3)} & {len(soc)} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("benchmark.tex", L)
    return sum(1 for n in runs)


def accuracy_macros():
    """Numbers quoted in the benchmark, cross-protocol and ablation prose, with guards."""
    from scipy import stats as sps
    M = {}
    ms_soc, ms_soh = uapi_seeds("mse")
    ng_soc, ng_soh = uapi_seeds("nig")
    M["MseSoc"], M["MseSocSd"] = f"{st.mean(ms_soc):.3f}", f"{st.stdev(ms_soc):.3f}"
    M["NigVsMsePSoc"] = f"{sps.ttest_rel(ng_soc, ms_soc).pvalue:.3f}"
    M["NigVsMsePSoh"] = f"{sps.ttest_rel(ng_soh, ms_soh).pvalue:.3f}"
    assert min(float(M["NigVsMsePSoc"]), float(M["NigVsMsePSoh"])) > 0.05   # "indistinguishable"

    c = load_protocol_acc()["cells"]
    v = [c[k]["soc_rmse"] for k in ("B0025", "B0026", "B0027", "B0028")]
    M["ProtSocMean"], M["ProtSocSd"] = f"{st.mean(v):.2f}", f"{st.stdev(v):.2f}"

    R, S = ablation_summary()
    sig = lambda t: {k for k, _ in ABL if k != "full" and S[k]["p_" + t] < 0.05}
    # Table S6 text and Figure S1 asterisks (five paired seeds, SP4)
    assert sig("soc") == {"no_eite", "no_dtag_ctba", "no_ctba"}, sig("soc")
    assert sig("soh") == {"no_eite", "no_dtag_ctba", "no_ctba"}, sig("soh")
    sh = sorted(set(R["full"]) & set(R["ctba_self"]) & set(R["no_ctba"]))
    assert sh == [0, 1, 2, 3, 4], sh
    full_sh = [R["full"][x][0] for x in sh]
    M["AblFullSocShared"] = f"{st.mean(full_sh):.3f}"
    M["AblSelfSoc"], M["AblSelfSocSd"] = f"{st.mean(S['ctba_self']['soc']):.3f}", f"{st.stdev(S['ctba_self']['soc']):.3f}"
    M["AblPSelfSoc"], M["AblPSelfSoh"] = f"{S['ctba_self']['p_soc']:.2f}", f"{S['ctba_self']['p_soh']:.2f}"
    assert S["ctba_self"]["p_soc"] > 0.05 and S["ctba_self"]["p_soh"] > 0.05        # "costs nothing measurable"
    M["AblCtbaCost"] = f"{st.mean(S['no_ctba']['soc']) - st.mean(full_sh):.2f}"
    p_cf = S["no_ctba"]["p_soc"]
    p_cs = float(sps.ttest_rel([R["no_ctba"][x][0] for x in sh], [R["ctba_self"][x][0] for x in sh]).pvalue)
    assert p_cf < 0.05 and p_cs < 0.05, (p_cf, p_cs)                                  # against both
    M["AblPCtba"], M["AblPCtbaSelf"] = fmt_p(p_cf), fmt_p(p_cs)
    M["AblPEiteSoc"], M["AblPEiteSoh"] = f"{S['no_eite']['p_soc']:.3f}", f"{S['no_eite']['p_soh']:.3f}"
    # Supplement S3.1 (iii) / S3.6 (R2.6): of the physics-derived components only EITE shows an effect;
    # on SOH it survives a Bonferroni correction over all tests, on SOC it does not; PRAP and IC are null
    n_tests = 2 * (len(ABL) - 1)
    assert all(S[k]["p_" + t] > 0.05 for k in ("no_prap", "no_ic") for t in ("soc", "soh"))
    assert S["no_eite"]["p_soh"] * n_tests < 0.05 < S["no_eite"]["p_soc"] * n_tests, (S["no_eite"], n_tests)
    M["AblNTests"] = str(n_tests)
    # App. C (R2.7): "its removal costs more SOC accuracy than removing any other single component"
    single = ("no_eite", "no_nig", "no_prap", "no_dct", "no_ctba", "no_ic")
    assert max(single, key=lambda k: st.mean(S[k]["soc"])) == "no_ctba"
    return M


def cp_and_chemistry_macros():
    """Sec. 4.6 and abstract numbers, all per seed and fold unless stated; with guards."""
    cp = [json.load(open(f)) for f in seed_files("results/revision/cp_baselines_s{s}.json")]
    per = lambda k, f: [r[c][k][f] for r in cp for c in CELLS]
    M = {"CpSeeds": {2: "two", 3: "three", 4: "four", 5: "five"}[len(cp)]}
    fm = lambda k, f, c: st.mean(r[c][k][f] for r in cp)                # fold mean over seeds
    tab = lambda k, c: round(100 * fm(k, "accepted_coverage", c), 1)    # as Table 8 displays it
    # CQR. In-distribution check first: conformalized on one random half of the source
    # calibration windows and evaluated on the other half, the heads must be calibrated
    # and narrow, otherwise the held-out-cell numbers say nothing about shift.
    dg = [r[c]["CQR"]["diag"] for r in cp for c in CELLS]
    src_c = [100 * g["src_half_coverage"] for g in dg]
    src_w = [g["src_half_width_pp"] for g in dg]
    assert min(src_c) > 80 and max(src_w) < 15, (src_c, src_w)
    # heads are sorted before conformalization; only a negative correction can empty an interval
    assert max(g["tgt_full_range_share"] for g in dg) == 0 and max(g["tgt_inverted_share"] for g in dg) < 0.01
    cq_c = [100 * x for x in per("CQR", "accepted_coverage")]
    fc_c, fc_w = [tab("CQR", c) for c in CELLS], [fm("CQR", "mean_width_pp", c) for c in CELLS]
    dl = [round(tab("CQR", c) - tab("S0", c), 1) for c in CELLS]
    assert min(dl) > 0 and max(fc_c) < 90, (dl, fc_c)           # "above S0, below nominal on every fold"
    nom = sorted({c for r in cp for c in CELLS if 100 * r[c]["CQR"]["accepted_coverage"] >= 90})
    assert len(nom) == 1, nom                                    # text names the one cell
    mlp = [tab("CQR_mlp", c) for c in CELLS]
    assert all(m < q for m, q in zip(mlp, fc_c)), mlp            # "MLP heads do no better"
    M.update(CqrSrcCovMin=f"{min(src_c):.1f}", CqrSrcCovMax=f"{max(src_c):.1f}",
             CqrSrcWidthMin=f"{min(src_w):.1f}", CqrSrcWidthMax=f"{max(src_w):.1f}",
             CqrFoldCovMin=f"{min(fc_c):.1f}", CqrFoldCovMax=f"{max(fc_c):.1f}",
             CqrFoldWidthMin=f"{min(fc_w):.1f}", CqrFoldWidthMax=f"{max(fc_w):.1f}",
             CqrDeltaMin=f"{min(dl):.1f}", CqrDeltaMax=f"{max(dl):.1f}",
             CqrNominalRuns=str(sum(x >= 90 for x in cq_c)), CqrRuns=str(len(cq_c)), CqrNominalCell=nom[0],
             CqrMlpFoldCovMin=f"{min(mlp):.1f}", CqrMlpFoldCovMax=f"{max(mlp):.1f}")
    # from the one-decimal means Table 8 displays, so a reader's subtraction matches
    d = {c: round(st.mean(100 * r[c]["Mondrian"]["accepted_coverage"] for r in cp), 1)
            - round(st.mean(100 * r[c]["S0"]["accepted_coverage"] for r in cp), 1) for c in CELLS}
    cmax = max(d, key=lambda c: abs(d[c]))
    M["MondrianDeltaMax"], M["MondrianDeltaMaxCell"] = f"{abs(d[cmax]):.1f}", cmax
    for k, t in (("S0", "Szero"), ("ACI", "Aci")):
        cv = [100 * x for x in per(k, "accepted_coverage")]
        w = per(k, "mean_width_pp")
        M[f"Cp{t}CovMin"], M[f"Cp{t}CovMax"] = f"{min(cv):.1f}", f"{max(cv):.1f}"
        M[f"Cp{t}WidthMin"], M[f"Cp{t}WidthMax"] = f"{min(w):.1f}", f"{max(w):.1f}"
    for c, t in (("B0006", "Bsix"), ("B0018", "Beighteen")):
        rf = [100 * r[c]["ACI"]["refusal"] for r in cp]
        M[f"AciRef{t}Min"], M[f"AciRef{t}Max"] = f"{min(rf):.1f}", f"{max(rf):.1f}"
    fc = [100 * x for x in per("ACI", "finite_and_correct")]
    M["AciFcMin"], M["AciFcMax"] = f"{min(fc):.1f}", f"{max(fc):.1f}"
    # ACI step-size sensitivity; the gamma = 0.01 entry is the primary run
    gs = sorted(cp[0][CELLS[0]]["ACI_gamma"], key=float)
    assert all(r[c]["ACI_gamma"]["0.01"] == {k: v for k, v in r[c]["ACI"].items()} for r in cp for c in CELLS)
    G = lambda g, f: [100 * r[c]["ACI_gamma"][g][f] for r in cp for c in CELLS]
    alt = [g for g in gs if g != "0.01"]                         # the two step sizes the text names
    fcg = [x for g in alt for x in G(g, "finite_and_correct")]
    fcf = [st.mean(100 * r[c]["ACI_gamma"][g]["finite_and_correct"] for r in cp) for g in alt for c in CELLS]
    assert max(fcf) < 90, fcf                                    # "below nominal on average on every held-out cell"
    M["AciGammaFcFoldMax"] = f"{max(fcf):.1f}"
    # "larger steps buy accepted coverage with more refusal": fold means over seeds, every cell
    fa = {g: [st.mean(100 * r[c]["ACI_gamma"][g]["accepted_coverage"] for r in cp) for c in CELLS] for g in gs}
    fr = {g: [st.mean(100 * r[c]["ACI_gamma"][g]["refusal"] for r in cp) for c in CELLS] for g in gs}
    for a, b in zip(gs, gs[1:]):
        assert all(x < y for x, y in zip(fa[a], fa[b])), (a, b, fa)
        assert all(x <= y for x, y in zip(fr[a], fr[b])) and sum(fr[b]) > sum(fr[a]), (a, b, fr)
    ref_hi = [max(G(g, "refusal")) for g in gs]
    assert ref_hi[-1] == max(ref_hi)                             # "up to X% of a cell's windows"
    M["AciGammaLo"], M["AciGammaHi"] = gs[0], gs[-1]
    M["AciGammaFcMax"], M["AciGammaRefMax"] = f"{max(fcg):.1f}", f"{ref_hi[-1]:.1f}"
    # SOC > 0 stratum (Sec. 3.1): size of the change and unchanged ranking of the four schemes
    prim = ("S0", "Mondrian", "CQR", "ACI")
    pm = lambda k, c: st.mean(100 * r[c]["soc_pos"][k]["accepted_coverage"] for r in cp)
    chg = [abs(tab(k, c) - pm(k, c)) for k in prim for c in CELLS]
    for c in CELLS:
        assert sorted(prim, key=lambda k: tab(k, c)) == sorted(prim, key=lambda k: pm(k, c)), c
    M["CpPosChangeMax"] = f"{max(chg):.1f}"
    s0 = [100 * x for x in per("S0", "finite_and_correct")]
    # text: once refusals count as failures, ACI stays below nominal on average on every held-out cell
    assert max(st.mean(100 * r[c]["ACI"]["finite_and_correct"] for r in cp) for c in CELLS) < 90
    # abstract: source-calibrated coverage across chemistries, zero-shot and EOT
    cc = load_cc()
    # Sec. 4.4 / abstract: S2 on the LOCO folds, from the same clean study as Table 5
    s2 = {c: [cc[f"loco_clean_s{i}"][c]["soc"]["cov90"] for i in SEEDS] for c in CELLS}
    assert all(e["S2"]["refusal"] == 0 for c in CELLS for e in s2[c])
    s2c = [100 * e["S2"]["accepted_coverage"] for c in CELLS for e in s2[c]]
    s2w = [100 * e["S2"]["mean_width"] for c in CELLS for e in s2[c]]
    s2d = [abs(st.mean(100 * e["S2"]["accepted_coverage"] for e in s2[c])
               - st.mean(100 * e["S0"]["coverage"] for e in s2[c])) for c in CELLS]
    M["STwoLocoCovMin"], M["STwoLocoCovMax"] = f"{min(s2c):.1f}", f"{max(s2c):.1f}"
    M["STwoLocoWidthMin"], M["STwoLocoWidthMax"] = f"{min(s2w):.1f}", f"{max(s2w):.1f}"
    M["STwoLocoDeltaMax"] = f"{max(abs(shown(st.mean(100 * e['S2']['accepted_coverage'] for e in s2[c])) - shown(st.mean(100 * e['S0']['coverage'] for e in s2[c]))) for c in CELLS):.1f}"
    # abstract: S2, CQR and Mondrian all stay within this many points of S0 on every held-out cell
    within = max(s2d + [abs(x) for x in dl] + [abs(tab("Mondrian", c) - tab("S0", c)) for c in CELLS])
    M["CpWithinMax"] = f"{within:.0f}"
    assert within <= int(M["CpWithinMax"])                       # "within N points" stays true after rounding
    # v12-NIG zero-shot (Table 5) and every v12-MSE source-calibrated row of the cross-chemistry
    # table; the former EOT-warped entries paired v12-MSE transducers with v12-NIG latents (invalid)
    chem = [100 * e["zero_shot"]["soc"]["cov90"]["S0"]["coverage"] for k in ("chemistry_calce", "chemistry_oxford")
            for e in cc[k + "_seeds"]]
    _, chs = load_seed_json(sp.chem_cov_json)
    chem += [100 * e["soc"]["S0"]["coverage"] for ch in chs for k in ch["source"] for e in ch["source"][k].values()]
    assert len(chem) == 8 * len(SEEDS), chem     # per seed: 2 NIG zero-shot + 6 MSE source rows
    M["ChemCovMax"] = f"{max(chem):.2f}"
    rm = cross_chemistry_rmse()
    red = [100 * (1 - rm[(k, "eot")] / rm[(k, "zero")]) for k in ("calce", "oxford", "mit_tri")]
    M["EotRedMin"], M["EotRedMax"] = f"{min(red):.0f}", f"{max(red):.0f}"
    assert min(red) > 0                          # abstract: "cuts mean point error" on every chemistry
    return M


def b0006_macros(diag):
    """Sec. 4.8 / abstract / Discussion / Conclusions: every B0006 number, with guards on
    the qualitative sentences. SOH labels are capacity / first-cycle capacity."""
    M = {}
    rng = lambda v, f="{:.1f}": (f.format(min(v)), f.format(max(v)))
    # fade, from the target SOH minima (identical across seeds)
    smin = {c: {round(d[c]["target_soh_min"], 6) for d in diag} for c in CELLS}
    assert all(len(v) == 1 for v in smin.values())
    smin = {c: v.pop() for c, v in smin.items()}
    assert min(smin, key=smin.get) == "B0006"                       # "further than any other NASA cell"
    for c, t in (("B0005", "Bfive"), ("B0006", "Bsix"), ("B0007", "Bseven"), ("B0018", "Beighteen")):
        M[f"SohMin{t}"] = f"{100 * smin[c]:.1f}"
    # text: "below 0.693, the lowest SOH of its three source cells (B0005's minimum), below which no
    # training cycle lies in any seed": the per-seed training floor is B0005's minimum or within 0.001 above
    # it (a seed whose LOCO partition puts that cycle outside training), and the out-of-range share is
    # the same in every seed
    floor = [d["B0006"]["source_soh_range"][0] for d in diag]
    assert all(0 <= f - smin["B0005"] < 1e-3 for f in floor), (floor, smin["B0005"])
    assert len({round(d["B0006"]["frac_target_below_source_range"], 9) for d in diag}) == 1
    assert all(d[c]["frac_target_below_source_range"] == 0 for d in diag for c in CELLS if c != "B0006")
    b6 = [d["B0006"] for d in diag]
    M["BsixInRmseMin"], M["BsixInRmseMax"] = rng([x["inside"]["S0"]["rmse_pct"] for x in b6])
    M["BsixInCovMin"], M["BsixInCovMax"] = rng([100 * x["inside"]["S0"]["coverage"] for x in b6])
    M["BsixBelowRmseMin"], M["BsixBelowRmseMax"] = rng([x["below"]["S0"]["rmse_pct"] for x in b6])
    M["BsixRatioBelowMin"], M["BsixRatioBelowMax"] = rng([x["below"]["mean_density_ratio"] for x in b6])
    M["BsixRatioInMin"], M["BsixRatioInMax"] = rng([x["inside"]["mean_density_ratio"] for x in b6])
    # "within the source range B0006 behaves like the other folds": inside the per-seed range of S0 elsewhere
    other = [100 * d[c]["inside"]["S0"]["coverage"] for d in diag for c in CELLS if c != "B0006"]
    assert min(other) <= float(M["BsixInCovMin"]) and float(M["BsixInCovMax"]) <= max(other)
    # per-window signals that track the error on B0006 (Spearman with |SOC error|)
    rho = {k: [x["spearman_abs_err"][k] for x in b6] for k in ("knn_dist", "density_ratio", "sigma")}
    assert all(v > 0 for vs in rho.values() for v in vs)                # "rise with the error"
    for k, t in (("knn_dist", "Knn"), ("density_ratio", "Ratio"), ("sigma", "Sigma")):
        M[f"BsixRho{t}Min"], M[f"BsixRho{t}Max"] = rng(rho[k], "{:.2f}")
    # support gate at p = 99
    gate = [json.load(open(f)) for f in seed_files("results/revision/support_gate_s{s}.json")]
    assert len(gate) == len(diag)
    g6 = [(x["B0006"]["p99.0"]["detection_rate_below"], x["B0006"]["p99.0"]["false_alarm_rate_inside"],
           x["B0006"]["ungated_coverage"], x["B0006"]["p99.0"]["accepted_coverage"]) for x in gate]
    det = [100 * g[0] for g in g6]
    gain = [100 * (g[3] - g[2]) for g in g6]
    n_half = sum(v > 50 for v in det)
    words = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    # text: "flags X-Y% of B0006's out-of-range windows depending on the training run (more than half in
    # N of the M runs)"; detection is run-dependent: it spans more than a factor of three
    assert max(det) > 3 * min(det) and 0 < n_half < len(det), det
    assert min(gain) > 0, gain                                       # coverage rises in every run
    M["GateDetMin"], M["GateDetMax"] = rng(det)
    M["GateFaMin"], M["GateFaMax"] = rng([100 * g[1] for g in g6])
    M["GateHalfRuns"], M["GateRuns"] = words[n_half], words[len(g6)]
    M["GateGainMin"], M["GateGainMax"] = rng(gain)
    M["GateBsixAccMax"] = f"{100 * max(h[3] for h in g6):.1f}"
    assert float(M["GateBsixAccMax"]) < 90                              # "does not restore nominal"
    # the gate on the three cells that have no out-of-range windows
    oth = [(100 * x[c]["p99.0"]["refusal_rate"], 100 * (x[c]["p99.0"]["accepted_coverage"] - x[c]["ungated_coverage"]), c)
           for x in gate for c in CELLS if c != "B0006"]
    top = max(oth)
    M["GateOtherRefMin"], M["GateOtherRefMax"], M["GateOtherRefMaxCell"] = f"{min(o[0] for o in oth):.2f}", f"{top[0]:.1f}", top[2]
    M["GateOtherCovGainMax"] = f"{max(o[1] for o in oth):.1f}"
    # "not specific to label-support extrapolation": it refuses a substantial share of windows on cells
    # that have no out-of-range windows (the coverage change there is quoted as a number, not a claim)
    assert top[0] > 10, top
    return M


def runtime_macros():
    """Sec. 4.12 runtime and storage. Backbone latency pooled over three 100-run sessions in one idle
    window (session means differ by up to ~0.4 ms, so one session understates the variation); the
    calibration stages were timed in the same window; serialized FP32 / INT8 model size."""
    import numpy as np
    files = ["results/revision/inference_benchmark_idle.json",
             "results/revision/inference_benchmark_idle_s2.json",
             "results/revision/inference_benchmark_idle_s3.json"]
    bbs = [json.load(open(f)) for f in files]
    rt = json.load(open("results/revision/runtime_s1.json"))
    sz = json.load(open("results/revision/model_storage.json"))["checkpoints/nasa_v12/best.pt"]
    for bb in bbs:                                   # same checkpoint, CPU threads, batch; recorded idle
        assert bb["checkpoint"] == "checkpoints/nasa_v12/best.pt" and bb["batch_size"] == 1
        assert bb["torch_num_threads"] == rt["threads"] and bb.get("other_jobs_running") == 0
    assert rt.get("other_jobs_running") == 0
    runs = np.concatenate([np.asarray(bb["all_runs_ms"]) for bb in bbs])
    sess = [bb["mean_ms"] for bb in bbs]
    mean = float(runs.mean())
    M = {"BbMean": f"{mean:.2f}", "BbPninetyfive": f"{np.percentile(runs, 95):.2f}", "BbMax": f"{runs.max():.2f}",
         "BbRuns": str(len(runs)), "BbSessions": ["", "one", "two", "three", "four"][len(bbs)],
         "BbSessMin": f"{min(sess):.2f}", "BbSessMax": f"{max(sess):.2f}",
         "BbThreads": str(bbs[0]["torch_num_threads"]),
         "RtNCal": f"{rt['n_cal']:,}".replace(",", "{,}"), "RtNTgt": f"{rt['n_target_batch']:,}".replace(",", "{,}"),
         "RtFitMean": f"{rt['s1_discriminator_fit_s']['mean']:.2f}", "RtFitMax": f"{rt['s1_discriminator_fit_s']['max']:.2f}",
         "RtPerMean": f"{rt['s1_per_window_ms']['mean']:.3f}", "RtPerPninetyfive": f"{rt['s1_per_window_ms']['p95']:.3f}",
         "RtStwo": f"{rt['s2_batch_weights_s']['mean']:.3f}",
         "RtPerShare": f"{100 * rt['s1_per_window_ms']['mean'] / mean:.0f}",
         "StorageFp": f"{sz['fp32_mb']:.2f}", "StorageInt": f"{sz['int8_mb']:.2f}"}
    assert len(runs) == 300
    assert rt["s0_per_window_ms"]["mean"] < 0.001                        # "scalar multiply (<0.001 ms)"
    assert rt["s1_per_window_ms"]["mean"] < 0.1 * mean                   # "computation is not the bottleneck"
    # the timed fold is a real clean-LOCO fold: its calibration size lies in the LOCO range
    lr = loco_runs()
    ncal = [r[c]["n_cal"] for r in lr.values() for c in CELLS if c in r]
    assert min(ncal) <= rt["n_cal"] <= max(ncal), (rt["n_cal"], min(ncal), max(ncal))
    return M


def sequential_macros():
    """Sequential vs transductive S1 (Section 3.3, abstract), per seed and fold, on the later-cycle
    windows (accepted-point coverage, the quantity in Table 5 panel B). Case-specific wording of the
    three-seed text was replaced by counts and ranges that hold at any number of seeds."""
    runs = seq_runs()
    A = lambda e: 100 * e["accepted_coverage"]
    ks = ("seq_k1", "seq_k5", "seq_k20")
    gain = {(s, c, k): A(r[c][k]) - A(r[c]["S0"]) for s, r in runs.items() for c in CELLS for k in ks + ("transductive",)}
    seqg = [v for (s, c, k), v in gain.items() if k in ks]
    word = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
            8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
            15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty"}
    cases = [(s, c) for s in runs for c in CELLS]
    big = [(s, c) for s, c in cases if max(gain[(s, c, k)] for k in ks) > 5]
    loss = [(s, c) for s, c in cases if min(gain[(s, c, k)] for k in ks) < -5]
    helped = [(s, c) for s, c in cases if gain[(s, c, "transductive")] > 5]
    tloss = [(s, c) for s, c in cases if gain[(s, c, "transductive")] < -5]
    # "does not reliably improve on doing nothing": large gains and large losses both occur
    assert big and loss and len(loss) >= len(big) - 1, (big, loss)
    sgn = lambda x: f"{x:+.1f}".replace("-", "$-$")
    # nominal level: the fold mean over seeds of every prefix length stays below it
    fold_mean = {(c, k): st.mean(A(r[c][k]) for r in runs.values()) for c in CELLS for k in ks}
    assert max(fold_mean.values()) < 90, fold_mean
    keep_small = [max(gain[(s, c, k)] for k in ("seq_k1", "seq_k5")) for s, c in helped]
    keep20 = [gain[(s, c, "seq_k20")] for s, c in helped]
    assert min(keep20) < 5 < max(keep20), keep20                       # "may or may not recover"
    refs = [100 * r[c][k]["refusal"] for r in runs.values() for c in CELLS for k in ks + ("transductive",)]
    assert sum(x == 0 for x in refs) >= 0.6 * len(refs), refs          # "mostly fails silently"
    M = {"SeqCases": word[len(cases)], "SeqBigGain": word[len(big)], "SeqBigLoss": word[len(loss)],
         "SeqDeltaMin": sgn(min(seqg)), "SeqDeltaMax": sgn(max(seqg)),
         "SeqFoldMeanMax": f"{max(fold_mean.values()):.1f}",
         "SeqCovMax": f"{max(A(r[c][k]) for r in runs.values() for c in CELLS for k in ks):.1f}",
         "TransBigGain": word[len(helped)], "TransBigLoss": word[len(tloss)],
         "TransGainMin": f"{min(gain[(s, c, 'transductive')] for s, c in helped):.1f}",
         "TransGainMax": f"{max(gain[(s, c, 'transductive')] for s, c in helped):.1f}",
         "SeqSmallMin": sgn(min(keep_small)), "SeqSmallMax": sgn(max(keep_small)),
         "SeqKtwentyMin": sgn(min(keep20)), "SeqKtwentyMax": sgn(max(keep20)),
         "SeqRefZero": str(sum(x == 0 for x in refs)), "SeqRefN": str(len(refs)),
         "SeqRefMax": f"{max(refs):.1f}"}
    # identical evaluation sets across seeds (text gives the two sizes)
    nev = {c: {r[c]["n_eval_windows"] for r in runs.values()} for c in CELLS}
    assert all(len(v) == 1 for v in nev.values()) and len({list(nev[c])[0] for c in CELLS[:3]}) == 1
    M["SeqEvalMain"] = f"{list(nev['B0005'])[0]:,}".replace(",", "{,}")
    M["SeqEvalBeighteen"] = f"{list(nev['B0018'])[0]:,}".replace(",", "{,}")
    M["SeqSeeds"] = word[len(runs)]
    # mechanism (k = 20): SOH displacement and latent separability of the first twenty cycles
    mk = [(r[c]["mechanism"], s, c) for s, r in runs.items() for c in CELLS]
    below = [m["k20"]["eval_below_fit_soh_pct"] for m, _, _ in mk]
    fit_auc = [m["k20"]["fit_vs_eval_auc"] for m, _, _ in mk]
    cal_auc = [m["cal_vs_eval_auc"] for m, _, _ in mk]
    higher = sum(f > c_ for f, c_ in zip(fit_auc, cal_auc))
    assert min(fit_auc) > 0.9 and higher >= len(mk) / 2                # "about as easily, more often more"
    M.update(SeqBelowMin=f"{min(below):.1f}", SeqBelowMax=f"{max(below):.1f}",
             SeqAucFitMin=f"{min(fit_auc):.3f}", SeqAucFitMax=f"{max(fit_auc):.3f}",
             SeqAucCalMin=f"{min(cal_auc):.3f}", SeqAucCalMax=f"{max(cal_auc):.3f}",
             SeqAucHigher=word[higher], SeqAucCases=word[len(mk)])
    return M


def baselines_sensitivity_macros():
    """R1.6 training-rule sensitivity: baselines retrained for a fixed 100 epochs with the
    best-validation checkpoint (results/revision/sensitivity/). Primary Table 7 keeps the
    rule UAPI-Former's LOCO models used; every sentence about the sensitivity run is
    asserted here."""
    import os as _os
    sd = "results/revision/sensitivity"
    files = sorted(glob.glob(f"{sd}/baseline_conformal_fixed100_s*.json"))
    if not files:
        return {}
    F = {f.split("_s")[-1].split(".")[0]: json.load(open(f)) for f in files}
    tr_f = json.load(open(f"{sd}/baselines_loco_fixed100.json"))
    tr_p = json.load(open("results/revision/baselines_loco.json"))
    seeds, S = baselines_shift_summary()
    assert sorted(F) == seeds, (sorted(F), seeds)          # same seeds as Table 7
    lc = loco_runs()
    E = [k for k, _ in BSHIFT_ESTS]
    def fx(k, c, f):
        if f == "rmse":
            return st.mean(lc[x][c]["soc_rmse"] if k == "uapi_former" else tr_f[f"{k}_s{x}_{c}"]["soc_rmse"] for x in seeds)
        key = {"s0": ("S0", "coverage"), "s1": ("S1", "accepted_coverage")}[f]
        return st.mean(100 * F[x][c][k][key[0]][key[1]] for x in seeds)
    pr = lambda k, c, f: st.mean(S[(k, c)][f])
    M = {}
    assert all(v["patience"] == 0 and v["epochs_trained"] == 100 for v in tr_f.values())
    be = [v["best_epoch"] for v in tr_f.values()]
    M["SensBestEpochMin"], M["SensBestEpochMax"] = str(min(be)), str(max(be))
    # pinned seeds only: baselines_loco.json also holds the seed 3-4 runs (SP2, not yet in the text)
    import re as _re
    es = [v["best_epoch"] for n, v in tr_p.items() if "best_epoch" in v
          and int(_re.search(r"_s(\d+)_B\d{4}$", n).group(1)) in SEEDS]
    M["EsBestEpochMin"], M["EsBestEpochMax"] = str(min(es)), str(max(es))
    ue = [r[c]["best_epoch"] for r in lc.values() for c in CELLS]
    M["UapiLocoBestEpochMin"], M["UapiLocoBestEpochMax"] = str(min(ue)), str(max(ue))
    # finding 1 still holds: every estimator below nominal on the two hardest folds
    for c, t in (("B0006", "Bsix"), ("B0018", "Beighteen")):
        v = max(fx(k, c, "s0") for k in E)
        assert v < 90, (c, v)
        M[f"SensCov{t}Max"] = f"{v:.1f}"
    # finding 2 still holds: weighting close to inert for the baselines
    d = max(abs(fx(k, c, "s1") - fx(k, c, "s0")) for k in E[1:] for c in CELLS)
    assert d < min(abs(pr("uapi_former", c, "s1") - pr("uapi_former", c, "s0")) for c in ("B0005", "B0007", "B0018"))
    M["SensBaseDeltaMax"] = f"{d:.1f}"
    # finding 3 still holds
    assert all(fx("uapi_former", c, "rmse") < fx("cnn_bilstm", c, "rmse") for c in CELLS)
    assert all(fx("uapi_former", c, "s0") < fx("cnn_bilstm", c, "s0") for c in CELLS)
    low = sum(fx("uapi_former", c, "s0") == min(fx(k, c, "s0") for k in E) for c in CELLS)
    M["SensUapiLowestCov"] = {4: "all four", 3: "three of the four", 2: "two of the four"}[low]
    # longer training made the physics-guided Transformer better on BOTH axes where it helped
    for c, t in (("B0007", "Bseven"), ("B0018", "Beighteen")):
        r0, r1 = pr("pi_transformer", c, "rmse"), fx("pi_transformer", c, "rmse")
        c0, c1 = pr("pi_transformer", c, "s0"), fx("pi_transformer", c, "s0")
        assert r0 - r1 > 0.5 and c1 > c0, (c, r0, r1, c0, c1)        # "both more accurate and better covered"
        M[f"SensPiRmse{t}Es"], M[f"SensPiRmse{t}Fixed"] = f"{r0:.2f}", f"{r1:.2f}"
        M[f"SensPiCov{t}Es"], M[f"SensPiCov{t}Fixed"] = f"{c0:.1f}", f"{c1:.1f}"
    # ...and on B0018 better than UAPI-Former on both
    assert fx("pi_transformer", "B0018", "rmse") < pr("uapi_former", "B0018", "rmse")
    assert fx("pi_transformer", "B0018", "s0") > pr("uapi_former", "B0018", "s0")
    return M


BSHIFT_ESTS = [("uapi_former", "UAPI-Former"), ("cnn_bilstm", "CNN--BiLSTM"),
               ("pi_transformer", "Physics-guided Transformer")]


def baselines_shift_summary():
    """R1.6 per (estimator, fold): lists over the seeds that have a conformal run.

    Seed-matched throughout: an estimator's RMSE and its coverage at a fold come
    from the same training runs, and all three estimators use the same seeds."""
    runs = {f.split("_s")[-1].split(".")[0]: json.load(open(f))
            for f in seed_files("results/revision/baseline_conformal_s{s}.json")}
    if not runs:
        return [], {}
    base = json.load(open("results/revision/baselines_loco.json"))
    lc = loco_runs()
    S = {}
    for key, _ in BSHIFT_ESTS:
        for c in CELLS:
            d = S.setdefault((key, c), {f: [] for f in ("rmse", "s0", "s1", "ref", "auc", "ess")})
            for seed, r in runs.items():                       # KeyError = missing run: fail loudly
                d["rmse"].append(lc[seed][c]["soc_rmse"] if key == "uapi_former"
                                 else base[f"{key}_s{seed}_{c}"]["soc_rmse"])
                e = r[c][key]
                d["s0"].append(100 * e["S0"]["coverage"])
                d["s1"].append(100 * e["S1"]["accepted_coverage"])
                d["ref"].append(100 * e["S1"]["refusal"])
                d["auc"].append(e["S1"]["auc"])
                d["ess"].append(100 * e["S1"]["ess_frac"])
    return sorted(runs), S


def table_baselines_shift():
    """R1.6: every estimator under the same clean LOCO folds and calibration."""
    seeds, S = baselines_shift_summary()
    if not S:
        return 0
    L = [r"\begin{tabular}{llcccc}", r"\toprule",
         r"Estimator & Quantity & " + " & ".join(CELLS) + r" \\", r"\midrule"]
    rows = [(r"SOC RMSE (\%)", "rmse", "{:.2f}"), (r"S0 coverage (\%)", "s0", "{:.1f}"),
            (r"S1 accepted coverage (\%)", "s1", "{:.1f}"),
            ("Discriminator AUC", "auc", "{:.2f}"), (r"S1 ESS (\% of $n_{\mathrm{cal}}$)", "ess", "{:.0f}")]
    for key, lab in BSHIFT_ESTS:
        for i, (rlab, f, fmt) in enumerate(rows):
            L.append((lab if i == 0 else "") + f" & {rlab} & "
                     + " & ".join(ms(S[(key, c)][f], fmt) for c in CELLS) + r" \\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("baselines_shift.tex", L)
    return len(seeds)


def baselines_shift_macros():
    """Numbers and counts quoted in the R1.6 paragraph, plus guards on its claims.

    Each qualitative sentence in that paragraph is asserted here, so a rerun that
    makes one untrue stops the build instead of leaving false prose in the paper."""
    seeds, S = baselines_shift_summary()
    if not S:
        return {}
    m = lambda k, c, f: st.mean(S[(k, c)][f])
    sd = lambda k, c, f: st.stdev(S[(k, c)][f])
    E = [k for k, _ in BSHIFT_ESTS]
    BASE = E[1:]
    words = {4: "all four", 3: "three of the four", 2: "two of the four", 1: "one of the four", 0: "none of the four"}
    tag = {"B0005": "Bfive", "B0006": "Bsix", "B0007": "Bseven", "B0018": "Beighteen"}
    M = {"BshiftSeeds": {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}[len(seeds)]}

    # (1) no estimator reaches nominal on the two hardest folds
    for c in ("B0006", "B0018"):
        v = [m(k, c, "s0") for k in E]
        assert max(v) < 80, ("claim 'every estimator well below nominal' fails at", c, v)
        M[f"BshiftCov{tag[c]}Min"], M[f"BshiftCov{tag[c]}Max"] = f"{min(v):.1f}", f"{max(v):.1f}"

    # (2) S1 barely moves the baselines ...
    # differences quoted in the text are taken between the one-decimal means Table 4 displays,
    # so a reader's subtraction matches; the assertions below use the unrounded values
    dbase = max(abs(m(k, c, "s1") - m(k, c, "s0")) for k in BASE for c in CELLS)
    M["BshiftBaseDeltaMax"] = f"{max(abs(shown(m(k, c, 's1')) - shown(m(k, c, 's0'))) for k in BASE for c in CELLS):.1f}"
    # ... but moves UAPI-Former a lot, in both directions, and not stably across seeds
    du = {c: [b - a for a, b in zip(S[("uapi_former", c)]["s0"], S[("uapi_former", c)]["s1"])] for c in CELLS}
    for c in CELLS:
        M[f"BshiftUapiDelta{tag[c]}"] = f"{abs(shown(m('uapi_former', c, 's1')) - shown(m('uapi_former', c, 's0'))):.1f}"
        M[f"BshiftUapiDelta{tag[c]}Min"] = f"{min(du[c]):+.1f}".replace("-", "$-$")
        M[f"BshiftUapiDelta{tag[c]}Max"] = f"{max(du[c]):+.1f}".replace("-", "$-$")
    assert st.mean(du["B0005"]) < -dbase and st.mean(du["B0007"]) < -dbase, du
    assert st.mean(du["B0018"]) > dbase, du

    # (3) accuracy ranking and calibration ranking disagree
    acc = sum(m("uapi_former", c, "rmse") < m("cnn_bilstm", c, "rmse") for c in CELLS)
    lowcov = sum(m("uapi_former", c, "s0") < m("cnn_bilstm", c, "s0") for c in CELLS)
    lowest = [c for c in CELLS if m("uapi_former", c, "s0") == min(m(k, c, "s0") for k in E)]
    worst_acc = [c for c in CELLS if m("uapi_former", c, "rmse") == max(m(k, c, "rmse") for k in E)]
    gaps = {c: m("cnn_bilstm", c, "s0") - m("uapi_former", c, "s0") for c in CELLS}
    shown_gaps = [shown(m("cnn_bilstm", c, "s0")) - shown(m("uapi_former", c, "s0")) for c in CELLS]
    M.update(BshiftUapiMoreAccurate=words[acc], BshiftUapiLowerCov=words[lowcov],
             BshiftUapiLowestCov=words[len(lowest)], BshiftUapiLeastAccurate=words[len(worst_acc)],
             BshiftCovGapMin=f"{min(shown_gaps):.1f}", BshiftCovGapMax=f"{max(shown_gaps):.1f}")
    for c in CELLS:
        for k, kt in (("uapi_former", "Uapi"), ("cnn_bilstm", "Cnn")):
            M[f"Bshift{kt}Cov{tag[c]}"] = f"{m(k, c, 's0'):.1f}"
            M[f"Bshift{kt}Rmse{tag[c]}"] = f"{m(k, c, 'rmse'):.2f}"
    gmin, gmax = min(gaps, key=gaps.get), max(gaps, key=gaps.get)
    M["BshiftCovGapMinCell"], M["BshiftCovGapMaxCell"] = gmin, gmax
    # text (SP4): "lower ... on three of the four, by X-Y points (largest on C); on D the two differ by Z
    # points, within the seed-to-seed spread" -- exactly one fold is not lower, and its gap is within the SD
    lower = [c for c in CELLS if gaps[c] > 0]
    even = [c for c in CELLS if gaps[c] <= 0]
    assert len(even) == 1 and len(lower) == lowcov, gaps
    sg = dict(zip(CELLS, shown_gaps))
    M["BshiftCovGapLowMin"] = f"{min(sg[c] for c in lower):.1f}"
    M["BshiftCovEvenCell"], M["BshiftCovEvenGap"] = even[0], f"{abs(sg[even[0]]):.1f}"
    assert abs(gaps[even[0]]) < max(sd("uapi_former", even[0], "s0"), sd("cnn_bilstm", even[0], "s0")), gaps
    # text: UAPI-Former "never has the highest RMSE" of the three
    assert not worst_acc, worst_acc

    # (4) representation separability, the proposed mechanism for (2)
    for k, kt in zip(E, ("Uapi", "Cnn", "Pi")):
        M[f"BshiftAuc{kt}Min"] = f"{min(m(k, c, 'auc') for c in CELLS):.2f}"
        M[f"BshiftAuc{kt}Max"] = f"{max(m(k, c, 'auc') for c in CELLS):.2f}"
        M[f"BshiftEss{kt}Min"] = f"{min(m(k, c, 'ess') for c in CELLS):.0f}"
        M[f"BshiftEss{kt}Max"] = f"{max(m(k, c, 'ess') for c in CELLS):.0f}"
    # text: UAPI-Former's latent is more separable than both baselines' on B0005-B0007,
    # while on B0018 all three are about equally separable
    other = [c for c in CELLS if c != "B0018"]
    for c in other:
        assert m("uapi_former", c, "auc") > max(m(k, c, "auc") for k in BASE), c
    M["BshiftAucUapiOtherMin"] = f"{min(m('uapi_former', c, 'auc') for c in other):.2f}"
    M["BshiftAucUapiOtherMax"] = f"{max(m('uapi_former', c, 'auc') for c in other):.2f}"
    M["BshiftAucBaseOtherMin"] = f"{min(m(k, c, 'auc') for k in BASE for c in other):.2f}"
    M["BshiftAucBaseOtherMax"] = f"{max(m(k, c, 'auc') for k in BASE for c in other):.2f}"
    a18 = [m(k, "B0018", "auc") for k in E]
    e18 = [m(k, "B0018", "ess") for k in E]
    assert max(a18) - min(a18) <= 0.05, a18
    M["BshiftAucBeighteenMin"], M["BshiftAucBeighteenMax"] = f"{min(a18):.2f}", f"{max(a18):.2f}"
    M["BshiftEssBeighteenMin"], M["BshiftEssBeighteenMax"] = f"{min(e18):.0f}", f"{max(e18):.0f}"
    # caption: S1 refusal is zero everywhere except one (estimator, fold, seed)
    # Table 4 caption: in how many estimator-fold-seed runs S1 refused any window, and the largest refusal
    allr = [(r, k, c) for k in E for c in CELLS for r in S[(k, c)]["ref"]]
    nz = [x for x in allr if x[0] > 0]
    assert len(nz) < len(allr) / 4, nz                                   # "refused no window in most runs"
    top = max(allr)
    M["BshiftRefRuns"], M["BshiftAllRuns"] = str(len(nz)), str(len(allr))
    M["BshiftMaxRefusal"] = f"{top[0]:.1f}"
    M["BshiftMaxRefusalCell"] = top[2]
    M["BshiftMaxRefusalEst"] = {"uapi_former": "UAPI-Former", "cnn_bilstm": "CNN--BiLSTM",
                                "pi_transformer": "physics-guided Transformer"}[top[1]]
    return M


def table_support_diagnostics():
    """ESS, discriminator AUC and overall inclusion (appendix, kept for completeness)."""
    d = load_cc()
    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"Shift level & Discriminator AUC & S1 ESS & SOC overall inclusion (\%) & SOH overall inclusion (\%) \\",
         r"\midrule"]
    for _, lab, spec in SHIFTS:
        try:
            es = _entries(spec, d, None)
        except KeyError:
            continue
        L.append(f"{lab} & {ms([e['discriminator_auc'] for e in es], '{:.3f}')} & "
                 f"{ms([e['ess']['S1'] for e in es], '{:.1f}')} & "
                 f"{ms([100*e['soc']['cov90']['S1']['coverage'] for e in es], '{:.1f}')} & "
                 f"{ms([100*e['soh']['cov90']['S1']['coverage'] for e in es], '{:.1f}')} " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("support_diagnostics.tex", L)
    return len(L)


if __name__ == "__main__":
    table_coverage("soc"); table_coverage("soh"); table_support_diagnostics(); table_cross_chemistry(); table_eot_variants()
    table_calce_fewshot(); table_coverage_merged(); table_accuracy_merged(); table_comparators()
    if os.path.exists("docs/revision_r1/supplement.tex"):    # manuscript source: not part of the code release
        supplement_list()
    n1 = table_loco_accuracy()
    n2 = table_sequential()
    n3 = table_cp_baselines(); table_benchmark(); table_baselines_shift()
    table_ablation(); table_cross_protocol(); table_baselines_protocol(); macros()
    print(f"LOCO seeds: {n1} | sequential seeds: {n2} | cp-baseline seeds: {n3}")
