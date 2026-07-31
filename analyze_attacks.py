#!/usr/bin/env python3
"""
analyze_attacks.py -- collapse the seeded attack CSVs + sweep summaries into
paper-ready tables.

Handles the two schema generations present in the attack CSVs: the older
single-run rows (no `seed` column) and the newer 5-seed rows (with `seed`).
Only the seeded rows are used for the mean +/- std tables; the legacy rows are
reported separately so you can see what the current Table 6 was built from.

The attack CSVs and the sweep summaries normally live in different folders, so
they get separate flags. --sweeps defaults to --in, which is only correct when
everything sits in one directory.

Usage:
    python analyze_attacks.py --in ./attacks --sweeps ./results --out ./results/attacks

Inputs:
    <--in>/mia_results.csv
    <--in>/recon_results.csv
    <--sweeps>/{dataset}_summary.csv   for each dataset in DATASETS

Outputs (CSV, in --out):
    mia_by_config.csv        MIA metrics, mean/std/min/max over seeds
    recon_by_config.csv      reconstruction metrics, mean/std over seeds
    recon_vs_uninformed.csv  reconstruction score minus the mean-init baseline
    sweep_frontier.csv       best non-collapsed utility per (dataset, mech, M, C)
    sweep_all.csv            full sweep, tidy, with base-rate/collapse flags
    seed_stability.csv       per-metric across-seed spread + t-test vs chance
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# display names / colours for the paper table + clarification figures
MECH_ORDER = ["plain", "ldp", "cdp", "example"]
MECH_LABEL = {"plain": "Plain FL", "ldp": "LDP", "cdp": "CDP", "example": "DP-SGD"}
MECH_COLOR = {"plain": "#555555", "ldp": "#1f77b4", "cdp": "#d62728", "example": "#2ca02c"}
CLIENT_COLORS = {3: "#4c72b0", 6: "#dd8452", 10: "#55a868"}
DS_LABEL = {
    "body_signal_of_smoking": "Smoking (MLP)", "cifar10": "CIFAR-10 (CNN)",
    "network_monitoring": "Network Monitoring (GRU)", "household_power": "Household Power (GRU)",
}

# columns that must never be coerced to numeric
STR_COLS = {
    "run_id", "name", "dataset", "aggregation_type", "partition_type",
    "export_path", "status", "error", "recon_base_detail", "recon_hard_detail",
    "targets", "target_split", "clipping", "dp", "local", "no_fl", "variant",
    "mech", "mechanism", "agg", "collapsed", "dp_level",
}

# Columns added to the results schema over time, in the order they were added.
# The pipelines now emit a fresh header row whenever the schema changes, so this
# only covers LEGACY rows appended under a stale header. Order matters: the
# schema grew strictly by prefix (gen1 -> +seed -> +dp_level), so candidate
# headers are built cumulatively, never as arbitrary combinations -- that keeps
# the field-count -> header mapping unambiguous.
_SCHEMA_EVOLUTION = [("seed", "dataset"), ("dp_level", "local")]

DATASETS = ["body_signal_of_smoking", "cifar10", "network_monitoring", "household_power"]

# majority-class / trivial-predictor reference for each classification task
BASE_RATE = {"body_signal_of_smoking": 0.6327, "cifar10": 0.10}

# Per-dataset attack configuration, taken verbatim from attack_experiments.yaml.
# Every attack run mis-logged learning_rate as 0.001, and the logged l2_norm_clip
# is unreliable; both are brute-force corrected to these values downstream. There
# is exactly one (lr, clip) pair per dataset because the attack config fixes one
# pair per dataset. NOTE: these are the example-level (dp_level=example) attack
# settings; network_monitoring/household_power ran the example attacks at lr=0.01,
# which is higher than the lr=0.001 main sweep for those tasks.
ATTACK_CONFIG = {
    "body_signal_of_smoking": {"learning_rate": 0.001,  "l2_norm_clip": 5.0},
    "cifar10":                {"learning_rate": 0.0005, "l2_norm_clip": 1.0},
    "network_monitoring":     {"learning_rate": 0.01,   "l2_norm_clip": 1.0},
    "household_power":        {"learning_rate": 0.01,   "l2_norm_clip": 1.0},
}


# ----------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------
def _candidate_headers(hdr: list[str]) -> dict[int, list[str]]:
    """Every header variant along the additive schema chain, keyed by field count.

    The schema only ever GREW, by inserting the _SCHEMA_EVOLUTION columns at fixed
    positions. A block's header may sit ANYWHERE on that chain, and its rows may be
    EARLIER (shorter -- e.g. the legacy no-`seed` single-run rows) or LATER (longer).
    We therefore reconstruct the full (newest) schema from `hdr`, then peel the
    evolution columns off in reverse historical order to recover every earlier
    length. This keeps the field-count -> header mapping unambiguous while handling
    a header that is newer than some of its rows (the case a purely additive
    expansion missed, raising "unrecognised field counts").
    """
    full = list(hdr)
    for col, after in _SCHEMA_EVOLUTION:
        if col not in full:
            idx = full.index(after) + 1 if after in full else len(full)
            full.insert(idx, col)
    out = {len(full): list(full)}
    cur = list(full)
    for col, _after in reversed(_SCHEMA_EVOLUTION):
        if col in cur:
            cur = [c for c in cur if c != col]
            out[len(cur)] = list(cur)
    return out


def _block_frame(hdr: list[str], rows: list[list[str]]) -> pd.DataFrame:
    """One (header, rows) block -> DataFrame, tolerating legacy short/long rows."""
    schemas = _candidate_headers(hdr)
    frames, unknown = [], {}
    by_len: dict[int, list[list[str]]] = {}
    for r in rows:
        by_len.setdefault(len(r), []).append(r)
    for n, rws in by_len.items():
        if n not in schemas:
            unknown[n] = len(rws)
            continue
        frames.append(pd.DataFrame(rws, columns=schemas[n]))
    if unknown:
        raise ValueError(
            f"rows with unrecognised field counts {unknown} under header of "
            f"{len(hdr)} fields; add the new column to _SCHEMA_EVOLUTION"
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=hdr)


def load_mixed_schema(path: str) -> pd.DataFrame:
    """Read a results CSV that may contain SEVERAL header blocks.

    The pipelines append across schema versions and now emit a new header row on
    each change, so the file is a sequence of (header, rows) blocks rather than
    one table -- a bare pd.read_csv cannot read it. Blocks are split on any row
    whose first cell repeats the first field name. Within a block, rows that
    predate that header are recovered via _candidate_headers().
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"attack CSV not found: {path}\n"
            f"  --in should point at the folder holding mia_results.csv and "
            f"recon_results.csv."
        )
    with open(path, newline="") as fh:
        rows = [r for r in csv.reader(fh) if r]
    if not rows:
        raise ValueError(f"{path} is empty")

    key = rows[0][0]                      # "run_id"
    blocks, hdr, body = [], None, []
    for r in rows:
        if r[0] == key:                   # header row -> start a new block
            if hdr is not None:
                blocks.append((hdr, body))
            hdr, body = r, []
        else:
            body.append(r)
    blocks.append((hdr, body))

    df = pd.concat([_block_frame(h, b) for h, b in blocks], ignore_index=True)
    for c in df.columns:
        if c not in STR_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Add `mech` (plain/ldp/cdp/example) and `variant` (full/canary) columns."""
    df = df.copy()

    def _mech(r):
        """plain / ldp / cdp / example, parsed from the run NAME.

        dp_level is NOT a column in the attack CSVs, so the earlier dp/local-based
        derivation collapsed example-level DP-SGD (dp=True, local=True) into LDP.
        The mechanism is instead encoded in the name token between the dataset
        prefix and the _{canary|full} suffix:
            <ds>_plain_<v>, <ds>_dp_local_<v>, <ds>_dp_central_<v>, <ds>_dp_example_<v>.
        We strip the known dataset prefix (from the dataset column) and the variant
        suffix, then map the remaining token; dp/local is only a last-resort fallback.
        """
        nm = str(r.get("name", "")); ds = str(r.get("dataset", ""))
        tok = nm
        for suf in ("_canary", "_full"):
            if tok.endswith(suf):
                tok = tok[:-len(suf)]; break
        if ds and tok.startswith(ds + "_"):
            tok = tok[len(ds) + 1:]
        tok = tok.lower()
        if "example" in tok:
            return "example"
        if "central" in tok:
            return "cdp"
        if "local" in tok:
            return "ldp"
        if "plain" in tok or str(r.get("dp", "")).lower() != "true":
            return "plain"
        return "ldp" if str(r.get("local", "")).lower() == "true" else "cdp"

    df["mech"] = df.apply(_mech, axis=1)
    df["variant"] = df["name"].str.extract(r"_(full|canary)$")
    return df


def apply_config_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """Brute-force learning_rate and l2_norm_clip to the per-dataset attack config.

    The attack pipeline logged learning_rate=0.001 for every run and the logged
    l2_norm_clip is unreliable; the true values are fixed per dataset in
    attack_experiments.yaml (mirrored in ATTACK_CONFIG). The override is applied
    to every row of the dataset. l2_norm_clip is a grouping key, so correcting it
    also re-groups any mis-split cells (mech stays a separate key, so plain/ldp/
    cdp/example never collapse together); learning_rate is added to KEYS so the
    corrected value is surfaced in every output table. Rows whose dataset is not
    in ATTACK_CONFIG are left untouched.
    """
    df = df.copy()
    if "learning_rate" not in df.columns:
        df["learning_rate"] = np.nan
    if "l2_norm_clip" not in df.columns:
        df["l2_norm_clip"] = np.nan
    df["learning_rate"] = pd.to_numeric(df["learning_rate"], errors="coerce")
    df["l2_norm_clip"] = pd.to_numeric(df["l2_norm_clip"], errors="coerce")
    n_total = 0
    for ds, cfg in ATTACK_CONFIG.items():
        m = df["dataset"] == ds
        n = int(m.sum())
        if n:
            df.loc[m, "learning_rate"] = cfg["learning_rate"]
            df.loc[m, "l2_norm_clip"] = cfg["l2_norm_clip"]
            n_total += n
    if n_total:
        print(f"[attacks] forced learning_rate + l2_norm_clip to per-dataset "
              f"attack config on {n_total} rows")
    return df


# ----------------------------------------------------------------------
# aggregation
# ----------------------------------------------------------------------
# learning_rate is a corrected, per-dataset-constant column (see ATTACK_CONFIG);
# it is kept in KEYS so the fixed value is surfaced in every keyed output table.
KEYS = ["dataset", "variant", "mech", "n_clients", "epsilon", "l2_norm_clip", "learning_rate"]


def agg_over_seeds(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    metrics = [m for m in metrics if m in df.columns]
    g = df.groupby(KEYS, dropna=False)[metrics].agg(["mean", "std", "min", "max", "count"])
    g.columns = [f"{a}__{b}" for a, b in g.columns]
    return g.reset_index()


def ttest_vs(values: np.ndarray, null: float) -> tuple[float, float]:
    """Two-sided one-sample t statistic and p-value against `null`.

    Implemented with a normal-approximation survival function so the script has
    no SciPy dependency; with n=5 this is indicative, not exact -- treat p as a
    rough screen and lean on the reported CI instead.
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n < 2:
        return np.nan, np.nan
    sd = v.std(ddof=1)
    if sd == 0:
        return np.inf if v.mean() != null else 0.0, 0.0 if v.mean() != null else 1.0
    t = (v.mean() - null) / (sd / np.sqrt(n))
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / np.sqrt(2))))
    return float(t), float(p)


def seed_stability(mia: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Per-config across-seed spread + a screen for 'is this above chance'."""
    out = []
    for key, sub in mia.groupby(KEYS, dropna=False):
        for m in metrics:
            if m not in sub.columns:
                continue
            v = sub[m].dropna().to_numpy(dtype=float)
            if len(v) < 2:
                continue
            t, p = ttest_vs(v, 0.5)
            half = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
            out.append(dict(
                zip(KEYS, key),
                metric=m, n_seeds=len(v),
                mean=v.mean(), std=v.std(ddof=1),
                ci95_lo=v.mean() - half, ci95_hi=v.mean() + half,
                t_vs_chance=t, p_vs_chance=p,
                sig_above_chance=bool(p < 0.05 and v.mean() > 0.5),
            ))
    return pd.DataFrame(out)


# ----------------------------------------------------------------------
# sweep summaries
# ----------------------------------------------------------------------
def tidy_sweep(sweepdir: str) -> pd.DataFrame:
    """Stack the per-dataset {dataset}_summary.csv files into one tidy frame.

    Returns an EMPTY frame (not an exception) when none are found, so the attack
    tables still get written; main() reports the searched paths so a wrong
    --sweeps is obvious rather than showing up as an empty concat deep in pandas.
    """
    frames, missing = [], []
    for ds in DATASETS:
        p = os.path.join(sweepdir, f"{ds}_summary.csv")
        if not os.path.exists(p):
            missing.append(p)
            continue
        d = pd.read_csv(p)
        d["dataset"] = ds
        # pick the headline utility column for this task type
        if "server/global_eval/accuracy_binary_mean" in d.columns:
            um, us = "server/global_eval/accuracy_binary_mean", "server/global_eval/accuracy_binary_std"
            kind, better = "accuracy", "higher"
        elif "server/global_eval/accuracy_mean" in d.columns:
            um, us = "server/global_eval/accuracy_mean", "server/global_eval/accuracy_std"
            kind, better = "accuracy", "higher"
        else:
            um, us = "server/global_eval/mse_mean", "server/global_eval/mse_std"
            kind, better = "mse", "lower"
        d["utility_mean"], d["utility_std"] = d[um], d[us]
        d["utility_kind"], d["utility_better"] = kind, better
        if kind == "accuracy":
            br = BASE_RATE.get(ds, np.nan)
            d["base_rate"] = br
            d["beats_trivial"] = d["utility_mean"] > br
        else:
            d["base_rate"] = np.nan
            d["beats_trivial"] = np.nan
        keep = ["dataset", "mechanism", "agg", "FL_N_CLIENTS", "epsilon", "l2_norm_clip",
                "utility_kind", "utility_mean", "utility_std", "utility_better",
                "base_rate", "beats_trivial", "collapsed"]
        frames.append(d[[c for c in keep if c in d.columns]])

    if missing:
        print(f"[sweep] {len(missing)} summary file(s) not found:", file=sys.stderr)
        for p in missing:
            print(f"          {p}", file=sys.stderr)
    if not frames:
        print(f"[sweep] no summaries under {os.path.abspath(sweepdir)} -- "
              f"skipping sweep_all.csv / sweep_frontier.csv. "
              f"Point --sweeps at the folder holding the *_summary.csv files.",
              file=sys.stderr)
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def frontier(sweep: pd.DataFrame) -> pd.DataFrame:
    """Best usable cell per (dataset, mechanism, agg, M, C): the tightest epsilon
    that still beats the trivial predictor (classification) or the lowest MSE."""
    rows = []
    for key, sub in sweep.groupby(["dataset", "mechanism", "agg", "FL_N_CLIENTS", "l2_norm_clip"], dropna=False):
        sub = sub[sub["utility_mean"].notna()]
        if sub.empty:
            continue
        if sub["utility_better"].iloc[0] == "higher":
            usable = sub[sub["beats_trivial"] == True]  # noqa: E712
            best = sub.loc[sub["utility_mean"].idxmax()]
            # plain-FL rows carry no epsilon, so there is no frontier point there
            with_eps = usable[usable["epsilon"].notna()]
            tightest = with_eps.loc[with_eps["epsilon"].idxmin()] if len(with_eps) else None
        else:
            usable = sub
            best = sub.loc[sub["utility_mean"].idxmin()]
            tightest = None
        rows.append({
            "dataset": key[0], "mechanism": key[1], "agg": key[2],
            "M": key[3], "clip": key[4],
            "n_cells": len(sub), "n_usable": int(len(usable)) if usable is not None else 0,
            "best_epsilon": best["epsilon"], "best_utility": best["utility_mean"],
            "best_utility_std": best["utility_std"],
            "tightest_usable_epsilon": tightest["epsilon"] if tightest is not None else np.nan,
            "tightest_usable_utility": tightest["utility_mean"] if tightest is not None else np.nan,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# paper table (attacks joined with utility) + clarification figures
# ----------------------------------------------------------------------
def _sweep_utility(sweep, ds, mech, M, eps, clip):
    """(utility_mean, utility_std, kind) for one config, pulled from the sweep frame
    (regular aggregation). For DP mechanisms it matches epsilon (=300) and, when
    available, the attack clip; plain FL carries no epsilon/clip."""
    if sweep is None or not len(sweep):
        return np.nan, np.nan, None
    s = sweep[(sweep["dataset"] == ds) & (sweep["mechanism"] == mech)
              & (sweep["FL_N_CLIENTS"] == M)]
    if "agg" in s.columns:
        s = s[s["agg"] == "regular"]
    if mech != "plain" and len(s):
        se = s[s["epsilon"] == eps]; s = se if len(se) else s
        sc = s[s["l2_norm_clip"] == clip]; s = sc if len(sc) else s
    if not len(s):
        return np.nan, np.nan, None
    r = s.iloc[0]
    return r.get("utility_mean", np.nan), r.get("utility_std", np.nan), r.get("utility_kind", None)


def privacy_utility_table(mia_seeded, rec_seeded, sweep, outdir):
    """Joined privacy-utility table (the Table-6 equivalent), per (dataset, mech, M).

    Attack metrics are taken from the CANARY stress regime (variant=canary -- the
    regime the paper's leakage numbers are measured under); MIA is reported on the
    real D' member set (dprime). Utility is joined from the sweep summaries at the
    attacked config (epsilon=300, per-dataset attack clip). Writes
    attack_privacy_utility.csv and .tex.
    """
    m_metrics = ["mia_base_dprime_auc", "mia_learned_dprime_auc",
                 "mia_learned_dprime_tpr_at_1pct_fpr"]
    r_metrics = ["recon_base_mean_score", "recon_base_mean_improvement",
                 "recon_hard_mean_improvement"]
    mia_c = agg_over_seeds(mia_seeded[mia_seeded["variant"] == "canary"], m_metrics)
    rec_c = agg_over_seeds(rec_seeded[rec_seeded["variant"] == "canary"], r_metrics)
    jkeys = [k for k in KEYS if k != "variant"]
    j = mia_c.merge(rec_c, on=jkeys, how="outer", suffixes=("", "_r"))

    rows = []
    for _, x in j.iterrows():
        ds, mech, M = x["dataset"], x["mech"], x["n_clients"]
        um, us, kind = _sweep_utility(sweep, ds, mech, M, x.get("epsilon"), x.get("l2_norm_clip"))
        rows.append(dict(
            dataset=ds, mechanism=mech, M=M, epsilon=x.get("epsilon"), clip=x.get("l2_norm_clip"),
            utility_kind=kind, utility_mean=um, utility_std=us,
            mia_base_auc=x.get("mia_base_dprime_auc__mean"),
            mia_learned_auc=x.get("mia_learned_dprime_auc__mean"),
            mia_learned_tpr1=x.get("mia_learned_dprime_tpr_at_1pct_fpr__mean"),
            recon_score=x.get("recon_base_mean_score__mean"),
            recon_base_improvement=x.get("recon_base_mean_improvement__mean"),
            recon_hard_improvement=x.get("recon_hard_mean_improvement__mean"),
        ))
    out = pd.DataFrame(rows)
    if out.empty:
        print("[attacks] privacy-utility table: no canary rows -- nothing written"); return out
    out["_mo"] = out["mechanism"].map({m: i for i, m in enumerate(MECH_ORDER)}).fillna(9)
    out = out.sort_values(["dataset", "_mo", "M"]).drop(columns="_mo")
    out.to_csv(os.path.join(outdir, "attack_privacy_utility.csv"), index=False)

    def _u(mu, sd):
        if pd.isna(mu):
            return "--"
        return f"{mu:.4f}$\\pm${sd:.4f}" if pd.notna(sd) else f"{mu:.4f}"
    tex = ["\\begin{tabular}{llccccc}", "\\toprule",
           "Dataset & Mech. & $M$ & Utility & MIA base & MIA learned & Recon.\\ impr. \\\\",
           "\\midrule"]
    for ds in [d for d in DATASETS if (out["dataset"] == d).any()]:
        sub = out[out["dataset"] == ds]
        tex.append(f"\\multicolumn{{7}}{{l}}{{\\textbf{{{DS_LABEL.get(ds, ds)}}}}} \\\\")
        for _, r in sub.iterrows():
            tex.append(" & ".join([
                "", MECH_LABEL.get(r["mechanism"], r["mechanism"]),
                (f"{int(r['M'])}" if pd.notna(r["M"]) else "--"),
                _u(r["utility_mean"], r["utility_std"]),
                (f"{r['mia_base_auc']:.3f}" if pd.notna(r["mia_base_auc"]) else "--"),
                (f"{r['mia_learned_auc']:.3f}" if pd.notna(r["mia_learned_auc"]) else "--"),
                (f"{r['recon_base_improvement']:+.4f}" if pd.notna(r["recon_base_improvement"]) else "--"),
            ]) + " \\\\")
        tex.append("\\midrule")
    tex[-1] = "\\bottomrule"; tex.append("\\end{tabular}")
    open(os.path.join(outdir, "attack_privacy_utility.tex"), "w").write("\n".join(tex) + "\n")
    print("[attacks] wrote attack_privacy_utility.csv + .tex")
    return out


def _mech_bar_panel(ax, data, mechs, Ms, hline, title, ylabel, fs=1.0):
    # fs = font scale (fs=2.0 => +100%); default 1.0 leaves other figures unchanged.
    width = 0.8 / max(1, len(Ms)); xpos = np.arange(len(mechs))
    for j, M in enumerate(Ms):
        ax.bar(xpos + (j - (len(Ms) - 1) / 2) * width,
               [data.get((m, M), np.nan) for m in mechs], width,
               color=CLIENT_COLORS.get(M), label=f"{M} clients")
    ax.axhline(hline, ls="--", lw=1, color="#666666")
    ax.set_xticks(xpos)
    ax.set_xticklabels([MECH_LABEL.get(m, m) for m in mechs], rotation=15, ha="right", fontsize=8 * fs)
    ax.set_ylabel(ylabel, fontsize=9 * fs); ax.set_title(title, fontsize=10 * fs)
    ax.tick_params(axis="both", labelsize=8 * fs)
    ax.grid(axis="y", ls=":", alpha=0.5)


def attack_figures(mia_seeded, rec_seeded, outdir):
    """Two clarification figures (2x2, one panel per dataset), canary regime:
      attack_mia_by_mechanism.png    learned-attacker MIA AUC on the real (D') member
                                     set, with the 0.5 chance line.
      attack_recon_by_mechanism.png  reconstruction score minus the uninformed mean-init
                                     baseline, with the 0 line (positive = real leakage).
    """
    from matplotlib.lines import Line2D
    mia = agg_over_seeds(mia_seeded[mia_seeded["variant"] == "canary"], ["mia_learned_dprime_auc"])
    rec = agg_over_seeds(rec_seeded[rec_seeded["variant"] == "canary"], ["recon_base_mean_improvement"])

    def _panels(agg, col, hline, ylabel, suptitle, fname, fs=1.0):
        dss = [d for d in DATASETS if (agg["dataset"] == d).any()]
        if not dss:
            print(f"[attacks] no data for {fname}"); return
        n = len(dss); ncol = min(2, n); nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(12.94 * ncol, 4.6 * nrow), squeeze=False)
        used_Ms = set()
        for i, ds in enumerate(dss):
            ax = axes[i // ncol][i % ncol]; s = agg[agg["dataset"] == ds]
            mechs = [m for m in MECH_ORDER if (s["mech"] == m).any()]
            Ms = sorted(int(x) for x in s["n_clients"].dropna().unique()); used_Ms.update(Ms)
            data = {(r["mech"], int(r["n_clients"])): r[col]
                    for _, r in s.iterrows() if pd.notna(r["n_clients"])}
            _mech_bar_panel(ax, data, mechs, Ms, hline, DS_LABEL.get(ds, ds), ylabel, fs=fs)
        for k in range(n, nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        handles = [Line2D([0], [0], marker="s", ls="", color=CLIENT_COLORS.get(M, "#333"),
                          label=f"{M} clients") for M in sorted(used_Ms)]
        handles.append(Line2D([0], [0], ls="--", color="#666666",
                              label=("chance (0.5)" if hline == 0.5 else "no advantage (0)")))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                   fontsize=9 * fs, bbox_to_anchor=(0.5, 0.01))
        fig.suptitle(suptitle, fontsize=12 * fs)
        # margins widen with fs so the larger suptitle/legend/labels don't collide
        fig.subplots_adjust(top=0.92 - 0.02 * (fs - 1.0), bottom=0.13 + 0.05 * (fs - 1.0),
                            hspace=0.42 + 0.08 * (fs - 1.0), wspace=0.22)
        fig.savefig(os.path.join(outdir, fname), dpi=200); plt.close(fig)
        print(f"[attacks] wrote {fname}")

    # attack_mia_by_mechanism gets +100% fonts (fs=2.0); recon figure stays at fs=1.0.
    _panels(mia, "mia_learned_dprime_auc__mean", 0.5, "MIA AUC (learned attacker)",
            "Membership inference by mechanism (canary regime)", "attack_mia_by_mechanism.png", fs=2.0)
    _panels(rec, "recon_base_mean_improvement__mean", 0.0, "Recon. score $-$ uninformed",
            "Reconstruction advantage over uninformed baseline (canary regime)",
            "attack_recon_by_mechanism.png", fs=2.0)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default="./attacks",
                    help="folder with mia_results.csv and recon_results.csv")
    ap.add_argument("--sweeps", dest="sweepdir", default="./results",
                    help="folder with the {dataset}_summary.csv files (default: same as --in)")
    ap.add_argument("--out", dest="outdir", default="./results/attacks")
    args = ap.parse_args()
    if args.sweepdir is None:
        args.sweepdir = args.indir
    os.makedirs(args.outdir, exist_ok=True)

    mia = apply_config_overrides(annotate(load_mixed_schema(os.path.join(args.indir, "mia_results.csv"))))
    rec = apply_config_overrides(annotate(load_mixed_schema(os.path.join(args.indir, "recon_results.csv"))))

    mia_seeded, rec_seeded = mia[mia.seed.notna()], rec[rec.seed.notna()]

    mia_metrics = [
        "mia_base_dprime_auc", "mia_learned_dprime_auc",
        "mia_base_orig_auc", "mia_learned_orig_auc",
        "mia_base_canary_auc", "mia_learned_canary_auc",
        "mia_base_dprime_tpr_at_1pct_fpr", "mia_learned_dprime_tpr_at_1pct_fpr",
        "mia_base_dprime_advantage", "mia_learned_dprime_advantage",
    ]
    rec_metrics = [
        "recon_base_mean_score", "recon_hard_mean_score",
        "recon_base_mean_uninf", "recon_hard_mean_uninf",
        "recon_base_mean_improvement", "recon_hard_mean_improvement",
        "recon_base_mean_dist", "recon_hard_mean_dist",
    ]

    agg_over_seeds(mia_seeded, mia_metrics).to_csv(os.path.join(args.outdir, "mia_by_config.csv"), index=False)
    agg_over_seeds(rec_seeded, rec_metrics).to_csv(os.path.join(args.outdir, "recon_by_config.csv"), index=False)

    # reconstruction is only evidence of leakage if it beats the mean-init attacker
    r = agg_over_seeds(rec_seeded, rec_metrics)
    r["base_beats_uninformed"] = r["recon_base_mean_improvement__mean"] > 0
    r["hard_beats_uninformed"] = r["recon_hard_mean_improvement__mean"] > 0
    r["max_improvement"] = r[["recon_base_mean_improvement__mean", "recon_hard_mean_improvement__mean"]].max(axis=1)
    cols = KEYS + ["recon_base_mean_score__mean", "recon_base_mean_uninf__mean",
                   "recon_base_mean_improvement__mean", "recon_base_mean_improvement__std",
                   "recon_hard_mean_improvement__mean", "recon_hard_mean_improvement__std",
                   "base_beats_uninformed", "hard_beats_uninformed", "max_improvement"]
    r[cols].to_csv(os.path.join(args.outdir, "recon_vs_uninformed.csv"), index=False)

    seed_stability(mia_seeded, mia_metrics).to_csv(os.path.join(args.outdir, "seed_stability.csv"), index=False)

    sweep = tidy_sweep(args.sweepdir)
    n_sweep = len(sweep)
    if n_sweep:
        sweep.to_csv(os.path.join(args.outdir, "sweep_all.csv"), index=False)
        frontier(sweep).to_csv(os.path.join(args.outdir, "sweep_frontier.csv"), index=False)

    # paper table (attacks joined with utility) + clarification figures
    privacy_utility_table(mia_seeded, rec_seeded, sweep, args.outdir)
    attack_figures(mia_seeded, rec_seeded, args.outdir)

    print(f"wrote outputs to {os.path.abspath(args.outdir)}")
    print(f"  attack rows: {len(mia)} MIA ({len(mia_seeded)} seeded), "
          f"{len(rec)} recon ({len(rec_seeded)} seeded)")
    print(f"  mechanisms : MIA={sorted(mia['mech'].unique())}  "
          f"variants={sorted(mia['variant'].dropna().unique())}")
    print(f"  sweep rows : {n_sweep}")


if __name__ == "__main__":
    main()
