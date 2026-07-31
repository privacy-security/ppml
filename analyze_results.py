#!/usr/bin/env python3
"""
analyze_results.py -- aggregate seeded FL+DP sweep results into mean +/- std
tables and privacy-utility figures.

Usage:
    python analyze_results.py --csv export.csv                 # ALL datasets
    python analyze_results.py --csv export.csv --dataset cifar10
    python analyze_results.py --csv export.csv --outdir results/

The tidy summary CSVs go straight into <outdir>/ so every dataset's summary sits
side by side for downstream analysis; everything else is per-dataset.

    <outdir>/<dataset>_summary.csv      tidy: one row per (mechanism, agg, M,
                                        epsilon, clip) with mean/std/min/max/n
                                        for every metric. Includes BOTH regular
                                        and secure (SecAgg) rows.

For each dataset it also writes (into <outdir>/<dataset>/):
    <dataset>_table.tex        compact table (primary metric, best clip),
                               LaTeX booktabs, for the paper
    <dataset>_<metric>.png     3 subplots (M=3/6/10): metric vs epsilon,
                               +/- std bands, one line per mechanism, best clip

It also writes a cross-dataset SecAgg-vs-regular utility comparison (into
<outdir>/secagg_DP/): secagg_dp_comparison.csv / .tex and secagg_dp_utility.png.
These test the claim that adding SecAgg to a DP pipeline changes only runtime,
not utility, by lining up matched regular vs secure DP cells.

Grouping key is (mechanism, agg, M, epsilon, l2_norm_clip) -- clip and
aggregation are real axes.
  * Mechanism: plain (dp False) / ldp (dp True, local True) / cdp (dp True, local False).
  * agg: regular / secure (SecAgg).  SecAgg is confidentiality-only and adds no
    DP noise, so the main privacy-utility tables/figures are built from the
    REGULAR rows only; the secure rows are still written to <dataset>_summary.csv
    for the separate SecAgg comparison.
Robust to partial data: cells report n; missing mechanisms/epsilons are skipped.
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Per-dataset metric maps. Each metric: (column, nice_name, direction).
# direction "max" = higher better; "min" = lower better.
# ----------------------------------------------------------------------------
DATASET_METRICS = {
    "body_signal_of_smoking": {
        "label": "Smoking (MLP)",
        "base_rate": 0.633,   # majority-class accuracy; usable models must beat this
        "metrics": [
            ("server/global_eval/accuracy_binary", "Accuracy", "max"),
            ("server/global_eval/f1_binary",       "F1",       "max"),
        ],
        "aux": ["server/global_eval/precision_binary",
                "server/global_eval/recall_binary"],
    },
    "cifar10": {
        "label": "CIFAR-10 (CNN)",
        "base_rate": 0.10,    # 10 balanced classes
        "metrics": [
            ("server/global_eval/accuracy", "Accuracy", "max"),
            ("server/global_eval/loss",     "Loss",     "min"),
        ],
        "aux": [],
    },
    "network_monitoring": {
        "label": "Network Monitoring (GRU)",
        "base_rate": None,    # regression: no majority baseline
        "metrics": [
            ("server/global_eval/mse", "MSE", "min"),
            ("server/global_eval/mae", "MAE", "min"),
        ],
        "aux": [],
    },
    "household_power": {
        "label": "Household Power (GRU)",
        "base_rate": None,
        "metrics": [
            ("server/global_eval/mse", "MSE", "min"),
            ("server/global_eval/mae", "MAE", "min"),
        ],
        "aux": [],
    },
}


PROBE_CLIPS = {70.0}

# "example" is a THIRD DP mechanism, not a variant of ldp: it protects a record
# rather than a silo, and its epsilon is a record-level budget. There is no
# central example-level branch by design, so the label needs no local/central
# qualifier.
MECH_ORDER = ["plain", "ldp", "cdp", "example"]
MECH_LABEL = {"plain": "No DP (plain FL)", "ldp": "LDP (local, client-level)",
              "cdp": "CDP (central, client-level)",
              "example": "DP-SGD (local, example level)"}
MECH_COLOR = {"plain": "#555555", "ldp": "#1f77b4", "cdp": "#d62728",
              "example": "#2ca02c"}
CLIENT_COLORS = {3: "#4c72b0", 6: "#dd8452", 10: "#55a868"}   # bar colour per client count


def _mech_table_name(mech):
    """Display name for generated tables. The internal key 'example' renders as
    the paper-facing mechanism name DP-SGD; all others keep their acronym. This
    affects generated .tex output only -- the 'example' key is left untouched in
    the file-reading and grouping logic."""
    return "DP-SGD" if mech == "example" else mech.upper()


def _to_bool(x):
    return str(x).strip().lower() in ("true", "1", "yes", "y", "t")


def _dp_level(row):
    """'client' or 'example'. Missing/NaN -> 'client' (pre-dp_level runs)."""
    v = row.get("dp_level", None)
    if v is None or (isinstance(v, float) and v != v):   # NaN
        return "client"
    v = str(v).strip().lower()
    return v if v in ("client", "example") else "client"


def derive_mechanism(row):
    """plain / ldp / cdp / example.

    dp_level is the unit of privacy, and it MUST be read here. Example-level
    DP-SGD is local-only by construction, so it logs local=True and dp=True --
    exactly like client-level LDP -- and both sweeps share the same epsilon,
    clip, n_clients and seed grids. Without this branch the two mechanisms land
    in an identical grouping key and get silently averaged into one cell.
    Runs predating dp_level have no such column and default to "client", so
    their labels are unchanged.
    """
    if not _to_bool(row.get("dp", False)):
        return "plain"
    if _dp_level(row) == "example":
        return "example"
    return "ldp" if _to_bool(row.get("local", row.get("local_dp", False))) else "cdp"


def derive_agg(row):
    """Aggregation axis: 'regular' or 'secure' (SecAgg).

    Coalesces the normalized internal field (FL_AGGREGATION_TYPE) and the raw
    sweep param (aggregation_type); one may be blank/NaN depending on what the
    run logged to wandb config. Defaults to 'regular' when neither is present.
    """
    for col in ("FL_AGGREGATION_TYPE", "aggregation_type"):
        v = row.get(col)
        if v is None:
            continue
        if isinstance(v, float) and v != v:     # NaN
            continue
        s = str(v).strip().lower()
        if s == "":
            continue
        return "secure" if s in ("secure", "secagg") else "regular"
    return "regular"


def load(csv_path):
    df = pd.read_csv(csv_path)
    n0 = len(df)

    # 1) drop wandb SWEEP-SUMMARY rows (Name="Sweep: xxx"): these are per-sweep
    #    aggregates, not runs -- they carry averaged garbage (e.g. M=6.33, eps=74)
    #    and compound States, and would corrupt the grouping.
    if "Name" in df.columns:
        df = df[~df["Name"].astype(str).str.startswith("Sweep:")].copy()
    n_sweep = n0 - len(df)

    # 2) keep only genuinely FINISHED runs. A failed/running/crashed run has empty
    #    metrics and must be omitted (this is the source of all-NaN cells). Handle
    #    both atomic State ("finished") and any compound value defensively.
    if "State" in df.columns:
        st = df["State"].astype(str).str.lower()
        keep = st.str.contains("finished") & ~st.str.contains("failed|running|crashed|killed")
        n_bad = (~keep).sum()
        df = df[keep].copy()
    else:
        n_bad = 0

    print(f"[load] {n0} rows -> dropped {n_sweep} sweep-summary + {n_bad} failed/running "
          f"-> {len(df)} finished runs")

    df["mechanism"] = df.apply(derive_mechanism, axis=1)
    df["agg"]       = df.apply(derive_agg, axis=1)
    for c in ("FL_N_CLIENTS", "epsilon", "l2_norm_clip", "seed"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 2b) Clipping norm is a DP-only parameter and is never applied in plain
    #     (non-DP) FL, so any l2_norm_clip on a plain row is an inert, mis-inherited
    #     config fragment. On CIFAR-10 the ONLY plain M=6 runs carry the junk
    #     l2_norm_clip=70, so the probe-drop below would delete every plain M=6 run
    #     and leave the M=6 panel with no baseline. Neutralize the clip on all plain
    #     rows (-> NaN) BEFORE the probe-drop, so no plain-FL run is thrown away and
    #     all plain runs collapse into one cell per (agg, M). This is what surfaces
    #     the CIFAR-10 M=6 baseline; it also merges the (previously clip-split) plain
    #     runs at M=3/M=10, so those baselines now average every plain run.
    if {"mechanism", "l2_norm_clip"}.issubset(df.columns):
        plain_clip = (df["mechanism"] == "plain") & df["l2_norm_clip"].notna()
        n_plain = int(plain_clip.sum())
        if n_plain:
            df.loc[plain_clip, "l2_norm_clip"] = np.nan
            print(f"[load] neutralized inert l2_norm_clip on {n_plain} plain (non-DP) "
                  f"rows so no plain-FL run is dropped by the probe filter")

    # 3) drop the failed high-clip probe (see PROBE_CLIPS). Done AFTER numeric
    #    coercion so the compare is float-vs-float, not str-vs-float.
    if "l2_norm_clip" in df.columns and PROBE_CLIPS:
        is_probe = df["l2_norm_clip"].isin(PROBE_CLIPS)
        n_probe = int(is_probe.sum())
        if n_probe:
            df = df[~is_probe].copy()
            print(f"[load] dropped {n_probe} failed-probe rows "
                  f"(l2_norm_clip in {sorted(PROBE_CLIPS)}) -> {len(df)} runs kept")

    # 4) CIFAR-10 legacy clip fix (now normally inert). Earlier CIFAR-10 client-level
    #    DP (CDP/LDP) runs were only run at clip C=1 but mis-logged as C=5; the fix
    #    duplicated those C=5 rows to C=1 so the client-level cells landed on the C=1
    #    grid the figures use. The experiments have since been RE-RUN with correct
    #    C=1 logging, so this should be a no-op. Guard it: only duplicate a
    #    mechanism's C=5 rows when that mechanism has NO C=1 rows already -- otherwise
    #    the genuine C=1 reruns would be double-counted. On the current (re-run) data
    #    there are no C=5 rows to begin with, so nothing happens. Scoped to CIFAR-10 +
    #    {ldp,cdp}; done after numeric coercion so the compare is float-vs-float.
    if {"DATASET", "mechanism", "l2_norm_clip"}.issubset(df.columns):
        dups = []
        for mech in ("ldp", "cdp"):
            base = (df["DATASET"] == "cifar10") & (df["mechanism"] == mech)
            has_c1 = bool((base & (df["l2_norm_clip"] == 1.0)).any())
            c5 = base & (df["l2_norm_clip"] == 5.0)
            n_c5 = int(c5.sum())
            if n_c5 and not has_c1:
                d = df[c5].copy(); d["l2_norm_clip"] = 1.0
                dups.append(d)
                print(f"[load] CIFAR-10 clip fix: duplicated {n_c5} {mech.upper()} "
                      f"rows C=5 -> C=1 (no C=1 rows present for this mechanism)")
            elif n_c5 and has_c1:
                print(f"[load] CIFAR-10 clip fix: {mech.upper()} already has C=1 rows "
                      f"-- skipped duplication of {n_c5} C=5 rows (no double-count)")
        if dups:
            df = pd.concat([df, *dups], ignore_index=True)

    return df


def aggregate(df_ds, metric_cols):
    """mean/std/min/max/n over seeds, grouped by (mechanism, agg, M, epsilon, clip).

    `agg` (regular/secure) is a real grouping axis so SecAgg runs are never
    averaged together with their regular-aggregation counterparts.
    """
    keys = ["mechanism", "agg", "FL_N_CLIENTS", "epsilon", "l2_norm_clip"]
    keys = [k for k in keys if k in df_ds.columns]
    present = [m for m in metric_cols if m in df_ds.columns]
    agg = {m: ["mean", "std", "min", "max", "count"] for m in present}
    g = df_ds.groupby(keys, dropna=False).agg(agg)
    g.columns = [f"{c}_{stat}" for c, stat in g.columns]
    return g.reset_index()


def _regular_only(summary):
    """Return the regular-aggregation subset (or the whole frame if no agg col)."""
    if "agg" in summary.columns:
        return summary[summary["agg"] == "regular"]
    return summary


def flag_collapse(summary, ds):
    """Mark degenerate models: predict-one-class (precision/recall) OR, for
    classification, mean accuracy at/below the majority baseline."""
    pcol = "server/global_eval/precision_binary_mean"
    rcol = "server/global_eval/recall_binary_mean"
    degenerate = pd.Series(False, index=summary.index)
    if pcol in summary.columns and rcol in summary.columns:
        rec, prec = summary[rcol], summary[pcol]
        degenerate = degenerate | ((rec > 0.98) | (rec < 0.02) | (prec < 0.02)).fillna(False)
    base = DATASET_METRICS.get(ds, {}).get("base_rate")
    primary = DATASET_METRICS[ds]["metrics"][0][0]
    amean = f"{primary}_mean"
    if base is not None and amean in summary.columns:
        degenerate = degenerate | (summary[amean] <= base).fillna(False)
    summary["collapsed"] = degenerate
    return summary


def pick_best_clip(summary, primary_col, direction):
    """Clip value with the best mean primary metric averaged over usable cells.

    Restricted to REGULAR-aggregation cells so the clip choice is not influenced
    by SecAgg runs (which duplicate regular utility), and to DP cells (epsilon
    not null) so a non-DP row's clip cannot win a value that no DP mechanism
    shares -- which would empty every DP cell in the downstream
    `l2_norm_clip == best_clip` filter. (The probe clips are already gone by
    load(); this DP-only restriction is the general guard.)"""
    summary = _regular_only(summary)
    if "epsilon" in summary.columns:
        summary = summary[summary["epsilon"].notna()]
    m = f"{primary_col}_mean"
    clips = [c for c in summary["l2_norm_clip"].dropna().unique()]
    fallback = sorted(clips)[0] if clips else None
    if m not in summary.columns:
        return fallback
    usable = summary[~summary.get("collapsed", False)]
    src = usable if len(usable) else summary
    by_clip = pd.to_numeric(src.groupby("l2_norm_clip")[m].mean(), errors="coerce").dropna()
    if by_clip.empty:
        return fallback
    return (by_clip.idxmax() if direction == "max" else by_clip.idxmin())


def write_summary(summary, ds, outroot):
    """Write the tidy per-dataset summary CSV to the outdir ROOT, not the
    per-dataset subfolder, so all four summaries sit together in results/.
    Both regular and secure rows are kept -- the SecAgg comparison consumes the
    secure ones."""
    path = os.path.join(outroot, f"{ds}_summary.csv")
    summary.to_csv(path, index=False)
    return path


def write_tables(summary, ds, label, metrics, best_clip, outdir):
    """Compact LaTeX booktabs table (regular-aggregation subset, best clip).

    The tidy summary CSV is written separately by write_summary(), which targets
    the outdir root."""
    reg = _regular_only(summary)

    # compact: primary metric, best clip, rows=epsilon, cols=mechanism x M
    pcol = metrics[0][0]
    m = f"{pcol}_mean"; s = f"{pcol}_std"
    sub = reg[(reg["l2_norm_clip"] == best_clip) | reg["epsilon"].isna()]
    combos = []
    for mech in MECH_ORDER:
        for M in sorted(x for x in reg["FL_N_CLIENTS"].dropna().unique()):
            if ((sub["mechanism"] == mech) & (sub["FL_N_CLIENTS"] == M)).any():
                combos.append((mech, M))
    eps_vals = sorted(x for x in sub["epsilon"].dropna().unique())

    # LaTeX booktabs
    tex = ["\\begin{tabular}{l" + "r" * len(combos) + "}", "\\toprule",
           " & ".join(["$\\varepsilon$"] + [f"{_mech_table_name(me)} $M{{=}}{int(M)}$" for me, M in combos]) + " \\\\",
           "\\midrule"]
    for eps in eps_vals:
        cells = [f"{eps:g}"]
        for mech, M in combos:
            cell = sub[(sub["mechanism"] == mech) & (sub["FL_N_CLIENTS"] == M) & (sub["epsilon"] == eps)]
            if len(cell) and pd.notna(cell.iloc[0].get(m)):
                mu, sd = cell.iloc[0][m], cell.iloc[0].get(s, np.nan)
                cells.append(f"{mu:.3f}$\\pm${sd:.3f}" if pd.notna(sd) else f"{mu:.3f}")
            else:
                cells.append("--")
        tex.append(" & ".join(cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(outdir, f"{ds}_table.tex"), "w").write("\n".join(tex) + "\n")


def plot_metric(summary, ds, label, metric, best_clip, outdir):
    # main privacy-utility figures use regular-aggregation rows only
    summary = _regular_only(summary)
    col, name, direction = metric
    mcol, scol = f"{col}_mean", f"{col}_std"
    if mcol not in summary.columns:
        return
    # skip entirely if this metric has no finite values anywhere
    if not np.isfinite(pd.to_numeric(summary[mcol], errors="coerce")).any():
        print(f"    [skip plot] {ds}/{name}: no finite values")
        return
    Ms = sorted(x for x in summary["FL_N_CLIENTS"].dropna().unique())
    if not Ms:
        return
    fig, axes = plt.subplots(1, len(Ms), figsize=(7 * len(Ms), 4.2),
                             sharey=True, constrained_layout=True)
    if len(Ms) == 1:
        axes = [axes]
    drew_any = False
    all_y = []  # collect finite y-values across panels to set an explicit ylim
    all_x = []
    for ax, M in zip(axes, Ms):
        for mech in MECH_ORDER:
            cell = summary[(summary["mechanism"] == mech) &
                           (summary["FL_N_CLIENTS"] == M) &
                           (summary["l2_norm_clip"] == best_clip)]
            if mech == "plain":  # plain has no epsilon -> horizontal reference
                pl = summary[(summary["mechanism"] == "plain") & (summary["FL_N_CLIENTS"] == M)]
                base = pd.to_numeric(pl[mcol], errors="coerce").mean() if len(pl) else np.nan
                if np.isfinite(base):
                    ax.axhline(base, ls="--", color=MECH_COLOR[mech],
                               lw=1.3, label=MECH_LABEL[mech]); drew_any = True
                    all_y.append(base)
                continue
            cell = cell.dropna(subset=["epsilon"]).sort_values("epsilon")
            if not len(cell):
                continue
            x = pd.to_numeric(cell["epsilon"], errors="coerce").to_numpy(dtype=float)
            mu = pd.to_numeric(cell[mcol], errors="coerce").to_numpy(dtype=float)
            sd = (pd.to_numeric(cell[scol], errors="coerce").to_numpy(dtype=float)
                  if scol in cell.columns else np.zeros_like(mu))
            # keep only points with finite x AND finite mu; NaN std -> 0 band
            keep = np.isfinite(x) & np.isfinite(mu)
            x, mu, sd = x[keep], mu[keep], np.nan_to_num(sd[keep], nan=0.0)
            if x.size == 0:
                continue
            ax.plot(x, mu, "-o", color=MECH_COLOR[mech], label=MECH_LABEL[mech], ms=5)
            ax.fill_between(x, mu - sd, mu + sd, color=MECH_COLOR[mech], alpha=0.18)
            all_y.extend((mu - sd).tolist()); all_y.extend((mu + sd).tolist())
            all_x.extend(x.tolist())
            drew_any = True
        ax.set_xscale("log")
        ax.set_xlabel(r"target $\varepsilon$")
        ax.set_title(f"M = {int(M)} clients")
        ax.grid(True, which="both", ls=":", alpha=0.5)
    if not drew_any:
        plt.close(fig)
        print(f"    [skip plot] {ds}/{name}: nothing to draw at clip={best_clip}")
        return
    # explicit, finite limits on EVERY axis so no panel has a degenerate
    # (NaN) transform -> tight-bbox layout can't produce a NaN axis length.
    all_y = [v for v in all_y if np.isfinite(v)]
    all_x = [v for v in all_x if np.isfinite(v) and v > 0]
    if all_y:
        lo, hi = min(all_y), max(all_y)
        pad = 0.05 * (hi - lo) if hi > lo else (abs(hi) * 0.05 + 1e-6)
        ylim = (lo - pad, hi + pad)
    else:
        ylim = (0.0, 1.0)
    xlim = (min(all_x) / 1.5, max(all_x) * 1.5) if all_x else (0.5, 500)
    for ax in axes:
        ax.set_ylim(*ylim)
        ax.set_xlim(*xlim)
    axes[0].set_ylabel(name + ("  (lower better)" if direction == "min" else "  (higher better)"))
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        axes[-1].legend(fontsize=9, loc="best")
    fig.suptitle(f"{label}: {name} vs privacy budget  (clip C = {best_clip})", fontsize=12)
    out = os.path.join(outdir, f"{ds}_{name.lower().replace(' ', '_')}.png")
    # NOTE: no bbox_inches="tight" / tight_layout -- those do fragile bbox math
    # that yields a NaN axis length on sparse panels. constrained_layout handles it.
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _fmt_pm(mu, sd):
    """mean$\\pm$std for a LaTeX cell; '--' when the mean is missing."""
    if pd.isna(mu):
        return "--"
    return f"{mu:.4f}$\\pm${sd:.4f}" if pd.notna(sd) else f"{mu:.4f}"


def secagg_dp_comparison(summaries, outroot):
    """SecAgg-vs-regular utility comparison for the DP mechanisms.

    Claim under test: adding SecAgg on top of a DP pipeline changes ONLY runtime,
    not utility. SecAgg reconstructs the exact aggregate the server would have seen
    without it, so at matched (dataset, DP-mech, M, epsilon, clip) the secure and
    regular models should score identically up to across-seed noise. For every such
    config that has BOTH a regular and a secure cell, we line up each utility metric
    and report the difference; if the claim holds, every secure value sits on top of
    its regular value (|Delta| well within the seed std, points on the diagonal).

    Consumes the secure rows already present in each dataset's summary (aggregate()
    keeps agg as a grouping axis). Writes into <outroot>/secagg_DP/:
        secagg_dp_comparison.csv   tidy: every matched config x every metric
        secagg_dp_comparison.tex   LaTeX, primary metric per config, by dataset
        secagg_dp_utility.png      regular-vs-secure scatter, one panel per dataset
    """
    outdir = os.path.join(outroot, "secagg_DP")
    os.makedirs(outdir, exist_ok=True)
    DP_MECHS = ["ldp", "cdp", "example"]
    KEYC = ["mechanism", "FL_N_CLIENTS", "epsilon", "l2_norm_clip"]

    rows = []
    for ds, sm in summaries.items():
        if ds not in DATASET_METRICS or "agg" not in sm.columns:
            continue
        prim_name = DATASET_METRICS[ds]["metrics"][0][1]
        reg = sm[(sm["agg"] == "regular") & sm["mechanism"].isin(DP_MECHS)]
        sec = sm[(sm["agg"] == "secure") & sm["mechanism"].isin(DP_MECHS)]
        if not len(sec):
            continue
        merged = reg.merge(sec, on=KEYC, suffixes=("_reg", "_sec"), how="inner")
        for _, r in merged.iterrows():
            for col, name, direction in DATASET_METRICS[ds]["metrics"]:
                rm, rs = r.get(f"{col}_mean_reg"), r.get(f"{col}_std_reg")
                cm, cs = r.get(f"{col}_mean_sec"), r.get(f"{col}_std_sec")
                rn, cn = r.get(f"{col}_count_reg"), r.get(f"{col}_count_sec")
                delta = (cm - rm) if (pd.notna(rm) and pd.notna(cm)) else np.nan
                noise = max([v for v in (rs, cs) if pd.notna(v)] or [np.nan])
                rows.append(dict(
                    dataset=ds, mechanism=r["mechanism"], M=r["FL_N_CLIENTS"],
                    epsilon=r["epsilon"], l2_norm_clip=r["l2_norm_clip"],
                    metric=name, direction=direction, is_primary=(name == prim_name),
                    regular_mean=rm, regular_std=rs, regular_n=rn,
                    secure_mean=cm, secure_std=cs, secure_n=cn, delta=delta,
                    abs_delta=(abs(delta) if pd.notna(delta) else np.nan),
                    within_seed_noise=(bool(abs(delta) <= noise)
                                       if (pd.notna(delta) and pd.notna(noise)) else np.nan),
                ))

    if not rows:
        print("[secagg_DP] no secure-aggregation DP cell matched a regular "
              "counterpart -- nothing written. (No SecAgg+DP runs in the summaries, "
              "or none share a (mech, M, epsilon, clip) with a regular run.)")
        return

    comp = pd.DataFrame(rows).sort_values(["dataset", "mechanism", "M", "epsilon", "metric"])
    comp.to_csv(os.path.join(outdir, "secagg_dp_comparison.csv"), index=False)
    prim = comp[comp["is_primary"]].copy()
    dss = [d for d in DATASET_METRICS if (prim["dataset"] == d).any()]

    # ---- LaTeX table: primary metric per matched config, grouped by dataset ----
    tex = ["\\begin{tabular}{llccrrr}", "\\toprule",
           "Dataset & Mech. & $M$ & $\\varepsilon$ & Regular & SecAgg & $\\Delta$ \\\\",
           "\\midrule"]
    for ds in dss:
        sub = prim[prim["dataset"] == ds]
        mname = DATASET_METRICS[ds]["metrics"][0][1]
        tex.append(f"\\multicolumn{{7}}{{l}}{{\\textbf{{{DATASET_METRICS[ds]['label']}}}"
                   f" --- {mname}}} \\\\")
        for _, r in sub.iterrows():
            tex.append(" & ".join([
                "", _mech_table_name(r["mechanism"]), f"{int(r['M'])}", f"{r['epsilon']:g}",
                _fmt_pm(r["regular_mean"], r["regular_std"]),
                _fmt_pm(r["secure_mean"], r["secure_std"]),
                (f"{r['delta']:+.4f}" if pd.notna(r["delta"]) else "--"),
            ]) + " \\\\")
        tex.append("\\midrule")
    tex[-1] = "\\bottomrule"
    tex.append("\\end{tabular}")
    open(os.path.join(outdir, "secagg_dp_comparison.tex"), "w").write("\n".join(tex) + "\n")

    # ---- scatter: regular vs secure primary metric, one panel per dataset ----
    from matplotlib.lines import Line2D
    n = len(dss)
    ncol = min(2, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10.19 * ncol, 4.8 * nrow), squeeze=False)
    M_MARK = {3.0: "o", 6.0: "s", 10.0: "^"}
    for i, ds in enumerate(dss):
        ax = axes[i // ncol][i % ncol]
        sub = prim[prim["dataset"] == ds]
        mname = DATASET_METRICS[ds]["metrics"][0][1]
        xr = pd.to_numeric(sub["regular_mean"], errors="coerce").dropna().to_numpy()
        yr = pd.to_numeric(sub["secure_mean"], errors="coerce").dropna().to_numpy()
        vals = np.concatenate([xr, yr]) if xr.size or yr.size else np.array([0.0, 1.0])
        lo, hi = float(vals.min()), float(vals.max())
        pad = 0.05 * (hi - lo) if hi > lo else (abs(hi) * 0.05 + 1e-6)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], ls="--", color="#999999", lw=1, zorder=0)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        for _, r in sub.iterrows():
            ax.errorbar(r["regular_mean"], r["secure_mean"],
                        xerr=(r["regular_std"] if pd.notna(r["regular_std"]) else 0.0),
                        yerr=(r["secure_std"] if pd.notna(r["secure_std"]) else 0.0),
                        fmt=M_MARK.get(float(r["M"]), "o"),
                        color=MECH_COLOR.get(r["mechanism"], "#333333"),
                        ms=7, capsize=2, lw=1, alpha=0.9, zorder=3)
        ax.set_xlabel(f"Regular-agg {mname}", fontsize=17.5)   # +75% (base 10)
        ax.set_ylabel(f"SecAgg {mname}", fontsize=17.5)
        ax.set_title(DATASET_METRICS[ds]["label"], fontsize=17.5)
        ax.tick_params(axis="both", labelsize=17.5)
        ax.grid(True, ls=":", alpha=0.5)
    for j in range(n, nrow * ncol):            # blank any unused panel
        axes[j // ncol][j % ncol].axis("off")
    mech_handles = [Line2D([0], [0], marker="o", ls="", color=MECH_COLOR[m],
                           label=_mech_table_name(m)) for m in DP_MECHS]
    m_handles = [Line2D([0], [0], marker=mk, ls="", color="#333333", label=f"M={int(mm)}")
                 for mm, mk in M_MARK.items()]
    fig.legend(handles=mech_handles + m_handles, loc="lower center",
               ncol=len(mech_handles) + len(m_handles), fontsize=15.75,   # +75% (base 9)
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("SecAgg vs regular aggregation under DP "
                 "(points on the diagonal = identical utility)", fontsize=19.25)   # +75% (base 11)
    fig.subplots_adjust(top=0.88, bottom=0.20, hspace=0.32, wspace=0.28)
    fig.savefig(os.path.join(outdir, "secagg_dp_utility.png"), dpi=150)
    plt.close(fig)

    for ds in dss:
        s = prim[prim["dataset"] == ds]
        md = pd.to_numeric(s["abs_delta"], errors="coerce").max()
        print(f"[secagg_DP] {ds:24} matched configs={len(s):2d}  "
              f"max|Δ {DATASET_METRICS[ds]['metrics'][0][1]}|={md:.5f}")
    print(f"[secagg_DP] wrote comparison.csv/.tex + secagg_dp_utility.png -> {outdir}/")


def _runtime_series(df):
    """Numeric wall-clock runtime, tolerant of the export's column name."""
    for c in ("Runtime", "runtime", "_runtime"):
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return None


def _runtime_prep(df_ds):
    """Numeric-runtime, finite-M subset of one dataset, or None."""
    rt = _runtime_series(df_ds)
    if rt is None or not rt.notna().any():
        return None
    d = df_ds.copy()
    d["runtime"] = rt
    d = d[d["runtime"].notna() & d["FL_N_CLIENTS"].notna()].copy()
    return d if len(d) else None


def runtime_figures(df, datasets, outroot):
    """Two COMBINED 2x2 runtime figures (one subplot per dataset), high-DPI for sharp
    PDF, plus a per-dataset <ds>_runtime_summary.csv.

      runtime_by_mechanism_regagg.png : relative runtime by mechanism (regular aggregation),
                                 normalized to plain FL at the smallest M. All mechanisms
                                 with regular runs are shown (plain included).
      runtime_by_mechanism_secagg.png : SecAgg overhead (secure / regular) per mechanism. ONLY
                                 mechanisms that actually have secure runs are drawn, so
                                 there is no empty leading group / rightward offset. Plain
                                 FL was not run under SecAgg, so it is represented by the
                                 1.00 baseline line (labelled in the legend), not a bar.

    Bars are grouped by client count (colour = M, consistent across subplots). The dashed
    1.00 line is added to the legend in both figures.
    """
    from matplotlib.lines import Line2D
    dss = [d for d in DATASET_METRICS if (df["DATASET"] == d).any() and d in datasets]
    prepped = {}
    for ds in dss:
        d = _runtime_prep(df[df["DATASET"] == ds])
        if d is None:
            print(f"[runtime] {ds}: no Runtime column/values -- skipped"); continue
        dd = os.path.join(outroot, ds); os.makedirs(dd, exist_ok=True)
        g = (d.groupby(["mechanism", "agg", "FL_N_CLIENTS"], dropna=False)["runtime"]
               .agg(mean_runtime="mean", std_runtime="std", n="count").reset_index())
        g.insert(0, "dataset", ds)
        g.to_csv(os.path.join(dd, f"{ds}_runtime_summary.csv"), index=False)
        prepped[ds] = d
    plot_dss = [d for d in dss if d in prepped]
    if not plot_dss:
        print("[runtime] no datasets with runtime -- no figures"); return
    n = len(plot_dss); ncol = min(2, n); nrow = int(np.ceil(n / ncol))

    def _make(kind):
        fig, axes = plt.subplots(nrow, ncol, figsize=(12.94 * ncol, 4.9 * nrow), squeeze=False)
        used_Ms = set()
        for i, ds in enumerate(plot_dss):
            ax = axes[i // ncol][i % ncol]; d = prepped[ds]
            mechs_all = [m for m in MECH_ORDER if (d["mechanism"] == m).any()]
            Ms = sorted(int(x) for x in d["FL_N_CLIENTS"].dropna().unique())
            used_Ms.update(Ms)
            if kind == "bymech":
                reg = d[d["agg"] == "regular"]
                bmask = (reg["mechanism"] == "plain") & (reg["FL_N_CLIENTS"] == Ms[0])
                base = reg[bmask]["runtime"].mean() if bmask.any() else np.nan
                mechs = mechs_all
                def val(m, M):
                    v = reg[(reg["mechanism"] == m) & (reg["FL_N_CLIENTS"] == M)]["runtime"].mean()
                    return v / base if (pd.notna(v) and pd.notna(base) and base > 0) else np.nan
                ylabel = "Relative runtime"
            else:
                def val(m, M):
                    r = d[(d["mechanism"] == m) & (d["FL_N_CLIENTS"] == M) & (d["agg"] == "regular")]["runtime"].mean()
                    s = d[(d["mechanism"] == m) & (d["FL_N_CLIENTS"] == M) & (d["agg"] == "secure")]["runtime"].mean()
                    if pd.notna(r) and pd.notna(s) and r > 0:
                        return s / r
                    # no secure run for this mechanism (e.g. plain FL was not run under
                    # SecAgg) -> show it at the 1.00 no-overhead reference so the x-axis
                    # stays consistent with the runtime-by-mechanism figure.
                    return 1.0 if (pd.notna(r) and r > 0) else np.nan
                mechs = mechs_all
                ylabel = "SecAgg / regular runtime"
            width = 0.8 / max(1, len(Ms)); xpos = np.arange(len(mechs))
            for j, M in enumerate(Ms):
                ax.bar(xpos + (j - (len(Ms) - 1) / 2) * width, [val(m, M) for m in mechs],
                       width, color=CLIENT_COLORS.get(M, None), label=f"M={M}")
            ax.axhline(1.0, ls="--", lw=1, color="#666666")
            ax.set_xticks(xpos)
            ax.set_xticklabels([MECH_LABEL.get(m, m) for m in mechs], rotation=15, ha="right", fontsize=14)
            ax.set_ylabel(ylabel, fontsize=18)
            ax.set_title(DATASET_METRICS[ds]["label"], fontsize=20)
            ax.tick_params(axis="both", labelsize=14)  # +100% tick labels (x base 7 -> 14, y default -> 14)
            ax.grid(axis="y", ls=":", alpha=0.5)
        for k in range(n, nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        Ms_all = sorted(used_Ms)
        handles = [Line2D([0], [0], marker="s", ls="", color=CLIENT_COLORS.get(M, "#333333"),
                          label=f"{M} clients") for M in Ms_all]
        base_lbl = ("plain FL @ M=%d (1.00)" % (Ms_all[0]) if kind == "bymech"
                    else "no overhead (secure = regular)")
        handles.append(Line2D([0], [0], ls="--", color="#666666", label=base_lbl))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                   fontsize=18, bbox_to_anchor=(0.5, 0.01))
        fig.suptitle("Relative runtime by mechanism (regular aggregation)" if kind == "bymech"
                     else "SecAgg runtime overhead (secure / regular)", fontsize=26)
        # margins relaxed so the +100% suptitle/legend/labels don't collide
        fig.subplots_adjust(top=0.90, bottom=0.18, hspace=0.50, wspace=0.22)
        fname = "runtime_by_mechanism_regagg.png" if kind == "bymech" else "runtime_by_mechanism_secagg.png"
        fig.savefig(os.path.join(outroot, fname), dpi=220)
        plt.close(fig)
        print(f"[runtime] wrote {fname} (2x2, dpi=220) -> {outroot}/")

    _make("bymech"); _make("secagg")


ROUNDS_CLIENT_PANELS = [3, 10]     # M=6 omitted from panels (falls between -- note in paper text)
ROUNDS_EPS_KEEP = [1.0, 300.0]     # DP epsilons drawn as lines (tightest + loosest budget)
# per-round wandb history key + axis label per dataset. metric_1 == the primary
# accuracy for the classification tasks (confirmed against the export); the GRU
# tasks use the eval loss curve, as in the original plotting script.
ROUNDS_HIST_KEY = {
    "body_signal_of_smoking": ("server/global_eval/accuracy_binary", "Evaluation accuracy"),
    "cifar10":                ("server/global_eval/accuracy", "Evaluation accuracy"),
    "network_monitoring":     ("server/global_eval/mse",     "Evaluation loss"),
    "household_power":        ("server/global_eval/mse",     "Evaluation loss"),
}


def _run_history(run, x_key, y_key):
    """(x_key, y_key) per-round history for a pre-fetched wandb run. Keyed scan first
    (fast), fall back to a full unkeyed scan on ANY failure (some wandb versions raise
    'Step column _step not found' on keyed scans); x resolves to `round` or `_step`.
    Returns a round-indexed Series or None; never raises."""
    h = None
    try:
        rows = list(run.scan_history(keys=[x_key, y_key]))
        if rows:
            h = pd.DataFrame(rows)
    except Exception:
        h = None
    if h is None or h.empty or x_key not in h.columns or y_key not in h.columns:
        try:
            h = pd.DataFrame(list(run.scan_history()))
        except Exception:
            return None
    if h is None or h.empty or y_key not in h.columns:
        return None
    xc = x_key if x_key in h.columns else ("_step" if "_step" in h.columns else None)
    if xc is None:
        return None
    h = h[[xc, y_key]].dropna().sort_values(xc).drop_duplicates(xc, keep="last")
    return h.set_index(xc)[y_key] if len(h) else None


def plot_rounds_from_wandb(df_ds, ds, label, best_clip, entity_project, outdir):
    """Utility-over-rounds figure (<ds>_rounds_side_by_side.png), rebuilt as a 2x2 grid:
    ROWS = epsilon (300 on top, 1 on the bottom), COLUMNS = client count (M=3, M=10).
    Each cell draws plain FL plus every DP mechanism (LDP, CDP, DP-SGD) that has runs at
    that (epsilon, M, best_clip); the 5 seeds of each config are averaged per round
    (mean curve + /- std band), colour = mechanism.

    Runs are SELECTED from the CSV (this script's reading standard); the per-round CURVE
    is pulled from wandb history. Only the SELECTED runs are fetched, in a SINGLE batched
    `$in` lookup per dataset (the previous per-name lookup was the slow part). Per-subplot
    y-limits are taken from the MEAN curves (not the +/-std bands), so one noisy seed can't
    blow up the axis (this is what broke the household-power panel before)."""
    try:
        import wandb
        api = wandb.Api()
    except Exception as e:
        print(f"[rounds] wandb unavailable ({e}) -- round-wise figure skipped "
              f"(pass --wandb '' to silence, or run `wandb login`)"); return
    if "Name" not in df_ds.columns:
        print(f"[rounds] {ds}: no run-name column -- cannot resolve wandb runs"); return

    x_key = "round"
    y_key, ylabel = ROUNDS_HIST_KEY.get(ds, ("server/global_eval/loss", "Evaluation loss"))
    reg = df_ds[df_ds["agg"] == "regular"]
    eps_rows = sorted(ROUNDS_EPS_KEEP, reverse=True)     # 300 top, 1 bottom
    M_cols = ROUNDS_CLIENT_PANELS

    def _names(mask):
        return reg[mask]["Name"].dropna().astype(str).tolist()

    # build the per-cell series and gather every run name we need
    cells, all_names = {}, set()
    for eps in eps_rows:
        for M in M_cols:
            series = [(MECH_LABEL["plain"], MECH_COLOR["plain"],
                       _names((reg["mechanism"] == "plain") & (reg["FL_N_CLIENTS"] == M)))]
            for mech in ("ldp", "cdp", "example"):
                nm = _names((reg["mechanism"] == mech) & (reg["FL_N_CLIENTS"] == M)
                            & (reg["epsilon"] == eps) & (reg["l2_norm_clip"] == best_clip))
                if nm:
                    series.append((MECH_LABEL[mech], MECH_COLOR[mech], nm))
            cells[(eps, M)] = series
            for _, _, nm in series:
                all_names.update(nm)

    # ONE batched lookup for all selected runs (per-name fallback only if needed)
    all_names = [n for n in all_names if n]
    run_by_name = {}
    print(f"[rounds] {ds}: {len(all_names)} runs selected across {len(cells)} cells; "
          f"batched wandb lookup ...")
    t_lookup = time.perf_counter()
    if all_names:
        try:
            for r in api.runs(entity_project, filters={"display_name": {"$in": all_names}}):
                run_by_name[r.name] = r
        except Exception as e:
            print(f"[rounds] {ds}: batched lookup failed ({e}); will resolve per-run")
    print(f"[rounds] {ds}: resolved {len(run_by_name)}/{len(all_names)} runs "
          f"in {_fmt_dt(time.perf_counter() - t_lookup)}")

    fetched = [0]                        # running count of REAL history pulls
    hist_cache = {}                      # run_name -> Series|None (pull each run at most once)

    def get_run(nm):
        if nm in run_by_name:
            return run_by_name[nm]
        try:
            rr = list(api.runs(entity_project, filters={"display_name": nm}))
            r = next((x for x in rr if x.name == nm), rr[0] if rr else None)
            if r is not None:
                run_by_name[nm] = r
            return r
        except Exception:
            return None

    def get_hist(nm):
        if nm in hist_cache:
            return hist_cache[nm]
        run = get_run(nm)
        s = _run_history(run, x_key, y_key) if run is not None else None
        hist_cache[nm] = s
        fetched[0] += 1
        return s

    t_fetch = time.perf_counter()
    fig, axes = plt.subplots(len(eps_rows), len(M_cols),
                             figsize=(8.68 * len(M_cols), 4.6 * len(eps_rows)), squeeze=False)
    any_curve = False
    for ri, eps in enumerate(eps_rows):
        for ci, M in enumerate(M_cols):
            t_cell = time.perf_counter()
            ax = axes[ri][ci]; mean_curves = []
            for line_label, color, names in cells[(eps, M)]:
                curves = [s for s in (get_hist(nm) for nm in names) if s is not None]
                if not curves:
                    print(f"[rounds] {ds} eps={eps:g} M={M}: no history for '{line_label}'")
                    continue
                allc = pd.concat(curves, axis=1)
                mean, std = allc.mean(axis=1), allc.std(axis=1)
                ax.plot(mean.index, mean.values, lw=2, color=color, label=line_label)
                if allc.shape[1] > 1:
                    ax.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.15)
                mean_curves.append(mean); any_curve = True
            if mean_curves:                              # y-lim from mean curves only
                allm = pd.concat(mean_curves)
                lo, hi = float(allm.min()), float(allm.max())
                pad = 0.05 * (hi - lo) if hi > lo else (abs(hi) * 0.05 + 1e-6)
                ax.set_ylim(lo - pad, hi + pad)
            ax.set_title(f"{M} clients, \u03b5={eps:g}", fontsize=20)   # +100% (base 10)
            ax.set_xlabel("Round", fontsize=20); ax.set_ylabel(ylabel, fontsize=20)
            ax.tick_params(axis="both", labelsize=20); ax.grid(alpha=0.3)
            print(f"[rounds] {ds}:   cell eps={eps:g} M={M} done "
                  f"({fetched[0]}/{len(all_names)} histories pulled, "
                  f"cell {_fmt_dt(time.perf_counter() - t_cell)})")
    print(f"[rounds] {ds}: all histories pulled in {_fmt_dt(time.perf_counter() - t_fetch)}")
    if not any_curve:
        plt.close(fig)
        print(f"[rounds] {ds}: no curves fetched from wandb -- figure not written"); return

    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    fig.suptitle(f"{label}: utility development over rounds", fontsize=26)   # +100% (base 13)
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)),
               fontsize=18, bbox_to_anchor=(0.5, -0.02))   # +100% (base 9)
    fig.tight_layout(rect=[0, 0.10, 1, 0.95])
    out = os.path.join(outdir, f"{ds}_rounds_side_by_side.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"[rounds] {ds}: wrote {out}")


def process_dataset(df, ds, outroot, wandb_project=None):
    if ds not in DATASET_METRICS:
        print(f"[skip] {ds}: no metric map"); return
    spec = DATASET_METRICS[ds]
    df_ds = df[df["DATASET"] == ds].copy()
    if not len(df_ds):
        print(f"[skip] {ds}: no rows"); return
    outdir = os.path.join(outroot, ds)
    os.makedirs(outdir, exist_ok=True)

    metric_cols = [m[0] for m in spec["metrics"]] + spec.get("aux", [])
    summary = aggregate(df_ds, metric_cols)
    summary = flag_collapse(summary, ds)

    primary = spec["metrics"][0]
    best_clip = pick_best_clip(summary, primary[0], primary[2])

    # summary CSV -> outdir ROOT; tables + figures -> per-dataset subfolder
    summary_path = write_summary(summary, ds, outroot)
    write_tables(summary, ds, spec["label"], spec["metrics"], best_clip, outdir)
    ts = time.perf_counter()
    for metric in spec["metrics"]:
        plot_metric(summary, ds, spec["label"], metric, best_clip, outdir)
    print(f"[time] {ds}: tables + metric figures {_fmt_dt(time.perf_counter() - ts)}")
    if wandb_project:
        tr = time.perf_counter()
        plot_rounds_from_wandb(df_ds, ds, spec["label"], best_clip, wandb_project, outdir)
        print(f"[time] {ds}: round-wise wandb figure {_fmt_dt(time.perf_counter() - tr)}")

    n_cells = len(summary)
    mechs = summary["mechanism"].unique().tolist()
    aggs  = summary["agg"].value_counts().to_dict() if "agg" in summary.columns else {}
    print(f"[ok] {ds:24} cells={n_cells:3d}  mechanisms={mechs}  agg={aggs}  "
          f"best_clip={best_clip}  -> {summary_path} + {outdir}/")

    # tagged copy for the combined SecAgg comparison (summary CSV on disk is
    # unchanged -- the tag is added only to the in-memory frame after writing)
    summary = summary.copy()
    summary["dataset"] = ds
    return summary


def _fmt_dt(sec):
    """Human-friendly elapsed time."""
    if sec < 60:
        return f"{sec:.1f}s"
    m, s = divmod(sec, 60)
    return f"{int(m)}m{s:04.1f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/project.csv", help="wandb export CSV (default = results/project.csv)")
    ap.add_argument("--dataset", default=None, help="one dataset; default = all present")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--wandb", default="dpnerds/privacy_preserving_federated_learning",
                    help="wandb 'entity/project' for round-wise utility curves. "
                         "Defaults to the reran project; pass --wandb '' to skip.")
    args = ap.parse_args()

    t_all = time.perf_counter()
    t = time.perf_counter()
    df = load(args.csv)
    print(f"[time] load CSV: {_fmt_dt(time.perf_counter() - t)}")

    datasets = [args.dataset] if args.dataset else \
        [d for d in DATASET_METRICS if (df["DATASET"] == d).any()]
    os.makedirs(args.outdir, exist_ok=True)
    summaries = {}
    for i, ds in enumerate(datasets, 1):
        t = time.perf_counter()
        print(f"[step] ({i}/{len(datasets)}) processing {ds} ...")
        sm = process_dataset(df, ds, args.outdir, wandb_project=args.wandb)
        if sm is not None:
            summaries[ds] = sm
        print(f"[time] {ds}: {_fmt_dt(time.perf_counter() - t)}")

    t = time.perf_counter()
    print("[step] SecAgg-vs-regular comparison ...")
    secagg_dp_comparison(summaries, args.outdir)
    print(f"[time] secagg_DP comparison: {_fmt_dt(time.perf_counter() - t)}")

    t = time.perf_counter()
    print("[step] runtime figures ...")
    runtime_figures(df, datasets, args.outdir)
    print(f"[time] runtime figures: {_fmt_dt(time.perf_counter() - t)}")

    print(f"\nDone in {_fmt_dt(time.perf_counter() - t_all)}. "
          f"Summary CSVs in {args.outdir}/, tables + figures under {args.outdir}/<dataset>/")


if __name__ == "__main__":
    main()
