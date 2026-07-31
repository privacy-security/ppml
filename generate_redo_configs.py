#!/usr/bin/env python3
"""
generate_redo_configs.py -- read project.csv, find FAILED runs, and emit one
wandb grid-sweep YAML per (dataset, mechanism, agg) so you can re-run just the
failures via run_sweep. Up to 24 files: 4 datasets x {fl, fl_dp_central,
fl_dp_local} x {regular, secure}. A file is written only if that group actually
has failures.

The YAMLs match your existing sweep schema (program/run.sh, method: grid,
parameters with value/values, command block) and are DATA-DRIVEN: each param's
`values` list is exactly the distinct values seen among that group's failed runs
(so DP mechanisms use `epsilon`, as your current runs do -- not the old
`noise_multiplier`).

AGGREGATION: failed SecAgg runs are grouped and redone as `aggregation_type:
secure`; regular-aggregation failures as `aggregation_type: regular`. Without
this, a failed secure run would silently be redone as a regular run.

NOTE ON GRID COVERAGE: wandb grid re-runs the cartesian product of the listed
values, so if failures are sparse a few already-successful configs may re-run
too. That is harmless (idempotent) and the script prints the over-coverage ratio.
Seeds are handled by run_sweep (not a sweep param here); the failed seeds are
listed as a comment + in redo_failed_manifest.csv for reference.

Usage:
    python generate_redo_configs.py --csv project.csv --outdir redo_configs
"""
import argparse
import os
import numpy as np
import pandas as pd

DS_SHORT = {
    "body_signal_of_smoking": "smoking",
    "cifar10": "cifar",
    "network_monitoring": "network",
    "household_power": "household",
}
DS_DEFAULT_BATCH  = {"body_signal_of_smoking": 128, "cifar10": 16,
                     "network_monitoring": 128, "household_power": 128}
DS_DEFAULT_EPOCHS = {"body_signal_of_smoking": 30, "cifar10": 50,
                     "network_monitoring": 30, "household_power": 30}


def _to_bool(x):
    return str(x).strip().lower() in ("true", "1", "yes", "y", "t")


def mechanism(row):
    if not _to_bool(row.get("dp", False)):
        return "fl"
    return "fl_dp_local" if _to_bool(row.get("local", row.get("local_dp", False))) else "fl_dp_central"


def agg_of(row):
    """'regular' or 'secure', coalescing the normalized and raw aggregation fields."""
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


def num(x):
    """Pretty number: 1.0 -> 1, 0.5 -> 0.5."""
    f = float(x)
    return int(f) if f == int(f) else f


def distinct(series):
    vals = sorted(set(num(v) for v in pd.to_numeric(series, errors="coerce").dropna()))
    return vals


def const(series, fallback):
    v = pd.to_numeric(series, errors="coerce").dropna()
    if len(v):
        return num(v.mode().iloc[0])
    return fallback


def yaml_block(dataset, mech, agg, g):
    """Build the YAML text for one (dataset, mechanism, agg) group of failed runs."""
    short = DS_SHORT.get(dataset, dataset)
    n_clients = distinct(g["FL_N_CLIENTS"]) or [3, 6, 10]
    clips = distinct(g["l2_norm_clip"]) or [1, 5]
    lr = const(g.get("learning_rate", pd.Series(dtype=float)), 0.001)
    bs = const(g.get("BATCH_SIZE", pd.Series(dtype=float)), DS_DEFAULT_BATCH.get(dataset, 128))
    ep = const(g.get("EPOCHS", pd.Series(dtype=float)), DS_DEFAULT_EPOCHS.get(dataset, 30))
    seeds = distinct(g["seed"]) if "seed" in g.columns else []

    L = []
    L.append("program: run.sh")
    L.append("project: privacy_preserving_federated_learning")
    L.append("method: grid")
    if seeds:
        L.append(f"# REDO of failed runs. failed seeds seen: {seeds}")
    L.append("")
    L.append("parameters:")
    L.append(f'  dataset:\n    value: "{dataset}"')
    L.append(f'  aggregation_type:\n    value: "{agg}"')

    if mech == "fl":
        L.append("  dp:\n    value: False")
    elif mech == "fl_dp_central":
        L.append("  dp:\n    value: True")
        L.append("  local:\n    value: False")
        L.append('  clipping:\n    value: "server"')
        eps = distinct(g["epsilon"])
        L.append(f"  epsilon:\n    values: {eps}")
    elif mech == "fl_dp_local":
        L.append("  dp:\n    value: True")
        L.append("  local:\n    value: True")
        eps = distinct(g["epsilon"])
        L.append(f"  epsilon:\n    values: {eps}")

    L.append(f"  l2_norm_clip:\n    values: {clips}")
    L.append('  partition_type:\n    value: "iid"')
    L.append(f"  n_clients:\n    values: {n_clients}")
    L.append(f"  learning_rate:\n    value: {lr}")
    L.append(f"  batch_size:\n    value: {bs}")
    L.append(f"  epochs:\n    value: {ep}")
    L.append("")
    L.append("command:")
    L.append("  - ${env}")
    L.append("  - bash")
    L.append("  - ${program}")
    L.append("  - ${args}")
    return "\n".join(L) + "\n", n_clients, clips, distinct(g.get("epsilon", pd.Series(dtype=float)))


def grid_size(mech, n_clients, clips, eps):
    g = len(n_clients) * len(clips)
    if mech != "fl":
        g *= max(1, len(eps))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/project.csv", help="wandb export CSV (default = results/project.csv)")
    ap.add_argument("--outdir", default="config/config_redo", help="output directory for redo YAMLs + manifest (default = config/config_redo)")
    ap.add_argument("--states", default="failed,crashed,killed",
                    help="comma list of run states to treat as failed")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    # drop any sweep-summary rows defensively
    if "Name" in df.columns:
        df = df[~df["Name"].astype(str).str.startswith("Sweep:")].copy()

    bad_states = set(s.strip().lower() for s in args.states.split(","))
    st = df["State"].astype(str).str.lower()
    failed = df[st.isin(bad_states) | st.str.contains("|".join(bad_states))].copy()
    if not len(failed):
        print("No failed runs found. Nothing to redo. (States present: "
              f"{df['State'].value_counts(dropna=False).to_dict()})")
        return

    failed["mechanism"] = failed.apply(mechanism, axis=1)
    failed["agg"]       = failed.apply(agg_of, axis=1)
    manifest = []
    n_written = 0
    print(f"Found {len(failed)} failed runs. Grouping by (dataset, mechanism, agg):\n")
    for (ds, mech, agg), g in failed.groupby(["DATASET", "mechanism", "agg"]):
        if ds not in DS_SHORT:
            print(f"  [skip] unknown dataset {ds}"); continue
        text, n_clients, clips, eps = yaml_block(ds, mech, agg, g)
        agg_sfx = "" if agg == "regular" else "_secure"
        fname = f"{DS_SHORT[ds]}_{mech}{agg_sfx}.yaml"
        with open(os.path.join(args.outdir, fname), "w", newline="\n") as f:
            f.write(text)
        gs = grid_size(mech, n_clients, clips, eps)
        print(f"  {fname:40} failed={len(g):3d}  grid={gs:3d}  "
              f"(M={n_clients}, clip={clips}" + (f", eps={eps}" if mech != 'fl' else "") + ")")
        for _, r in g.iterrows():
            manifest.append({
                "dataset": ds, "mechanism": mech, "agg": agg,
                "M": num(r["FL_N_CLIENTS"]) if pd.notna(r.get("FL_N_CLIENTS")) else None,
                "l2_norm_clip": num(r["l2_norm_clip"]) if pd.notna(r.get("l2_norm_clip")) else None,
                "epsilon": num(r["epsilon"]) if pd.notna(r.get("epsilon")) else None,
                "seed": num(r["seed"]) if pd.notna(r.get("seed")) else None,
                "Name": r.get("Name"), "State": r.get("State"),
            })
        n_written += 1

    pd.DataFrame(manifest).to_csv(os.path.join(args.outdir, "redo_failed_manifest.csv"), index=False)
    print(f"\nWrote {n_written} redo YAML(s) + redo_failed_manifest.csv to {args.outdir}/")


if __name__ == "__main__":
    main()
