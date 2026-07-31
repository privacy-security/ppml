#!/usr/bin/env python3
"""
select_attack_configs.py -- from the seeded sweep results, pick the DP configs
worth attacking (example-level MIA + targeted reconstruction) and emit:

  attack_experiments.yaml         one file, all datasets, CURRENT epsilon-based
                                  format. Per dataset: a plain (no-DP) baseline
                                  plus LDP and CDP at the selected usable epsilon
                                  ladder, each in TWO variants:
                                    *_canary   (leaky+canaries -> D' member set)
                                    *_full     (full train     -> full-train members)
                                  so both attack regimes run, per your request.

  attack_selection_summary.csv    audit trail: every (dataset, mech, M, eps, clip)
                                  with its utility and the usable/selected flags.

Usability (per mechanism/clip, seed-averaged, regular agg only):
  * classification: config is usable if NOT collapsed (predict-one-class or
    <= majority baseline).
  * regression: usable if mean error is within --reg-tol of the best (lowest)
    error for that mechanism/clip.

Epsilon selection among the usable set is controlled by --eps-select:
  * "best"   (default): the SINGLE epsilon with the best utility -- argmax mean
    accuracy (classification) or argmin mean error (regression), averaged over
    the attacked client counts, taken over ALL non-collapsed epsilons. It does
    NOT use the reg_tol band: that band is anchored to the global-minimum error
    and can exclude the genuine best-utility epsilon. Concrete case this fixes:
    network_monitoring example dropped eps=300 because a seed-noisy M=10/eps=1
    cell owned the global-min MSE and shrank the +20% band below eps=300's clean
    M=3 minimum -- even though eps=300 has the lowest MEAN MSE over M={3,10} and
    is the model you'd actually ship. argmin-mean-error also never picks a
    blown-up model, so no tolerance gate is needed here.
  * "ladder": [most-private, middle, near-baseline] across the reg_tol-usable
    epsilons. Restores the within-mechanism privacy->utility/leakage frontier at
    len(ladder)x the attacks. NOTE: example-level epsilon is a RECORD-level
    budget and client-level epsilon is a CLIENT-CONTRIBUTION-level budget --
    different privacy units -- so a longer example ladder does not make it "more
    comparable" to a single client point.

Selection consumes analyze_results.aggregate(), whose grouping key is
(mechanism, agg, M, epsilon, clip) -- SEED is NOT a grouping key, so the utility
numbers driving selection are already seed-averaged (mean over the 5 seeds).
Selection is restricted to the REGULAR-aggregation rows (agg == "regular"):
SecAgg adds no DP noise, so it does not change which epsilon ladder is usable,
and mixing secure rows into the selection would double-count cells. The emitted
configs carry a `seed` axis in `defaults` so the attack pipeline grid-expands
each config across every seed and attacks one model per seed (option A: 1 attack
per model, variance from training).
"""
import argparse
import os
import numpy as np
import pandas as pd
import yaml

from analyze_results import (load, aggregate, flag_collapse, pick_best_clip,
                             DATASET_METRICS)
# single source of truth for the pfor/map_fn choice; duplicating it here would
# let the attack runs and the sweep drift apart.
from generate_configs import EXAMPLE_VECTORIZED

DATASET_BATCH  = {"body_signal_of_smoking": 128, "cifar10": 16,
                  "network_monitoring": 128, "household_power": 128}
DATASET_EPOCHS = {"body_signal_of_smoking": 30, "cifar10": 50,
                  "network_monitoring": 30, "household_power": 30}
MECH_TAG = {"ldp": "dp_local", "cdp": "dp_central", "example": "dp_example"}


def _regular_only(summary):
    """Restrict a summary frame to regular-aggregation rows (guarded)."""
    if "agg" in summary.columns:
        return summary[summary["agg"] == "regular"]
    return summary


def usable_epsilons(summary, ds, mech, best_clip, target_Ms, reg_tol):
    """reg_tol-usable epsilons -- the candidate range for the LADDER.

    classification: non-collapsed. regression: mean error within reg_tol of the
    best (lowest) error. Anchored to the global min, so on a task with a flat,
    seed-noisy low-error regime this band can be tight and exclude the genuine
    best-utility epsilon -- which is why --eps-select best does NOT use it."""
    col, _, direction = DATASET_METRICS[ds]["metrics"][0]
    mcol = f"{col}_mean"
    summary = _regular_only(summary)
    sub = summary[(summary["mechanism"] == mech) &
                  (summary["l2_norm_clip"] == best_clip) &
                  (summary["FL_N_CLIENTS"].isin(target_Ms))].copy()
    if not len(sub):
        return []
    if direction == "max":
        good = sub[~sub["collapsed"]]
    else:  # regression: within reg_tol of the best (lowest) error
        best = sub[mcol].min()
        good = sub[sub[mcol] <= best * (1.0 + reg_tol)]
    return sorted(set(float(e) for e in good["epsilon"].dropna().unique()))


def all_epsilons(summary, ds, mech, best_clip, target_Ms):
    """Every non-collapsed epsilon for this (mech, clip) across the attacked
    client counts -- the candidate set for --eps-select best.

    For regression there is no collapse flag, so this is every present epsilon;
    strongest_epsilon's argmin-mean-error then picks the best and never a
    blown-up one, so no reg_tol gate is needed (and using it would wrongly drop
    the best-utility epsilon, e.g. network_monitoring eps=300)."""
    summary = _regular_only(summary)
    sub = summary[(summary["mechanism"] == mech) &
                  (summary["l2_norm_clip"] == best_clip) &
                  (summary["FL_N_CLIENTS"].isin(target_Ms))]
    sub = sub[~sub["collapsed"]]
    return sorted(set(float(e) for e in sub["epsilon"].dropna().unique()))


def ladder(eps):
    eps = sorted(eps)
    if len(eps) <= 3:
        return eps
    return [eps[0], eps[len(eps) // 2], eps[-1]]


def strongest_epsilon(summary, ds, mech, best_clip, target_Ms, cand):
    """Single epsilon with the best seed-averaged primary-metric utility.

    "One strongest meaningful operating point": among the candidate epsilons
    `cand`, return the one whose utility (mean over the attacked client counts)
    is best -- argmax for accuracy, argmin for error. Returns [] if none, else a
    one-element list so the downstream emit path (which expects a list) is
    unchanged."""
    if not cand:
        return []
    col, _, direction = DATASET_METRICS[ds]["metrics"][0]
    mcol = f"{col}_mean"
    summary = _regular_only(summary)
    sub = summary[(summary["mechanism"] == mech) &
                  (summary["l2_norm_clip"] == best_clip) &
                  (summary["FL_N_CLIENTS"].isin(target_Ms)) &
                  (summary["epsilon"].isin(cand))]
    util = pd.to_numeric(sub.groupby("epsilon")[mcol].mean(), errors="coerce").dropna()
    if util.empty:
        return []
    best = util.idxmax() if direction == "max" else util.idxmin()
    return [float(best)]


def build_entry(ds, mech_tag, epsilons, clip, canary, defaults):
    example = (mech_tag == "dp_example")
    e = {
        "dataset": ds,
        "batch_size": DATASET_BATCH[ds],
        "epochs": DATASET_EPOCHS[ds],
        "learning_rate": 0.001,
        "dp": True,
        # The unit of privacy MUST be emitted: run.sh applies
        # DP_LEVEL="${DP_LEVEL:-client}", so an example-level entry that omits
        # this trains a client-level model under an example-level name -- no
        # error, wrong experiment.
        "dp_level": "example" if example else "client",
        # example-level DP-SGD is local-only by construction (noise lives in
        # DPSGDModel.train_step, which only ever runs on a client).
        "local": example or (mech_tag == "dp_local"),
        "epsilon": [float(x) for x in epsilons],
        "l2_norm_clip": [float(clip)],
    }
    if mech_tag == "dp_central":
        e["clipping"] = "server"
    if example:
        # pfor cannot convert the GRU's TensorList (variant) ops -- see the
        # dp_sgd.py docstring -- so the GRU datasets must take the sequential
        # map_fn path or they die on the first batch.
        e["example_vectorized"] = EXAMPLE_VECTORIZED[ds]
    if not canary:  # full-train variant: no leaky subsample, no canaries
        e["leaky_train_frac"] = 1.0
        e["canary_frac"] = 0.0
        e["canary_dups"] = 0
        e["canary_flip"] = False
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/project.csv",
                    help="seeded sweep results CSV (from fetch_runs.py)")
    ap.add_argument("--outdir", default="config",
                    help="output directory for attack_experiments.yaml and audit CSV")
    ap.add_argument("--attack-clients", default="3,10",
                    help="M values the attack runs on (default 3,10)")
    ap.add_argument("--seeds", default="0,1,2,3,4",
                    help="model seeds the attack retrains + attacks per config "
                         "(default 0,1,2,3,4). Emitted as a grid axis in defaults; "
                         "5 seeds x len(attack-clients) models per named config.")
    ap.add_argument("--reg-tol", type=float, default=0.20,
                    help="regression LADDER usability: usable if error within this "
                         "fraction of best (ignored by --eps-select best)")
    ap.add_argument("--eps-select", choices=["best", "ladder"], default="best",
                    help="'best' (default): single epsilon with the best mean "
                         "utility over all non-collapsed epsilons -- the strongest "
                         "meaningful operating point. 'ladder': [most-private, "
                         "middle, near-baseline] over the reg_tol-usable epsilons "
                         "to trace the privacy-utility frontier (len(ladder)x the "
                         "attacks).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    target_Ms = [int(x) for x in args.attack_clients.split(",")]
    seeds     = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    df = load(args.csv)

    defaults = {
        "aggregation_type": "regular",
        "partition_type": "iid",
        "n_clients": target_Ms,
        "seed": seeds,
        "leaky_train_frac": 0.1,
        "canary_frac": 0.01,
        "canary_dups": 5,
        "canary_flip": True,
    }
    experiments = {}
    audit_rows = []

    for ds in DATASET_METRICS:
        df_ds = df[df["DATASET"] == ds]
        if not len(df_ds):
            continue
        spec = DATASET_METRICS[ds]
        metric_cols = [m[0] for m in spec["metrics"]] + spec.get("aux", [])
        summary = flag_collapse(aggregate(df_ds, metric_cols), ds)
        primary = spec["metrics"][0]
        best_clip = pick_best_clip(summary, primary[0], primary[2])

        # audit/selection use regular-agg rows only
        summary_reg = _regular_only(summary)

        # plain baseline (undefended upper bound on leakage). Train fresh full data.
        experiments[f"{ds}_plain_full"] = {
            "dataset": ds, "batch_size": DATASET_BATCH[ds], "epochs": DATASET_EPOCHS[ds],
            "learning_rate": 0.001, "dp": False,
            "leaky_train_frac": 1.0, "canary_frac": 0.0, "canary_dups": 0, "canary_flip": False,
        }
        experiments[f"{ds}_plain_canary"] = {
            "dataset": ds, "batch_size": DATASET_BATCH[ds], "epochs": DATASET_EPOCHS[ds],
            "learning_rate": 0.001, "dp": False,
        }

        for mech in ("ldp", "cdp", "example"):
            # `ref` = the epsilons eligible for selection under the chosen mode:
            #   best   -> all non-collapsed epsilons (argmin/argmax mean utility)
            #   ladder -> the reg_tol-usable band ([most-private, middle, base])
            if args.eps_select == "best":
                ref = all_epsilons(summary, ds, mech, best_clip, target_Ms)
                sel = strongest_epsilon(summary, ds, mech, best_clip, target_Ms, ref)
            else:
                ref = usable_epsilons(summary, ds, mech, best_clip, target_Ms, args.reg_tol)
                sel = ladder(ref)

            # audit every cell (regular agg only, so no regular/secure duplication)
            mcol = f"{primary[0]}_mean"
            for _, r in summary_reg[(summary_reg["mechanism"] == mech) &
                                    (summary_reg["l2_norm_clip"] == best_clip)].iterrows():
                eps = r["epsilon"]
                audit_rows.append({
                    "dataset": ds, "mechanism": mech, "M": r["FL_N_CLIENTS"],
                    "epsilon": eps, "clip": best_clip,
                    "utility_mean": r.get(mcol), "collapsed": bool(r.get("collapsed", False)),
                    "usable": (float(eps) in ref) if pd.notna(eps) else False,
                    "selected": (float(eps) in sel) if pd.notna(eps) else False,
                })
            if not sel:
                print(f"[{ds}/{mech}] no usable configs at clip={best_clip}, M={target_Ms} -- skipped")
                continue
            tag = MECH_TAG[mech]
            experiments[f"{ds}_{tag}_canary"] = build_entry(ds, tag, sel, best_clip, True, defaults)
            experiments[f"{ds}_{tag}_full"]   = build_entry(ds, tag, sel, best_clip, False, defaults)
            print(f"[{ds}/{mech}] best_clip={best_clip} candidates={ref} "
                  f"-> selected ({args.eps_select}) {sel}")

    out_yaml = os.path.join(args.outdir, "attack_experiments.yaml")
    with open(out_yaml, "w") as f:
        yaml.safe_dump({"defaults": defaults, "experiments": experiments},
                       f, sort_keys=False, default_flow_style=False)

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(
        os.path.join("results", "attack_selection_summary.csv"), index=False)

    print(f"\nWrote {out_yaml}  ({len(experiments)} experiments, "
          f"seeds={seeds}, M={target_Ms} -> {len(seeds)*len(target_Ms)} models each)")
    print(f"Wrote {os.path.join('results', 'attack_selection_summary.csv')}")


if __name__ == "__main__":
    main()
