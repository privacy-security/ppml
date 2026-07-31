#!/usr/bin/env python3
import os
import csv
import json
import time
import hashlib
import subprocess
from typing import Dict, Any, List, Tuple

import itertools
import yaml
import re
import argparse
from pathlib import Path
import logging
import random
import statistics
import numpy as np

PROJECT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.abspath(os.path.join(PROJECT_DIR, ".."))
EXPORT_DIR   = os.path.join(PROJECT_DIR, "exports_recon")
RESULTS_CSV  = os.path.join(PROJECT_DIR, "recon_results.csv")

DEFAULT_DATASET    = "body_signal_of_smoking"
DEFAULT_VENV_SIZES = "../venv"

_DATASET_ALPHA = {
    "cifar10":               0.5,
    "body_signal_of_smoking": 0.1,
    "network_monitoring":    0.1,
    "household_power":       0.1,
}


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")


def extract_json_object(text: str) -> dict:
    matches = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    if not matches:
        raise ValueError(f"No JSON object found:\n{text[:2000]}")
    return json.loads(matches[-1])


def ensure_csv_header(path: str, fieldnames: List[str]) -> None:
    """Write the header, or append a NEW header row when the schema has changed.

    The results CSV is append-only across pipeline versions, so its schema drifts
    (seed was added once; dp_level now). Previously the header was written only
    for a new file, leaving later rows sitting under a stale header and readable
    only by guessing from field counts. Emitting a fresh header row on change
    makes the file self-describing: it becomes a sequence of (header, rows)
    blocks, which analyze_attacks.load_mixed_schema splits on. Note this is
    deliberately NOT plain-CSV-readable by a bare pd.read_csv.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        return
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    last_header = next((r for r in reversed(rows) if r and r[0] == fieldnames[0]), None)
    if last_header != list(fieldnames):
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def append_row(path: str, row: Dict[str, Any], fieldnames: List[str]) -> None:
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerow(
            {k: row.get(k, "") for k in fieldnames})


def short_hash(d: Dict[str, Any]) -> str:
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def run_capture(cmd: str, env: Dict[str, str], verbose: bool = False) -> str:
    if verbose:
        logging.debug(f"[CMD] {cmd}")
    p = subprocess.run(
        ["bash", "-lc", cmd],
        cwd=PROJECT_DIR, env={**os.environ, **env},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    out = (p.stdout or "").strip()
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed rc={p.returncode}\n--- OUTPUT TAIL ---\n{out[-4000:]}")
    if verbose and out:
        logging.debug(f"[OUT_TAIL]\n{out[-1200:]}")
    return out


def get_tf_keras_versions(venv: str) -> str:
    cmd = (
        f"source {venv}/bin/activate && python - <<'PY'\n"
        "import tensorflow as tf\n"
        "try:\n  import keras\n  kv = keras.__version__\n"
        "except Exception:\n  kv = 'unknown'\n"
        "print('TF=' + tf.__version__ + ' Keras=' + str(kv))\nPY"
    )
    return subprocess.check_output(["bash", "-lc", cmd],
                                   cwd=PROJECT_DIR).decode().strip()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attack-saved",         action="store_true")
    p.add_argument("--exports-dir",          type=str, default=None)
    p.add_argument("--dataset",              type=str, default=None)
    p.add_argument("--verbose",              action="store_true")
    p.add_argument("--config",               type=str,
                   default="config/attack_experiments.yaml")
    p.add_argument("--num-targets",          type=int, default=10)
    p.add_argument("--targets-seed",         type=int, default=42)
    p.add_argument("--target-split",         type=str, default="train",
                   choices=["train", "test"])
    p.add_argument("--targets",              type=str, default="")
    p.add_argument("--venv-for-sizes",       type=str, default=DEFAULT_VENV_SIZES)
    # FIX: stress params for target selection + attack_saved mode
    p.add_argument("--stress-alpha",         type=float, default=None)
    p.add_argument("--stress-beta",          type=float, default=0.01)
    p.add_argument("--stress-k",             type=int,   default=5)
    p.add_argument("--stress-seed",          type=int,   default=42)
    p.add_argument("--targets-from-dprime",  action="store_true",
                   help="Sample reconstruction targets only from D' "
                        "(records the model actually trained on).  "
                        "Requires stress_alpha to be set or auto-detected.")
    return p.parse_args()


def _split_fixed_and_grid(d: Dict[str, Any]):
    fixed, grid = {}, {}
    for k, v in (d or {}).items():
        (grid if isinstance(v, list) else fixed)[k] = v
    return fixed, grid


def load_experiments_from_config(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    default_dataset  = cfg.get("dataset", DEFAULT_DATASET)
    defaults         = cfg.get("defaults", {})
    defaults_fixed, defaults_grid = _split_fixed_and_grid(defaults)
    experiments = []

    for name, spec in cfg["experiments"].items():
        spec_fixed, spec_grid = _split_fixed_and_grid(spec)
        combined_fixed = {**defaults_fixed, **spec_fixed}
        combined_grid  = {**defaults_grid,  **spec_grid}
        exp_dataset    = combined_fixed.get("dataset", default_dataset)

        if bool(combined_fixed.get("no_fl", False)):
            combined_grid.pop("n_clients", None)
            combined_fixed["n_clients"] = 1

        if combined_grid:
            keys, values = list(combined_grid), list(combined_grid.values())
            for combo in itertools.product(*values):
                experiments.append({
                    "name": name, "dataset": exp_dataset,
                    **combined_fixed, **dict(zip(keys, combo)),
                })
        else:
            experiments.append({"name": name, "dataset": exp_dataset,
                                 **combined_fixed})

    return experiments


def pick_venv(exp: Dict[str, Any]) -> str:
    return "../venv_dp" if bool(exp.get("no_fl", False)) else "../venv"


def pick_venv_for_saved_model(model_filename: str) -> str:
    name = model_filename.lower()
    return "../venv_dp" if ("centralized" in name or "no_fl" in name) else "../venv"


# args_for_run_sh() filters the experiment dict against this whitelist, so a key
# missing HERE is silently dropped rather than rejected. dp_level and
# example_vectorized must be present: run.sh defaults DP_LEVEL to "client", and
# leaves EXAMPLE_VECTORIZED empty (-> model.py's True default), so an
# example-level experiment would otherwise train a client-level model, and the
# GRU datasets would crash in pfor.
VALID_RUN_SH_ARGS = {
    "aggregation_type", "dp", "dp_level", "partition_type", "l2_norm_clip",
    "noise_multiplier", "learning_rate", "batch_size", "epochs",
    "local", "clipping", "epsilon", "delta", "no_fl", "n_clients", "dataset",
    "seed", "example_vectorized",
}


def args_for_run_sh(exp: Dict[str, Any]) -> List[str]:
    return [f"--{k}={v}" for k, v in exp.items()
            if k in VALID_RUN_SH_ARGS and v is not None]


_DATASET_CACHE: Dict[Tuple[str, str], Tuple[int, int]] = {}


def _get_split_sizes_via_venv(dataset: str, venv: str) -> Tuple[int, int]:
    cmd = (
        f"source {venv}/bin/activate && "
        f"PYTHONPATH='{REPO_ROOT}':$PYTHONPATH "
        "python - <<'PY'\n"
        "import json, contextlib, io\n"
        "from data.dataset_loader import load_dataset\n"
        f"ds = {json.dumps(dataset)}\n"
        "buf = io.StringIO()\n"
        "with contextlib.redirect_stdout(buf):\n"
        "    (x_tr, _), (x_te, _), _ = load_dataset(ds, partition_type='centralized')\n"
        "print(json.dumps({'n_train': int(len(x_tr)), 'n_test': int(len(x_te))}))\nPY"
    )
    out  = subprocess.check_output(["bash", "-lc", cmd],
                                   cwd=PROJECT_DIR).decode().strip()
    info = json.loads(out)
    return int(info["n_train"]), int(info["n_test"])


def _get_split_sizes(dataset: str, venv: str) -> Tuple[int, int]:
    key = (dataset, venv)
    if key not in _DATASET_CACHE:
        _DATASET_CACHE[key] = _get_split_sizes_via_venv(dataset, venv)
    return _DATASET_CACHE[key]


def _reconstruct_dprime_indices(n_train: int, alpha: float,
                                beta: float, seed: int) -> List[int]:
    """
    Re-run Algorithm 1 deterministically to identify which training
    indices were in D'.  Returned list is used to restrict target sampling.
    """
    rng       = np.random.default_rng(seed)
    n_dprime  = int(np.floor(alpha * n_train))
    dprime_idx = rng.choice(n_train, size=n_dprime, replace=False)
    return dprime_idx.tolist()


def pick_target_indices(
    dataset: str, split: str, k: int, seed: int,
    explicit: str = "",
    venv_for_sizes: str = DEFAULT_VENV_SIZES,
    targets_from_dprime: bool = False,
    stress_alpha: float = 0.1,
    stress_beta:  float = 0.01,
    stress_seed:  int   = 42,
) -> List[int]:
    explicit_idx = [int(t.strip()) for t in explicit.split(",")
                    if t.strip()] if explicit.strip() else []
    if explicit_idx:
        return explicit_idx

    n_train, n_test = _get_split_sizes(dataset, venv_for_sizes)
    n = n_train if split == "train" else n_test

    if targets_from_dprime and split == "train":
        # sample only from records the model actually trained on
        pool = _reconstruct_dprime_indices(n_train, stress_alpha,
                                           stress_beta, stress_seed)
        logging.info(f"[TARGET-SEL] D'-restricted pool: {len(pool)} / {n_train} records")
    else:
        pool = list(range(n))

    k = max(1, min(int(k), len(pool)))
    return random.Random(int(seed)).sample(pool, k)


def _safe_json_loads(s: Any) -> Dict[str, Any]:
    try:
        return json.loads(s) if s else {}
    except Exception:
        return {}


def _agg(xs: List[float]) -> Dict[str, float]:
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float)
                                                          and x != x)]
    if not xs:
        return {"mean": float("nan"), "median": float("nan"),
                "min": float("nan"), "max": float("nan")}
    return {"mean": statistics.mean(xs), "median": statistics.median(xs),
            "min": min(xs), "max": max(xs)}


def build_export_filename(exp: Dict[str, Any], rid: str) -> str:
    dataset_tag = str(exp.get("dataset", DEFAULT_DATASET))
    privacy_tag = "plain"
    if exp.get("dp", False):
        # example-level runs carry local=True, so test dp_level FIRST or every
        # example model lands on disk tagged "_ldp_".
        if str(exp.get("dp_level", "client")).strip().lower() == "example":
            privacy_tag = "example"
        else:
            privacy_tag = "ldp" if exp.get("local", False) else "cdp"
    parts = []
    for k, sfx in [("noise_multiplier", "nm"), ("epsilon", "eps"),
                   ("l2_norm_clip", "clip")]:
        if exp.get(k) is not None:
            parts.append(f"{sfx}{exp[k]}")
    return (f"{dataset_tag}_{exp['name']}_{privacy_tag}_"
            f"{'_'.join(parts) or 'noparams'}_{rid}.keras")


# ──────────────────────────────────────────────────────────────────────────────
# run reconstruction using scripts and capture fields
# ──────────────────────────────────────────────────────────────────────────────

def run_recon_on_model(
    venv: str, model_path: str, dataset: str,
    verbose: bool, targets: List[int], target_split: str,
    seed: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    env = {"DATASET": dataset}

    base_scores, base_dists, base_uninfs, base_improvements = [], [], [], []
    hard_scores, hard_dists, hard_uninfs, hard_improvements = [], [], [], []

    for tidx in targets:
        # use baseline script
        base_cmd = (
            f"source {venv}/bin/activate && "
            f"python reconstruction_attack_baseline.py "
            f"--model_path '{model_path}' "
            f"--dataset '{dataset}' "
            f"--output_json_stdout true "
            f"--seed {seed} "
            f"--target_split '{target_split}' "
            f"--target_idx {tidx} "
            f"{'--verbose' if verbose else ''}"
        )
        out  = run_capture(base_cmd, env, verbose)
        base = extract_json_object(out)

        base_scores.append(float(base.get("score", float("nan"))))
        # capture fields
        base_uninfs.append(float(base.get("s_uninf", float("nan"))))
        base_improvements.append(float(base.get("improvement", float("nan"))))
        bdet = _safe_json_loads(base.get("detail"))
        if "dist_to_target" in bdet:
            base_dists.append(float(bdet["dist_to_target"]))

        # use extended script
        hard_cmd = (
            f"source {venv}/bin/activate && "
            f"python reconstruction_attack_extended.py "
            f"--model_path '{model_path}' "
            f"--dataset '{dataset}' "
            f"--output_json_stdout true "
            f"--seed {seed} "
            f"--target_split '{target_split}' "
            f"--target_idx {tidx} "
            f"{'--verbose' if verbose else ''}"
        )
        out2 = run_capture(hard_cmd, env, verbose)
        hard = extract_json_object(out2)

        hard_scores.append(float(hard.get("score", float("nan"))))
        # capture  fields
        hard_uninfs.append(float(hard.get("s_uninf", float("nan"))))
        hard_improvements.append(float(hard.get("improvement", float("nan"))))
        hdet = _safe_json_loads(hard.get("detail"))
        if "dist_to_target" in hdet:
            hard_dists.append(float(hdet["dist_to_target"]))

    base_summary = {
        "k": len(base_scores),
        "score":          _agg(base_scores),
        "dist_to_target": _agg(base_dists),
        "s_uninf":        _agg(base_uninfs),
        "improvement":    _agg(base_improvements),
    }
    hard_summary = {
        "k": len(hard_scores),
        "score":          _agg(hard_scores),
        "dist_to_target": _agg(hard_dists),
        "s_uninf":        _agg(hard_uninfs),
        "improvement":    _agg(hard_improvements),
    }
    return base_summary, hard_summary


# ──────────────────────────────────────────────────────────────────────────────
# CSV fieldnames
# ──────────────────────────────────────────────────────────────────────────────

def make_fieldnames() -> List[str]:
    return [
        "run_id", "name", "dataset", "seed",
        # dp_level distinguishes client-level LDP from example-level DP-SGD:
        # both carry dp=True/local=True, so without it they are indistinguishable
        # downstream. row.update(**exp) already carries the value; listing it
        # here is what lets append_row through it.
        "no_fl", "dp", "local", "dp_level", "aggregation_type", "partition_type",
        "n_clients", "learning_rate", "batch_size", "epochs",
        "noise_multiplier", "l2_norm_clip", "clipping", "epsilon",
        "export_path",
        "target_split", "num_targets", "targets_seed", "targets",
        "stress_alpha", "targets_from_dprime",

        # original fields
        "recon_base_mean_score", "recon_base_median_score",
        "recon_base_mean_dist",
        "recon_hard_mean_score", "recon_hard_median_score",
        "recon_hard_mean_dist",

        # uninformed-baseline and improvement fields
        "recon_base_mean_uninf",    "recon_base_mean_improvement",
        "recon_hard_mean_uninf",    "recon_hard_mean_improvement",

        "recon_base_detail", "recon_hard_detail",
        "status", "error",
    ]


def _row_from_summaries(base: Dict, hard: Dict) -> Dict:
    """Flatten summary dicts into CSV columns."""
    return {
        "recon_base_mean_score":   base["score"]["mean"],
        "recon_base_median_score": base["score"]["median"],
        "recon_base_mean_dist":    base["dist_to_target"]["mean"],
        "recon_hard_mean_score":   hard["score"]["mean"],
        "recon_hard_median_score": hard["score"]["median"],
        "recon_hard_mean_dist":    hard["dist_to_target"]["mean"],
        
        "recon_base_mean_uninf":       base["s_uninf"]["mean"],
        "recon_base_mean_improvement": base["improvement"]["mean"],
        "recon_hard_mean_uninf":       hard["s_uninf"]["mean"],
        "recon_hard_mean_improvement": hard["improvement"]["mean"],
        "recon_base_detail": json.dumps(base, sort_keys=True),
        "recon_hard_detail": json.dumps(hard, sort_keys=True),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args       = parse_args()
    setup_logging(args.verbose)

    exports_dir = args.exports_dir or EXPORT_DIR
    os.makedirs(exports_dir, exist_ok=True)

    fieldnames = make_fieldnames()
    ensure_csv_header(RESULTS_CSV, fieldnames)

    # ── attack_saved mode ────────────────────────────────────────────────────
    if args.attack_saved:
        dataset      = args.dataset or DEFAULT_DATASET
        model_paths  = sorted(Path(exports_dir).glob("*.keras"))
        if not model_paths:
            logging.warning(f"[ATTACK-SAVED] No .keras models in {exports_dir}")
            return

        stress_alpha = (args.stress_alpha if args.stress_alpha is not None
                        else _DATASET_ALPHA.get(dataset, 0.1))

        targets = pick_target_indices(
            dataset=dataset, split=args.target_split,
            k=args.num_targets, seed=args.targets_seed,
            explicit=args.targets,
            venv_for_sizes=args.venv_for_sizes,
            targets_from_dprime=args.targets_from_dprime,
            stress_alpha=stress_alpha,
            stress_seed=args.stress_seed,
        )
        logging.info(f"[ATTACK-SAVED] {len(model_paths)} models, "
                     f"targets={targets}, stress_alpha={stress_alpha}")

        for mp in model_paths:
            rid  = f"{now_tag()}_{short_hash({'model': mp.name, 'dataset': dataset})}"
            venv = pick_venv_for_saved_model(mp.name)

            row = {k: "" for k in fieldnames}
            row.update({
                "run_id": rid, "name": "attack_saved", "dataset": dataset,
                "export_path": mp.name,
                "target_split": args.target_split,
                "num_targets": len(targets),
                "targets_seed": args.targets_seed,
                "targets": ",".join(map(str, targets)),
                "stress_alpha": stress_alpha,
                "targets_from_dprime": args.targets_from_dprime,
                "status": "started",
            })

            try:
                # attack-saved: model seed unknown -> run_recon_on_model default 42
                base_s, hard_s = run_recon_on_model(
                    venv=venv, model_path=str(mp), dataset=dataset,
                    verbose=args.verbose, targets=targets,
                    target_split=args.target_split,
                )
                row.update(_row_from_summaries(base_s, hard_s))
                row["status"] = "ok"
            except Exception as e:
                row["status"] = "failed"
                row["error"]  = str(e)
                logging.error(f"[ATTACK-SAVED] FAILED {mp.name}: {e}")

            append_row(RESULTS_CSV, row, fieldnames)

        logging.info(f"[DONE] → {RESULTS_CSV}")
        return

    # ── train + attack mode ──────────────────────────────────────────────────
    exps = load_experiments_from_config(args.config)
    logging.info(f"[TRAIN+ATTACK] {len(exps)} experiments")

    for exp in exps:
        dataset      = exp.get("dataset", DEFAULT_DATASET)
        # Model seed: axis from config (defaults 42 if absent). Flows to run.sh
        # via args_for_run_sh AND into every recon subprocess via --seed.
        seed         = int(exp.get("seed", 42))
        stress_alpha = float(exp.get("leaky_train_frac",
                                     _DATASET_ALPHA.get(dataset, 0.1)))
        stress_beta  = float(exp.get("canary_frac",  0.01))
        stress_k     = int(exp.get("canary_dups",    5))
        # stress_seed defines the D' member set; held FIXED across model seeds.
        stress_seed  = 42

        # targets_seed is held fixed (args default 42) so the SAME 10 target
        # records are attacked for every model seed -> mean±std reflects model
        # variance, not target variance.
        targets = pick_target_indices(
            dataset=dataset, split=args.target_split,
            k=args.num_targets, seed=args.targets_seed,
            explicit=args.targets,
            venv_for_sizes=args.venv_for_sizes,
            targets_from_dprime=args.targets_from_dprime,
            stress_alpha=stress_alpha,
            stress_beta=stress_beta,
            stress_seed=stress_seed,
        )

        conf_for_hash   = {k: v for k, v in exp.items() if k != "name"}
        rid             = f"{now_tag()}_{short_hash(conf_for_hash)}"
        export_filename = build_export_filename(exp, rid)
        export_path     = os.path.join(exports_dir, export_filename)

        env = {
            "MODEL_SAVE":      "true",
            "EXPORT_PATH":     export_path,
            "DATASET":         dataset,
            "LEAKY_TRAIN_FRAC": str(stress_alpha),
            "CANARY_FRAC":     str(stress_beta),
            "CANARY_DUPS":     str(stress_k),
            "CANARY_FLIP":     str(exp.get("canary_flip", True)).lower(),
            "LEAKY_SEED":      str(stress_seed),
        }

        venv     = pick_venv(exp)
        run_args = " ".join(args_for_run_sh(exp))
        logging.info(f"[EXP] {exp['name']} rid={rid} seed={seed} "
                     f"stress_alpha={stress_alpha} targets={targets}")

        row = {k: "" for k in fieldnames}
        row.update({
            "run_id": rid, "name": exp.get("name", ""), "dataset": dataset,
            "export_path": os.path.basename(export_path),
            "target_split": args.target_split,
            "num_targets": len(targets),
            "targets_seed": args.targets_seed,
            "targets": ",".join(map(str, targets)),
            "stress_alpha": stress_alpha,
            "targets_from_dprime": args.targets_from_dprime,
            "status": "started",
            **exp,
        })

        try:
            run_capture(
                f"source {venv}/bin/activate && bash ../run.sh {run_args}",
                env, args.verbose)

            base_s, hard_s = run_recon_on_model(
                venv=venv, model_path=export_path, dataset=dataset,
                verbose=args.verbose, targets=targets,
                target_split=args.target_split, seed=seed,
            )
            row.update(_row_from_summaries(base_s, hard_s))
            row["status"] = "ok"

        except Exception as e:
            row["status"] = "failed"
            row["error"]  = str(e)
            logging.error(f"[EXP] FAILED {exp['name']}: {e}")

        append_row(RESULTS_CSV, row, fieldnames)

    logging.info(f"[DONE] → {RESULTS_CSV}")


if __name__ == "__main__":
    main()
