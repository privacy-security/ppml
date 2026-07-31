import os
import csv
import json
import time
import hashlib
import subprocess
from typing import Dict, Any, List

import itertools
import yaml
import re
import argparse
from pathlib import Path
import logging

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR  = os.path.join(PROJECT_DIR, "exports_mia")
RESULTS_CSV = os.path.join(PROJECT_DIR, "mia_results.csv")

DEFAULT_DATASET = "body_signal_of_smoking"

# Default stress_alpha per dataset (matches attack_experiments.yaml)
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
    """
    Extract the outermost JSON object from stdout.
    Uses greedy DOTALL match — works correctly when the script prints exactly
    one JSON object.
    """
    matches = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    if not matches:
        raise ValueError(f"No JSON object found in output:\n{text[:2000]}")
    return json.loads(matches[-1])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attack-saved",  action="store_true")
    p.add_argument("--exports-dir",   type=str, default=None)
    p.add_argument("--dataset",       type=str, default=None)
    p.add_argument("--verbose",       action="store_true")
    p.add_argument("--config",        type=str,
                   default="config/attack_experiments.yaml")
    # CLI overrides for stress regime (used in --attack-saved mode primarily)
    p.add_argument("--stress-alpha",  type=float, default=None,
                   help="Override stress_alpha (leaky_train_frac). "
                        "Auto-detected from dataset if not set.")
    p.add_argument("--stress-beta",   type=float, default=0.01)
    p.add_argument("--stress-k",      type=int,   default=5)
    p.add_argument("--stress-seed",   type=int,   default=42)
    return p.parse_args()


def pick_venv(exp: Dict[str, Any]) -> str:
    return "../venv_dp" if bool(exp.get("no_fl", False)) else "../venv"


def pick_venv_for_saved_model(model_filename: str) -> str:
    name = model_filename.lower()
    return "../venv_dp" if ("centralized" in name or "no_fl" in name) else "../venv"


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


def short_hash(d: Dict[str, Any]) -> str:
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


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


# args_for_run_sh() filters the experiment dict against this whitelist, so a key
# missing HERE is silently dropped rather than rejected. dp_level and
# example_vectorized must be present: run.sh defaults DP_LEVEL to "client", and
# leaves EXAMPLE_VECTORIZED empty (-> model.py's True default), so an
# example-level experiment would otherwise train a client-level model, and the
# GRU datasets would crash in pfor.
VALID_RUN_SH_ARGS = {
    "aggregation_type", "dp", "dp_level", "partition_type", "l2_norm_clip",
    "noise_multiplier", "learning_rate", "batch_size", "epochs",
    "local", "clipping", "epsilon", "no_fl", "n_clients", "dataset",
    "seed", "example_vectorized",
}


def args_for_run_sh(exp: Dict[str, Any]) -> List[str]:
    return [f"--{k}={v}" for k, v in exp.items()
            if k in VALID_RUN_SH_ARGS and v is not None]


def build_export_filename(exp: Dict[str, Any], rid: str) -> str:
    dataset_tag  = str(exp.get("dataset", DEFAULT_DATASET))
    privacy_tag  = "plain"
    if exp.get("dp", False):
        # example-level runs carry local=True, so test dp_level FIRST or every
        # example model lands on disk tagged "_ldp_".
        if str(exp.get("dp_level", "client")).strip().lower() == "example":
            privacy_tag = "example"
        else:
            privacy_tag = "ldp" if exp.get("local", False) else "cdp"
    parts = []
    for k, suffix in [("noise_multiplier", "nm"), ("epsilon", "eps"),
                       ("l2_norm_clip", "clip")]:
        if exp.get(k) is not None:
            parts.append(f"{suffix}{exp[k]}")
    param_tag = "_".join(parts) if parts else "noparams"
    return f"{dataset_tag}_{exp['name']}_{privacy_tag}_{param_tag}_{rid}.keras"


# ──────────────────────────────────────────────────────────────────────────────
# Build attack commands
# ──────────────────────────────────────────────────────────────────────────────

def build_mia_base_cmd(venv, model_path, dataset, stress_alpha,
                       stress_beta, stress_k, stress_seed, verbose, seed=42):
    return (
        f"source {venv}/bin/activate && "
        f"python mia_attack_baseline.py "
        f"--model_path '{model_path}' "
        f"--dataset '{dataset}' "
        f"--seed {seed} "
        f"--stress_alpha {stress_alpha} "
        f"--stress_beta {stress_beta} "
        f"--stress_k {stress_k} "
        f"--stress_seed {stress_seed} "
        f"--output_json_stdout true "
        f"{'--verbose' if verbose else ''}"
    )


def build_mia_learned_cmd(venv, model_path, dataset, stress_alpha,
                          stress_beta, stress_k, stress_seed, verbose, seed=42):
    return (
        f"source {venv}/bin/activate && "
        f"python mia_attack_learned.py "
        f"--model_path '{model_path}' "
        f"--dataset '{dataset}' "
        f"--stress_alpha {stress_alpha} "
        f"--stress_beta {stress_beta} "
        f"--stress_k {stress_k} "
        f"--stress_seed {stress_seed} "
        f"--output_json_stdout true "
        f"--seed {seed} "
        f"--max_samples 0 "
        f"{'--verbose' if verbose else ''}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Extract metrics from the new nested JSON — FIX: use nested keys
# ──────────────────────────────────────────────────────────────────────────────

def extract_mia_metrics(result: dict, prefix: str) -> dict:
    """
    prefix is 'base' or 'learned'.
    result is the full JSON dict from the attack script.
    Returns a flat dict of CSV columns.
    """
    row = {}

    def _get(sub_key, metric):
        sub = result.get(sub_key, {})
        return sub.get(metric)

    for source, col_suffix in [
        ("mia_dprime",             "dprime"),
        ("mia_canary",             "canary"),
        ("mia_original_contaminated", "orig"),
    ]:
        sub = result.get(source, {})
        for metric in ["auc", "advantage", "accuracy",
                       "tpr_at_1pct_fpr", "tpr_at_10pct_fpr"]:
            row[f"mia_{prefix}_{col_suffix}_{metric}"] = sub.get(metric)

    # Structural info (same in both base and learned, store once per prefix)
    row[f"mia_{prefix}_n_dprime"] = result.get("n_dprime")
    row[f"mia_{prefix}_n_canary"] = result.get("n_canary")
    row[f"mia_{prefix}_stress_alpha"] = result.get("alpha")

    return row


# ──────────────────────────────────────────────────────────────────────────────
# CSV fieldnames — FIX: expanded to cover new metrics
# ──────────────────────────────────────────────────────────────────────────────

def make_fieldnames() -> List[str]:
    base_fields = [
        "run_id", "name", "dataset", "seed",
        # dp_level distinguishes client-level LDP from example-level DP-SGD:
        # both carry dp=True/local=True, so without it they are indistinguishable
        # downstream. row.update(**exp) already carries the value; listing it
        # here is what lets append_row through it.
        "no_fl", "dp", "local", "dp_level", "aggregation_type", "partition_type",
        "n_clients", "learning_rate", "batch_size", "epochs",
        "noise_multiplier", "l2_norm_clip", "clipping", "epsilon", "delta",
        "export_path",
        # stress regime metadata
        "stress_alpha", "stress_beta", "stress_k", "stress_seed",
        "n_dprime", "n_canary",
    ]
    metric_fields = []
    for prefix in ["base", "learned"]:
        for source in ["dprime", "canary", "orig"]:
            for metric in ["auc", "advantage", "accuracy",
                           "tpr_at_1pct_fpr", "tpr_at_10pct_fpr"]:
                metric_fields.append(f"mia_{prefix}_{source}_{metric}")
    return base_fields + metric_fields + ["status", "error"]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    setup_logging(args.verbose)

    exports_dir = args.exports_dir or EXPORT_DIR
    os.makedirs(exports_dir, exist_ok=True)

    fieldnames = make_fieldnames()
    ensure_csv_header(RESULTS_CSV, fieldnames)

    # ── attack_saved mode ────────────────────────────────────────────────────
    if args.attack_saved:
        dataset = args.dataset or DEFAULT_DATASET
        model_paths = sorted(Path(exports_dir).glob("*.keras"))

        if not model_paths:
            logging.warning(f"[ATTACK-SAVED] No .keras models in: {exports_dir}")
            return

        # FIX: derive stress_alpha from dataset if not explicitly provided
        stress_alpha = args.stress_alpha \
                       if args.stress_alpha is not None \
                       else _DATASET_ALPHA.get(dataset, 0.1)
        stress_beta  = args.stress_beta
        stress_k     = args.stress_k
        stress_seed  = args.stress_seed

        logging.info(f"[ATTACK-SAVED] {len(model_paths)} models, "
                     f"dataset={dataset}, stress_alpha={stress_alpha}")

        for mp in model_paths:
            rid  = f"{now_tag()}_{short_hash({'model': mp.name, 'dataset': dataset})}"
            venv = pick_venv_for_saved_model(mp.name)
            versions = get_tf_keras_versions(venv)
            logging.info(f"[ATTACK-SAVED] {mp.name} ({versions})")

            row = {k: "" for k in fieldnames}
            row.update({
                "run_id": rid, "name": "attack_saved", "dataset": dataset,
                "export_path": mp.name, "stress_alpha": stress_alpha,
                "stress_beta": stress_beta, "stress_k": stress_k,
                "stress_seed": stress_seed, "status": "started",
            })

            env = {"DATASET": dataset}
            try:
                t0 = time.time()

                # Baseline MIA  (attack-saved: model seed unknown -> default 42)
                base_cmd = build_mia_base_cmd(
                    venv, str(mp), dataset,
                    stress_alpha, stress_beta, stress_k, stress_seed,
                    args.verbose)
                base_result = extract_json_object(
                    run_capture(base_cmd, env, args.verbose))
                row.update(extract_mia_metrics(base_result, "base"))

                # Learned MIA
                learned_cmd = build_mia_learned_cmd(
                    venv, str(mp), dataset,
                    stress_alpha, stress_beta, stress_k, stress_seed,
                    args.verbose)
                learned_result = extract_json_object(
                    run_capture(learned_cmd, env, args.verbose))
                row.update(extract_mia_metrics(learned_result, "learned"))

                # Store structural info once (same for both attackers)
                row["n_dprime"] = base_result.get("n_dprime")
                row["n_canary"] = base_result.get("n_canary")

                row["status"] = "ok"
                logging.info(f"[ATTACK-SAVED] done {mp.name} in "
                             f"{time.time()-t0:.1f}s")

            except Exception as e:
                row["status"] = "failed"
                row["error"]  = str(e)
                logging.error(f"[ATTACK-SAVED] FAILED {mp.name}: {e}")

            append_row(RESULTS_CSV, row, fieldnames)

        logging.info(f"[DONE] {len(model_paths)} models → {RESULTS_CSV}")
        return

    # ── train + attack mode ──────────────────────────────────────────────────
    exps = load_experiments_from_config(args.config)
    logging.info(f"[TRAIN+ATTACK] {len(exps)} experiments from {args.config}")

    for exp in exps:
        conf_for_hash   = {k: v for k, v in exp.items() if k != "name"}
        rid             = f"{now_tag()}_{short_hash(conf_for_hash)}"
        export_filename = build_export_filename(exp, rid)
        export_path     = os.path.join(exports_dir, export_filename)
        dataset         = exp.get("dataset", DEFAULT_DATASET)

        # Model seed: axis from config (defaults 42 if absent). Flows to run.sh
        # via args_for_run_sh (seed is in VALID_RUN_SH_ARGS) AND into every
        # attack subprocess via --seed below.
        seed = int(exp.get("seed", 42))

        # FIX: derive stress params from experiment config
        stress_alpha = float(exp.get("leaky_train_frac",
                                     _DATASET_ALPHA.get(dataset, 0.1)))
        stress_beta  = float(exp.get("canary_frac",  0.01))
        stress_k     = int(exp.get("canary_dups",    5))
        # stress_seed defines the D' member set and is held FIXED across all
        # model seeds (all 5 seeded models leak the same records).
        stress_seed  = 42

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
        versions = get_tf_keras_versions(venv)
        run_args = " ".join(args_for_run_sh(exp))
        logging.info(f"[EXP] {exp['name']} rid={rid} venv={venv} ({versions}) "
                     f"seed={seed} stress_alpha={stress_alpha}")

        row = {k: "" for k in fieldnames}
        row.update({
            "run_id": rid, "export_path": os.path.basename(export_path),
            "stress_alpha": stress_alpha, "stress_beta": stress_beta,
            "stress_k": stress_k, "stress_seed": stress_seed,
            "status": "started",
            **exp,
        })

        try:
            t0 = time.time()

            # Train
            run_capture(
                f"source {venv}/bin/activate && bash ../run.sh {run_args}",
                env, args.verbose)

            # Baseline MIA: use script + stress params + model seed
            base_result = extract_json_object(run_capture(
                build_mia_base_cmd(venv, export_path, dataset,
                                   stress_alpha, stress_beta, stress_k,
                                   stress_seed, args.verbose, seed=seed),
                env, args.verbose))
            row.update(extract_mia_metrics(base_result, "base"))

            # Learned MIA: use script + stress params + model seed
            learned_result = extract_json_object(run_capture(
                build_mia_learned_cmd(venv, export_path, dataset,
                                      stress_alpha, stress_beta, stress_k,
                                      stress_seed, args.verbose, seed=seed),
                env, args.verbose))
            row.update(extract_mia_metrics(learned_result, "learned"))

            row["n_dprime"] = base_result.get("n_dprime")
            row["n_canary"] = base_result.get("n_canary")
            row["status"]   = "ok"
            logging.info(f"[EXP] {exp['name']} done in {time.time()-t0:.1f}s")

        except Exception as e:
            row["status"] = "failed"
            row["error"]  = str(e)
            logging.error(f"[EXP] FAILED {exp['name']}: {e}")

        append_row(RESULTS_CSV, row, fieldnames)

    logging.info(f"[DONE] → {RESULTS_CSV}")


if __name__ == "__main__":
    main()
