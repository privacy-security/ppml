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
EXPORT_DIR = os.path.join(PROJECT_DIR, "exports_mia")
RESULTS_CSV = os.path.join(PROJECT_DIR, "mia_results.csv")

DEFAULT_DATASET = "body_signal_of_smoking"


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def extract_json_object(text: str) -> dict:
    matches = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    if not matches:
        raise ValueError(f"No JSON object found in output:\n{text[:2000]}")
    return json.loads(matches[-1])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attack-saved", action="store_true")
    p.add_argument("--exports-dir", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--config", type=str, default="config/attack_experiments.yaml")
    return p.parse_args()


def pick_venv_for_saved_model(model_filename: str) -> str:
    name = model_filename.lower()
    if "centralized" in name or "no_fl" in name:
        return "../venv_dp"
    return "../venv"


def _split_fixed_and_grid(d: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, List[Any]]]:
    fixed = {}
    grid = {}
    for k, v in (d or {}).items():
        if isinstance(v, list):
            grid[k] = v
        else:
            fixed[k] = v
    return fixed, grid


def load_experiments_from_config(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    default_dataset = cfg.get("dataset", DEFAULT_DATASET)
    defaults = cfg.get("defaults", {})

    defaults_fixed, defaults_grid = _split_fixed_and_grid(defaults)
    experiments = []

    for name, spec in cfg["experiments"].items():
        spec_fixed, spec_grid = _split_fixed_and_grid(spec)

        combined_fixed = {
            **defaults_fixed,
            **spec_fixed,
        }
        combined_grid = {
            **defaults_grid,
            **spec_grid,
        }

        exp_dataset = combined_fixed.get("dataset", default_dataset)

        no_fl = bool(combined_fixed.get("no_fl", False))
        if no_fl:
            combined_grid.pop("n_clients", None)
            combined_fixed["n_clients"] = 1

        if combined_grid:
            keys = list(combined_grid.keys())
            values = list(combined_grid.values())
            for combo in itertools.product(*values):
                exp = {
                    "name": name,
                    "dataset": exp_dataset,
                    **combined_fixed,
                    **dict(zip(keys, combo)),
                }
                experiments.append(exp)
        else:
            experiments.append({
                "name": name,
                "dataset": exp_dataset,
                **combined_fixed,
            })

    return experiments


def short_hash(d: Dict[str, Any]) -> str:
    s = json.dumps(d, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_csv_header(path: str, fieldnames: List[str]) -> None:
    write_header = False
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        write_header = True

    if write_header:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()


def append_row(path: str, row: Dict[str, Any], fieldnames: List[str]) -> None:
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow({k: row.get(k, "") for k in fieldnames})


def run_capture(cmd: str, env: Dict[str, str], verbose: bool = False) -> str:
    if verbose:
        logging.debug(f"[CMD] {cmd}")
        show_keys = [
            "DATASET", "MODEL_SAVE", "EXPORT_PATH",
            "LEAKY_TRAIN_FRAC", "CANARY_FRAC", "CANARY_DUPS", "CANARY_FLIP", "LEAKY_SEED",
        ]
        shown = {k: env.get(k) for k in show_keys if k in env}
        logging.debug(f"[ENV] {shown}")

    p = subprocess.run(
        ["bash", "-lc", cmd],
        cwd=PROJECT_DIR,
        env={**os.environ, **env},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = (p.stdout or "").strip()

    if p.returncode != 0:
        tail = out[-4000:] if out else ""
        raise RuntimeError(f"Command failed rc={p.returncode}\n--- OUTPUT TAIL ---\n{tail}")

    if verbose and out:
        logging.debug(f"[OUT_TAIL]\n{out[-1200:]}")
    return out


def get_tf_keras_versions(venv: str) -> str:
    cmd = (
        f"source {venv}/bin/activate && "
        "python - <<'PY'\n"
        "import tensorflow as tf\n"
        "try:\n"
        "  import keras\n"
        "  kv = getattr(keras, '__version__', 'unknown')\n"
        "except Exception:\n"
        "  kv = 'unknown'\n"
        "print('TF=' + getattr(tf, '__version__', 'unknown') + ' Keras=' + str(kv))\n"
        "PY"
    )
    out = subprocess.check_output(["bash", "-lc", cmd], cwd=PROJECT_DIR).decode().strip()
    return out


def pick_venv(exp: Dict[str, Any]) -> str:
    no_fl = bool(exp.get("no_fl", False))
    return "../venv_dp" if no_fl else "../venv"


VALID_RUN_SH_ARGS = {
    "aggregation_type", "dp", "partition_type", "l2_norm_clip",
    "noise_multiplier", "learning_rate", "batch_size", "epochs",
    "local", "clipping", "epsilon", "no_fl", "n_clients", "dataset",
}


def args_for_run_sh(exp: Dict[str, Any]) -> List[str]:
    args = []
    for k, v in exp.items():
        if k in VALID_RUN_SH_ARGS and v is not None:
            args.append(f"--{k}={v}")
    return args


def build_export_filename(exp: Dict[str, Any], rid: str) -> str:
    dataset_tag = str(exp.get("dataset", DEFAULT_DATASET))
    privacy_tag = "plain"
    if exp.get("dp", False):
        privacy_tag = "ldp" if exp.get("local", False) else "cdp"

    param_parts = []
    if exp.get("noise_multiplier") is not None:
        param_parts.append(f"nm{exp['noise_multiplier']}")
    if exp.get("epsilon") is not None:
        param_parts.append(f"eps{exp['epsilon']}")
    if exp.get("l2_norm_clip") is not None:
        param_parts.append(f"clip{exp['l2_norm_clip']}")

    param_tag = "_".join(param_parts) if param_parts else "noparams"
    return f"{dataset_tag}_{exp['name']}_{privacy_tag}_{param_tag}_{rid}.keras"


def main():
    args = parse_args()
    setup_logging(args.verbose)

    exports_dir = args.exports_dir or EXPORT_DIR
    os.makedirs(exports_dir, exist_ok=True)

    fieldnames = [
        "run_id", "name", "dataset",
        "no_fl", "dp", "local", "aggregation_type", "partition_type", "n_clients",
        "learning_rate", "batch_size", "epochs",
        "noise_multiplier", "l2_norm_clip", "clipping", "epsilon", "delta",
        "export_path",

        "mia_base_auc", "mia_base_advantage", "mia_base_accuracy", "mia_base_tpr_at_1pct_fpr",
        "mia_learned_auc", "mia_learned_advantage", "mia_learned_accuracy", "mia_learned_tpr_at_1pct_fpr",

        "status", "error",
    ]
    ensure_csv_header(RESULTS_CSV, fieldnames)

    if args.attack_saved:
        dataset = args.dataset or DEFAULT_DATASET
        model_paths = sorted(Path(exports_dir).glob("*.keras"))

        if not model_paths:
            logging.warning(f"[ATTACK-SAVED] No .keras models found in: {exports_dir}")
            return

        logging.info(f"[ATTACK-SAVED] Found {len(model_paths)} models in {exports_dir}")
        for mp in model_paths:
            rid = f"{now_tag()}_{short_hash({'model': mp.name, 'dataset': dataset})}"
            venv = pick_venv_for_saved_model(mp.name)
            versions = get_tf_keras_versions(venv)

            logging.info(f"[ATTACK-SAVED] model={mp.name} venv={venv} ({versions})")

            row = {k: "" for k in fieldnames}
            row.update({
                "run_id": rid,
                "name": "attack_saved",
                "dataset": dataset,
                "export_path": mp.name,
                "status": "started",
                "error": "",
            })

            env = {"DATASET": dataset}

            try:
                t0 = time.time()

                mia_base_cmd = (
                    f"source {venv}/bin/activate && "
                    f"python mia_attack_baseline.py "
                    f"--model_path '{str(mp)}' "
                    f"--dataset '{dataset}' "
                    f"--output_json_stdout true "
                    f"{'--verbose' if args.verbose else ''}"
                )
                out = run_capture(mia_base_cmd, env, verbose=args.verbose)
                mia_base = extract_json_object(out)

                row.update({
                    "mia_base_auc": mia_base.get("auc"),
                    "mia_base_advantage": mia_base.get("advantage"),
                    "mia_base_accuracy": mia_base.get("accuracy"),
                    "mia_base_tpr_at_1pct_fpr": mia_base.get("tpr_at_1pct_fpr"),
                })

                mia_learned_cmd = (
                    f"source {venv}/bin/activate && "
                    f"python mia_attack_learned.py "
                    f"--model_path '{str(mp)}' "
                    f"--dataset '{dataset}' "
                    f"--output_json_stdout true "
                    f"--seed 42 "
                    f"--max_samples 0 "
                    f"{'--verbose' if args.verbose else ''}"
                )
                out2 = run_capture(mia_learned_cmd, env, verbose=args.verbose)
                mia_learned = extract_json_object(out2)

                row.update({
                    "mia_learned_auc": mia_learned.get("auc"),
                    "mia_learned_advantage": mia_learned.get("advantage"),
                    "mia_learned_accuracy": mia_learned.get("accuracy"),
                    "mia_learned_tpr_at_1pct_fpr": mia_learned.get("tpr_at_1pct_fpr"),
                })

                row["status"] = "ok"
                dt = time.time() - t0
                logging.info(f"[ATTACK-SAVED] done model={mp.name} in {dt:.2f}s")

            except Exception as e:
                row["status"] = "failed"
                row["error"] = str(e)
                logging.error(f"[ATTACK-SAVED] FAILED model={mp.name}: {e}")

            append_row(RESULTS_CSV, row, fieldnames)

        logging.info(f"[DONE] Attacked {len(model_paths)} saved models. Results: {RESULTS_CSV}")
        return

    exps = load_experiments_from_config(args.config)
    logging.info(f"[TRAIN+ATTACK] Loaded {len(exps)} experiments from {args.config}")

    for exp in exps:
        conf_for_hash = {k: v for k, v in exp.items() if k != "name"}
        rid = f"{now_tag()}_{short_hash(conf_for_hash)}"
        export_filename = build_export_filename(exp, rid)
        export_path = os.path.join(exports_dir, export_filename)

        env = {
            "MODEL_SAVE": "true",
            "EXPORT_PATH": export_path,
            "DATASET": exp.get("dataset", DEFAULT_DATASET),
            "LEAKY_TRAIN_FRAC": str(exp.get("leaky_train_frac", 1.0)),
            "CANARY_FRAC": str(exp.get("canary_frac", 0.0)),
            "CANARY_DUPS": str(exp.get("canary_dups", 0)),
            "CANARY_FLIP": str(exp.get("canary_flip", False)).lower(),
            "LEAKY_SEED": "42",
        }

        venv = pick_venv(exp)
        versions = get_tf_keras_versions(venv)
        run_args = " ".join(args_for_run_sh(exp))

        logging.info(f"[EXP] {exp.get('name')} rid={rid} venv={venv} ({versions})")

        row = {k: "" for k in fieldnames}
        row.update({
            "run_id": rid,
            "export_path": os.path.basename(export_path),
            "status": "started",
            "error": "",
            **exp,
        })

        try:
            t0 = time.time()

            train_cmd = f"source {venv}/bin/activate && bash ../run.sh {run_args}"
            run_capture(train_cmd, env, verbose=args.verbose)

            mia_base_cmd = (
                f"source {venv}/bin/activate && "
                f"python mia_attack_baseline.py "
                f"--model_path '{export_path}' "
                f"--dataset '{exp.get('dataset', DEFAULT_DATASET)}' "
                f"--output_json_stdout true "
                f"{'--verbose' if args.verbose else ''}"
            )
            out = run_capture(mia_base_cmd, env, verbose=args.verbose)
            mia_base = extract_json_object(out)

            row.update({
                "mia_base_auc": mia_base.get("auc"),
                "mia_base_advantage": mia_base.get("advantage"),
                "mia_base_accuracy": mia_base.get("accuracy"),
                "mia_base_tpr_at_1pct_fpr": mia_base.get("tpr_at_1pct_fpr"),
            })

            mia_learned_cmd = (
                f"source {venv}/bin/activate && "
                f"python mia_attack_learned.py "
                f"--model_path '{export_path}' "
                f"--dataset '{exp.get('dataset', DEFAULT_DATASET)}' "
                f"--output_json_stdout true "
                f"--seed 42 "
                f"--max_samples 0 "
                f"{'--verbose' if args.verbose else ''}"
            )
            out2 = run_capture(mia_learned_cmd, env, verbose=args.verbose)
            mia_learned = extract_json_object(out2)

            row.update({
                "mia_learned_auc": mia_learned.get("auc"),
                "mia_learned_advantage": mia_learned.get("advantage"),
                "mia_learned_accuracy": mia_learned.get("accuracy"),
                "mia_learned_tpr_at_1pct_fpr": mia_learned.get("tpr_at_1pct_fpr"),
            })

            row["status"] = "ok"
            dt = time.time() - t0
            logging.info(f"[EXP] finished exp={exp.get('name')} in {dt:.2f}s")

        except Exception as e:
            row["status"] = "failed"
            row["error"] = str(e)
            logging.error(f"[EXP] FAILED exp={exp.get('name')}: {e}")

        append_row(RESULTS_CSV, row, fieldnames)

    logging.info(f"[DONE] Wrote results to: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
