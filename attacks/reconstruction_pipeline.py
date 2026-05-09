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


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(PROJECT_DIR, ".."))
EXPORT_DIR = os.path.join(PROJECT_DIR, "exports_recon")
RESULTS_CSV = os.path.join(PROJECT_DIR, "recon_results.csv")

DEFAULT_DATASET = "body_signal_of_smoking"
DEFAULT_VENV_SIZES = "../venv"


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")


def extract_json_object(text: str) -> dict:
    matches = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    if not matches:
        raise ValueError(f"No JSON object found in output:\n{text[:2000]}")
    return json.loads(matches[-1])


def ensure_csv_header(path: str, fieldnames: List[str]) -> None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()


def append_row(path: str, row: Dict[str, Any], fieldnames: List[str]) -> None:
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow({k: row.get(k, "") for k in fieldnames})


def short_hash(d: Dict[str, Any]) -> str:
    s = json.dumps(d, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


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
    return subprocess.check_output(["bash", "-lc", cmd], cwd=PROJECT_DIR).decode().strip()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attack-saved", action="store_true")
    p.add_argument("--exports-dir", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--config", type=str, default="config/attack_experiments.yaml")
    p.add_argument("--num-targets", type=int, default=10)
    p.add_argument("--targets-seed", type=int, default=42)
    p.add_argument("--target-split", type=str, default="train", choices=["train", "test"])
    p.add_argument("--targets", type=str, default="")
    p.add_argument("--venv-for-sizes", type=str, default=DEFAULT_VENV_SIZES)
    return p.parse_args()


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


def pick_venv(exp: Dict[str, Any]) -> str:
    no_fl = bool(exp.get("no_fl", False))
    return "../venv_dp" if no_fl else "../venv"


def pick_venv_for_saved_model(model_filename: str) -> str:
    name = model_filename.lower()
    if "centralized" in name or "no_fl" in name:
        return "../venv_dp"
    return "../venv"


VALID_RUN_SH_ARGS = {
    "aggregation_type", "dp", "partition_type", "l2_norm_clip",
    "noise_multiplier", "learning_rate", "batch_size", "epochs",
    "local", "clipping", "epsilon", "delta",
    "no_fl", "n_clients", "dataset",
}


def args_for_run_sh(exp: Dict[str, Any]) -> List[str]:
    args = []
    for k, v in exp.items():
        if k in VALID_RUN_SH_ARGS and v is not None:
            args.append(f"--{k}={v}")
    return args


_DATASET_CACHE: Dict[Tuple[str, str], Tuple[int, int]] = {}


def _get_split_sizes_via_venv(dataset: str, venv: str) -> Tuple[int, int]:
    cmd = (
        f"source {venv}/bin/activate && "
        f"PYTHONPATH='{REPO_ROOT}':$PYTHONPATH "
        "python - <<'PY'\n"
        "import json\n"
        "import contextlib, io\n"
        "from data.dataset_loader import load_dataset\n"
        "dataset = " + json.dumps(dataset) + "\n"
        "buf = io.StringIO()\n"
        "with contextlib.redirect_stdout(buf):\n"
        "    (x_train, _), (x_test, _), _ = load_dataset(dataset, partition_type='centralized')\n"
        "print(json.dumps({'n_train': int(len(x_train)), 'n_test': int(len(x_test))}))\n"
        "PY"
    )

    out = subprocess.check_output(["bash", "-lc", cmd], cwd=PROJECT_DIR).decode("utf-8").strip()
    try:
        info = json.loads(out)
    except Exception:
        info = extract_json_object(out)

    return int(info["n_train"]), int(info["n_test"])


def _get_split_sizes(dataset: str, venv_for_sizes: str) -> Tuple[int, int]:
    key = (dataset, venv_for_sizes)
    if key in _DATASET_CACHE:
        return _DATASET_CACHE[key]
    n_train, n_test = _get_split_sizes_via_venv(dataset, venv_for_sizes)
    _DATASET_CACHE[key] = (n_train, n_test)
    return n_train, n_test


def _parse_targets_arg(targets: str) -> List[int]:
    if not targets.strip():
        return []
    return [int(tok.strip()) for tok in targets.split(",") if tok.strip()]


def pick_target_indices(
    dataset: str,
    split: str,
    k: int,
    seed: int,
    explicit: str = "",
    venv_for_sizes: str = DEFAULT_VENV_SIZES,
) -> List[int]:
    explicit_idx = _parse_targets_arg(explicit)
    if explicit_idx:
        return explicit_idx

    n_train, n_test = _get_split_sizes(dataset, venv_for_sizes)
    n = n_train if split == "train" else n_test
    if n <= 0:
        raise RuntimeError(f"Empty split: dataset={dataset} split={split}")

    k = max(1, min(int(k), n))
    rng = random.Random(int(seed))
    return rng.sample(range(n), k)


def _safe_json_loads(s: Any) -> Dict[str, Any]:
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}


def _agg(xs: List[float]) -> Dict[str, float]:
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return {"mean": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(statistics.mean(xs)),
        "median": float(statistics.median(xs)),
        "min": float(min(xs)),
        "max": float(max(xs)),
    }


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
        "noise_multiplier", "l2_norm_clip", "clipping", "epsilon", "export_path",

        "target_split", "num_targets", "targets_seed", "targets",

        "recon_base_mean_score", "recon_base_median_score", "recon_base_mean_dist",
        "recon_hard_mean_score", "recon_hard_median_score", "recon_hard_mean_dist",

        "recon_base_detail", "recon_hard_detail",

        "status", "error",
    ]
    ensure_csv_header(RESULTS_CSV, fieldnames)

    def run_recon_on_model(venv: str, model_path: str, dataset: str, verbose: bool, targets: List[int]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        env = {"DATASET": dataset}

        base_scores, base_dists = [], []
        hard_scores, hard_dists = [], []

        for tidx in targets:
            base_cmd = (
                f"source {venv}/bin/activate && "
                f"python reconstruction_attack_baseline.py "
                f"--model_path '{model_path}' "
                f"--dataset '{dataset}' "
                f"--output_json_stdout true "
                f"--target_split '{args.target_split}' "
                f"--target_idx {tidx} "
                f"{'--verbose' if verbose else ''}"
            )
            out = run_capture(base_cmd, env, verbose=verbose)
            base = extract_json_object(out)
            base_scores.append(float(base.get("score", float("nan"))))
            bdet = _safe_json_loads(base.get("detail"))
            if "dist_to_target" in bdet:
                base_dists.append(float(bdet["dist_to_target"]))

            hard_cmd = (
                f"source {venv}/bin/activate && "
                f"python reconstruction_attack_extended.py "
                f"--model_path '{model_path}' "
                f"--dataset '{dataset}' "
                f"--output_json_stdout true "
                f"--seed 42 "
                f"--target_split '{args.target_split}' "
                f"--target_idx {tidx} "
                f"{'--verbose' if verbose else ''}"
            )
            out2 = run_capture(hard_cmd, env, verbose=verbose)
            hard = extract_json_object(out2)
            hard_scores.append(float(hard.get("score", float("nan"))))
            hdet = _safe_json_loads(hard.get("detail"))
            if "dist_to_target" in hdet:
                hard_dists.append(float(hdet["dist_to_target"]))

        base_summary = {
            "k": len(base_scores),
            "score": _agg(base_scores),
            "dist_to_target": _agg(base_dists),
        }
        hard_summary = {
            "k": len(hard_scores),
            "score": _agg(hard_scores),
            "dist_to_target": _agg(hard_dists),
        }
        return base_summary, hard_summary

    if args.attack_saved:
        dataset = args.dataset or DEFAULT_DATASET
        model_paths = sorted(Path(exports_dir).glob("*.keras"))
        if not model_paths:
            logging.warning(f"[ATTACK-SAVED] No .keras models found in: {exports_dir}")
            return

        targets = pick_target_indices(
            dataset=dataset,
            split=args.target_split,
            k=args.num_targets,
            seed=args.targets_seed,
            explicit=args.targets,
            venv_for_sizes=args.venv_for_sizes,
        )

        for mp in model_paths:
            rid = f"{now_tag()}_{short_hash({'model': mp.name, 'dataset': dataset})}"
            venv = pick_venv_for_saved_model(mp.name)

            row = {k: "" for k in fieldnames}
            row.update({
                "run_id": rid,
                "name": "attack_saved",
                "dataset": dataset,
                "export_path": mp.name,
                "target_split": args.target_split,
                "num_targets": len(targets),
                "targets_seed": args.targets_seed,
                "targets": ",".join(map(str, targets)),
                "status": "started",
                "error": "",
            })

            try:
                base_summary, hard_summary = run_recon_on_model(
                    venv=venv,
                    model_path=str(mp),
                    dataset=dataset,
                    verbose=args.verbose,
                    targets=targets,
                )

                row.update({
                    "recon_base_mean_score": base_summary["score"]["mean"],
                    "recon_base_median_score": base_summary["score"]["median"],
                    "recon_base_mean_dist": base_summary["dist_to_target"]["mean"],
                    "recon_hard_mean_score": hard_summary["score"]["mean"],
                    "recon_hard_median_score": hard_summary["score"]["median"],
                    "recon_hard_mean_dist": hard_summary["dist_to_target"]["mean"],
                    "recon_base_detail": json.dumps(base_summary, sort_keys=True),
                    "recon_hard_detail": json.dumps(hard_summary, sort_keys=True),
                    "status": "ok",
                })

            except Exception as e:
                row["status"] = "failed"
                row["error"] = str(e)

            append_row(RESULTS_CSV, row, fieldnames)

        logging.info(f"[DONE] Results: {RESULTS_CSV}")
        return

    exps = load_experiments_from_config(args.config)
    logging.info(f"[TRAIN+ATTACK] Loaded {len(exps)} experiments from {args.config}")

    for exp in exps:
        dataset = exp.get("dataset", DEFAULT_DATASET)
        targets = pick_target_indices(
            dataset=dataset,
            split=args.target_split,
            k=args.num_targets,
            seed=args.targets_seed,
            explicit=args.targets,
            venv_for_sizes=args.venv_for_sizes,
        )

        conf_for_hash = {k: v for k, v in exp.items() if k != "name"}
        rid = f"{now_tag()}_{short_hash(conf_for_hash)}"
        export_filename = build_export_filename(exp, rid)
        export_path = os.path.join(exports_dir, export_filename)

        env = {
            "MODEL_SAVE": "true",
            "EXPORT_PATH": export_path,
            "DATASET": dataset,
            "LEAKY_TRAIN_FRAC": str(exp.get("leaky_train_frac", 1.0)),
            "CANARY_FRAC": str(exp.get("canary_frac", 0.0)),
            "CANARY_DUPS": str(exp.get("canary_dups", 0)),
            "CANARY_FLIP": str(exp.get("canary_flip", False)).lower(),
            "LEAKY_SEED": "42",
        }

        venv = pick_venv(exp)
        run_args = " ".join(args_for_run_sh(exp))

        row = {k: "" for k in fieldnames}
        row.update({
            "run_id": rid,
            "name": exp.get("name", ""),
            "dataset": dataset,
            "export_path": os.path.basename(export_path),
            "target_split": args.target_split,
            "num_targets": len(targets),
            "targets_seed": args.targets_seed,
            "targets": ",".join(map(str, targets)),
            "status": "started",
            "error": "",
            **exp,
        })

        try:
            train_cmd = f"source {venv}/bin/activate && bash ../run.sh {run_args}"
            run_capture(train_cmd, env, verbose=args.verbose)

            base_summary, hard_summary = run_recon_on_model(
                venv=venv,
                model_path=export_path,
                dataset=dataset,
                verbose=args.verbose,
                targets=targets,
            )

            row.update({
                "recon_base_mean_score": base_summary["score"]["mean"],
                "recon_base_median_score": base_summary["score"]["median"],
                "recon_base_mean_dist": base_summary["dist_to_target"]["mean"],
                "recon_hard_mean_score": hard_summary["score"]["mean"],
                "recon_hard_median_score": hard_summary["score"]["median"],
                "recon_hard_mean_dist": hard_summary["dist_to_target"]["mean"],
                "recon_base_detail": json.dumps(base_summary, sort_keys=True),
                "recon_hard_detail": json.dumps(hard_summary, sort_keys=True),
                "status": "ok",
            })

        except Exception as e:
            row["status"] = "failed"
            row["error"] = str(e)

        append_row(RESULTS_CSV, row, fieldnames)

    logging.info(f"[DONE] Wrote results to: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
