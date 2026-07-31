import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

from data.dataset_loader import load_dataset
from seeding import set_global_seed


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output_json_stdout", type=str, default="false")
    p.add_argument("--verbose", action="store_true")

    # Global determinism seed. Must match the training SEED of the attacked
    # model so the attack's own RNG-dependent steps are reproducible per seed.
    # NOTE: this is the *model* seed, distinct from --stress_seed (which defines
    # the D' member set and is held fixed across all model seeds).
    p.add_argument("--seed", type=int, default=42,
                   help="Global seed for determinism (matches training SEED).")

    # FIX 1 & 2: stress regime parameters (must match training configuration)
    p.add_argument("--stress_alpha",  type=float, default=None,
                   help="Training fraction used in stress regime (e.g. 0.1 or 0.5). "
                        "Auto-detected by dataset if not set.")
    p.add_argument("--stress_beta",   type=float, default=0.01,
                   help="Canary fraction of D'  (default 0.01).")
    p.add_argument("--stress_k",      type=int,   default=5,
                   help="Canary duplication factor (default 5).")
    p.add_argument("--stress_seed",   type=int,   default=42,
                   help="Random seed used during stress regime (default 42).")
    p.add_argument("--member_indices_file", type=str, default=None,
                   help="Optional path to a JSON file containing the list of "
                        "x_train indices that were in D' during training.  "
                        "When provided this is used instead of seed-based reconstruction.")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def str2bool(x):
    return str(x).lower() in ("1", "true", "yes", "y", "t")


def _dbg(verbose, msg):
    if verbose:
        print(f"[MIA_BASE][DBG] {msg}", file=sys.stderr, flush=True)


def to_class_indices(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim > 1 and y.shape[-1] > 1:
        return np.argmax(y, axis=1).astype(int)
    return y.reshape(-1).astype(int)


def _default_alpha(dataset: str) -> float:
    """Return the stress-regime alpha used in the paper for each dataset."""
    return 0.5 if dataset == "cifar10" else 0.1


def regression_loss_per_sample(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    y    = np.asarray(y,    dtype=np.float64)
    diff = (pred - y).reshape(pred.shape[0], -1)
    return np.mean(diff ** 2, axis=1).astype(np.float32)
    

def reconstruct_stress_indices(n_train: int, alpha: float, beta: float,
                                seed: int, verbose: bool):
    """
    Deterministically reconstruct which x_train indices were in D' and which
    were canaries, using the same operations as Algorithm 1 in the paper.

    Returns
    -------
    dprime_idx  : np.ndarray of int  – indices of ALL records in D'
    canary_idx  : np.ndarray of int  – indices of the canary records inside D'
    """
    rng = np.random.default_rng(seed)

    n_dprime  = int(np.floor(alpha * n_train))
    dprime_idx = rng.choice(n_train, size=n_dprime, replace=False)

    n_canary  = int(np.floor(beta * n_dprime))
    canary_positions = rng.choice(n_dprime, size=n_canary, replace=False)
    canary_idx = dprime_idx[canary_positions]

    _dbg(verbose, f"Reconstructed D': {n_dprime} records, {n_canary} canaries "
                  f"(seed={seed}, alpha={alpha}, beta={beta})")
    return dprime_idx, canary_idx


def load_member_indices(path: str, n_train: int, verbose: bool) -> np.ndarray:
    with open(path) as f:
        idx = np.asarray(json.load(f), dtype=int)
    _dbg(verbose, f"Loaded {len(idx)} member indices from {path}")
    assert idx.max() < n_train, "member_indices_file contains out-of-range index"
    return idx


# ──────────────────────────────────────────────────────────────────────────────
# Membership score computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_loss_scores(dataset: str, model, x_data: np.ndarray,
                        y_data: np.ndarray) -> np.ndarray:
    """Return per-sample membership score = -loss (higher = more likely member)."""
    eps = 1e-7
    pred = np.asarray(model.predict(x_data, verbose=0))

    if dataset in ("network_monitoring", "household_power"):
        return -regression_loss_per_sample(pred, y_data)

    elif pred.ndim == 2 and pred.shape[1] > 1:          # multiclass
        y_int = to_class_indices(y_data)
        p     = np.clip(pred[np.arange(len(y_int)), y_int], eps, 1.0)
        return np.log(p).astype(np.float32)             # = -NLL

    else:                                                # binary
        y_int = to_class_indices(y_data)
        p     = np.clip(pred.reshape(-1), eps, 1.0 - eps)
        p_true = np.where(y_int == 1, p, 1.0 - p)
        return np.log(np.clip(p_true, eps, 1.0)).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_mia(member_scores: np.ndarray,
                 nonmember_scores: np.ndarray,
                 label: str,
                 verbose: bool) -> dict:
    scores = np.concatenate([member_scores, nonmember_scores])
    labels = np.concatenate([
        np.ones(len(member_scores),    dtype=int),
        np.zeros(len(nonmember_scores), dtype=int),
    ])

    auc       = float(roc_auc_score(labels, scores))
    advantage = float(2 * auc - 1)
    threshold = float(np.median(scores))
    preds     = (scores >= threshold).astype(int)
    acc       = float(accuracy_score(labels, preds))

    fpr, tpr, _ = roc_curve(labels, scores)
    valid          = np.where(fpr <= 0.01)[0]
    idx_01         = int(valid[-1]) if len(valid) else int(np.argmin(fpr))
    tpr_at_1pct    = float(tpr[idx_01])

    valid_10       = np.where(fpr <= 0.10)[0]
    idx_10         = int(valid_10[-1]) if len(valid_10) else int(np.argmin(fpr))
    tpr_at_10pct   = float(tpr[idx_10])

    _dbg(verbose, f"[{label}] n_mem={len(member_scores)} n_non={len(nonmember_scores)} "
                  f"AUC={auc:.4f} Adv={advantage:.4f} TPR@1%FPR={tpr_at_1pct:.4f}")

    return {
        "auc":             auc,
        "advantage":       advantage,
        "accuracy":        acc,
        "tpr_at_1pct_fpr": tpr_at_1pct,
        "tpr_at_10pct_fpr": tpr_at_10pct,
        "n_members":       int(len(member_scores)),
        "n_nonmembers":    int(len(nonmember_scores)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    set_global_seed(args.seed)
    json_out = str2bool(args.output_json_stdout)

    _dbg(args.verbose, f"Global seed set to {args.seed}")
    _dbg(args.verbose, f"Loading model: {args.model_path}")
    model = tf.keras.models.load_model(
        os.path.abspath(args.model_path), compile=False)

    _dbg(args.verbose, f"Loading dataset={args.dataset} partition_type=centralized")
    (x_train, y_train), (x_test, y_test), _ = load_dataset(
        args.dataset, partition_type="centralized")

    x_train = np.asarray(x_train, dtype=np.float32)
    x_test  = np.asarray(x_test,  dtype=np.float32)
    y_train = np.asarray(y_train)
    y_test  = np.asarray(y_test)

    n_train = len(x_train)
    _dbg(args.verbose, f"x_train={x_train.shape} x_test={x_test.shape}")

    alpha = args.stress_alpha if args.stress_alpha is not None \
            else _default_alpha(args.dataset)

    if args.member_indices_file:
        dprime_idx = load_member_indices(
            args.member_indices_file, n_train, args.verbose)
        # Reconstruct canaries separately (need seed-based reconstruction)
        _, canary_idx = reconstruct_stress_indices(
            n_train, alpha, args.stress_beta, args.stress_seed, args.verbose)
        # Keep only canary indices that are actually in the provided dprime_idx
        dprime_set = set(dprime_idx.tolist())
        canary_idx = np.array([i for i in canary_idx if i in dprime_set], dtype=int)
    else:
        dprime_idx, canary_idx = reconstruct_stress_indices(
            n_train, alpha, args.stress_beta, args.stress_seed, args.verbose)

    _dbg(args.verbose,
         f"Member set: {len(dprime_idx)} records ({100*len(dprime_idx)/n_train:.1f}% "
         f"of full training set)  |  Canary records: {len(canary_idx)}")

    # ── Compute scores ────────────────────────────────────────────────────────
    _dbg(args.verbose, "Computing scores for full training set …")
    all_train_scores = compute_loss_scores(
        args.dataset, model, x_train, y_train)
    _dbg(args.verbose, "Computing scores for test set …")
    test_scores = compute_loss_scores(
        args.dataset, model, x_test, y_test)

    member_scores = all_train_scores[dprime_idx]

    canary_scores = all_train_scores[canary_idx]

    # An attacker who knows nothing about membership would see identical
    # score distributions for members and non-members → AUC = 0.50.
    # Report as calibration reference alongside actual results.
    uninformed_auc = 0.5

    # ── Evaluate: full D' vs test ─────────────────────────────────────────────
    result_full = evaluate_mia(
        member_scores, test_scores, "full_D'", args.verbose)

    # ── Evaluate: canary records only vs test ─────────────────────────────────
    result_canary = {}
    if len(canary_scores) > 0:
        result_canary = evaluate_mia(
            canary_scores, test_scores, "canary_only", args.verbose)
    else:
        _dbg(args.verbose, "No canary records found – skipping canary eval")

    # ── Comparison with original (contaminated) evaluation ───────────────────
    # For reference: what the original script would have reported
    result_original_style = evaluate_mia(
        all_train_scores, test_scores, "ORIGINAL_full_train", args.verbose)

    result = {
        "attack": "baseline_loss",
        "dataset": args.dataset,
        "seed": int(args.seed),
        "alpha": float(alpha),
        "beta": float(args.stress_beta),
        "k": int(args.stress_k),
        "stress_seed": int(args.stress_seed),
        "n_train_full": int(n_train),
        "n_dprime": int(len(dprime_idx)),
        "n_canary": int(len(canary_idx)),
        "uninformed_auc": uninformed_auc,
        # ── correct member set ──
        "mia_dprime": result_full,
        # ── canary-only ──
        "mia_canary": result_canary,
        # ── For reference: original ──
        "mia_original_contaminated": result_original_style,
    }

    if json_out:
        print(json.dumps(result), flush=True)
    else:
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
