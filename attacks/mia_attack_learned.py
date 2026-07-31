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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

from data.dataset_loader import load_dataset
from seeding import set_global_seed


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",            required=True)
    p.add_argument("--dataset",               required=True)
    p.add_argument("--output_json_stdout",    type=str, default="false")
    # Global determinism seed (matches training SEED of the attacked model).
    # Drives set_global_seed, the member/non-member split RNG, and the
    # attacker's random_state.  Distinct from --stress_seed.
    p.add_argument("--seed",                  type=int, default=42)
    p.add_argument("--max_samples",           type=int, default=0)
    p.add_argument("--verbose",               action="store_true")
    # FIX 1 & 2
    p.add_argument("--stress_alpha",          type=float, default=None)
    p.add_argument("--stress_beta",           type=float, default=0.01)
    p.add_argument("--stress_k",              type=int,   default=5)
    p.add_argument("--stress_seed",           type=int,   default=42)
    p.add_argument("--member_indices_file",   type=str,   default=None)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def str2bool(x):
    return str(x).lower() in ("1", "true", "yes", "y", "t")


def _dbg(verbose, msg):
    if verbose:
        print(f"[MIA_LEARNED][DBG] {msg}", file=sys.stderr, flush=True)


def _default_alpha(dataset: str) -> float:
    return 0.5 if dataset == "cifar10" else 0.1


def to_class_indices(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim > 1 and y.shape[-1] > 1:
        return np.argmax(y, axis=1).astype(int)
    return y.reshape(-1).astype(int)


def reconstruct_stress_indices(n_train, alpha, beta, seed, verbose):
    rng       = np.random.default_rng(seed)
    n_dprime  = int(np.floor(alpha * n_train))
    dprime_idx = rng.choice(n_train, size=n_dprime, replace=False)
    n_canary  = int(np.floor(beta * n_dprime))
    canary_pos = rng.choice(n_dprime, size=n_canary, replace=False)
    canary_idx = dprime_idx[canary_pos]
    _dbg(verbose, f"D': {n_dprime} records, canaries: {n_canary}")
    return dprime_idx, canary_idx


def load_member_indices(path, n_train, verbose):
    with open(path) as f:
        idx = np.asarray(json.load(f), dtype=int)
    _dbg(verbose, f"Loaded {len(idx)} member indices from {path}")
    return idx


def _subsample(X, y, rng, max_samples):
    if max_samples and len(X) > max_samples:
        idx = rng.choice(len(X), size=max_samples, replace=False)
        return X[idx], y[idx]
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def _binary_features(probs, y_true):
    probs  = np.asarray(probs).reshape(-1)
    y_true = to_class_indices(y_true)
    eps    = 1e-7
    p      = np.clip(probs, eps, 1.0 - eps)
    p_true = np.where(y_true == 1, p, 1.0 - p)
    loss   = -np.log(np.clip(p_true, eps, 1.0))
    conf   = np.maximum(p, 1.0 - p)
    margin = np.abs(p - 0.5)
    pred   = (p >= 0.5).astype(int)
    correct = (pred == y_true).astype(float)
    entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return np.stack([loss, -loss, p_true, conf, margin, entropy, correct],
                    axis=1).astype(np.float32)


def _multiclass_features(probs, y_true):
    probs  = np.asarray(probs)
    y_true = to_class_indices(y_true)
    eps    = 1e-7
    probs  = np.clip(probs, eps, 1.0)
    p_true = probs[np.arange(len(y_true)), y_true]
    loss   = -np.log(p_true)
    top1   = np.max(probs, axis=1)
    top2   = np.sort(probs, axis=1)[:, -2]
    margin = top1 - top2
    pred   = np.argmax(probs, axis=1)
    correct = (pred == y_true).astype(float)
    entropy = -np.sum(probs * np.log(probs), axis=1)
    return np.stack([loss, -loss, p_true, top1, margin, entropy, correct],
                    axis=1).astype(np.float32)


def _regression_features(pred, y_true):
    pred   = np.asarray(pred,   dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    pred_f = pred.reshape(pred.shape[0], -1)
    y_f    = y_true.reshape(y_true.shape[0], -1)
    diff   = pred_f - y_f
    mse    = np.mean(diff ** 2, axis=1)
    mae    = np.mean(np.abs(diff), axis=1)
    l2_r   = np.linalg.norm(diff, axis=1)
    pn     = np.linalg.norm(pred_f, axis=1)
    tn     = np.linalg.norm(y_f,    axis=1)
    return np.stack([mse, -mse, mae, -mae, l2_r, pn, tn],
                    axis=1).astype(np.float32)


def extract_features(dataset, pred, y):
    if dataset in ("network_monitoring", "household_power"):
        return _regression_features(pred, y)
    elif np.asarray(pred).ndim == 2 and np.asarray(pred).shape[1] > 1:
        return _multiclass_features(pred, y)
    else:
        return _binary_features(pred, y)


# ──────────────────────────────────────────────────────────────────────────────
# Attack evaluation helper
# ──────────────────────────────────────────────────────────────────────────────

def run_attack(X_mem, X_non, rng, max_samples, label, verbose, seed):
    """
    Train logistic-regression attacker on half the data, evaluate on the other
    half.  Returns result dict.  `seed` pins the attacker's random_state so the
    fit is reproducible per model seed.
    """
    mem_idx = rng.permutation(len(X_mem))
    non_idx = rng.permutation(len(X_non))
    mh, nh  = len(mem_idx) // 2, len(non_idx) // 2

    X_mem_tr, X_mem_te = X_mem[mem_idx[:mh]],  X_mem[mem_idx[mh:]]
    X_non_tr, X_non_te = X_non[non_idx[:nh]],  X_non[non_idx[nh:]]

    y_mem_tr = np.ones(len(X_mem_tr),  dtype=int)
    y_non_tr = np.zeros(len(X_non_tr), dtype=int)
    y_mem_te = np.ones(len(X_mem_te),  dtype=int)
    y_non_te = np.zeros(len(X_non_te), dtype=int)

    X_mem_tr, y_mem_tr = _subsample(X_mem_tr, y_mem_tr, rng, max_samples)
    X_non_tr, y_non_tr = _subsample(X_non_tr, y_non_tr, rng, max_samples)
    X_mem_te, y_mem_te = _subsample(X_mem_te, y_mem_te, rng, max_samples)
    X_non_te, y_non_te = _subsample(X_non_te, y_non_te, rng, max_samples)

    X_tr = np.concatenate([X_mem_tr, X_non_tr])
    y_tr = np.concatenate([y_mem_tr, y_non_tr])
    X_te = np.concatenate([X_mem_te, X_non_te])
    y_te = np.concatenate([y_mem_te, y_non_te])

    shuf = rng.permutation(len(X_tr))
    clf  = LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=-1,
                              class_weight="balanced", random_state=int(seed))
    clf.fit(X_tr[shuf], y_tr[shuf])

    scores    = clf.predict_proba(X_te)[:, 1]
    auc       = float(roc_auc_score(y_te, scores))
    advantage = float(2 * auc - 1)
    threshold = float(np.median(scores))
    acc       = float(accuracy_score(y_te, (scores >= threshold).astype(int)))

    fpr, tpr, _ = roc_curve(y_te, scores)
    v1  = np.where(fpr <= 0.01)[0]
    v10 = np.where(fpr <= 0.10)[0]
    tpr_1  = float(tpr[int(v1[-1])  if len(v1)  else np.argmin(fpr)])
    tpr_10 = float(tpr[int(v10[-1]) if len(v10) else np.argmin(fpr)])

    _dbg(verbose, f"[{label}] n_mem_te={len(X_mem_te)} n_non_te={len(X_non_te)} "
                  f"AUC={auc:.4f} Adv={advantage:.4f}")

    return {
        "auc": auc, "advantage": advantage, "accuracy": acc,
        "tpr_at_1pct_fpr": tpr_1, "tpr_at_10pct_fpr": tpr_10,
        "n_members_train":     int(len(X_mem_tr)),
        "n_nonmembers_train":  int(len(X_non_tr)),
        "n_members_eval":      int(len(X_mem_te)),
        "n_nonmembers_eval":   int(len(X_non_te)),
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

    (x_train, y_train), (x_test, y_test), _ = load_dataset(
        args.dataset, partition_type="centralized")
    x_train = np.asarray(x_train, dtype=np.float32)
    x_test  = np.asarray(x_test,  dtype=np.float32)
    y_train = np.asarray(y_train)
    y_test  = np.asarray(y_test)
    n_train = len(x_train)

    # ── FIX 1: reconstruct correct member set ────────────────────────────────
    alpha = args.stress_alpha if args.stress_alpha is not None \
            else _default_alpha(args.dataset)

    if args.member_indices_file:
        dprime_idx = load_member_indices(
            args.member_indices_file, n_train, args.verbose)
        _, canary_idx = reconstruct_stress_indices(
            n_train, alpha, args.stress_beta, args.stress_seed, args.verbose)
        dprime_set = set(dprime_idx.tolist())
        canary_idx = np.array([i for i in canary_idx if i in dprime_set], dtype=int)
    else:
        dprime_idx, canary_idx = reconstruct_stress_indices(
            n_train, alpha, args.stress_beta, args.stress_seed, args.verbose)

    _dbg(args.verbose,
         f"D': {len(dprime_idx)} records, canaries: {len(canary_idx)}, "
         f"test: {len(x_test)}")

    # ── Compute features ─────────────────────────────────────────────────────
    _dbg(args.verbose, "Predicting on training set …")
    train_pred = np.asarray(model.predict(x_train, verbose=0))
    _dbg(args.verbose, "Predicting on test set …")
    test_pred  = np.asarray(model.predict(x_test,  verbose=0))

    all_X_train = extract_features(args.dataset, train_pred, y_train)
    X_test      = extract_features(args.dataset, test_pred,  y_test)

    rng = np.random.default_rng(args.seed)

    # ── correct D' members ────────────────────────────────────────────
    X_mem_dprime = all_X_train[dprime_idx]
    result_dprime = run_attack(
        X_mem_dprime, X_test, rng, args.max_samples, "D'", args.verbose, args.seed)

    # ── canary-only ───────────────────────────────────────────────────
    result_canary = {}
    if len(canary_idx) > 0:
        X_mem_canary = all_X_train[canary_idx]
        result_canary = run_attack(
            X_mem_canary, X_test, rng, args.max_samples, "canary", args.verbose,
            args.seed)
    else:
        _dbg(args.verbose, "No canary records — skipping canary eval")

    # ── Original (contaminated) for reference ────────────────────────────────
    result_original = run_attack(
        all_X_train, X_test, rng, args.max_samples, "ORIGINAL", args.verbose,
        args.seed)

    result = {
        "attack":               "learned_logreg",
        "dataset":              args.dataset,
        "seed":                 int(args.seed),
        "alpha":                float(alpha),
        "beta":                 float(args.stress_beta),
        "stress_seed":          int(args.stress_seed),
        "n_train_full":         int(n_train),
        "n_dprime":             int(len(dprime_idx)),
        "n_canary":             int(len(canary_idx)),
        "uninformed_auc":       0.5,
        "mia_dprime":           result_dprime,
        "mia_canary":           result_canary,
        "mia_original_contaminated": result_original,
    }

    if json_out:
        print(json.dumps(result), flush=True)
    else:
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
