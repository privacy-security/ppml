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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output_json_stdout", type=str, default="false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=0, help="Optional cap per split (0 = no cap).")
    p.add_argument("--verbose", action="store_true", help="Print debug info to stderr.")
    return p.parse_args()


def str2bool(x):
    return str(x).lower() in ("1", "true", "yes", "y", "t")


def _dbg(verbose: bool, msg: str) -> None:
    if verbose:
        print(f"[MIA_LEARNED][DBG] {msg}", file=sys.stderr, flush=True)


def to_class_indices(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim > 1 and y.shape[-1] > 1:
        return np.argmax(y, axis=1).astype(int)
    return y.reshape(-1).astype(int)


def _binary_features(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs).reshape(-1)
    y_true = to_class_indices(y_true)

    eps = 1e-7
    p = np.clip(probs, eps, 1.0 - eps)
    p_true = np.where(y_true == 1, p, 1.0 - p)
    loss = -np.log(np.clip(p_true, eps, 1.0))

    conf = np.maximum(p, 1.0 - p)
    margin = np.abs(p - 0.5)
    pred = (p >= 0.5).astype(int)
    correct = (pred == y_true).astype(float)
    entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))

    X = np.stack(
        [loss, -loss, p_true, conf, margin, entropy, correct],
        axis=1,
    ).astype(np.float32)

    return X


def _multiclass_features(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs)
    y_true = to_class_indices(y_true)

    eps = 1e-7
    probs = np.clip(probs, eps, 1.0)

    p_true = probs[np.arange(len(y_true)), y_true]
    loss = -np.log(p_true)

    top1 = np.max(probs, axis=1)
    sorted_probs = np.sort(probs, axis=1)
    top2 = sorted_probs[:, -2]
    margin = top1 - top2
    pred = np.argmax(probs, axis=1)
    correct = (pred == y_true).astype(float)
    entropy = -np.sum(probs * np.log(probs), axis=1)

    X = np.stack(
        [loss, -loss, p_true, top1, margin, entropy, correct],
        axis=1,
    ).astype(np.float32)

    return X


def _regression_features(pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)

    pred_flat = pred.reshape(pred.shape[0], -1)
    y_flat = y_true.reshape(y_true.shape[0], -1)
    diff = pred_flat - y_flat

    mse = np.mean(diff ** 2, axis=1)
    mae = np.mean(np.abs(diff), axis=1)
    l2_resid = np.linalg.norm(diff, axis=1)
    pred_norm = np.linalg.norm(pred_flat, axis=1)
    target_norm = np.linalg.norm(y_flat, axis=1)

    X = np.stack(
        [mse, -mse, mae, -mae, l2_resid, pred_norm, target_norm],
        axis=1,
    ).astype(np.float32)

    return X


def _subsample(X, y, rng, max_samples: int):
    if max_samples and len(X) > max_samples:
        idx = rng.choice(len(X), size=max_samples, replace=False)
        return X[idx], y[idx]
    return X, y


def main():
    args = parse_args()
    json_out = str2bool(args.output_json_stdout)

    _dbg(args.verbose, f"Loading model: {args.model_path}")
    model_path = os.path.abspath(args.model_path)
    model = tf.keras.models.load_model(model_path, compile=False)

    _dbg(args.verbose, f"Loading dataset={args.dataset} partition_type=centralized")
    (x_train, y_train), (x_test, y_test), _ = load_dataset(
        args.dataset,
        partition_type="centralized",
    )
    _dbg(args.verbose, f"x_train={x_train.shape} x_test={x_test.shape}")

    train_pred = np.asarray(model.predict(x_train, verbose=0))
    test_pred = np.asarray(model.predict(x_test, verbose=0))

    # ---------------------------------------------------------
    # Network monitoring: regression
    # ---------------------------------------------------------
    if args.dataset == "network_monitoring":
        X_mem = _regression_features(train_pred, y_train)
        X_non = _regression_features(test_pred, y_test)

    # ---------------------------------------------------------
    # Multiclass classification
    # ---------------------------------------------------------
    elif train_pred.ndim == 2 and train_pred.shape[1] > 1:
        X_mem = _multiclass_features(train_pred, y_train)
        X_non = _multiclass_features(test_pred, y_test)

    # ---------------------------------------------------------
    # Binary classification
    # ---------------------------------------------------------
    else:
        X_mem = _binary_features(train_pred, y_train)
        X_non = _binary_features(test_pred, y_test)

    rng = np.random.default_rng(args.seed)
    mem_idx = rng.permutation(len(X_mem))
    non_idx = rng.permutation(len(X_non))

    mem_half = len(mem_idx) // 2
    non_half = len(non_idx) // 2

    X_mem_tr, X_mem_te = X_mem[mem_idx[:mem_half]], X_mem[mem_idx[mem_half:]]
    X_non_tr, X_non_te = X_non[non_idx[:non_half]], X_non[non_idx[non_half:]]

    y_mem_tr = np.ones(len(X_mem_tr), dtype=int)
    y_non_tr = np.zeros(len(X_non_tr), dtype=int)
    y_mem_te = np.ones(len(X_mem_te), dtype=int)
    y_non_te = np.zeros(len(X_non_te), dtype=int)

    X_mem_tr, y_mem_tr = _subsample(X_mem_tr, y_mem_tr, rng, args.max_samples)
    X_non_tr, y_non_tr = _subsample(X_non_tr, y_non_tr, rng, args.max_samples)
    X_mem_te, y_mem_te = _subsample(X_mem_te, y_mem_te, rng, args.max_samples)
    X_non_te, y_non_te = _subsample(X_non_te, y_non_te, rng, args.max_samples)

    X_train_attack = np.concatenate([X_mem_tr, X_non_tr], axis=0)
    y_train_attack = np.concatenate([y_mem_tr, y_non_tr], axis=0)

    X_test_attack = np.concatenate([X_mem_te, X_non_te], axis=0)
    y_test_attack = np.concatenate([y_mem_te, y_non_te], axis=0)

    _dbg(args.verbose, f"attack train: X={X_train_attack.shape} y={y_train_attack.shape}")
    _dbg(args.verbose, f"attack test : X={X_test_attack.shape} y={y_test_attack.shape}")

    shuf = rng.permutation(len(X_train_attack))
    X_train_attack = X_train_attack[shuf]
    y_train_attack = y_train_attack[shuf]

    clf = LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=-1)
    clf.fit(X_train_attack, y_train_attack)

    scores = clf.predict_proba(X_test_attack)[:, 1]

    auc = float(roc_auc_score(y_test_attack, scores))
    thr = float(np.median(scores))
    preds = (scores >= thr).astype(int)
    acc = float(accuracy_score(y_test_attack, preds))
    advantage = float(2 * auc - 1)

    fpr, tpr, _ = roc_curve(y_test_attack, scores)
    target_fpr = 0.01
    valid = np.where(fpr <= target_fpr)[0]
    idx = int(valid[-1]) if len(valid) else int(np.argmin(fpr))
    tpr_at_1pct_fpr = float(tpr[idx])

    result = {
        "auc": auc,
        "advantage": advantage,
        "accuracy": acc,
        "tpr_at_1pct_fpr": tpr_at_1pct_fpr,
        "attack": "learned_logreg",
        "seed": int(args.seed),
        "max_samples": int(args.max_samples),
    }

    if json_out:
        print(json.dumps(result), flush=True)
    else:
        print(result, flush=True)


if __name__ == "__main__":
    main()
