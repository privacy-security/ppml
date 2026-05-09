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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output_json_stdout", type=str, default="false")
    p.add_argument("--verbose", action="store_true", help="Print debug info to stderr.")
    return p.parse_args()


def str2bool(x):
    return str(x).lower() in ("1", "true", "yes", "y", "t")


def _dbg(verbose: bool, msg: str) -> None:
    if verbose:
        print(f"[MIA_BASE][DBG] {msg}", file=sys.stderr, flush=True)


def _summ(x: np.ndarray) -> str:
    x = np.asarray(x).reshape(-1)
    q = np.quantile(x, [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0])
    return f"n={len(x)} mean={x.mean():.4g} std={x.std():.4g} q={q.round(4).tolist()}"


def to_class_indices(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim > 1 and y.shape[-1] > 1:
        return np.argmax(y, axis=1).astype(int)
    return y.reshape(-1).astype(int)


def regression_loss_per_sample(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    diff = pred - y
    diff = diff.reshape(diff.shape[0], -1)
    mse = np.mean(diff ** 2, axis=1)
    return mse.astype(np.float32)


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

    eps = 1e-7

    # ---------------------------------------------------------
    # Network monitoring: regression / sequence prediction
    # ---------------------------------------------------------
    if args.dataset == "network_monitoring":
        train_loss = regression_loss_per_sample(train_pred, y_train)
        test_loss = regression_loss_per_sample(test_pred, y_test)

    # ---------------------------------------------------------
    # Multiclass classification (e.g. CIFAR-10)
    # ---------------------------------------------------------
    elif train_pred.ndim == 2 and train_pred.shape[1] > 1:
        y_train_int = to_class_indices(y_train)
        y_test_int = to_class_indices(y_test)

        p_train = train_pred[np.arange(len(y_train_int)), y_train_int]
        p_test = test_pred[np.arange(len(y_test_int)), y_test_int]

        p_train = np.clip(p_train, eps, 1.0)
        p_test = np.clip(p_test, eps, 1.0)

        train_loss = -np.log(p_train)
        test_loss = -np.log(p_test)

    # ---------------------------------------------------------
    # Binary classification (e.g. body smoking)
    # ---------------------------------------------------------
    else:
        y_train_int = to_class_indices(y_train)
        y_test_int = to_class_indices(y_test)

        train_pred = train_pred.reshape(-1)
        test_pred = test_pred.reshape(-1)

        p_train = np.clip(train_pred, eps, 1.0 - eps)
        p_test = np.clip(test_pred, eps, 1.0 - eps)

        ptrue_train = np.where(y_train_int == 1, p_train, 1.0 - p_train)
        ptrue_test = np.where(y_test_int == 1, p_test, 1.0 - p_test)

        train_loss = -np.log(np.clip(ptrue_train, eps, 1.0))
        test_loss = -np.log(np.clip(ptrue_test, eps, 1.0))

    train_score = -train_loss
    test_score = -test_loss

    _dbg(args.verbose, f"train_score: {_summ(train_score)}")
    _dbg(args.verbose, f"test_score : {_summ(test_score)}")

    scores = np.concatenate([train_score, test_score], axis=0)
    labels = np.concatenate(
        [np.ones(len(train_score), dtype=int), np.zeros(len(test_score), dtype=int)],
        axis=0,
    )

    auc = float(roc_auc_score(labels, scores))

    threshold = float(np.median(scores))
    preds = (scores >= threshold).astype(int)
    acc = float(accuracy_score(labels, preds))
    advantage = float(2 * auc - 1)

    fpr, tpr, _ = roc_curve(labels, scores)
    target_fpr = 0.01
    valid = np.where(fpr <= target_fpr)[0]
    idx = int(valid[-1]) if len(valid) else int(np.argmin(fpr))
    tpr_at_1pct_fpr = float(tpr[idx])

    result = {
        "auc": auc,
        "advantage": advantage,
        "accuracy": acc,
        "tpr_at_1pct_fpr": tpr_at_1pct_fpr,
        "attack": "baseline_loss",
    }

    if json_out:
        print(json.dumps(result), flush=True)
    else:
        print(result, flush=True)


if __name__ == "__main__":
    main()
