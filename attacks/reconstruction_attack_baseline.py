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

from data.dataset_loader import load_dataset
from seeding import set_global_seed


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",         required=True)
    p.add_argument("--dataset",            required=True)
    p.add_argument("--output_json_stdout", type=str, default="false")
    # Global determinism seed (matches training SEED of the attacked model).
    # Pins TF op-determinism so the Adam optimisation is reproducible per seed.
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--steps",              type=int,   default=500)
    p.add_argument("--lr",                 type=float, default=0.05)
    p.add_argument("--prior_l2",           type=float, default=1e-2)
    p.add_argument("--verbose",            action="store_true")
    p.add_argument("--target_split",       type=str,   default="train",
                   choices=["train", "test"])
    p.add_argument("--target_idx",         type=int,   default=0)
    p.add_argument("--target_label",       type=str,   default="auto")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def str2bool(x):
    return str(x).lower() in ("1", "true", "yes", "y", "t")


def _dbg(verbose, msg):
    if verbose:
        print(f"[RECON_BASE][DBG] {msg}", file=sys.stderr, flush=True)


def to_scalar_label(y_val) -> int:
    y_val = np.asarray(y_val)
    if y_val.ndim > 0 and y_val.shape[-1] > 1:
        return int(np.argmax(y_val))
    return int(np.asarray(y_val).reshape(-1)[0])


def _predict(model, x: tf.Tensor) -> tf.Tensor:
    y = model(tf.expand_dims(x, axis=0), training=False)
    return tf.reshape(tf.convert_to_tensor(y), [-1])


def get_dataset_bounds(x_train: np.ndarray):
    lo = x_train.min(axis=0).astype(np.float32)
    hi = x_train.max(axis=0).astype(np.float32)
    return lo, np.maximum(hi, lo)


def is_regression(dataset: str) -> bool:
    """Return True for tasks with continuous vector targets (not class labels)."""
    return dataset in ("network_monitoring", "household_power")


def _classification_loss(model, x: tf.Tensor, target_label: int) -> tf.Tensor:
    """Original cross-entropy loss — correct for classification."""
    probs = _predict(model, x)
    eps   = tf.constant(1e-7, dtype=probs.dtype)
    if tf.size(probs) == 1:
        p = tf.clip_by_value(probs[0], eps, 1.0 - eps)
        p_true = p if target_label == 1 else (1.0 - p)
    else:
        p_true = tf.clip_by_value(probs[target_label], eps, 1.0)
    return -tf.math.log(p_true)


def _regression_output_matching_loss(model, x: tf.Tensor,
                                     target_output: tf.Tensor) -> tf.Tensor:
    """
    Minimises MSE between model(x_candidate) and model(x_target).
    This directly asks: which input produces the same model behaviour as the
    target record?  This is the correct objective for model-inversion on a
    regression model, unlike cross-entropy on a continuous output.
    """
    pred = _predict(model, x)
    return tf.reduce_mean(tf.square(pred - target_output))


def build_loss_fn(dataset, model, target_label, target_output):
    """Return a zero-argument callable that captures the current x variable."""
    if is_regression(dataset):
        def loss_fn(x):
            return _regression_output_matching_loss(model, x, target_output)
    else:
        def loss_fn(x):
            return _classification_loss(model, x, target_label)
    return loss_fn


def compute_uninformed_score(feat_mean: np.ndarray, x_target: np.ndarray) -> float:
    """
    Score achieved by an attacker who returns the dataset mean mu without
    optimising.  s_uninf = 1 / (1 + ||mu - x_target||_2).
    Any reported s_recon should exceed this to be meaningful.
    """
    dist = float(np.linalg.norm((feat_mean - x_target).reshape(-1)))
    return float(1.0 / (1.0 + dist))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    set_global_seed(args.seed)
    json_out = str2bool(args.output_json_stdout)

    model_path = os.path.abspath(args.model_path)
    _dbg(args.verbose, f"Global seed set to {args.seed}")
    _dbg(args.verbose, f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    (x_train, y_train), (x_test, y_test), _ = load_dataset(
        args.dataset, partition_type="centralized")
    x_train = np.asarray(x_train, dtype=np.float32)
    x_test  = np.asarray(x_test,  dtype=np.float32)
    y_train = np.asarray(y_train)
    y_test  = np.asarray(y_test)

    X_split, y_split = (x_train, y_train) if args.target_split == "train" \
                                           else (x_test, y_test)
    if not (0 <= args.target_idx < len(X_split)):
        raise ValueError(f"target_idx {args.target_idx} out of range "
                         f"(split size={len(X_split)})")

    x_target    = X_split[args.target_idx].astype(np.float32)
    y_target    = y_split[args.target_idx]
    target_label = to_scalar_label(y_target) \
                   if args.target_label.lower() == "auto" \
                   else int(args.target_label)

    feat_mean        = x_train.mean(axis=0).astype(np.float32)
    lower_np, upper_np = get_dataset_bounds(x_train)
    lower = tf.constant(lower_np, dtype=tf.float32)
    upper = tf.constant(upper_np, dtype=tf.float32)

    s_uninf = compute_uninformed_score(feat_mean, x_target)
    _dbg(args.verbose,
         f"s_uninf (dataset mean → target) = {s_uninf:.6f}  "
         f"||mu - x_target|| = {1/s_uninf - 1:.6f}")

    if is_regression(args.dataset):
        _dbg(args.verbose, "Regression task: using output-matching loss")
        target_output = _predict(
            model, tf.constant(x_target, dtype=tf.float32))
        _dbg(args.verbose, f"Target model output: {target_output.numpy()}")
    else:
        target_output = None
        _dbg(args.verbose,
             f"Classification task: using cross-entropy, label={target_label}")

    loss_fn_builder = build_loss_fn(
        args.dataset, model, target_label, target_output)

    # ── Optimisation ─────────────────────────────────────────────────────────
    x  = tf.Variable(feat_mean, dtype=tf.float32)
    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)

    for step in range(int(args.steps)):
        with tf.GradientTape() as tape:
            data_loss = loss_fn_builder(x)
            # L2 prior toward mean (unchanged from original for classification;
            # still used for regression to keep candidate in a plausible region)
            prior = args.prior_l2 * tf.reduce_mean(tf.square(x - feat_mean))
            total = data_loss + prior

        grads = tape.gradient(total, [x])
        opt.apply_gradients(zip(grads, [x]))
        x.assign(tf.clip_by_value(x, lower, upper))

        if args.verbose and step in (0, 10, 50, 100, 200, args.steps - 1):
            _dbg(args.verbose,
                 f"step={step} data_loss={float(data_loss):.6g} "
                 f"prior={float(prior):.6g}")

    x_star       = x.numpy().astype(np.float32)
    dist_target  = float(np.linalg.norm((x_star - x_target).reshape(-1)))
    s_recon      = float(1.0 / (1.0 + dist_target))
    improvement  = s_recon - s_uninf        # positive = model helped

    train_flat   = x_train.reshape(len(x_train), -1)
    d2           = np.sum((train_flat - x_star.reshape(-1)) ** 2, axis=1)
    min_d_train  = float(np.sqrt(np.min(d2)))

    _dbg(args.verbose,
         f"s_recon={s_recon:.6f}  s_uninf={s_uninf:.6f}  "
         f"improvement={improvement:+.6f}  dist_to_target={dist_target:.6g}")

    detail = {
        "target_split":       args.target_split,
        "target_idx":         int(args.target_idx),
        "y_target":           int(to_scalar_label(y_target)),
        "used_label":         int(target_label),
        "task_type":          "regression" if is_regression(args.dataset)
                              else "classification",
        "dist_to_target":     dist_target,
        "min_l2_dist_to_train": min_d_train,
        "steps":              int(args.steps),
        "lr":                 float(args.lr),
        "prior_l2":           float(args.prior_l2),
    }

    result = {
        "score":            s_recon,
        "s_uninf":          s_uninf,
        "improvement":      improvement,
        "seed":             int(args.seed),
        "detail":           json.dumps(detail, sort_keys=True),
        "attack":           "recon_targeted_baseline",
    }

    if json_out:
        print(json.dumps(result), flush=True)
    else:
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
