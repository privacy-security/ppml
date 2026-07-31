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
    p.add_argument("--output_json_stdout", type=str,   default="false")
    # Global determinism seed (matches training SEED of the attacked model).
    # Drives set_global_seed AND the restart-noise RNG below.
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--restarts",           type=int,   default=10)
    p.add_argument("--steps",              type=int,   default=800)
    p.add_argument("--lr",                 type=float, default=0.05)
    p.add_argument("--prior_l2z",          type=float, default=1e-2)
    p.add_argument("--prior_l1",           type=float, default=1e-3)
    p.add_argument("--noise_scale",        type=float, default=0.15)
    p.add_argument("--verbose",            action="store_true")
    p.add_argument("--target_split",       type=str,   default="train",
                   choices=["train", "test"])
    p.add_argument("--target_idx",         type=int,   default=0)
    p.add_argument("--target_label",       type=str,   default="auto")
    p.add_argument("--try_both_labels",    action="store_true",
                   help="Try all output classes as label.  "
                        "Auto-disabled for regression tasks.")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def str2bool(x):
    return str(x).lower() in ("1", "true", "yes", "y", "t")


def _dbg(verbose, msg):
    if verbose:
        print(f"[RECON_HARD][DBG] {msg}", file=sys.stderr, flush=True)


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
    return dataset in ("network_monitoring", "household_power")


def _classification_loss(model, x, target_label):
    probs = _predict(model, x)
    eps   = tf.constant(1e-7, dtype=probs.dtype)
    if tf.size(probs) == 1:
        p = tf.clip_by_value(probs[0], eps, 1.0 - eps)
        p_true = p if target_label == 1 else (1.0 - p)
    else:
        p_true = tf.clip_by_value(probs[target_label], eps, 1.0)
    return -tf.math.log(p_true)


def _regression_output_matching_loss(model, x, target_output):
    """FIX: minimise distance between model outputs, not cross-entropy."""
    pred = _predict(model, x)
    return tf.reduce_mean(tf.square(pred - target_output))


def compute_uninformed_score(feat_mean, x_target):
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
        raise ValueError(f"target_idx {args.target_idx} out of range")

    x_target     = X_split[args.target_idx].astype(np.float32)
    y_target     = y_split[args.target_idx]
    label_auto   = to_scalar_label(y_target) \
                   if args.target_label.lower() == "auto" \
                   else int(args.target_label)

    mean    = x_train.mean(axis=0).astype(np.float32)
    std     = x_train.std(axis=0).astype(np.float32)
    std     = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    lo_np, hi_np = get_dataset_bounds(x_train)
    lower   = tf.constant(lo_np, dtype=tf.float32)
    upper   = tf.constant(hi_np, dtype=tf.float32)

    # ──  uninformed baseline ────────────────────────────────────────────
    s_uninf = compute_uninformed_score(mean, x_target)
    _dbg(args.verbose,
         f"s_uninf={s_uninf:.6f}  ||mu-x_target||={1/s_uninf-1:.6f}")

    # ── pre-compute target model output for regression ─────────────────
    if is_regression(args.dataset):
        _dbg(args.verbose, "Regression task → output-matching loss")
        target_output = _predict(model, tf.constant(x_target, dtype=tf.float32))
        _dbg(args.verbose, f"model(x_target) = {target_output.numpy()}")
    else:
        target_output = None

    # ── disable label search for regression ───────────────────────────
    regression = is_regression(args.dataset)
    if regression:
        labels_to_try = [0]           # label is irrelevant for regression
        _dbg(args.verbose, "try_both_labels suppressed for regression task")
    elif args.try_both_labels:
        probe = _predict(model, tf.constant(mean)).numpy().reshape(-1)
        labels_to_try = list(range(probe.size)) if probe.size > 1 else [0, 1]
    else:
        labels_to_try = [label_auto]

    rng        = np.random.default_rng(int(args.seed))
    train_flat = x_train.reshape(len(x_train), -1)
    x_tgt_flat = x_target.reshape(-1)

    best = {
        "dist_to_target":   float("inf"),
        "score":            0.0,
        "improvement":      float("-inf"),
        "used_label":       None,
        "restart":          None,
        "x_star":           None,
        "p_final":          None,
        "min_dist_to_train": None,
    }

    for used_label in labels_to_try:
        for r in range(int(args.restarts)):
            x0 = mean + rng.normal(
                0.0, float(args.noise_scale), size=mean.shape).astype(np.float32)
            x0 = np.clip(x0, lo_np, hi_np)
            x  = tf.Variable(x0, dtype=tf.float32)
            opt = tf.keras.optimizers.Adam(learning_rate=args.lr)

            for _ in range(int(args.steps)):
                with tf.GradientTape() as tape:
                    # ── choose loss based on task type ────────────────
                    if regression:
                        data_loss = _regression_output_matching_loss(
                            model, x, target_output)
                    else:
                        data_loss = _classification_loss(
                            model, x, int(used_label))

                    z         = (x - mean) / std
                    prior_l2z = args.prior_l2z * tf.reduce_mean(tf.square(z))
                    prior_l1  = args.prior_l1  * tf.reduce_mean(tf.abs(x - mean))
                    total     = data_loss + prior_l2z + prior_l1

                grads = tape.gradient(total, [x])
                opt.apply_gradients(zip(grads, [x]))
                x.assign(tf.clip_by_value(x, lower, upper))

            x_star       = x.numpy().astype(np.float32)
            x_star_flat  = x_star.reshape(-1)
            dist_target  = float(np.linalg.norm(x_star_flat - x_tgt_flat))
            s_recon      = float(1.0 / (1.0 + dist_target))
            improvement  = s_recon - s_uninf

            d2             = np.sum((train_flat - x_star_flat) ** 2, axis=1)
            min_d_train    = float(np.sqrt(np.min(d2)))
            p_final        = _predict(model, tf.constant(x_star)).numpy()

            # verbose shows improvement on every restart
            _dbg(args.verbose,
                 f"label={used_label} r={r} s_recon={s_recon:.4f} "
                 f"s_uninf={s_uninf:.4f} improvement={improvement:+.4f} "
                 f"dist={dist_target:.4g}")

            if dist_target < best["dist_to_target"]:
                best.update({
                    "dist_to_target":    dist_target,
                    "score":             s_recon,
                    "improvement":       improvement,
                    "used_label":        int(used_label),
                    "restart":           int(r),
                    "x_star":            x_star,
                    "p_final":           p_final,
                    "min_dist_to_train": min_d_train,
                })

    _dbg(args.verbose,
         f"BEST s_recon={best['score']:.6f} s_uninf={s_uninf:.6f} "
         f"improvement={best['improvement']:+.6f} "
         f"label={best['used_label']} restart={best['restart']}")

    detail = {
        "target_split":       args.target_split,
        "target_idx":         int(args.target_idx),
        "y_target":           int(to_scalar_label(y_target)),
        "label_auto":         int(label_auto),
        "task_type":          "regression" if regression else "classification",
        "try_both_labels":    bool(args.try_both_labels and not regression),
        "dist_to_target":     float(best["dist_to_target"]),
        "min_l2_dist_to_train": float(best["min_dist_to_train"])
                              if best["min_dist_to_train"] is not None else None,
        "best_label":         best["used_label"],
        "best_restart":       best["restart"],
        "restarts":           int(args.restarts),
        "steps":              int(args.steps),
        "p_final":            np.asarray(best["p_final"]).reshape(-1).tolist()
                              if best["p_final"] is not None else None,
    }

    result = {
        "score":       float(best["score"]),
        "s_uninf":     s_uninf,
        "improvement": float(best["improvement"]),
        "detail":      json.dumps(detail, sort_keys=True),
        "attack":      "recon_targeted_harder",
        "seed":        int(args.seed),
    }

    if json_out:
        print(json.dumps(result), flush=True)
    else:
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
