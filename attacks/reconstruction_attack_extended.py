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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output_json_stdout", type=str, default="false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--restarts", type=int, default=10)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--prior_l2z", type=float, default=1e-2)
    p.add_argument("--prior_l1", type=float, default=1e-3)
    p.add_argument("--noise_scale", type=float, default=0.15)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--target_split", type=str, default="train", choices=["train", "test"])
    p.add_argument("--target_idx", type=int, default=0)
    p.add_argument("--target_label", type=str, default="auto", help="auto | integer label")
    p.add_argument("--try_both_labels", action="store_true")
    return p.parse_args()


def str2bool(x):
    return str(x).lower() in ("1", "true", "yes", "y", "t")


def _dbg(verbose: bool, msg: str) -> None:
    if verbose:
        print(f"[RECON_HARD][DBG] {msg}", file=sys.stderr, flush=True)


def to_scalar_label(y_val) -> int:
    y_val = np.asarray(y_val)
    if y_val.ndim > 0 and y_val.shape[-1] > 1:
        return int(np.argmax(y_val))
    return int(np.asarray(y_val).reshape(-1)[0])


def _predict_prob(model: tf.keras.Model, x: tf.Tensor) -> tf.Tensor:
    y = model(tf.expand_dims(x, axis=0), training=False)
    y = tf.convert_to_tensor(y)
    return tf.reshape(y, [-1])


def _loss_for_label(probs: tf.Tensor, target_label: int) -> tf.Tensor:
    eps = tf.constant(1e-7, dtype=probs.dtype)
    if tf.size(probs) == 1:
        p = tf.clip_by_value(probs[0], eps, 1.0 - eps)
        p_true = p if target_label == 1 else (1.0 - p)
        return -tf.math.log(p_true)
    else:
        p = tf.clip_by_value(probs[target_label], eps, 1.0)
        return -tf.math.log(p)


def _parse_target_label(spec: str, y_val) -> int:
    if spec.lower() == "auto":
        return to_scalar_label(y_val)
    return int(spec)


def get_dataset_bounds(x_train: np.ndarray):
    lower = x_train.min(axis=0).astype(np.float32)
    upper = x_train.max(axis=0).astype(np.float32)
    upper = np.maximum(upper, lower)
    return lower, upper


def main():
    args = parse_args()
    json_out = str2bool(args.output_json_stdout)

    model_path = os.path.abspath(args.model_path)
    _dbg(args.verbose, f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    _dbg(args.verbose, f"Loading dataset={args.dataset} partition_type=centralized")
    (x_train, y_train), (x_test, y_test), _ = load_dataset(args.dataset, partition_type="centralized")

    x_train = np.asarray(x_train, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    if args.target_split == "train":
        X_split, y_split = x_train, y_train
    else:
        X_split, y_split = x_test, y_test

    if args.target_idx < 0 or args.target_idx >= len(X_split):
        raise ValueError(f"target_idx out of range: {args.target_idx} (split size={len(X_split)})")

    x_target = X_split[args.target_idx].astype(np.float32)
    y_target = y_split[args.target_idx]
    label_auto = _parse_target_label(args.target_label, y_target)

    mean = x_train.mean(axis=0).astype(np.float32)
    std = x_train.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    lower_np, upper_np = get_dataset_bounds(x_train)
    lower = tf.constant(lower_np, dtype=tf.float32)
    upper = tf.constant(upper_np, dtype=tf.float32)

    _dbg(args.verbose, f"x_train={x_train.shape} x_test={x_test.shape}")
    _dbg(args.verbose, f"Target: split={args.target_split} idx={args.target_idx} y={to_scalar_label(y_target)} auto_label={label_auto}")
    _dbg(args.verbose, f"Bounds: lower_min={lower_np.min():.6g} lower_max={lower_np.max():.6g} upper_min={upper_np.min():.6g} upper_max={upper_np.max():.6g}")

    rng = np.random.default_rng(int(args.seed))

    if args.try_both_labels:
        probs0 = _predict_prob(model, tf.convert_to_tensor(mean)).numpy().reshape(-1)
        if probs0.size == 1:
            labels_to_try = [0, 1]
        else:
            labels_to_try = list(range(probs0.size))
    else:
        labels_to_try = [label_auto]

    best = {
        "dist_to_target": float("inf"),
        "score": 0.0,
        "used_label": None,
        "restart": None,
        "x_star": None,
        "p": None,
        "min_dist_to_train": None,
    }

    train_flat = x_train.reshape(len(x_train), -1)
    x_target_flat = x_target.reshape(-1)

    for used_label in labels_to_try:
        for r in range(int(args.restarts)):
            x0 = mean + rng.normal(0.0, float(args.noise_scale), size=mean.shape).astype(np.float32)
            x0 = np.clip(x0, lower_np, upper_np)

            x = tf.Variable(x0, dtype=tf.float32)
            opt = tf.keras.optimizers.Adam(learning_rate=args.lr)

            for _ in range(int(args.steps)):
                with tf.GradientTape() as tape:
                    probs = _predict_prob(model, x)
                    ce = _loss_for_label(probs, int(used_label))

                    z = (x - mean) / std
                    prior_l2z = args.prior_l2z * tf.reduce_mean(tf.square(z))
                    prior_l1 = args.prior_l1 * tf.reduce_mean(tf.abs(x - mean))
                    loss = ce + prior_l2z + prior_l1

                grads = tape.gradient(loss, [x])
                opt.apply_gradients(zip(grads, [x]))
                x.assign(tf.clip_by_value(x, lower, upper))

            x_star = x.numpy().astype(np.float32)
            x_star_flat = x_star.reshape(-1)

            dist_to_target = float(np.linalg.norm(x_star_flat - x_target_flat))
            score = float(1.0 / (1.0 + dist_to_target))

            d2 = np.sum((train_flat - x_star_flat) ** 2, axis=1)
            min_dist_to_train = float(np.sqrt(np.min(d2)))

            p_final = _predict_prob(model, tf.convert_to_tensor(x_star)).numpy()

            if args.verbose:
                _dbg(
                    args.verbose,
                    f"label={used_label} restart={r} dist_to_target={dist_to_target:.6g} "
                    f"score={score:.6g} (debug min_dist_to_train={min_dist_to_train:.6g}) p={p_final}"
                )

            if dist_to_target < best["dist_to_target"]:
                best.update(
                    {
                        "dist_to_target": dist_to_target,
                        "score": score,
                        "used_label": int(used_label),
                        "restart": int(r),
                        "x_star": x_star,
                        "p": p_final,
                        "min_dist_to_train": min_dist_to_train,
                    }
                )

    detail = {
        "target_split": args.target_split,
        "target_idx": int(args.target_idx),
        "y_target": int(to_scalar_label(y_target)),
        "label_auto": int(label_auto),
        "try_both_labels": bool(args.try_both_labels),
        "dist_to_target": float(best["dist_to_target"]),
        "min_l2_dist_to_train": float(best["min_dist_to_train"]) if best["min_dist_to_train"] is not None else None,
        "best_label": best["used_label"],
        "best_restart": best["restart"],
        "restarts": int(args.restarts),
        "steps": int(args.steps),
        "lr": float(args.lr),
        "prior_l2z": float(args.prior_l2z),
        "prior_l1": float(args.prior_l1),
        "noise_scale": float(args.noise_scale),
        "p_final": np.asarray(best["p"]).reshape(-1).tolist() if best["p"] is not None else None,
        "bounds_lower_min": float(lower_np.min()),
        "bounds_lower_max": float(lower_np.max()),
        "bounds_upper_min": float(upper_np.min()),
        "bounds_upper_max": float(upper_np.max()),
    }

    _dbg(
        args.verbose,
        f"BEST dist_to_target={best['dist_to_target']:.6g} score={best['score']:.6g} "
        f"label={best['used_label']} restart={best['restart']}"
    )

    result = {
        "score": float(best["score"]),
        "detail": json.dumps(detail, sort_keys=True),
        "attack": "recon_targeted_harder",
        "seed": int(args.seed),
    }

    if json_out:
        print(json.dumps(result), flush=True)
    else:
        print(result, flush=True)


if __name__ == "__main__":
    main()
