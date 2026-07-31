import flwr as fl
import wandb
import numpy as np

from config import get_config
from federation import build_model, build_gru_model

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

ExpConfig = get_config()


# ================================================================
# Safe metric extraction from Keras model.evaluate
# ================================================================
def evaluate_with_metrics(model, x, y):
    """
    Runs model.evaluate safely and returns a clean dict of metrics.
    Uses model.metrics_names, which is more stable than iterating model.metrics.
    """
    raw = model.evaluate(x, y, verbose=0, return_dict=True)

    # Keras may return dict directly
    if isinstance(raw, dict):
        return {k: float(v) for k, v in raw.items()}

    # Keras may return scalar
    if not isinstance(raw, (list, tuple)):
        return {"loss": float(raw)}

    raw = list(raw)
    names = list(model.metrics_names)

    # Fallback if names are missing/misaligned
    if len(names) == len(raw):
        return {name: float(v) for name, v in zip(names, raw)}
    out = {"loss": float(raw[0])}
    for i, v in enumerate(raw[1:], start=1):
        out[f"metric_{i}"] = float(v)
    return out

# ================================================================
# Manual binary metrics (Smoking dataset)
# ================================================================
def compute_binary_metrics(y_true, y_pred_prob):
    y_true = y_true.astype(int).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)

    return accuracy, precision, recall, f1


def log_server_eval_to_wandb(server_round, metrics: dict, prefix="server/global_eval"):
    if wandb.run is None:
        return

    payload = {"round": int(server_round)}
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            payload[f"{prefix}/{k}"] = float(v)

    wandb.log(payload)


# ================================================================
# Unified horizontal evaluation function
# ================================================================
def get_evaluate_fn(testset, model_name=ExpConfig.MODEL_NAME):
    x_test, y_test = testset
    dataset = ExpConfig.DATASET

    def evaluate(server_round, parameters, config):
        # For tabular datasets, multivariate should be feature count
        multivariate = None
        if hasattr(x_test, "shape") and len(x_test.shape) >= 2:
            multivariate = x_test.shape[-1]

        model = build_model(
            dataset_name=dataset,
            model_name=model_name,
            multivariate=multivariate,
            sequence_len=ExpConfig.SEQUENCE_LEN,
            learning_rate=ExpConfig.learning_rate,
            dp=ExpConfig.dp,
            l2_norm_clip=ExpConfig.l2_norm_clip,
            noise_multiplier=ExpConfig.noise_multiplier,
            num_microbatches=ExpConfig.num_microbatches,
            config=ExpConfig,
        )

        model.set_weights(parameters)

        if server_round == ExpConfig.FL_ROUNDS:
            os.makedirs(ExpConfig.EXPORT_DIR, exist_ok=True)
            export_path = os.path.join(
                ExpConfig.EXPORT_DIR,
                f"fl_{ExpConfig.DATASET}_{ExpConfig.PARTITION_TYPE}.keras"
            )
            model.save(export_path)
            print(f"[SERVER] Final global model saved to {export_path}")

        metrics = evaluate_with_metrics(model, x_test, y_test)
        server_metrics = dict(metrics)

        # ---- Network Monitoring ----
        if dataset in ["network_monitoring", "household_power"]:
            log_server_eval_to_wandb(server_round, server_metrics)

            print("\n[DEBUG SERVER-EVAL] evaluate() returning:")
            print("  metrics_names =", model.metrics_names)
            print("  loss =", server_metrics.get("loss"))
            print("  metrics =", server_metrics, "\n")

            return float(server_metrics["loss"]), server_metrics

        # ---- CIFAR10 ----
        if dataset == "cifar10":
            log_server_eval_to_wandb(server_round, server_metrics)

            print("\n[DEBUG SERVER-EVAL] evaluate() returning:")
            print("  metrics_names =", model.metrics_names)
            print("  loss =", server_metrics.get("loss"))
            print("  metrics =", server_metrics, "\n")

            return float(server_metrics["loss"]), server_metrics

        # ---- Body signals of smoking ----
        if dataset == "body_signal_of_smoking":
            y_pred_prob = model.predict(x_test, verbose=0).flatten()
            acc_b, prec_b, rec_b, f1_b = compute_binary_metrics(
                y_test.flatten(), y_pred_prob
            )

            server_metrics.update({
                "accuracy_binary": float(acc_b),
                "precision_binary": float(prec_b),
                "recall_binary": float(rec_b),
                "f1_binary": float(f1_b),
            })

            log_server_eval_to_wandb(server_round, server_metrics)

            print("\n[DEBUG SERVER-EVAL] evaluate() returning:")
            print("  metrics_names =", model.metrics_names)
            print("  loss =", server_metrics.get("loss"))
            print("  metrics =", server_metrics, "\n")

            return float(server_metrics["loss"]), server_metrics

        return float(server_metrics["loss"]), server_metrics

    return evaluate


# ================================================================
# Vertical FL evaluation
# ================================================================
def get_evaluate_fn_v(testsets, model_name=ExpConfig.MODEL_NAME):
    if ExpConfig.DATASET != "network_monitoring":
        raise ValueError("Vertical evaluation is only supported for network_monitoring.")

    (http_x_test, http_y_test), (ssl_x_test, ssl_y_test) = testsets

    def evaluate(server_round, parameters, config):
        model = build_gru_model(
            model_name=model_name,
            multivariate=ExpConfig.MULTIVARIATE,
            sequence_len=ExpConfig.SEQUENCE_LEN,
            learning_rate=ExpConfig.learning_rate,
            dp=ExpConfig.dp,
            no_fl=ExpConfig.no_fl,
            l2_norm_clip=ExpConfig.l2_norm_clip,
            noise_multiplier=ExpConfig.noise_multiplier,
            num_microbatches=ExpConfig.num_microbatches,
        )
        model.set_weights(parameters)

        http_metrics = evaluate_with_metrics(model, http_x_test, http_y_test)
        ssl_metrics = evaluate_with_metrics(model, ssl_x_test, ssl_y_test)

        def avg(metric):
            return (
                http_metrics.get(metric, 0.0) +
                ssl_metrics.get(metric, 0.0)
            ) / 2.0

        merged = {
            "loss": avg("loss"),
            "mse": avg("mse"),
            "mae": avg("mae"),
        }

        log_server_eval_to_wandb(server_round, {
            "http_mse": http_metrics.get("mse", 0.0),
            "ssl_mse": ssl_metrics.get("mse", 0.0),
            **merged,
        })

        return float(merged["loss"]), {
            "mse": float(merged["mse"]),
            "mae": float(merged["mae"]),
        }

    return evaluate
