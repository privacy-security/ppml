from flwr.client import NumPyClient
from config import get_config
from federation.model import build_model
import numpy as np


class FlwrClient(NumPyClient):
    def __init__(self, x_train, y_train, x_val, y_val):
        self.ExpConfig = get_config()

        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [
                        tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=1024
                        )
                    ],
                )
            except RuntimeError:
                pass

        # Build model dynamically
        self.model = build_model(
            dataset_name=self.ExpConfig.DATASET,
            model_name=self.ExpConfig.MODEL_NAME,
            multivariate=self.ExpConfig.MULTIVARIATE,
            sequence_len=self.ExpConfig.SEQUENCE_LEN,
        )

        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val

    # ------------------------------------------------------------
    # Helper: extract compile-time metric names (automatically)
    # ------------------------------------------------------------
    def _metric_names(self):
        """
        Returns list of metric names defined in:
        model.compile(..., metrics=[...])
        """
        names = []
        for m in self.model.metrics:
            if hasattr(m, "name"):
                names.append(m.name)
            else:
                # Fallback for simple metrics like "accuracy"
                names.append(str(m))
        return names

    # ------------------------------------------------------------
    # Helper: history → metrics dict
    # ------------------------------------------------------------
    def _history_to_metrics(self, history):
        """Extract ALL metrics from Keras history object."""
        metrics = {}
        metric_keys = history.history.keys()

        for key in metric_keys:
            val = history.history[key][-1]
            try:
                metrics[key] = float(np.asarray(val))
            except:
                pass  # skip non-numeric

        return metrics

    # ------------------------------------------------------------
    # Helper: evaluate output → metrics dict
    # ------------------------------------------------------------
    def _evaluate_to_metrics(self, raw):
        """
        Standardizes evaluate() output to a metric dict.
        Handles both dict and list formats.
        """

        # Case 1: Keras returns dict
        if isinstance(raw, dict):
            metrics = {k: float(v) for k, v in raw.items()}
            loss = float(metrics.get("loss", 0.0))
            return loss, metrics

        # Case 2: Keras returns list → [loss, m1, m2, ...]
        raw = list(raw)
        loss = float(raw[0])

        metrics = {"loss": loss}
        metric_names = self._metric_names()

        # Ensure matching lengths
        for name, val in zip(metric_names, raw[1:]):
            try:
                metrics[name] = float(np.asarray(val))
            except:
                continue

        return loss, metrics

    # ------------------------------------------------------------
    # Flower API
    # ------------------------------------------------------------
    def get_parameters(self, config):
        return self.model.get_weights()
    def fit(self, parameters, config):
        # Set incoming weights
        self.model.set_weights(parameters)

        # Perform local training
        history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=1,
            batch_size=self.ExpConfig.BATCH_SIZE,
            verbose=0,
        )

        # Prepare proper return values
        weights = self.model.get_weights()
        num_examples = len(self.x_train)

        # Extract ALL Keras metrics
        metrics = {}

        for key, values in history.history.items():
            if len(values) > 0:
                try:
                    metrics[key] = float(values[-1])
                except:
                    pass

        # Ensure loss exists
        if "loss" not in metrics:
            metrics["loss"] = float(history.history.get("loss", [0])[-1])

        # Cleanup noise keys
        metrics.pop("val_loss", None)
        metrics.pop("lr", None)

        # DEBUG LOG
        print(f"[DEBUG CLIENT] Fit returning: num_examples={num_examples}")
        print("[DEBUG CLIENT] Fit metrics:", metrics)

        return weights, num_examples, metrics


    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)

        raw = self.model.evaluate(
            self.x_val,
            self.y_val,
            verbose=0,
            return_dict=True  # works on TF 2.9+; fallback handled below
        )

        # Standardize to metric dict
        loss, metrics = self._evaluate_to_metrics(raw)
        
        print("[DEBUG CLIENT] Evaluate returning:", {
            "loss": loss,
            "metrics": metrics
        })
        return loss, len(self.x_val), metrics
        