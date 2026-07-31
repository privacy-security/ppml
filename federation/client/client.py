from flwr.client import NumPyClient
from config import get_config
from federation.model import build_model
import numpy as np
import os


def _global_l2_norm(arrays):
    """L2 norm of a list of arrays treated as one concatenated vector."""
    return float(np.sqrt(sum(float(np.sum(np.square(a))) for a in arrays)))


class FlwrClient(NumPyClient):
    def __init__(self, x_train, y_train, x_val, y_val):
        self.ExpConfig = get_config()

        # Client id (partition index) -- used to give each client an INDEPENDENT
        # DP noise stream. Falls back to 0 if CID is not set (single-client).
        try:
            self.client_id = int(os.getenv("CID", "0"))
        except (TypeError, ValueError):
            self.client_id = 0
        self._local_round = 0  # fallback round counter if server_round is absent

        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            # memory growth is set process-wide in client_app.py; do NOT pin a
            # hard per-process memory_limit here (it caused OOM at 4.5/32 GB).
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass

        # Build model dynamically. Pass the config so build_model can construct a
        # DPSGDModel when dp_level == "example"; the per-client seed_base makes the
        # example-level DP-SGD noise independent across clients and reproducible.
        dp_seed_base = int(self.ExpConfig.SEED) * 100003 + int(self.client_id)
        self.model = build_model(
            dataset_name=self.ExpConfig.DATASET,
            model_name=self.ExpConfig.MODEL_NAME,
            multivariate=self.ExpConfig.MULTIVARIATE,
            sequence_len=self.ExpConfig.SEQUENCE_LEN,
            config=self.ExpConfig,
            dp_seed_base=dp_seed_base,
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

    # ------------------------------------------------------------
    # Local Differential Privacy: clip the update to C and add Gaussian
    # noise with the calibrated multiplier z, ON THE CLIENT, so the
    # released update is (eps, delta)-DP w.r.t. this client's data
    # (untrusted-server model). Noise is drawn from a per-(SEED, client,
    # round) stream: INDEPENDENT across clients (so the aggregate carries
    # sqrt(M) noise, as the theory requires) and reproducible given SEED.
    # ------------------------------------------------------------
    def _apply_local_dp(self, global_weights, new_weights, server_round):
        C = float(self.ExpConfig.l2_norm_clip)
        z = float(self.ExpConfig.noise_multiplier)

        # update = locally-trained weights - incoming global weights (per layer)
        update = [np.asarray(nw) - np.asarray(gw)
                  for nw, gw in zip(new_weights, global_weights)]

        # clip the GLOBAL L2 norm of the whole update to C (sensitivity = C)
        total_norm = _global_l2_norm(update)
        scale = min(1.0, C / (total_norm + 1e-12))
        clipped = [u * scale for u in update]

        # independent + reproducible Gaussian noise; std = z*C per coordinate
        seq = np.random.SeedSequence([int(self.ExpConfig.SEED),
                                      int(self.client_id),
                                      int(server_round)])
        rng = np.random.default_rng(seq)
        noised_update = [
            c + rng.normal(loc=0.0, scale=z * C, size=c.shape).astype(np.asarray(c).dtype)
            for c in clipped
        ]

        # weights the server will aggregate = global + noised, clipped update
        noised_weights = [np.asarray(gw) + n
                          for gw, n in zip(global_weights, noised_update)]
        return noised_weights, total_norm, scale

    def fit(self, parameters, config):
        # Keep a copy of the incoming GLOBAL weights (needed for the DP update)
        global_weights = [np.asarray(p).copy() for p in parameters]

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

        # ------------------------------------------------------------
        # LOCAL DIFFERENTIAL PRIVACY  (client-side clip + Gaussian noise)
        # ------------------------------------------------------------
        # CLIENT-LEVEL LDP only. For example-level DP the privacy is already in
        # DPSGDModel.train_step (per-example clip + noise), so do NOT also add
        # update noise here — that would double-apply DP.
        if (getattr(self.ExpConfig, "dp", False)
                and getattr(self.ExpConfig, "local_dp", False)
                and getattr(self.ExpConfig, "dp_level", "client") == "client"):
            self._local_round += 1
            server_round = int(config.get("server_round", self._local_round))
            weights, preclip_norm, scale = self._apply_local_dp(
                global_weights, weights, server_round
            )
            print(f"[DP CLIENT {self.client_id}] LDP @round {server_round}: "
                  f"z={float(self.ExpConfig.noise_multiplier):.4f}, "
                  f"C={float(self.ExpConfig.l2_norm_clip)}, "
                  f"preclip_update_norm={preclip_norm:.3f}, clip_scale={scale:.3f}")

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
        
