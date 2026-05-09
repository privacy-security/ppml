import argparse
import wandb
import numpy as np
import logging
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


import tensorflow as tf
from data.dataset_loader import load_dataset
from config import get_config
from federation.model import build_model

def str2bool(x: str) -> bool:
    return str(x).lower() in ("1", "true", "yes", "y", "t")

parser = argparse.ArgumentParser()
parser.add_argument("--model_save", type=str2bool, default=str2bool(os.getenv("MODEL_SAVE", "false")))
parser.add_argument("--export_path", type=str, default=os.getenv("EXPORT_PATH", ""))
args, _ = parser.parse_known_args()

# ------------------------------------------------------------------
# LOGGING SETUP
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logging.info("=== Starting centralized training script ===")

ExpConfig = get_config()
batch_size = ExpConfig.BATCH_SIZE
epochs = ExpConfig.EPOCHS
logging.info(f"Using epochs={epochs}, batch_size={batch_size}")

# ------------------------------------------------------------------
# WANDB INIT
# ------------------------------------------------------------------
logging.info("Initializing Weights & Biases")

wandb_run = wandb.init(
    project="privacy_preserving_federated_learning",
    name=f"centralized_dp_{ExpConfig.DATASET}",
)

logging.info(f"WANDB run initialized: {wandb_run.name}")

# ------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------
logging.info(f"Loading dataset: {ExpConfig.DATASET} (centralized)")

(train, test, _) = load_dataset(
    ExpConfig.DATASET,
    partition_type="centralized"
)

x_train, y_train = train
x_test, y_test = test

logging.debug(f"x_train shape: {x_train.shape}")
logging.debug(f"y_train shape: {y_train.shape}")
logging.debug(f"x_test shape:  {x_test.shape}")
logging.debug(f"y_test shape:  {y_test.shape}")

# ------------------------------------------------------------------
# MULTIVARIATE DETECTION
# ------------------------------------------------------------------
try:
    multivariate = x_train.shape[-1]
    logging.info(f"Multivariate input detected: {multivariate}")
except:
    multivariate = None
    logging.warning("Could not infer multivariate dimensions")

# ------------------------------------------------------------------
# DP SAFETY CHECKS
# ------------------------------------------------------------------
dp = getattr(ExpConfig, "dp", False)
num_microbatches = getattr(ExpConfig, "num_microbatches", 1)

if dp:
    if batch_size % num_microbatches != 0:
        logging.error(
            "DP is enabled but batch_size % num_microbatches != 0. "
            "num_microbatches must divide batch_size for per-microbatch gradient computation."
        )
        raise ValueError("batch_size must be divisible by num_microbatches when dp=True")

# ------------------------------------------------------------------
# MODEL BUILDING
# ------------------------------------------------------------------
logging.info("Building model...")

model = build_model(
    dataset_name=ExpConfig.DATASET,
    model_name="centralized_model",
    multivariate=multivariate,
    sequence_len=getattr(ExpConfig, "SEQUENCE_LEN", None),
    config=ExpConfig,   # needed for DP optimizer and other settings
)

logging.info("Model successfully built:")

# ------------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------------
logging.info("Starting model.fit()")

try:
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        verbose=2,
        callbacks=[
            wandb.keras.WandbMetricsLogger(),
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: logging.debug(
                    f"Epoch {epoch} logs: {logs}"
                )
            ),
        ],
    )
    logging.info("Training completed successfully")
except Exception as e:
    logging.error("Training failed!", exc_info=e)
    raise e

# ------------------------------------------------------------------
# EPSILON ACCOUNTANT (DP only)
# ------------------------------------------------------------------
try:
    if getattr(ExpConfig, "dp", False):
        logging.info("Computing differential privacy epsilon...")

        from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy import (
            compute_dp_sgd_privacy,
        )

        dataset_size = len(x_train)
        noise = ExpConfig.noise_multiplier
        l2_clip = ExpConfig.l2_norm_clip
        microbatches = ExpConfig.num_microbatches
        delta = 1 / dataset_size

        eps, _, _ = compute_dp_sgd_privacy(
            n=dataset_size,
            batch_size=batch_size,
            noise_multiplier=noise,
            epochs=epochs,
            delta=delta,
        )

        logging.info(f"Final epsilon computed: epsilon = {eps}")

        wandb.log({"dp_epsilon": eps})

except Exception as e:
    logging.error("Failed to compute epsilon!", exc_info=e)

# ------------------------------------------------------------------
# EVALUATION + WANDB LOGGING
# ------------------------------------------------------------------
logging.info("Evaluating model...")

try:
    metrics = model.evaluate(x_test, y_test, verbose=0, return_dict=True)
    final_metrics = {f"final_{k}": float(v) for k, v in metrics.items()}
    # final_metrics["epochs"] = int(epochs)
    # final_metrics["batch_size"] = int(batch_size)

    wandb.log(final_metrics)

    # ------------------------------------------------------------------
    # SAVE FINAL MODEL (OPTIONAL)
    # ------------------------------------------------------------------
    if args.model_save:
        os.makedirs(ExpConfig.EXPORT_DIR, exist_ok=True)

        export_path = args.export_path.strip()
        if not export_path:
            # fallback default, but prefer explicit EXPORT_PATH from pipeline
            export_path = os.path.join(
                ExpConfig.EXPORT_DIR,
                f"centralized_{ExpConfig.DATASET}.keras"
            )

        model.save(export_path)
        logging.info(f"Final centralized model saved to {export_path}")
        wandb.log({"export_path": export_path})
    else:
        logging.info("MODEL_SAVE=false, skipping model export")

except Exception as e:
    logging.error("Evaluation/logging failed!", exc_info=e)
    raise e



wandb.finish()
logging.info("=== Finished centralized training script ===")
