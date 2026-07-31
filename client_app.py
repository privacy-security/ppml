from flwr.client import ClientApp
from flwr.client.mod import secaggplus_mod, fixedclipping_mod

from config import get_config
from federation.client import client_fn

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
import wandb
import os

import tensorflow as tf

tf.get_logger().setLevel("ERROR")


# ==========================================================
# GPU / CPU AUTO-SELECTION LOGIC
# ==========================================================
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # print("[GPU] CUDA-compatible GPU found. Enabling memory growth...")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except (RuntimeError, ValueError):
                pass

        # optional: limit memory if needed
        #tf.config.set_logical_device_configuration(
        #    gpus[0],
        #    [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]
        #)

        print("[GPU] GPU activated for TensorFlow")
    except RuntimeError as e:
        print("[GPU-ERROR] Could not enable GPU memory growth:", e)
else:
    print("[CPU] No GPU detected — using CPU")


# Start with an empty mod list
mods = []

ExpConfig = get_config()

# ==========================================================
# 🔹 SECURE AGGREGATION
# ==========================================================

if ExpConfig.FL_AGGREGATION_TYPE == "secure":
    print("[CLIENT_APP] Using Built-in Secure Aggregation (SecAggPlus).")
    mods.append(secaggplus_mod)

# ==========================================================
# 🔹 LOCAL DIFFERENTIAL PRIVACY
# ==========================================================
# Local DP is applied INSIDE the client's fit() (transparent clip-to-C +
# Gaussian noise with the calibrated multiplier z), NOT via LocalDpMod.
# So there is no mod to append here; we just report the configuration.
if ExpConfig.dp and ExpConfig.local_dp:
    print(
        "[CLIENT_APP] Local DP active (noise applied in client.fit()): "
        f"z={ExpConfig.noise_multiplier:.4f}, C={ExpConfig.l2_norm_clip}, "
        f"target_eps={ExpConfig.epsilon}, "
        f"accountant_eps={ExpConfig.dp_epsilon_achieved:.4f}"
    )

# ==========================================================
# 🔹 CLIENT-SIDE CLIPPING (for central DP client variant)
# ==========================================================
if ExpConfig.dp and not ExpConfig.local_dp and ExpConfig.clipping == "client":
    print("[CLIENT_APP] Using Client-Side Fixed Differential Privacy (clipping).")
    mods.append(fixedclipping_mod)

# ==========================================================
# 🔹 CLIENT APP INITIALIZATION
# ==========================================================
app = ClientApp(client_fn=client_fn, mods=mods)
