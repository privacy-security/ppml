from flwr.client import ClientApp
from flwr.client.mod import secaggplus_mod, fixedclipping_mod, LocalDpMod

from config import get_config
from federation.client import client_fn

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
import math
import wandb
import os

import tensorflow as tf

tf.get_logger().setLevel("ERROR")


def compute_local_dp_noise(
    epsilon_total: float,
    delta_total: float,
    clipping_norm: float,
    num_releases: int,
):
    if num_releases <= 0:
        raise ValueError("num_releases must be positive")
    if clipping_norm <= 0:
        raise ValueError("clipping_norm must be positive")

    epsilon_round = epsilon_total / num_releases
    delta_round = delta_total / num_releases
    sensitivity = clipping_norm

    std_dev = (
        sensitivity * math.sqrt(2 * math.log(1.25 / delta_round)) / epsilon_round
    )
    noise_multiplier = std_dev / clipping_norm

    return {
        "epsilon_round": epsilon_round,
        "delta_round": delta_round,
        "std_dev": std_dev,
        "noise_multiplier": noise_multiplier,
    }

def log_ldp_info_to_wandb(ldp_info: dict, exp_config, R: int) -> None:
    if wandb.run is None:
        return

    wandb.log({
        "ldp/epsilon_total": float(exp_config.epsilon),
        "ldp/delta_total": float(exp_config.delta),
        "ldp/R": int(R),
        "ldp/clip_norm": float(exp_config.l2_norm_clip),
        "ldp/epsilon_round": float(ldp_info["epsilon_round"]),
        "ldp/delta_round": float(ldp_info["delta_round"]),
        "ldp/std_dev": float(ldp_info["std_dev"]),
        "ldp/noise_multiplier": float(ldp_info["noise_multiplier"]),
    })

# ==========================================================
# GPU / CPU AUTO-SELECTION LOGIC
# ==========================================================
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # print("[GPU] CUDA-compatible GPU found. Enabling memory growth...")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # optional: limit memory if needed
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]
        )

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
if ExpConfig.dp and ExpConfig.local_dp:
    print("[CLIENT_APP] Using Local Differential Privacy (client-side).")

    R = ExpConfig.FL_ROUNDS
    M = ExpConfig.FL_N_CLIENTS
    delta_total = 1.0 / (10*float(M))

    ldp_info = compute_local_dp_noise(
        epsilon_total=ExpConfig.epsilon,
        delta_total=delta_total,
        clipping_norm=ExpConfig.l2_norm_clip,
        num_releases=R,
    )

    log_ldp_info_to_wandb(ldp_info, ExpConfig, R)

    print(
        "[CLIENT_APP] LDP accounting: "
        f"epsilon_total={ExpConfig.epsilon}, "
        f"delta_total={delta_total}, "
        f"R={R}, "
        f"epsilon_round={ldp_info['epsilon_round']:.8f}, "
        f"delta_round={ldp_info['delta_round']:.8e}, "
        f"clip_norm={ExpConfig.l2_norm_clip}, "
        f"std_dev={ldp_info['std_dev']:.6f}, "
        f"noise_multiplier={ldp_info['noise_multiplier']:.6f}"
    )

    mods.append(LocalDpMod(
        clipping_norm=ExpConfig.l2_norm_clip,
        sensitivity=ExpConfig.l2_norm_clip,
        epsilon=ldp_info['epsilon_round'],
        delta=ldp_info['delta_round'],
    ))

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
