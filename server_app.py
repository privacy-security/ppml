import csv
import logging
import traceback
import flwr as fl
import math

from flwr.common import parameters_to_ndarrays
from federation.model import build_model
import numpy as np
import os

import sys
import traceback

from flwr.common import Parameters
from flwr.server.strategy import FedAvg

from typing import Optional
try:
    import dp_accounting
    DP_ACCOUNTING_AVAILABLE = True
except Exception:
    dp_accounting = None
    DP_ACCOUNTING_AVAILABLE = False

def str2bool(x: str) -> bool:
    return str(x).lower() in ("1", "true", "yes", "y", "t")

def get_total_client_count() -> int:
    return int(getattr(ExpConfig, "FL_TOTAL_CLIENTS", ExpConfig.FL_N_CLIENTS))

def compute_fl_user_level_epsilon(
    num_clients_total: int,
    num_clients_sampled: int,
    num_rounds: int,
    noise_multiplier: float,
    delta: float,
):
    """
    Approximate user-level epsilon for federated central DP.

    Assumptions:
    - one client = one protected unit
    - same Gaussian mechanism each round
    - Poisson subsampling approximation with q = sampled/total
    """
    if not DP_ACCOUNTING_AVAILABLE:
        logger.warning("dp-accounting not available; cannot compute FL epsilon.")
        return None

    if noise_multiplier <= 0:
        logger.warning("Noise multiplier must be > 0.")
        return None

    if num_clients_total <= 0 or num_clients_sampled <= 0 or num_rounds <= 0:
        logger.warning("Invalid accountant inputs.")
        return None

    q = min(1.0, float(num_clients_sampled) / float(num_clients_total))

    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)) + [128, 256, 512]

    try:
        accountant = dp_accounting.rdp.RdpAccountant(orders)

        event = dp_accounting.SelfComposedDpEvent(
            dp_accounting.PoissonSampledDpEvent(
                sampling_probability=q,
                event=dp_accounting.GaussianDpEvent(noise_multiplier),
            ),
            num_rounds,
        )

        accountant.compose(event)
        eps = accountant.get_epsilon(target_delta=delta)

        logger.info(
            f"[FL ACCOUNTANT] epsilon={eps:.6f}, delta={delta}, "
            f"q={q:.6f}, rounds={num_rounds}, noise={noise_multiplier}"
        )
        return float(eps)

    except Exception as e:
        logger.error(f"Failed to compute FL epsilon: {e}")
        logger.error(traceback.format_exc())
        return None


def excepthook(exc_type, exc_value, exc_tb):
    print("=== PYTHON EXCEPTION (GLOBAL) ===")
    traceback.print_exception(exc_type, exc_value, exc_tb)
    print("=== END OF EXCEPTION ===")

sys.excepthook = excepthook


from flwr.common import Context
from flwr.server import Driver, LegacyContext, ServerApp, ServerConfig
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow
from flwr.server.strategy import (
    FedAvg,
    DifferentialPrivacyServerSideFixedClipping,
    DifferentialPrivacyClientSideFixedClipping,
)

# Suppress TensorFlow spam BEFORE any tf import
import os

import tensorflow as tf



# ================================================================
# Logging Setup
# ================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("server_app")


logger.info("=" * 60)
logger.info("SERVER_APP MODULE LOADING")
logger.info("=" * 60)


# ================================================================
# Imports that may fail — handle gracefully
# ================================================================
try:
    import wandb
    logger.info("W&B imported")

    from config import get_config
    ExpConfig = get_config()
    logger.info("ExpConfig imported")

    from federation.server import get_evaluate_fn, get_evaluate_fn_v, weighted_average
    logger.info("Federation helpers imported")

except Exception as e:
    logger.error(f"Import failure: {e}")
    logger.error(traceback.format_exc())
    raise


# ================================================================
# GPU memory configuration
# ================================================================
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info("GPU memory growth enabled")

        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]
        )
    except RuntimeError as e:
        logger.warning(f"Cannot enable memory growth: {e}")
else:
    logger.info("No GPU detected – running on CPU")


# ================================================================
# Saving Strategy
# ================================================================   

class SavingFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latest_parameters: Parameters | None = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        # aggregated is either None or (parameters, metrics)
        if aggregated is not None:
            params, metrics = aggregated
            self.latest_parameters = params
        return aggregated

# ================================================================
# Flower ServerApp (new API)
# ================================================================
app = ServerApp()
logger.info("ServerApp instance created")


@app.main()
def main(driver: Driver, ctx: Context) -> None:
    """
    Main server function executed by Flower.
    """
    try:
        logger.info("=" * 60)
        logger.info("STARTING FEDERATED SERVER")
        logger.info("=" * 60)

        # --------------------------------------------------------
        # W&B INIT
        # --------------------------------------------------------
        wandb.login()
        wandb.init(project="privacy_preserving_federated_learning", config=ExpConfig.__dict__)
        wandb_config = wandb.config
        logger.info(f"W&B config initialized")

        # --------------------------------------------------------
        # Load dataset
        # --------------------------------------------------------
        logger.info("Loading dataset...")
        from data import load_dataset

        train_data, test_data, is_vertical = load_dataset(
            ExpConfig.DATASET,
            ExpConfig.PARTITION_TYPE
        )
        logger.info("Dataset loaded")

        # --------------------------------------------------------
        # Evaluation function selection
        # --------------------------------------------------------
        if ExpConfig.FL_N_CLIENTS <= 1:
            evaluate_fn = None
            logger.info("Single-client mode: evaluation disabled")
        else:
            evaluate_fn = (
                get_evaluate_fn_v(test_data)
                if is_vertical else
                get_evaluate_fn(test_data)
            )
            logger.info("Evaluation function prepared")

        # --------------------------------------------------------
        # Base FedAvg strategy
        # --------------------------------------------------------

        base_strategy = SavingFedAvg(
            evaluate_fn=evaluate_fn,
            fit_metrics_aggregation_fn=(weighted_average if ExpConfig.FL_N_CLIENTS > 1 else None),
            evaluate_metrics_aggregation_fn=(weighted_average if ExpConfig.FL_N_CLIENTS > 1 else None),
        )

        strategy = base_strategy
        # strategy = FedAvg(
        #     evaluate_fn=evaluate_fn,
        #     fit_metrics_aggregation_fn=(
        #         weighted_average if ExpConfig.FL_N_CLIENTS > 1 else None
        #     ),
        #     evaluate_metrics_aggregation_fn=(
        #         weighted_average if ExpConfig.FL_N_CLIENTS > 1 else None
        #     ),
        # )

        logger.info("FedAvg strategy created")

        # --------------------------------------------------------
        # Differential Privacy Wrapping
        # --------------------------------------------------------
        dp_enabled = wandb_config.get("dp", False)
        local_dp = wandb_config.get("local", False)

        if dp_enabled:
            noise = wandb_config.get("noise_multiplier", 1.0)
            clip = wandb_config.get("l2_norm_clip", 1.0)
            clipping_type = wandb_config.get("clipping", "server")

            if not local_dp:
                if clipping_type == "server":
                    logger.info("Applying Server-side (Server Clipping) Differential Privacy")
                    strategy = DifferentialPrivacyServerSideFixedClipping(
                        strategy,
                        noise,
                        clip,
                        ExpConfig.FL_N_CLIENTS,
                    )

                elif clipping_type == "client":
                    # if noise is None:
                    #     # Compute noise from sensitivity, epsilon, delta
                    #     eps = wandb_config["epsilon"]
                    #     delta = wandb_config["delta"]
                    #     clip = wandb_config["l2_norm_clip"]
                    #     noise = clip * math.sqrt(2 * math.log(1.25 / delta)) / eps
                    #     logger.info(f"[LOCAL DP] Computed noise multiplier = {noise}")
                    logger.info("Applying Server-side (Client Clipping) Differential Privacy")
                    strategy = DifferentialPrivacyClientSideFixedClipping(
                        strategy,
                        noise,
                        clip,
                        ExpConfig.FL_N_CLIENTS,
                    )

            logger.info(f"DP enabled: noise={noise}, clip={clip}, type={clipping_type}")

        # --------------------------------------------------------
        # Build LegacyContext (required for workflows)
        # --------------------------------------------------------
        server_config = ServerConfig(num_rounds=ExpConfig.FL_ROUNDS)

        legacy_context = LegacyContext(
            state=ctx.state,
            config=server_config,
            strategy=strategy,
        )
        logger.info("LegacyContext created")

        # --------------------------------------------------------
        # Select workflow: secure or regular
        # --------------------------------------------------------
        if ExpConfig.FL_AGGREGATION_TYPE == "secure":
            logger.info("Using SecAgg+ Workflow")

            base = SecAggPlusWorkflow(
                num_shares=ExpConfig.AGG_N_SHARES,
                reconstruction_threshold=ExpConfig.AGG_REC_SHARES,
                timeout=40,
            )

            workflow = DefaultWorkflow(base)
        else:
            logger.info("Using Regular DefaultWorkflow")
            workflow = DefaultWorkflow()

        # --------------------------------------------------------
        # Execute Workflow
        # --------------------------------------------------------
        logger.info("🚀 Starting FL training")
        workflow(driver, legacy_context, ExpConfig.MODEL_HISTORY_SAVE_PATH)

        # --------------------------------------------------------
        # Approximate FL DP accounting + W&B logging
        # --------------------------------------------------------
        try:
            if dp_enabled and not local_dp:
                noise = float(wandb_config.get("noise_multiplier", 1.0))
                num_rounds = int(ExpConfig.FL_ROUNDS)

                # If you do not have a separate total client count, this assumes full participation
                num_clients_sampled = int(ExpConfig.FL_N_CLIENTS)
                num_clients_total = int(getattr(ExpConfig, "FL_TOTAL_CLIENTS", ExpConfig.FL_N_CLIENTS))

                # For user-level FL accounting, delta should usually be tied to number of clients
                # delta = wandb_config.get("delta", None)
                # if delta is None:
                delta = 1.0 / (10*float(num_clients_total))
                delta = float(delta)

                fl_epsilon = compute_fl_user_level_epsilon(
                    num_clients_total=num_clients_total,
                    num_clients_sampled=num_clients_sampled,
                    num_rounds=num_rounds,
                    noise_multiplier=noise,
                    delta=delta,
                )

                if fl_epsilon is not None:
                    logger.info(
                        f"[FL ACCOUNTANT] Approximate central-FL DP epsilon={fl_epsilon:.6f}"
                    )
                    try:
                        wandb.log({
                            "fl_dp_epsilon_approx": fl_epsilon,
                            "fl_dp_delta_used": delta,
                            "fl_dp_noise_multiplier": noise,
                            "fl_dp_rounds": num_rounds,
                            "fl_dp_sampled_clients": num_clients_sampled,
                            "fl_dp_total_clients": num_clients_total,
                            "fl_dp_sampling_rate": num_clients_sampled / num_clients_total,
                        })
                    except Exception:
                        logger.warning("Failed to log FL accountant metrics to W&B.")

            elif dp_enabled and local_dp:
                # Local DP: do not run the central accountant, just log configured budget
                total_eps = wandb_config.get("epsilon", None)
                # For user-level FL accounting, delta should usually be tied to number of clients
                total_delta = 1.0 / (10*float(num_clients_total))
                
                if total_eps is not None and total_delta is not None:
                    total_eps = float(total_eps)
                    total_delta = float(total_delta)
                    num_rounds = int(ExpConfig.FL_ROUNDS)

                    eps_per_round = total_eps / num_rounds
                    delta_per_round = total_delta / num_rounds

                    logger.info(
                        f"[LDP CONFIG] total_eps={total_eps}, total_delta={total_delta}, "
                        f"eps_per_round={eps_per_round}, delta_per_round={delta_per_round}"
                    )
                    try:
                        wandb.log({
                            "ldp_epsilon_total": total_eps,
                            "ldp_delta_total": total_delta,
                            "ldp_epsilon_per_round": eps_per_round,
                            "ldp_delta_per_round": delta_per_round,
                            "ldp_rounds": num_rounds,
                        })
                    except Exception:
                        logger.warning("Failed to log LDP metrics to W&B.")

        except Exception as e:
            logger.error(f"DP accountant/logging block failed: {e}")
            logger.error(traceback.format_exc())


        # ==========================================================
        # SAVE FINAL GLOBAL MODEL (OPTIONAL, FOR MIA)
        # ==========================================================
        model_save = str2bool(os.getenv("MODEL_SAVE", "false"))
        export_path_env = os.getenv("EXPORT_PATH", "").strip()

        if model_save:
            try:
                logger.info("Saving final FL global model for MIA...")

                os.makedirs(ExpConfig.EXPORT_DIR, exist_ok=True)

                # 1) Extract final global parameters
                final_params = base_strategy.latest_parameters
                if final_params is None:
                    raise RuntimeError("No aggregated parameters captured (latest_parameters is None).")
                ndarrays = parameters_to_ndarrays(final_params)

                ndarrays = parameters_to_ndarrays(final_params)

                # 2) Rebuild model
                try:
                    x_test, y_test = test_data
                    multivariate = x_test.shape[-1]
                except Exception:
                    multivariate = None

                model = build_model(
                    dataset_name=ExpConfig.DATASET,
                    model_name="fl_global_model",
                    multivariate=multivariate,
                    sequence_len=getattr(ExpConfig, "SEQUENCE_LEN", None),
                    config=ExpConfig,
                )

                model.set_weights(ndarrays)

                # 3) Pick export path
                export_path = export_path_env
                if not export_path:
                    export_path = os.path.join(
                        ExpConfig.EXPORT_DIR,
                        f"fl_{ExpConfig.DATASET}_{ExpConfig.PARTITION_TYPE}.keras"
                    )

                model.save(export_path)

                logger.info(f"FL global model saved to {export_path}")
                try:
                    wandb.log({"export_path": export_path})
                except Exception:
                    pass

            except Exception as e:
                logger.error("FAILED to save FL global model")
                logger.error(str(e))
                logger.error(traceback.format_exc())
        else:
            logger.info("MODEL_SAVE=false, skipping global model export")

    except Exception as e:
        logger.error("FATAL SERVER ERROR")
        logger.error(str(e))
        logger.error(traceback.format_exc())
        raise

