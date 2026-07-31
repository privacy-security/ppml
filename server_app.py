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
    except RuntimeError as e:
        pass
        #logger.warning(f"Cannot enable memory growth: {e}")
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

        # Build the initial global model so BOTH FedAvg and the DP wrapper start
        # from the SAME concrete reference. Without this, Flower 1.9.0's
        # DifferentialPrivacyServerSideFixedClipping has no reference model, clips
        # each client update against a missing baseline, and training never learns
        # (accuracy stuck at random) -- while plain FL is unaffected.
        from flwr.common import ndarrays_to_parameters

        init_model = build_model(
            dataset_name=ExpConfig.DATASET,
            model_name=ExpConfig.MODEL_NAME,
            multivariate=ExpConfig.MULTIVARIATE,
            sequence_len=ExpConfig.SEQUENCE_LEN,
            config=ExpConfig,
        )
        initial_parameters = ndarrays_to_parameters(init_model.get_weights())
        logger.info("Initial global parameters built and passed to strategy.")

        base_strategy = SavingFedAvg(
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=lambda rnd: {"server_round": int(rnd)},
            fit_metrics_aggregation_fn=(weighted_average if ExpConfig.FL_N_CLIENTS > 1 else None),
            evaluate_metrics_aggregation_fn=(weighted_average if ExpConfig.FL_N_CLIENTS > 1 else None),
            initial_parameters=initial_parameters,
        )

        strategy = base_strategy
        logger.info("FedAvg strategy created")

        # --------------------------------------------------------
        # Differential Privacy Wrapping
        # --------------------------------------------------------
        dp_enabled = wandb_config.get("dp", False)
        local_dp = wandb_config.get("local", False)
        dp_level = wandb_config.get("dp_level", "client")

        # Server-side DP wrapping (CDP) applies ONLY to client-level central DP.
        # Example-level DP is enforced inside each client's DP-SGD train_step, and
        # client-level LDP adds noise in the client update -- neither needs the
        # server to wrap the strategy.
        if dp_enabled and dp_level == "client":
            noise = float(ExpConfig.noise_multiplier)   # calibrated from target epsilon
            clip = float(ExpConfig.l2_norm_clip)
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
                    init_p = strategy.initialize_parameters(client_manager=None) if hasattr(strategy, "initialize_parameters") else "N/A"
                    logger.info(f"[DIAG] DP wrapper initial_parameters is None? "
                                f"{init_p is None}  (base has initial_parameters? "
                                f"{getattr(base_strategy, 'initial_parameters', None) is not None})")

                elif clipping_type == "client":
                    logger.info("Applying Server-side (Client Clipping) Differential Privacy")
                    strategy = DifferentialPrivacyClientSideFixedClipping(
                        strategy,
                        noise,
                        clip,
                        ExpConfig.FL_N_CLIENTS,
                    )

            logger.info(f"DP enabled: noise={noise}, clip={clip}, type={clipping_type}")
        elif dp_enabled and dp_level == "example":
            logger.info("Example-level DP: privacy enforced in client DP-SGD; "
                        "no server-side DP wrapping.")

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
        # DP accounting + W&B logging (one clean block for CDP and LDP)
        # --------------------------------------------------------
        try:
            if dp_enabled:
                if dp_level == "example":
                    mech = "example"
                else:
                    mech = "ldp" if local_dp else "cdp"
                z = float(ExpConfig.noise_multiplier)
                C = float(ExpConfig.l2_norm_clip)
                R = int(ExpConfig.FL_ROUNDS)
                M = int(ExpConfig.FL_N_CLIENTS)
                delta = float(ExpConfig.delta)

                # client-level: eps from the client accountant (already on ExpConfig).
                # example-level: eps is the record-level accountant value, also on
                # ExpConfig (dp_epsilon_achieved), computed with q=batch/n_client.
                eps = getattr(ExpConfig, "dp_epsilon_achieved", None)
                if eps is None and dp_level == "client":
                    eps = compute_fl_user_level_epsilon(M, M, R, z, delta)

                # sampling rate: 1.0 at client level (full participation);
                # q=batch/n_client at example level.
                q = float(getattr(ExpConfig, "example_q", 1.0)) if dp_level == "example" else 1.0

                logger.info(
                    f"[DP ACCOUNTANT] level={dp_level}, mechanism={mech}, "
                    f"epsilon={eps:.6f}, delta={delta}, z={z:.6f}, C={C}, R={R}, M={M}, q={q:.6f}"
                )
                try:
                    # run-level constants -> summary (avoids empty per-step columns)
                    wandb.run.summary.update({
                        "dp/level": dp_level,
                        "dp/mechanism": mech,
                        "dp/epsilon": float(eps),
                        "dp/target_epsilon": float(ExpConfig.epsilon),
                        "dp/delta": delta,
                        "dp/noise_multiplier": z,
                        "dp/clip_norm": C,
                        "dp/rounds": R,
                        "dp/clients": M,
                        "dp/sampling_rate": q,
                        "dp/seed": int(ExpConfig.SEED),
                    })
                except Exception:
                    logger.warning("Failed to log DP metrics to W&B.")

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

