import os
import sys
from datetime import datetime
from pathlib import Path

from data.dataset_loader import load_dataset
from seeding import set_global_seed
import dp_utils

DATASETS = ["network_monitoring", "cifar10", "body_signal_of_smoking", "household_power"]
PARTITION_TYPES = ["noniid", "iid", "vertical", "centralized"]
AGGREGATION_TYPES = ["regular", "secure"]

DEFAULT_BATCH_SIZES = {
    "network_monitoring": 256,
    "cifar10": 16,
    "body_signal_of_smoking": 128,
    "household_power": 256,
}

EXPORT_DIR = "artifacts"
FINAL_MODEL_NAME = "final_model.keras"


def _log(msg):
    print(f"[CONFIG] {msg}")


class DefaultConfig:
    PARTITION_TYPE = "iid"
    AGGREGATION_TYPE = "regular"
    N_CLIENTS = 3
    BATCH_SIZE = 32
    EPOCHS = 10


class Config:
    SAVE_DIR = Path(__file__).parent.parent.joinpath("saves")
    SEQUENCE_LEN = 24
    FL_ROUNDS = None
    FL_SAVE_ON_ROUND = None

    def __init__(self):
        _log("Initializing Config...")

        self.__set_main_settings()
        self.__set_model_settings()
        self._set_secure_agg_settings()

        # Sync FL save round once FL_ROUNDS is known
        self.FL_SAVE_ON_ROUND = self.FL_ROUNDS

        self.num_microbatches = 1
        self.experiment_name = None
        self._partitions_prepared = False

        _log("Initialization completed")

    def prepare_partitions(self):
        self._prepare_partitions_impl()
        self._maybe_calibrate_example_dp()

    def _maybe_calibrate_example_dp(self):
        """Example-level (record) DP-SGD calibration: needs per-client sizes."""
        if not (self.dp and self.dp_level == "example"):
            return
        if getattr(self, "PARTITIONS", None) is None:
            return
        if getattr(self, "_example_dp_calibrated", False):
            return
        # conservative: smallest client (largest q => highest privacy cost)
        n_examples_per_client = min(len(p[0]) for p in self.PARTITIONS)
        q = float(self.BATCH_SIZE) / float(n_examples_per_client)
        T = dp_utils.example_level_steps(n_examples_per_client, self.BATCH_SIZE, self.FL_ROUNDS, local_epochs=1)
        if self.delta is None:
            self.delta = dp_utils.example_delta(n_examples_per_client)
        self.noise_multiplier = dp_utils.example_level_noise_for_epsilon(
            target_epsilon=self.epsilon, steps=T, sample_rate=q, delta=self.delta)
        self.dp_epsilon_achieved = dp_utils.example_level_epsilon(
            self.noise_multiplier, T, q, self.delta)
        self.example_q = q
        self.example_steps = T
        self.example_n_examples_per_client = n_examples_per_client
        self._example_dp_calibrated = True
        _log(f"DP calibration [example]: target_eps={self.epsilon}, "
             f"n_examples_per_client(min)={n_examples_per_client}, batch={self.BATCH_SIZE}, q={q:.5f}, T={T}, "
             f"delta={self.delta:.3g} -> z={self.noise_multiplier:.4f} "
             f"(accountant eps={self.dp_epsilon_achieved:.4f})")

    # -----------------------------------------------------------
    # PARTITION PREPARATION
    # -----------------------------------------------------------
    def _prepare_partitions_impl(self):
        if self._partitions_prepared:
            _log("Partitions already prepared, skipping.")
            return

        _log(f"Loading dataset: {self.DATASET}, partition_type={self.PARTITION_TYPE}")
        train_data, test_data, is_vertical = load_dataset(self.DATASET, self.PARTITION_TYPE)

        self.TRAIN_DATA = train_data
        self.TEST_DATA = test_data
        self.IS_VERTICAL = is_vertical

        # --------------------------------------------------------
        # VERTICAL FL (only network_monitoring)
        # --------------------------------------------------------
        if is_vertical:
            _log("Vertical FL detected.")
            # train_data is already tuple-of-tuples: (client1, client2, ...)
            self.MULTIVARIATE = 1
            self.PARTITIONS = train_data
            self.FL_N_CLIENTS = len(train_data)
            _log(f"Vertical mode → FL_N_CLIENTS = {self.FL_N_CLIENTS}")
            self._partitions_prepared = True
            return

        # --------------------------------------------------------
        # HORIZONTAL
        # --------------------------------------------------------
        x, y = train_data
        self.MULTIVARIATE = x.shape[-1] if x.ndim >= 2 else 1

        # CENTRALIZED or SINGLE-PARTITION
        if self.PARTITION_TYPE in ["centralized"]:
            _log("Centralized  mode detected.")
            self.PARTITIONS = [train_data]
            self.FL_N_CLIENTS = 1
            _log("FL_N_CLIENTS overridden → 1")
            self._partitions_prepared = True
            return

        # IID SPLITTING
        n = self.FL_N_CLIENTS
        _log(f"Preparing IID partitions with n_clients={n}")

        if n <= 0:
            raise ValueError("N_CLIENTS must be >= 1")

        if n == 1:
            _log("Detected 1 client → using centralized partition")
            self.PARTITIONS = [train_data]
            self._partitions_prepared = True
            return

        # Normal IID K-Fold split
        from sklearn.model_selection import KFold
        import numpy as np

        kf = KFold(n_splits=n, shuffle=True, random_state=self.SEED)
        parts = []

        for i, (_, idx) in enumerate(kf.split(x)):
            px, py = x[idx], y[idx]
            parts.append((px, py))
            _log(f"Client {i}: partition size = {len(px)}")

        self.PARTITIONS = parts
        self._partitions_prepared = True

        _log(f"Generated {len(parts)} IID partitions.")
        _log(f"Final FL_N_CLIENTS = {self.FL_N_CLIENTS}")

    # -----------------------------------------------------------
    # ENV VARIABLE HELPER
    # -----------------------------------------------------------
    def get_env_var(self, name, typev):
        var = os.getenv(name)
        if var is None:
            return None

        if typev is bool:
            return var.lower() in ("true", "1", "yes", "y")

        return typev(var)

    # -----------------------------------------------------------
    # MAIN SETTINGS (Dataset, partition type, DP, batch sizes)
    # -----------------------------------------------------------
    def __set_main_settings(self):
        _log("Setting main environment-based settings...")

        # -----------------------------------------------------------
        # GLOBAL SEED (must be set before any model build / data shuffle)
        # -----------------------------------------------------------
        seed_env = self.get_env_var("SEED", int)
        self.SEED = seed_env if seed_env is not None else 42
        set_global_seed(self.SEED)
        _log(f"SEED = {self.SEED} (global determinism set)")

        p_type = os.getenv("PARTITION_TYPE")
        self.PARTITION_TYPE = p_type if p_type in PARTITION_TYPES else DefaultConfig.PARTITION_TYPE
        _log(f"ENV PARTITION_TYPE = {p_type}")
        _log(f"PARTITION_TYPE = {self.PARTITION_TYPE}")

        agg_type = os.getenv("AGGREGATION_TYPE")
        self.FL_AGGREGATION_TYPE = (
            agg_type if agg_type in AGGREGATION_TYPES else DefaultConfig.AGGREGATION_TYPE
        )
        _log(f"FL_AGGREGATION_TYPE = {self.FL_AGGREGATION_TYPE}")

        dataset = os.getenv("DATASET")
        self.DATASET = dataset if dataset in DATASETS else "network_monitoring"
        _log(f"DATASET = {self.DATASET}")

        # number of clients
        n_clients = os.getenv("N_CLIENTS")
        self.FL_N_CLIENTS = int(n_clients) if n_clients else DefaultConfig.N_CLIENTS
        _log(f"Initial FL_N_CLIENTS = {self.FL_N_CLIENTS}")

        # batch size
        dataset_default_bs = DEFAULT_BATCH_SIZES.get(self.DATASET, DefaultConfig.BATCH_SIZE)
        env_bs = self.get_env_var("BATCH_SIZE", int)
        self.BATCH_SIZE = env_bs if env_bs else dataset_default_bs
        _log(f"BATCH_SIZE = {self.BATCH_SIZE}")

        self.learning_rate = self.get_env_var("LEARNING_RATE", float)
        if self.learning_rate is None:
            self.learning_rate = 0.0005

        env_epochs = self.get_env_var("EPOCHS", int)
        self.EPOCHS = env_epochs if env_epochs else DefaultConfig.EPOCHS
        _log(f"EPOCHS = {self.EPOCHS}")

        self.FL_ROUNDS = self.EPOCHS
        _log(f"FL_ROUNDS synchronized with EPOCHS = {self.FL_ROUNDS}")

        self.no_fl = self.get_env_var("NO_FL", bool)
        if self.no_fl is None:
            self.no_fl = False
        _log(f"NO_FL = {self.no_fl}")


        self.EXPORT_DIR = EXPORT_DIR
        _log(f"EXPORT_DIR = {self.EXPORT_DIR}")

        # DP SETTINGS
        self.dp = self.get_env_var("DP", bool)
        self.local_dp = self.get_env_var("LOCAL_DP", bool)
        self.clipping = self.get_env_var("CLIPPING", str)
        self.epsilon = self.get_env_var("EPSILON", float)
        self.delta = self.get_env_var("DELTA", float)
        self.l2_norm_clip = self.get_env_var("L2_NORM_CLIP", float)

        # unit of privacy: "client" (silo-level, current) or "example" (record-level DP-SGD)
        self.dp_level = self.get_env_var("DP_LEVEL", str) or "client"

        # -----------------------------------------------------------
        # DP NOISE CALIBRATION  (target epsilon -> noise multiplier z)
        # -----------------------------------------------------------
        # CLIENT level: z from the client-level accountant (q=1), computed HERE
        #   (needs only M and R). LDP and CDP share the same z; differ in placement.
        # EXAMPLE level: local DP-SGD, z from the subsampled accountant (q=batch/
        #   n_client) -> needs per-client sizes, so it is computed in
        #   prepare_partitions() once PARTITIONS exist.
        self.noise_multiplier = None
        self.dp_epsilon_achieved = None
        if self.dp or self.local_dp:
            if self.l2_norm_clip is None:
                self.l2_norm_clip = 5.0
                _log(f"L2_NORM_CLIP defaulted to {self.l2_norm_clip}")
            if self.epsilon is None:
                raise ValueError("EPSILON must be set when DP or LOCAL_DP is enabled.")

            if self.dp_level == "client":
                if self.delta is None:
                    self.delta = dp_utils.delta_for_clients(self.FL_N_CLIENTS)
                self.noise_multiplier = dp_utils.noise_multiplier_for_epsilon(
                    target_epsilon=self.epsilon,
                    num_rounds=self.FL_ROUNDS,
                    num_clients_total=self.FL_N_CLIENTS,
                    num_clients_sampled=self.FL_N_CLIENTS,   # q = 1 (full participation)
                    delta=self.delta,
                )
                self.dp_epsilon_achieved = dp_utils.client_level_epsilon(
                    self.noise_multiplier, self.FL_ROUNDS,
                    self.FL_N_CLIENTS, self.FL_N_CLIENTS, self.delta,
                )
                mech = "LDP" if self.local_dp else "CDP"
                _log(
                    f"DP calibration [client/{mech}]: target_eps={self.epsilon}, "
                    f"R={self.FL_ROUNDS}, M={self.FL_N_CLIENTS}, delta={self.delta:.6g}, "
                    f"C={self.l2_norm_clip} -> z={self.noise_multiplier:.4f} "
                    f"(accountant eps={self.dp_epsilon_achieved:.4f})"
                )
            elif self.dp_level == "example":
                # local DP-SGD is inherently local; z is computed in prepare_partitions
                self.local_dp = True
                _log("DP calibration [example]: deferred to prepare_partitions "
                     "(needs per-client sizes for q=batch/n_examples_per_client).")
            else:
                raise ValueError(f"Unknown DP_LEVEL={self.dp_level!r} (use 'client' or 'example').")

        _log("Main settings complete.")

    # -----------------------------------------------------------
    # SECURE AGG SETTINGS
    # -----------------------------------------------------------
    def _set_secure_agg_settings(self):
        self.AGG_N_SHARES = self.FL_N_CLIENTS if self.FL_N_CLIENTS >= 3 else 3
        self.AGG_REC_SHARES = self.AGG_N_SHARES - 1
        _log(f"Secure agg shares: N={self.AGG_N_SHARES}, REC={self.AGG_REC_SHARES}")

    # -----------------------------------------------------------
    # MODEL NAME & PATHS
    # -----------------------------------------------------------
    def __set_model_settings(self):
        exp_name = self.__get_experiment_name()
        self.experiment_name = exp_name

        self.MODEL_NAME = exp_name
        self.MODEL_SAVE_PATH = self.SAVE_DIR.joinpath(f"{exp_name}.weights.h5")
        self.MODEL_HISTORY_SAVE_PATH = self.SAVE_DIR.joinpath(f"{exp_name}.history.pkl")
        self.METRICS_SAVE_PATH = self.SAVE_DIR.joinpath(f"{exp_name}.lclmetrics.csv")

        _log(f"Model Name: {exp_name}")
        _log(f"Model Save Path: {self.MODEL_SAVE_PATH}")

    # -----------------------------------------------------------
    # EXPERIMENT NAME
    # -----------------------------------------------------------
    def __get_experiment_name(self):
        name = (
            f"{self.DATASET}_fl_{self.FL_AGGREGATION_TYPE}_"
            f"{self.PARTITION_TYPE}_clients_{self.FL_N_CLIENTS}_rounds_{self.FL_ROUNDS}"
        )
        if self.dp or self.local_dp:
            mech = "ldp" if self.local_dp else "cdp"
            name += f"_{mech}_eps_{self.epsilon}"
        name += f"_seed_{self.SEED}"
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        return f"{name}_{timestamp}"


# ================================================================
# GLOBAL CONFIG CREATION
# ================================================================
ExpConfig = None

def get_config():
    global ExpConfig
    if ExpConfig is None:
        _log("Creating global ExpConfig...")
        ExpConfig = Config()
        ExpConfig.prepare_partitions()
        if ExpConfig.FL_AGGREGATION_TYPE == "secure":
            ExpConfig._set_secure_agg_settings()
    return ExpConfig
