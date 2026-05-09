import os
import sys
from datetime import datetime
from pathlib import Path

from data.dataset_loader import load_dataset

DATASETS = ["network_monitoring", "cifar10", "body_signal_of_smoking"]
PARTITION_TYPES = ["noniid", "iid", "vertical", "centralized"]
AGGREGATION_TYPES = ["regular", "secure"]

DEFAULT_BATCH_SIZES = {
    "network_monitoring": 256,
    "cifar10": 16,
    "body_signal_of_smoking": 128,
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

    # -----------------------------------------------------------
    # PARTITION PREPARATION
    # -----------------------------------------------------------
    def prepare_partitions(self):
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

        kf = KFold(n_splits=n, shuffle=True, random_state=42)
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
        self.noise_multiplier = self.get_env_var("NOISE_MULTIPLIER", float)
        self.epsilon = self.get_env_var("EPSILON", float)
        self.delta = self.get_env_var("DELTA", float)
        self.l2_norm_clip = self.get_env_var("L2_NORM_CLIP", float)

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
