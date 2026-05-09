from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# -----------------------------
# Paths
# -----------------------------
CURRENT_DIR = Path(__file__).parent
NETWORK_FILE = CURRENT_DIR / "./network_monitoring/buffer-20210415-20210526.tsv"
CIFAR_FILE = CURRENT_DIR / "./cifar10/cifar10.npz"
BODY_SMOKING_FILE = CURRENT_DIR / "./body_signal_of_smoking/data.csv"

# -----------------------------
# Shared config
# -----------------------------
CFG_DATA_SPLIT = 0.8
CFG_SEQUENCE_LEN = 24
CFG_SAMPLING_RATE = 1
CFG_STRIDE = 1
CFG_BATCH_SIZE = 1

CFG_SPLIT = 0.8


def _env_float(name, default):
    v = os.getenv(name)
    return default if v is None or v == "" else float(v)


def _env_int(name, default):
    v = os.getenv(name)
    return default if v is None or v == "" else int(v)


def _env_bool(name, default=False):
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return str(v).lower() in ("1", "true", "yes", "y", "t")


def describe_array(name, arr):
    arr = np.asarray(arr)
    flat = arr.astype(np.float64).reshape(-1)
    print(f"{name}:")
    print(f"  shape: {arr.shape}")
    print(f"  min:   {flat.min():.8f}")
    print(f"  max:   {flat.max():.8f}")
    print(f"  mean:  {flat.mean():.8f}")
    print(f"  std:   {flat.std():.8f}")


def get_network_sizes():
    if not NETWORK_FILE.exists():
        raise FileNotFoundError(f"Network file not found: {NETWORK_FILE}")

    data_df = pd.read_csv(NETWORK_FILE, sep="\t")
    data_df = data_df.set_index("ts")
    data_df.index = pd.to_datetime(data_df.index)

    cols = ["http_count_uid_in", "ssl_count_uid_in"]
    data_df = data_df[cols]
    data_df = data_df.interpolate(method="time")

    rows = len(data_df)
    train_size = int(rows * CFG_DATA_SPLIT)
    test_size = rows - train_size

    np_train_raw = data_df.iloc[:train_size].to_numpy(dtype="float32")
    np_test_raw = data_df.iloc[train_size:].to_numpy(dtype="float32")

    train_effective = max(train_size - CFG_SEQUENCE_LEN, 0)
    test_effective = max(test_size - CFG_SEQUENCE_LEN, 0)

    tsg_params = {
        "length": CFG_SEQUENCE_LEN,
        "sampling_rate": CFG_SAMPLING_RATE,
        "stride": CFG_STRIDE,
        "batch_size": CFG_BATCH_SIZE,
    }

    tsg_train = TimeseriesGenerator(np_train_raw, np_train_raw, **tsg_params)
    tsg_test = TimeseriesGenerator(np_test_raw, np_test_raw, **tsg_params)

    x_train = np.array([tsg_train[i][0][0] for i in range(len(tsg_train))], dtype=np.float32)
    y_train = np.array([tsg_train[i][1][0] for i in range(len(tsg_train))], dtype=np.float32)
    x_test = np.array([tsg_test[i][0][0] for i in range(len(tsg_test))], dtype=np.float32)
    y_test = np.array([tsg_test[i][1][0] for i in range(len(tsg_test))], dtype=np.float32)

    print("=== Network monitoring dataset ===")
    print(f"Raw total rows:        {rows}")
    print(f"Raw train rows:        {train_size}")
    print(f"Raw test rows:         {test_size}")
    print(f"Sequence length:       {CFG_SEQUENCE_LEN}")
    print(f"Effective train size:  {train_effective}")
    print(f"Effective test size:   {test_effective}")
    print(f"TSG train len:         {len(tsg_train)}")
    print(f"TSG test len:          {len(tsg_test)}")
    describe_array("network train raw", np_train_raw)
    describe_array("network test raw", np_test_raw)
    describe_array("network x_train", x_train)
    describe_array("network y_train", y_train)
    describe_array("network x_test", x_test)
    describe_array("network y_test", y_test)

    return {
        "raw_total": rows,
        "raw_train": train_size,
        "raw_test": test_size,
        "effective_train": len(tsg_train),
        "effective_test": len(tsg_test),
    }


def get_cifar10_sizes():
    if not CIFAR_FILE.exists():
        raise FileNotFoundError(f"CIFAR file not found: {CIFAR_FILE}")

    data = np.load(CIFAR_FILE)

    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]

    print("\n=== CIFAR-10 dataset ===")
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape:  {x_test.shape}")
    print(f"y_test shape:  {y_test.shape}")
    print(f"Train samples: {len(x_train)}")
    print(f"Test samples:  {len(x_test)}")
    describe_array("cifar10 x_train", x_train)
    describe_array("cifar10 x_test", x_test)
    describe_array("cifar10 y_train", y_train)
    describe_array("cifar10 y_test", y_test)

    return {
        "train": len(x_train),
        "test": len(x_test),
    }


def get_body_smoking_sizes():
    if not BODY_SMOKING_FILE.exists():
        raise FileNotFoundError(f"Body smoking file not found: {BODY_SMOKING_FILE}")

    df = pd.read_csv(BODY_SMOKING_FILE)

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    raw_total = len(df)

    cat_cols = df.select_dtypes(include=["object"]).columns
    encoders = {}
    for col in cat_cols:
        enc = LabelEncoder()
        df[col] = enc.fit_transform(df[col].astype(str))
        encoders[col] = enc

    y = df["smoking"].values.astype("float32")
    X = df.drop(columns=["smoking"]).values.astype("float32")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=CFG_SPLIT, shuffle=True, random_state=42
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    raw_train = len(X_train)
    raw_test = len(X_test)

    train_frac = _env_float("LEAKY_TRAIN_FRAC", 1.0)
    seed = _env_int("LEAKY_SEED", 42)

    if train_frac < 1.0:
        rng = np.random.default_rng(seed)
        n = len(X_train)
        k = max(1, int(train_frac * n))
        idx = rng.choice(n, size=k, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    canary_frac = _env_float("CANARY_FRAC", 0.0)
    canary_dups = _env_int("CANARY_DUPS", 0)
    flip = _env_bool("CANARY_FLIP", True)

    added_canaries = 0
    selected_canaries = 0

    if canary_frac > 0.0 and canary_dups > 0:
        rng = np.random.default_rng(seed + 1)
        n = len(X_train)
        c = max(1, int(canary_frac * n))
        c_idx = rng.choice(n, size=c, replace=False)

        X_can = X_train[c_idx]
        y_can = y_train[c_idx].copy()

        if flip:
            y_can = 1.0 - y_can

        X_rep = np.repeat(X_can, repeats=canary_dups, axis=0)
        y_rep = np.repeat(y_can, repeats=canary_dups, axis=0)

        X_train = np.concatenate([X_train, X_rep], axis=0)
        y_train = np.concatenate([y_train, y_rep], axis=0)

        selected_canaries = c
        added_canaries = len(X_rep)

    effective_train = len(X_train)
    effective_test = len(X_test)

    print("\n=== Body signals of smoking dataset ===")
    print(f"Raw total rows:              {raw_total}")
    print(f"Raw train rows:              {raw_train}")
    print(f"Raw test rows:               {raw_test}")
    print(f"Leaky train fraction:        {train_frac}")
    print(f"Canary fraction:             {canary_frac}")
    print(f"Canary duplicates:           {canary_dups}")
    print(f"Selected canaries:           {selected_canaries}")
    print(f"Added canary rows:           {added_canaries}")
    print(f"Effective train samples:     {effective_train}")
    print(f"Effective test samples:      {effective_test}")
    print(f"Feature count:               {X_train.shape[1]}")
    describe_array("smoking X_train", X_train)
    describe_array("smoking X_test", X_test)
    describe_array("smoking y_train", y_train)
    describe_array("smoking y_test", y_test)

    return {
        "raw_total": raw_total,
        "raw_train": raw_train,
        "raw_test": raw_test,
        "effective_train": effective_train,
        "effective_test": effective_test,
        "feature_count": X_train.shape[1],
        "train_frac": train_frac,
        "canary_frac": canary_frac,
        "canary_dups": canary_dups,
        "selected_canaries": selected_canaries,
        "added_canaries": added_canaries,
    }


if __name__ == "__main__":
    network_sizes = get_network_sizes()
    cifar_sizes = get_cifar10_sizes()
    body_smoking_sizes = get_body_smoking_sizes()