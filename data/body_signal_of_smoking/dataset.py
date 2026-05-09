import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

CURREN_DIR = Path(__file__).parent
FILE_PATH = CURREN_DIR.joinpath("data.csv")

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

def load_body_smoking():
    df = pd.read_csv(FILE_PATH)

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

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

    # ---------------------------------------------------------
    # LEAKY MODE CONTROLS (env-driven)
    # ---------------------------------------------------------
    # Use only a fraction of training data -> promotes overfitting/memorization
    train_frac = _env_float("LEAKY_TRAIN_FRAC", 1.0)  # e.g. 0.05
    seed = _env_int("LEAKY_SEED", 42)

    if train_frac < 1.0:
        rng = np.random.default_rng(seed)
        n = len(X_train)
        k = max(1, int(train_frac * n))
        idx = rng.choice(n, size=k, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    # Optional: canaries (very effective leakage amplifier)
    # - Pick a fraction of training points
    # - Flip their labels
    # - Duplicate them many times
    canary_frac = _env_float("CANARY_FRAC", 0.0)       # e.g. 0.01
    canary_dups = _env_int("CANARY_DUPS", 0)          # e.g. 50
    flip = _env_bool("CANARY_FLIP", True)             # default True

    if canary_frac > 0.0 and canary_dups > 0:
        rng = np.random.default_rng(seed + 1)
        n = len(X_train)
        c = max(1, int(canary_frac * n))
        c_idx = rng.choice(n, size=c, replace=False)

        X_can = X_train[c_idx]
        y_can = y_train[c_idx].copy()

        if flip:
            # assumes binary labels {0,1}
            y_can = 1.0 - y_can

        # duplicate canaries
        X_rep = np.repeat(X_can, repeats=canary_dups, axis=0)
        y_rep = np.repeat(y_can, repeats=canary_dups, axis=0)

        # append to training set
        X_train = np.concatenate([X_train, X_rep], axis=0)
        y_train = np.concatenate([y_train, y_rep], axis=0)

    return (X_train, y_train), (X_test, y_test)
