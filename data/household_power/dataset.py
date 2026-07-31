"""
data/household_power/dataset.py

UCI Individual Household Electric Power Consumption -- independent public
time-series benchmark added for reviewer R2.6.

Mirrors the network_monitoring pipeline so the new dataset is the ONLY changed
variable in the time-series comparison:
  - two related continuous signals (global active + reactive power),
    analogous to the two protocol-count signals in network_monitoring
  - hourly resampling  -> period 24 = one day, matching SEQUENCE_LEN = 24
  - time interpolation of the ~1.25% missing rows
  - per-feature MinMax normalization to [0, 1]  (the GRU output is sigmoid,
    so targets MUST live in [0, 1])
  - 80/20 chronological train/test split, scaler fit on train only
  - length-24 sliding windows via TimeseriesGenerator (one-step-ahead targets)

Household power is HORIZONTAL-only: there is no vertical / per-feature split
(unlike network_monitoring), so this module exposes only horizontal artifacts.

Public names (consumed by __init__.py, notebooks, and dataset_loader):
    load_household_power()  -> ((train_x, train_y), (test_x, test_y))
    household_data_train, household_data_test   # the horizontal (x, y) tuples
    data_train_df, data_test_df                 # normalized hourly dataframes
All heavy work is cached in _BUNDLE, so touching several of these names loads
and processes the file exactly once.

DATA FILE (place here; do not commit the ~127 MB file):
    data/household_power/household_power_consumption.txt
DOWNLOAD (UCI dataset id 235, CC BY 4.0, DOI 10.24432/C58K54):
    https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
    -> download the .zip, unzip 'household_power_consumption.txt' into this folder.
    Alternatively: `pip install ucimlrepo` and set USE_UCIMLREPO = True below.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

CURRENT_DIR = Path(__file__).parent
FILE_PATH = CURRENT_DIR.joinpath("household_power_consumption.txt")

# ---- pipeline constants (kept identical to network_monitoring) ----
CFG_DATA_SPLIT = 0.8
CFG_SEQUENCE_LEN = 24
CFG_SAMPLING_RATE = 1
CFG_STRIDE = 1
CFG_BATCH_SIZE = 1
CFG_RESAMPLE = "60min"          # "60min" (not "1H") to avoid pandas H/h deprecation

# two related continuous signals (both kW), mirroring the two network counts
FEATURES = ["Global_active_power", "Global_reactive_power"]

USE_UCIMLREPO = False           # True -> fetch via `ucimlrepo` instead of local file

_BUNDLE = None                  # cache: file is read + processed at most once


# ------------------------------------------------------------------
# small helpers (self-contained so importing this module never triggers
# the network_monitoring module's top-level file read)
# ------------------------------------------------------------------
def _normalize(data, scaler=None):
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        norm = scaler.fit_transform(data)
        norm = np.where(norm <= 0, 1e-3, norm)   # same positivity guard as network
    else:
        norm = scaler.transform(data)
    return norm, scaler


def _tsg_to_dataset(tsg):
    tsg_len = len(tsg)
    x_shape = tsg[0][0][0].shape
    y_shape = tsg[0][1][0].shape
    xs, ys = [], []
    for seq, target in tsg:
        xs.append(seq)
        ys.append(target)
    xs = np.array(xs).reshape((tsg_len, *x_shape))
    ys = np.array(ys).reshape((tsg_len, *y_shape))
    return xs, ys


def _load_raw():
    if USE_UCIMLREPO:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=235)
        df = ds.data.features.copy()
    else:
        if not FILE_PATH.exists():
            raise FileNotFoundError(
                f"Expected household power file at {FILE_PATH}. Download the zip from "
                "UCI dataset id 235 "
                "(https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption), "
                "unzip 'household_power_consumption.txt' into that folder, or set "
                "USE_UCIMLREPO = True after `pip install ucimlrepo`."
            )
        df = pd.read_csv(FILE_PATH, sep=";", na_values=["?"], low_memory=False)

    # build a datetime index from the Date + Time columns
    dt = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    df = df.set_index(dt)

    # keep only the two target features, coerce to float (handles '?'/blank)
    df = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    return df


def _prepare():
    """Read + process the file once; cache the full artifact bundle."""
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE

    df = _load_raw()

    # hourly mean -> daily period (24); then fill the small gaps by time
    df = df.resample(CFG_RESAMPLE).mean()
    df = df.interpolate(method="time").dropna()

    values = df.values.astype("float32")
    rows = len(values)
    train_size = int(rows * CFG_DATA_SPLIT)
    raw_train, raw_test = values[:train_size], values[train_size:]

    train_n, scaler = _normalize(raw_train)
    test_n, _ = _normalize(raw_test, scaler)

    # normalized hourly dataframes with datetime index (analogue of network data_*_df)
    idx = df.index
    train_df = pd.DataFrame(train_n, columns=FEATURES, index=idx[:train_size])
    test_df = pd.DataFrame(test_n, columns=FEATURES, index=idx[train_size:])

    tsg_params = dict(
        length=CFG_SEQUENCE_LEN,
        sampling_rate=CFG_SAMPLING_RATE,
        stride=CFG_STRIDE,
        batch_size=CFG_BATCH_SIZE,
    )
    train_x, train_y = _tsg_to_dataset(TimeseriesGenerator(train_n, train_n, **tsg_params))
    test_x, test_y = _tsg_to_dataset(TimeseriesGenerator(test_n, test_n, **tsg_params))

    print(
        f"[DATASET] household_power: train={train_x.shape} test={test_x.shape} "
        f"features={FEATURES} resample={CFG_RESAMPLE}"
    )

    _BUNDLE = {
        "train": (train_x, train_y),
        "test": (test_x, test_y),
        "train_df": train_df,
        "test_df": test_df,
    }
    return _BUNDLE


def load_household_power():
    b = _prepare()
    return b["train"], b["test"]


# lazy, cached module-level names (mirrors network_monitoring's eager exports,
# but horizontal-only). Touching any of these loads the file once via _prepare().
_LAZY = {
    "household_data_train": "train",
    "household_data_test": "test",
    "data_train_df": "train_df",
    "data_test_df": "test_df",
}


def __getattr__(name):
    if name in _LAZY:
        return _prepare()[_LAZY[name]]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
