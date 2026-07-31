"""
data/household_power/__init__.py

Mirrors data/network_monitoring/__init__.py, adapted to household power.

Differences from network monitoring (both intentional):
  * household power is HORIZONTAL-only -> there are no vertical exports
    (no df_*_v, no *_feature_data_*); dataset_loader returns is_vertical=False.
  * its dataset module is function-based, so we also export the loader
    load_household_power().

Touching household_data_train / household_data_test / data_train_df /
data_test_df reads + processes the file once (cached in dataset._BUNDLE).
"""
from .dataset import (
    load_household_power,
    household_data_train,
    household_data_test,
    data_train_df,
    data_test_df,
)
from .partition_functions import partition_iid, partition_noniid
