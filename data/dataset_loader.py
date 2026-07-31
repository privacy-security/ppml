def load_dataset(dataset_name: str, partition_type: str):
    dataset_name = dataset_name.lower()
    print(f"[DATASET] Loading dataset='{dataset_name}', partition_type='{partition_type}'")

    # ======================================================
    # NETWORK MONITORING
    # ======================================================
    if dataset_name == "network_monitoring":
        from data.network_monitoring.dataset import (
            network_data_train,
            network_data_test,
            network_feature_data_train,
            network_feature_data_test,
        )

        if partition_type == "vertical":
            print("[DATASET] Vertical FL mode → returning feature-split partitions.")
            print(f"[DATASET] Vertical partitions: {len(network_feature_data_train)} clients")
            return network_feature_data_train, network_feature_data_test, True

        print("[DATASET] Returning horizontal data (x,y).")
        return network_data_train, network_data_test, False

    # ======================================================
    # BODY SIGNAL
    # ======================================================
    if dataset_name == "body_signal_of_smoking":
        from data.body_signal_of_smoking.dataset import load_body_smoking
        train, test = load_body_smoking()
        print("[DATASET] Loaded Body Signal (horizontal only).")
        print(f"[DATASET] Train size = {len(train[0])}, Test size = {len(test[0])}")
        return train, test, False

    # ======================================================
    # CIFAR10
    # ======================================================
    if dataset_name == "cifar10":
        from data.cifar10.dataset import load_cifar10
        train, test = load_cifar10()
        print("[DATASET] Loaded CIFAR10.")
        print(f"[DATASET] Train size = {len(train[0])}, Test size = {len(test[0])}")
        return train, test, False

    # ======================================================
    # HOUSEHOLD POWER  (independent public time-series benchmark)
    # ======================================================
    if dataset_name == "household_power":
        from data.household_power.dataset import load_household_power
        train, test = load_household_power()
        print("[DATASET] Loaded Household Power (horizontal time-series).")
        print(f"[DATASET] Train size = {len(train[0])}, Test size = {len(test[0])}")
        return train, test, False

    # ======================================================
    # UNKNOWN
    # ======================================================
    raise ValueError(f"Unknown dataset: {dataset_name}")