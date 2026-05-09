import numpy as np
import math

def partition_iid(n_partitions: int, train_data):
    x, y = train_data

    if n_clients == 1:
        return [(x, y)]

    # Shuffle
    idx = np.random.permutation(len(x))
    x = x[idx]
    y = y[idx]

    size = len(x) // n_clients

    partitions = []
    for i in range(n_clients):
        start = i * size
        end = (i + 1) * size
        partitions.append((x[start:end], y[start:end]))
    return partitions


def partition_noniid(n_partitions: int, train_data):
    x, y = train_data

    if n_clients == 1:
        return [(x, y)]

    # naive non-IID: sort by label then chunk
    sorted_idx = np.argsort(y)
    x = x[sorted_idx]
    y = y[sorted_idx]

    size = len(x) // n_clients
    partitions = []
    for i in range(n_clients):
        start = i * size
        end = (i + 1) * size
        partitions.append((x[start:end], y[start:end]))
    return partitions

