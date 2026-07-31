"""
seeding.py -- single entry point for global determinism.

set_global_seed(seed) is called once per process (in Config.__init__) BEFORE any
model is built or any data is shuffled, so weight init, dropout, and data shuffling
are reproducible for a given SEED and vary across SEEDs (which is what the multi-seed
protocol in R2.4 needs).

NOTE: the DP noise is NOT drawn from this global stream. It is drawn from a dedicated
per-(SEED, client_id, round) stream in the client, so that (a) each client's noise is
independent -- required for the sqrt(M) aggregate-noise behaviour to hold -- and
(b) it is still reproducible given SEED.
"""
from __future__ import annotations
import os
import random


def set_global_seed(seed: int) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

    try:
        import tensorflow as tf
        # TF 2.9+: seeds python, numpy and tf in one call.
        try:
            tf.keras.utils.set_random_seed(seed)
        except Exception:
            tf.random.set_seed(seed)
        # Best-effort determinism for GPU ops; harmless if unsupported.
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except Exception:
        pass
