# federation/model.py
from typing import Optional
import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam


logger = logging.getLogger(__name__)

# default hyperparams
cfg_sequence_len = 24
cfg_learning_rate = 0.001


class F1Metric(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * (p * r) / (p + r + 1e-7)

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()


def make_optimizer(
    learning_rate: float = 0.001,
    dp: bool = False,
    no_fl: bool = False,
    l2_norm_clip: Optional[float] = None,
    noise_multiplier: Optional[float] = None,
    num_microbatches: int = 1,
):
    """
    Return either a DP Keras optimizer (from tensorflow-privacy)
    or a standard Adam optimizer.

    Logs exactly which optimizer is used and why.
    """

    logger.info(
        "[OPTIMIZER] Requested optimizer config: "
        f"dp={dp}, no_fl={no_fl}, "
        f"lr={learning_rate}, "
        f"l2_clip={l2_norm_clip}, "
        f"noise={noise_multiplier}, "
        f"microbatches={num_microbatches}"
    )

    if dp and no_fl:
        import tensorflow_privacy as tf_privacy
        # dp==True and no fl-> build DPKerasAdamOptimizer (TF-Privacy)

        # default clip & noise if None (you can override from config)
        if l2_norm_clip is None:
            l2_norm_clip = 1.0
        if noise_multiplier is None:
            noise_multiplier = 1.0
        if num_microbatches is None:
            num_microbatches = 1

        # instantiate DP optimizer
        opt = tf_privacy.DPKerasAdamOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=num_microbatches,
            learning_rate=learning_rate,
        )

        logger.warning(
            "[OPTIMIZER] USING DP OPTIMIZER: DPKerasAdamOptimizer | "
            f"l2_clip={l2_norm_clip}, "
            f"noise={noise_multiplier}, "
            f"microbatches={num_microbatches}"
        )

        return opt

    # --------------------------------------------------
    # Standard (non-DP) optimizer
    # --------------------------------------------------
    opt = Adam(learning_rate)

    logger.warning(
        "[OPTIMIZER] USING NON-DP OPTIMIZER: Adam | "
        f"lr={learning_rate} | "
        f"reason: dp={dp}, no_fl={no_fl}"
    )

    return opt


def build_gru_model(
    model_name: str,
    multivariate: int,
    sequence_len: int,
    sequence_len_y: int = 1,
    learning_rate: float = cfg_learning_rate,
    dp: bool = False,
    no_fl: bool = False,
    l2_norm_clip: Optional[float] = None,
    noise_multiplier: Optional[float] = None,
    num_microbatches: int = 1,
):
    inputs = layers.Input(shape=(sequence_len, multivariate))
    h = layers.GRU(units=16, return_sequences=True)(inputs)
    h = layers.Dropout(0.1)(h)
    h = layers.GRU(units=80, return_sequences=True)(h)
    h = layers.Dropout(0.05)(h)
    h = layers.GRU(units=40, return_sequences=True)(h)
    h = layers.Dropout(0.25)(h)
    h = layers.GRU(units=96, return_sequences=False)(h)
    h = layers.Dropout(0.2)(h)

    outputs = layers.Dense(units=multivariate * sequence_len_y, activation="sigmoid")(h)

    model = Model(inputs=inputs, outputs=outputs, name=model_name)

    opt = make_optimizer(
        learning_rate=learning_rate,
        dp=dp,
        no_fl=no_fl,
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=num_microbatches,
    )

    # For regression/time-series MSE is common
    model.compile(loss="mean_squared_error", optimizer=opt, metrics=["mse", "mae"])

    return model


def build_cifar10_cnn(
    model_name: str,
    learning_rate: float = 0.001,
    dp: bool = False,
    no_fl: bool = False,
    l2_norm_clip: Optional[float] = None,
    noise_multiplier: Optional[float] = None,
    num_microbatches: int = 1,
):
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )

    inputs = layers.Input(shape=(32, 32, 3))

    # Apply augmentation only in training
    x = data_augmentation(inputs)

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", kernel_initializer="he_normal")(x)
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding="same", kernel_initializer="he_normal")(x)
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same", kernel_initializer="he_normal")(x)
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, padding="same", kernel_initializer="he_normal")(x)
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same", kernel_initializer="he_normal")(x)
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, 3, padding="same", kernel_initializer="he_normal")(x)
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)

    # Dense
    x = layers.Dense(128, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(10, activation="softmax")(x)

    model = Model(inputs, outputs, name=model_name)

    opt = make_optimizer(
        learning_rate=learning_rate,
        dp=dp,
        no_fl=no_fl,
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=num_microbatches,
    )

    # Note: keep categorical_crossentropy for softmax outputs.
    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            F1Metric(name="f1_score"),
            keras.metrics.AUC(name="auc", multi_label=True),
        ],
    )

    return model


def build_tabular_classifier(
    model_name: str,
    num_features: int,
    learning_rate: float,
    dp: bool = False,
    no_fl: bool = False,
    l2_norm_clip: Optional[float] = None,
    noise_multiplier: Optional[float] = None,
    num_microbatches: int = 1,
):
    """
    Build a tabular binary classifier.
    DP-related imports are done lazily via make_optimizer.
    """
    inputs = layers.Input(shape=(num_features,))

    x = layers.Dense(256, activation="swish")(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(128, activation="swish")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(64, activation="swish")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Dense(32, activation="swish")(x)
    x = layers.LayerNormalization()(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name=model_name)

    optimizer = make_optimizer(
        learning_rate=learning_rate,
        dp=dp,
        no_fl=no_fl,
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=num_microbatches,
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
       metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )

    return model


def build_model(dataset_name: str, **kwargs):
    """
    dataset_name determines which architecture to return.
    kwargs may include:
        model_name
        multivariate
        sequence_len
        learning_rate
        dp (bool)
        l2_norm_clip
        noise_multiplier
        num_microbatches
        config (optional ExpConfig with same fields)
    """

    # --------------------------------------------------------
    # Extract config or fall back to kwargs
    # --------------------------------------------------------
    cfg = kwargs.get("config", None)

    # learning rate
    learning_rate = kwargs.get("learning_rate", None)
    if learning_rate is None and cfg is not None:
        learning_rate = getattr(cfg, "learning_rate", None)
    if learning_rate is None:
        learning_rate = 0.001

    # differential privacy flags
    dp = kwargs.get("dp", None)
    if dp is None and cfg is not None:
        dp = getattr(cfg, "dp", False)
    if dp is None:
        dp = False

    no_fl = kwargs.get("no_fl", None)
    if no_fl is None and cfg is not None:
        no_fl = getattr(cfg, "no_fl", False)
    if no_fl is None:
        no_fl = False

    l2_norm_clip = kwargs.get("l2_norm_clip", None)
    if l2_norm_clip is None and cfg is not None:
        l2_norm_clip = getattr(cfg, "l2_norm_clip", None)

    noise_multiplier = kwargs.get("noise_multiplier", None)
    if noise_multiplier is None and cfg is not None:
        noise_multiplier = getattr(cfg, "noise_multiplier", None)

    num_microbatches = kwargs.get("num_microbatches", None)
    if num_microbatches is None and cfg is not None:
        num_microbatches = getattr(cfg, "num_microbatches", 1)
    if num_microbatches is None:
        num_microbatches = 1

    # --------------------------------------------------------
    # Select model by dataset
    # --------------------------------------------------------
    if dataset_name in ["body_signal", "body_signal_of_smoking", "adult_census"]:
        return build_tabular_classifier(
            model_name=kwargs["model_name"],
            num_features=kwargs["multivariate"],
            learning_rate=learning_rate,
            dp=dp,
            no_fl=no_fl,
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=num_microbatches,
        )

    if dataset_name == "cifar10":
        return build_cifar10_cnn(
            model_name=kwargs["model_name"],
            learning_rate=learning_rate,
            dp=dp,
            no_fl=no_fl,
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=num_microbatches,
        )

    if dataset_name == "network_monitoring":
        return build_gru_model(
            model_name=kwargs["model_name"],
            multivariate=kwargs["multivariate"],
            sequence_len=kwargs["sequence_len"],
            learning_rate=learning_rate,
            dp=dp,
            no_fl=no_fl,
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=num_microbatches,
        )

    raise ValueError(f"Unknown dataset: {dataset_name}")