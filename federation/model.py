# federation/model.py
from typing import Optional
import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam

from federation.dp_sgd import make_dp_sgd_model


def _construct(inputs, outputs, model_name, loss_id, example_dp):
    """Plain Model, or a DPSGDModel when example-level DP is active."""
    if example_dp is not None:
        return make_dp_sgd_model(
            inputs, outputs, name=model_name,
            l2_norm_clip=example_dp["l2_norm_clip"],
            noise_multiplier=example_dp["noise_multiplier"],
            loss_fn=loss_id,
            seed_base=example_dp["seed_base"],
            use_vectorized=example_dp.get("use_vectorized", True),
        )
    return Model(inputs, outputs, name=model_name)


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
    Always returns a standard Adam optimizer.

    NOTE: DP is no longer implemented via a DP optimizer. Client-level DP adds
    noise to the update after training; example-level DP is DP-SGD implemented in
    DPSGDModel.train_step. tensorflow_privacy's DP optimizers depend on
    keras.optimizers.legacy, which Keras 3 removed, so they are not used.
    The dp/no_fl/clip/noise args are accepted for signature compatibility.
    """
    logger.info(
        f"[OPTIMIZER] Adam (lr={learning_rate}); DP handled outside the optimizer "
        f"(dp={dp}, no_fl={no_fl})."
    )
    return Adam(learning_rate)


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
    example_dp: Optional[dict] = None,
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

    model = _construct(inputs, outputs, model_name, "mean_squared_error", example_dp)

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
    example_dp: Optional[dict] = None,
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

    model = _construct(inputs, outputs, model_name, "categorical_crossentropy", example_dp)

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
    example_dp: Optional[dict] = None,
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

    model = _construct(inputs, outputs, model_name, "binary_crossentropy", example_dp)

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
    # Example-level DP: build a DPSGDModel (per-example clip + noise) instead
    # of a plain Model. Only when dp is on AND dp_level == "example".
    # --------------------------------------------------------
    dp_level = kwargs.get("dp_level", None)
    if dp_level is None and cfg is not None:
        dp_level = getattr(cfg, "dp_level", "client")
    if dp_level is None:
        dp_level = "client"

    example_dp = None
    if dp and dp_level == "example":
        dp_seed_base = kwargs.get("dp_seed_base", 0)
        use_vec = kwargs.get("example_use_vectorized", None)
        if use_vec is None and cfg is not None:
            use_vec = getattr(cfg, "example_use_vectorized", True)
        if use_vec is None:
            use_vec = True
        example_dp = {
            "l2_norm_clip": float(l2_norm_clip if l2_norm_clip is not None else 1.0),
            "noise_multiplier": float(noise_multiplier if noise_multiplier is not None else 1.0),
            "seed_base": int(dp_seed_base),
            "use_vectorized": bool(use_vec),
        }

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
            example_dp=example_dp,
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
            example_dp=example_dp,
        )

    if dataset_name in ["network_monitoring", "household_power"]:
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
            example_dp=example_dp,
        )

    raise ValueError(f"Unknown dataset: {dataset_name}")
