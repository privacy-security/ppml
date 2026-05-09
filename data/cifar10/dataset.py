import numpy as np
import tensorflow as tf
from pathlib import Path


CIFAR_FILE = Path(__file__).parent / "cifar10.npz"

def load_cifar10(normalize=True, one_hot=True):
    if not CIFAR_FILE.exists():
        raise FileNotFoundError(
            f"CIFAR10 not found at {CIFAR_FILE}. "
            f"Run prepare_cifar.py once to download it."
        )

    data = np.load(CIFAR_FILE)

    x_train = data["x_train"]
    y_train = data["y_train"].squeeze()
    x_test = data["x_test"]
    y_test = data["y_test"].squeeze()

    if normalize:
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

    if one_hot:
       y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
       y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    return (x_train, y_train), (x_test, y_test)
