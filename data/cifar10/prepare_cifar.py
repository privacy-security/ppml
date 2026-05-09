from tensorflow.keras.datasets import cifar10
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent / "cifar10.npz"

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

np.savez_compressed(
    OUT,
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test
)

print(f"✅ CIFAR10 saved locally at: {OUT}")
