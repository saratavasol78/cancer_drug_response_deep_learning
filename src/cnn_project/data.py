import numpy as np
import tensorflow as tf
from tensorflow import keras


def load_datasets(cfg):
    """بارگذاری CIFAR-10 و ساخت دیتاست‌های train / validation / test.

    خروجی: train_ds, val_ds, test_ds, y_test
    (y_test برای محاسبه‌ی confusion matrix نگه داشته می‌شود)
    """
    seed = cfg["seed"]
    batch_size = cfg["data"]["batch_size"]
    val_split = cfg["data"]["validation_split"]

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train, y_test = y_train.flatten(), y_test.flatten()

    # جداسازی داده‌ی اعتبارسنجی با یک جایگشت تصادفی قابل‌تکرار
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x_train))
    n_val = int(len(x_train) * val_split)
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    def make_dataset(x, y, shuffle):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            ds = ds.shuffle(len(x), seed=seed)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = make_dataset(x_train[train_idx], y_train[train_idx], shuffle=True)
    val_ds = make_dataset(x_train[val_idx], y_train[val_idx], shuffle=False)
    test_ds = make_dataset(x_test, y_test, shuffle=False)
    return train_ds, val_ds, test_ds, y_test
