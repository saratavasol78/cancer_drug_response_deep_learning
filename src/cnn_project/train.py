import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras

from cnn_project.config import DEFAULT_CONFIG_PATH, load_config
from cnn_project.data import load_datasets
from cnn_project.evaluate import evaluate_model
from cnn_project.model import build_model, get_backbone


def compile_model(model, learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def unfreeze_top_layers(model, num_layers):
    """باز کردن آخرین لایه‌های backbone برای fine-tuning (BatchNorm فریز می‌ماند)."""
    backbone = get_backbone(model)
    backbone.trainable = True
    for layer in backbone.layers[:-num_layers]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False


def plot_history(history, save_path):
    """رسم نمودار دقت و خطا برای کل دوره‌ی آموزش."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ("accuracy", "loss")):
        ax.plot(history[metric], label="train")
        ax.plot(history[f"val_{metric}"], label="validation")
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model", choices=["custom_cnn", "mobilenet_v2"],
                        help="override model name from the config file")
    parser.add_argument("--epochs", type=int, help="override number of epochs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["model"]["name"] = args.model
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs

    keras.utils.set_random_seed(cfg["seed"])

    model_name = cfg["model"]["name"]
    out_dir = Path(cfg["paths"]["artifacts_dir"]) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.keras"

    train_ds, val_ds, test_ds, y_test = load_datasets(cfg)
    model = build_model(cfg)
    compile_model(model, cfg["training"]["learning_rate"])
    model.summary()

    callbacks = [
        # ذخیره‌ی بهترین مدل بر اساس دقت اعتبارسنجی
        keras.callbacks.ModelCheckpoint(
            best_path, monitor="val_accuracy", save_best_only=True
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=cfg["training"]["early_stopping_patience"],
            restore_best_weights=True,
        ),
        # کاهش نرخ یادگیری در صورت ثابت‌ماندن خطا
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        keras.callbacks.TensorBoard(log_dir=out_dir / "logs"),
    ]

    # فاز اول: آموزش اصلی
    histories = [
        model.fit(train_ds, validation_data=val_ds,
                  epochs=cfg["training"]["epochs"], callbacks=callbacks)
    ]

    # فاز دوم: fine-tuning (فقط برای یادگیری انتقالی)
    fine_tune_cfg = cfg["training"]["fine_tune"]
    if model_name == "mobilenet_v2" and fine_tune_cfg["epochs"] > 0:
        unfreeze_top_layers(model, fine_tune_cfg["unfreeze_layers"])
        compile_model(model, fine_tune_cfg["learning_rate"])
        histories.append(
            model.fit(train_ds, validation_data=val_ds,
                      epochs=fine_tune_cfg["epochs"], callbacks=callbacks)
        )

    # ترکیب تاریخچه‌ی فازها و ذخیره‌ی نمودار
    history = {}
    for h in histories:
        for key, values in h.history.items():
            history.setdefault(key, []).extend(values)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_history(history, out_dir / "training_curves.png")

    # ارزیابی بهترین مدل ذخیره‌شده روی داده‌ی تست
    best_model = keras.models.load_model(best_path)
    evaluate_model(best_model, test_ds, y_test, out_dir)
    print(f"Artifacts saved in: {out_dir}")


if __name__ == "__main__":
    main()
