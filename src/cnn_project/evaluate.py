import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

from cnn_project.config import DEFAULT_CONFIG_PATH, load_config
from cnn_project.data import load_datasets
from cnn_project.model import CLASS_NAMES


def plot_confusion_matrix(cm, save_path):
    """رسم و ذخیره‌ی confusion matrix (نرمال‌شده بر اساس هر کلاس)."""
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized confusion matrix")
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color=color, fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def evaluate_model(model, test_ds, y_test, out_dir):
    """ارزیابی مدل روی داده‌ی تست و ذخیره‌ی گزارش‌ها در out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    accuracy = float(np.mean(y_pred == y_test))

    report_text = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
    report_dict = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)

    (out_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    (out_dir / "metrics.json").write_text(
        json.dumps({"test_accuracy": accuracy, "report": report_dict}, indent=2),
        encoding="utf-8",
    )
    plot_confusion_matrix(cm, out_dir / "confusion_matrix.png")

    print(report_text)
    print(f"Test accuracy: {accuracy:.4f}")
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test set.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    _, _, test_ds, y_test = load_datasets(cfg)
    model = keras.models.load_model(args.model_path)
    evaluate_model(model, test_ds, y_test, Path(args.model_path).parent)


if __name__ == "__main__":
    main()
