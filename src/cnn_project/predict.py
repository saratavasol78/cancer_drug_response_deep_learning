import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras

from cnn_project.gradcam import make_gradcam_heatmap, overlay_heatmap
from cnn_project.model import CLASS_NAMES

DEFAULT_MODEL_PATH = "artifacts/custom_cnn/best.keras"


def preprocess_image(model, image):
    """تغییر اندازه‌ی عکس مطابق ورودی مدل و اضافه‌کردن بُعد batch."""
    height, width = model.input_shape[1:3]
    resized = image.convert("RGB").resize((width, height))
    return np.expand_dims(np.asarray(resized, dtype="float32"), axis=0)


def predict_top_k(model, image, k=3):
    """پیش‌بینی k کلاس برتر برای یک عکس PIL."""
    probabilities = model.predict(preprocess_image(model, image), verbose=0)[0]
    top_indices = np.argsort(probabilities)[::-1][:k]
    return [
        {"class": CLASS_NAMES[i], "probability": float(probabilities[i])}
        for i in top_indices
    ]


def main():
    parser = argparse.ArgumentParser(description="Classify an image with a trained model.")
    parser.add_argument("image_path")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--gradcam", action="store_true",
                        help="save a Grad-CAM overlay next to the image")
    args = parser.parse_args()

    model = keras.models.load_model(args.model_path)
    image = Image.open(args.image_path)

    print("Top predictions:")
    for item in predict_top_k(model, image, args.top_k):
        print(f"  {item['class']:<12} {item['probability'] * 100:.1f}%")

    if args.gradcam:
        heatmap, _ = make_gradcam_heatmap(model, preprocess_image(model, image))
        out_path = Path("outputs") / f"{Path(args.image_path).stem}_gradcam.png"
        out_path.parent.mkdir(exist_ok=True)
        overlay_heatmap(image, heatmap).save(out_path)
        print(f"Grad-CAM saved to: {out_path}")


if __name__ == "__main__":
    main()
