from pathlib import Path

from PIL import Image
from tensorflow import keras

from cnn_project.model import CLASS_NAMES


def main():
    """ذخیره‌ی ۱۰ عکس نمونه (از هر کلاس یکی) از داده‌ی تست در پوشه‌ی samples/."""
    out_dir = Path("samples")
    out_dir.mkdir(exist_ok=True)

    (_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
    saved = set()
    for image, label in zip(x_test, y_test.flatten()):
        label = int(label)
        if label in saved:
            continue
        Image.fromarray(image).resize((256, 256), Image.NEAREST).save(
            out_dir / f"{CLASS_NAMES[label]}.png"
        )
        saved.add(label)
        if len(saved) == len(CLASS_NAMES):
            break
    print(f"{len(saved)} sample images saved in {out_dir}/")


if __name__ == "__main__":
    main()
