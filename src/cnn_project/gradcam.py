import numpy as np
import tensorflow as tf
from matplotlib import colormaps
from PIL import Image
from tensorflow import keras


def find_last_conv_layer(model):
    """پیدا کردن آخرین لایه‌ی Conv2D در مدل."""
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
    raise ValueError("Grad-CAM needs a model with a top-level Conv2D layer (use custom_cnn).")


def make_gradcam_heatmap(model, image_batch, class_index=None):
    """محاسبه‌ی Grad-CAM؛ خروجی: (heatmap با مقادیر ۰ تا ۱, کلاس هدف)."""
    last_conv_name = find_last_conv_layer(model)
    grad_model = keras.Model(
        model.inputs, [model.get_layer(last_conv_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model([image_batch])
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        class_score = predictions[:, class_index]

    # گرادیان امتیاز کلاس نسبت به نقشه‌ی ویژگی‌ها
    grads = tape.gradient(class_score, conv_output)
    channel_weights = tf.reduce_mean(grads, axis=(1, 2))
    heatmap = tf.reduce_sum(conv_output * channel_weights[:, None, None, :], axis=-1)[0]
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), class_index


def overlay_heatmap(image, heatmap, alpha=0.4, size=256):
    """قراردادن heatmap روی عکس اصلی؛ ورودی و خروجی از نوع PIL.Image."""
    base = np.asarray(image.convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
    heat = Image.fromarray(np.uint8(255 * heatmap)).resize((size, size), Image.BILINEAR)
    colored = colormaps["jet"](np.asarray(heat) / 255.0)[..., :3]
    blended = (1 - alpha) * base + alpha * colored
    return Image.fromarray(np.uint8(255 * np.clip(blended, 0, 1)))
