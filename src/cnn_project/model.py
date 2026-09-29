from tensorflow import keras
from tensorflow.keras import layers

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def build_augmentation():
    """لایه‌های افزایش داده (فقط در زمان آموزش فعال هستند)."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomRotation(0.05),
        ],
        name="augmentation",
    )


def conv_block(x, filters):
    """Conv -> BatchNorm -> ReLU"""
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)


def build_custom_cnn(input_shape, num_classes, dropout):
    """CNN سفارشی به سبک VGG: سه بلوک، هر بلوک دو لایه‌ی کانولوشنی."""
    inputs = keras.Input(shape=input_shape, name="image")
    x = build_augmentation()(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    for filters in (32, 64, 128):
        x = conv_block(x, filters)
        x = conv_block(x, filters)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probabilities")(x)
    return keras.Model(inputs, outputs, name="custom_cnn")


def build_mobilenet_v2(input_shape, num_classes, dropout, pretrained, image_size):
    """یادگیری انتقالی با MobileNetV2 (backbone در ابتدا فریز می‌شود)."""
    inputs = keras.Input(shape=input_shape, name="image")
    x = build_augmentation()(inputs)
    x = layers.Resizing(image_size, image_size)(x)
    # معادل preprocess_input مخصوص MobileNetV2 (بازه‌ی -۱ تا ۱)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)

    backbone = keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet" if pretrained else None,
    )
    backbone.trainable = False
    x = backbone(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probabilities")(x)
    return keras.Model(inputs, outputs, name="mobilenet_v2_transfer")


def get_backbone(model):
    """پیدا کردن زیرمدل MobileNetV2 داخل مدل اصلی."""
    for layer in model.layers:
        if layer.name.startswith("mobilenetv2"):
            return layer
    raise ValueError("This model has no MobileNetV2 backbone.")


def build_model(cfg):
    """ساخت مدل بر اساس نام مدل در فایل تنظیمات."""
    model_cfg = cfg["model"]
    input_shape = tuple(model_cfg["input_shape"])
    name = model_cfg["name"]

    if name == "custom_cnn":
        return build_custom_cnn(input_shape, model_cfg["num_classes"], model_cfg["dropout"])
    if name == "mobilenet_v2":
        return build_mobilenet_v2(
            input_shape,
            model_cfg["num_classes"],
            model_cfg["dropout"],
            model_cfg["pretrained"],
            model_cfg["transfer_image_size"],
        )
    raise ValueError(f"Unknown model name: {name}")
