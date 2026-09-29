from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cnn_project.config import load_config
from cnn_project.gradcam import make_gradcam_heatmap, overlay_heatmap
from cnn_project.model import CLASS_NAMES, build_model
from cnn_project.predict import predict_top_k

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


@pytest.fixture
def cfg():
    config = load_config(CONFIG_PATH)
    config["model"]["pretrained"] = False  # جلوگیری از دانلود وزن‌ها در تست
    return config


def random_batch(n=2):
    return np.random.randint(0, 256, size=(n, 32, 32, 3)).astype("float32")


@pytest.mark.parametrize("name", ["custom_cnn", "mobilenet_v2"])
def test_model_output_is_probability_distribution(cfg, name):
    cfg["model"]["name"] = name
    model = build_model(cfg)
    output = model.predict(random_batch(), verbose=0)
    assert output.shape == (2, len(CLASS_NAMES))
    np.testing.assert_allclose(output.sum(axis=1), 1.0, atol=1e-4)


def test_gradcam_heatmap_range(cfg):
    model = build_model(cfg)
    heatmap, class_index = make_gradcam_heatmap(model, random_batch(1))
    assert heatmap.ndim == 2
    assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0
    assert 0 <= class_index < len(CLASS_NAMES)


def test_overlay_returns_image(cfg):
    heatmap = np.random.rand(8, 8).astype("float32")
    image = Image.fromarray(np.uint8(random_batch(1)[0]))
    assert overlay_heatmap(image, heatmap).size == (256, 256)


def test_predict_top_k_sorted(cfg):
    model = build_model(cfg)
    image = Image.fromarray(np.uint8(random_batch(1)[0]))
    results = predict_top_k(model, image, k=3)
    probs = [r["probability"] for r in results]
    assert len(results) == 3
    assert probs == sorted(probs, reverse=True)
