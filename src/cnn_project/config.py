from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = "configs/config.yaml"


def load_config(path=DEFAULT_CONFIG_PATH):
    """خواندن فایل تنظیمات YAML و برگرداندن آن به‌صورت دیکشنری."""
    with open(Path(path), "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
