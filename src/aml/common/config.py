import yaml
from .paths import CONFIGS


def load_config() -> dict:
    with open(CONFIGS / "config.yaml") as f:
        return yaml.safe_load(f)
