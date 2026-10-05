import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get("AML_DATA_DIR", ROOT / "data"))
RAW_CSV = DATA_DIR / "raw" / "HI-Small_Trans.csv"
PROCESSED = DATA_DIR / "processed"
STUB = ROOT / "data_stub"
RESULTS = ROOT / "results"
RESULTS_STUB = ROOT / "results_stub"
CONFIGS = ROOT / "configs"


def processed_dir(use_stub: bool = False) -> Path:
    """Stub mode: AML_USE_STUB=1 or use_stub=True."""
    return STUB if (use_stub or os.environ.get("AML_USE_STUB") == "1") else PROCESSED
