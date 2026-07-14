import os
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def _project_root() -> str:
    env_root = os.environ.get("SAM3_GUI_ROOT")
    if env_root:
        return os.path.abspath(os.path.expanduser(env_root))
    if (SOURCE_ROOT / "pyproject.toml").exists() or (
        SOURCE_ROOT / "data_root"
    ).exists():
        return str(SOURCE_ROOT)
    return os.getcwd()


REPO_ROOT = _project_root()
DATA_ROOT = os.environ.get("SAM3_GUI_DATA_ROOT", os.path.join(REPO_ROOT, "data_root"))
RUNTIME_DIR = os.environ.get("SAM3_GUI_RUNTIME_DIR", os.path.join(REPO_ROOT, "runtime"))
OUTPUT_DIR = os.path.join(RUNTIME_DIR, "output")


def image_sequence_dir(name: str) -> str:
    return os.path.join(DATA_ROOT, "images", name)


def mask_sequence_dir(name: str) -> str:
    return os.path.join(DATA_ROOT, "masks", name)


def ensure_output_dir(*parts: str) -> str:
    path = os.path.join(OUTPUT_DIR, *parts)
    os.makedirs(path, exist_ok=True)
    return path
