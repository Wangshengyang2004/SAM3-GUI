import os
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_DIR.parent


def project_root() -> Path:
    env_root = os.environ.get("SAM3_GUI_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    if (SOURCE_ROOT / "pyproject.toml").exists() or (
        SOURCE_ROOT / "data_root"
    ).exists():
        return SOURCE_ROOT
    return Path.cwd()


RUNTIME_DIR = Path(
    os.environ.get("SAM3_GUI_RUNTIME_DIR", project_root() / "runtime")
).expanduser()
RUNTIME_OUTPUT_DIR = RUNTIME_DIR / "output"
RUNTIME_LOG_DIR = RUNTIME_DIR / "logs"


def runtime_output_path(*parts: str) -> str:
    path = RUNTIME_OUTPUT_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)
