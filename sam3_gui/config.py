import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_module
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from sam3_gui.sam31_constants import SAM31_CHECKPOINT_NAME, SAM31_HF_REPO


@dataclass
class ServerConfig:
    port: int = 8890
    name: str = "127.0.0.1"


@dataclass
class DataConfig:
    root_dir: str = "data_root"
    vid_name: str = "videos"
    img_name: str = "images"
    mask_name: str = "masks"


@dataclass
class SamConfig:
    checkpoint_path: str | None = None
    checkpoint_candidates: list[str] = field(
        default_factory=lambda: [f"~/sam3/model/{SAM31_CHECKPOINT_NAME}"]
    )
    gpus: list[int] = field(default_factory=list)
    use_fa3: bool = False


@dataclass
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sam: SamConfig = field(default_factory=SamConfig)


DEFAULT_CONFIG_NAME = "config"
DEFAULT_PORT = ServerConfig.port
DEFAULT_SERVER_NAME = ServerConfig.name
DEFAULT_VID_NAME = DataConfig.vid_name
DEFAULT_IMG_NAME = DataConfig.img_name
DEFAULT_MASK_NAME = DataConfig.mask_name
DEFAULT_API_PREFIX = "/api"
CONFIG_MODULE = "sam3_gui.conf"


def repo_root(script_path: str) -> str:
    env_root = os.environ.get("SAM3_GUI_ROOT")
    if env_root:
        return os.path.abspath(os.path.expanduser(env_root))

    script_dir = Path(script_path).resolve().parent
    if (script_dir / "pyproject.toml").exists() or (script_dir / "data_root").exists():
        return str(script_dir)
    return os.getcwd()


def default_root_dir(script_path: str) -> str:
    return os.path.join(repo_root(script_path), DataConfig.root_dir)


def _register_config_schema() -> None:
    store = ConfigStore.instance()
    store.store(name="config_schema", node=AppConfig)


def _resolve_repo_path(path: str, script_path: str) -> str:
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    return os.path.join(repo_root(script_path), expanded)


def default_checkpoint_candidates(script_path: str) -> list[str]:
    cfg = load_app_config(script_path)
    return cfg.sam.checkpoint_candidates


def load_app_config(
    script_path: str,
    overrides: list[str] | None = None,
    config_name: str = DEFAULT_CONFIG_NAME,
) -> AppConfig:
    _register_config_schema()
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_module(
        version_base="1.3", config_module=CONFIG_MODULE, job_name="sam3_gui"
    ):
        raw = compose(config_name=config_name, overrides=overrides or [])

    cfg = OmegaConf.merge(OmegaConf.structured(AppConfig), raw)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    app_config = OmegaConf.to_object(OmegaConf.create(resolved))
    if not isinstance(app_config, AppConfig):
        app_config = OmegaConf.to_object(
            OmegaConf.merge(OmegaConf.structured(AppConfig), app_config)
        )

    app_config.data.root_dir = _resolve_repo_path(app_config.data.root_dir, script_path)
    app_config.sam.checkpoint_candidates = [
        _resolve_repo_path(candidate, script_path)
        for candidate in app_config.sam.checkpoint_candidates
    ]
    return app_config


def resolve_checkpoint_path(explicit_checkpoint_path, candidates, logger=None):
    if explicit_checkpoint_path:
        return str(Path(explicit_checkpoint_path).expanduser().resolve())

    for candidate in candidates:
        if os.path.exists(candidate):
            if logger is not None:
                logger.info(f"Using default checkpoint: {candidate}")
            return candidate
    if logger is not None:
        logger.info(
            f"No local SAM 3.1 checkpoint found; SAM3 will download {SAM31_HF_REPO} on first load."
        )
    return None


def legacy_args_to_overrides(args: Any) -> list[str]:
    overrides = []
    if getattr(args, "port", None) is not None:
        overrides.append(f"server.port={args.port}")
    if getattr(args, "server_name", None) is not None:
        overrides.append(f"server.name={args.server_name}")
    if getattr(args, "root_dir", None) is not None:
        overrides.append(f"data.root_dir={args.root_dir}")
    if getattr(args, "vid_name", None) is not None:
        overrides.append(f"data.vid_name={args.vid_name}")
    if getattr(args, "img_name", None) is not None:
        overrides.append(f"data.img_name={args.img_name}")
    if getattr(args, "mask_name", None) is not None:
        overrides.append(f"data.mask_name={args.mask_name}")
    if getattr(args, "checkpoint_path", None) is not None:
        overrides.append(f"sam.checkpoint_path={args.checkpoint_path}")
    if getattr(args, "gpus", None) is not None:
        gpus = [int(gpu.strip()) for gpu in args.gpus.split(",") if gpu.strip()]
        overrides.append(f"sam.gpus={gpus}")
    if getattr(args, "use_fa3", False):
        overrides.append("sam.use_fa3=true")
    return overrides
