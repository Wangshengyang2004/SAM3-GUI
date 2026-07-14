import argparse

from loguru import logger as guru

from sam3_gui.api_app import create_app
from sam3_gui.config import (
    legacy_args_to_overrides,
    load_app_config,
    resolve_checkpoint_path,
)


def build_parser(script_path=None):
    parser = argparse.ArgumentParser(
        description=(
            "Launch SAM3-GUI. Prefer Hydra overrides such as "
            "`server.port=8891 sam.use_fa3=true data.root_dir=data_root`."
        )
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Deprecated; use server.port=<port>"
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default=None,
        help="Deprecated; use server.name=<host>",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Deprecated; use sam.checkpoint_path=<path>",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help="Deprecated; use data.root_dir=<path>",
    )
    parser.add_argument(
        "--vid_name",
        type=str,
        default=None,
        help="Deprecated; use data.vid_name=<name>",
    )
    parser.add_argument(
        "--img_name",
        type=str,
        default=None,
        help="Deprecated; use data.img_name=<name>",
    )
    parser.add_argument(
        "--mask_name",
        type=str,
        default=None,
        help="Deprecated; use data.mask_name=<name>",
    )
    parser.add_argument(
        "--gpus", type=str, default=None, help="Deprecated; use sam.gpus=[0,1]"
    )
    parser.add_argument(
        "--use_fa3",
        action="store_true",
        help="Enable FlashAttention 3 in the SAM 3.1 backend",
    )
    return parser


def main(argv=None):
    parser = build_parser(__file__)
    args, hydra_overrides = parser.parse_known_args(argv)
    cfg = load_app_config(__file__, [*hydra_overrides, *legacy_args_to_overrides(args)])

    checkpoint_path = resolve_checkpoint_path(
        cfg.sam.checkpoint_path,
        cfg.sam.checkpoint_candidates,
        logger=guru,
    )
    if cfg.sam.gpus:
        guru.info(f"Using GPUs: {cfg.sam.gpus}")

    app = create_app(
        cfg.data.root_dir,
        checkpoint_path,
        cfg.sam.gpus,
        cfg.data.vid_name,
        cfg.data.img_name,
        cfg.data.mask_name,
        use_fa3=cfg.sam.use_fa3,
    )

    import uvicorn

    uvicorn.run(
        app,
        host=cfg.server.name,
        port=cfg.server.port,
    )


if __name__ == "__main__":
    main()
