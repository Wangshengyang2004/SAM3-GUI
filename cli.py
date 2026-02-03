import argparse
import sys

from loguru import logger as guru

from config import (
    DEFAULT_IMG_NAME,
    DEFAULT_MASK_NAME,
    DEFAULT_PORT,
    DEFAULT_SERVER_NAME,
    DEFAULT_VID_NAME,
    default_checkpoint_candidates,
    default_root_dir,
    resolve_checkpoint_path,
)
from mask_app import make_demo


def parse_gpus(gpus_arg):
    if gpus_arg is None:
        return None
    return [int(gpu.strip()) for gpu in gpus_arg.split(",") if gpu.strip()]


def build_parser(script_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--server_name", type=str, default=DEFAULT_SERVER_NAME)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--root_dir", type=str, default=default_root_dir(script_path))
    parser.add_argument("--vid_name", type=str, default=DEFAULT_VID_NAME)
    parser.add_argument("--img_name", type=str, default=DEFAULT_IMG_NAME)
    parser.add_argument("--mask_name", type=str, default=DEFAULT_MASK_NAME)
    parser.add_argument("--gpus", type=str, default=None)
    return parser


def main(argv=None):
    parser = build_parser(__file__)
    args = parser.parse_args(argv)

    checkpoint_path = resolve_checkpoint_path(
        args.checkpoint_path,
        default_checkpoint_candidates(__file__),
        logger=guru,
    )
    if not checkpoint_path:
        guru.error(
            "No checkpoint found. Set --checkpoint_path or place sam3.pt in a default path (see config.default_checkpoint_candidates)."
        )
        sys.exit(1)
    gpus_to_use = parse_gpus(args.gpus)
    if gpus_to_use is not None:
        guru.info(f"Using GPUs: {gpus_to_use}")

    demo = make_demo(
        args.root_dir,
        checkpoint_path,
        gpus_to_use,
        args.vid_name,
        args.img_name,
        args.mask_name,
    )
    share = args.share
    port = args.port
    for attempt in range(21):
        try:
            demo.launch(
                server_name=args.server_name,
                server_port=port,
                share=share,
                show_error=True,
            )
            break
        except OSError as e:
            if attempt < 20 and ("empty port" in str(e) or "port" in str(e).lower()):
                guru.warning(f"Port {port} in use, trying {port + 1}")
                port += 1
            else:
                raise


if __name__ == "__main__":
    main()
