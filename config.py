import os


DEFAULT_PORT = 8890
DEFAULT_SERVER_NAME = "0.0.0.0"
DEFAULT_VID_NAME = "videos"
DEFAULT_IMG_NAME = "images"
DEFAULT_MASK_NAME = "masks"


def default_root_dir(script_path):
    script_dir = os.path.dirname(os.path.abspath(script_path))
    return os.path.join(script_dir, "data_root")


def default_checkpoint_candidates(script_path):
    script_dir = os.path.dirname(os.path.abspath(script_path))
    return [
        os.path.expanduser("~/sam3/model/sam3.pt"),
        os.path.join(os.path.dirname(script_dir), "dir", "sam3.pt"),
    ]


def resolve_checkpoint_path(explicit_checkpoint_path, candidates, logger=None):
    if explicit_checkpoint_path:
        return explicit_checkpoint_path

    for candidate in candidates:
        if os.path.exists(candidate):
            if logger is not None:
                logger.info(f"Using default checkpoint: {candidate}")
            return candidate
    return None
