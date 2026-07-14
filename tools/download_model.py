#!/usr/bin/env python3
"""Download the SAM 3.1 Object Multiplex checkpoint."""

import argparse
import importlib
import os
import shutil
from pathlib import Path

from sam3_gui.sam31_constants import (
    SAM31_CHECKPOINT_NAME,
    SAM31_HF_REPO,
    SAM31_MODELSCOPE_REPO,
)


DEFAULT_OUTPUT_DIR = "~/sam3/model"
SOURCE_HUGGINGFACE = "huggingface"
SOURCE_MODELSCOPE = "modelscope"
SOURCES = (SOURCE_HUGGINGFACE, SOURCE_MODELSCOPE)


def _prepare_output_dir(output_dir: str | os.PathLike[str] | None) -> Path:
    path = Path(os.path.expanduser(output_dir or DEFAULT_OUTPUT_DIR))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _print_success(checkpoint: str | os.PathLike[str]) -> None:
    print(f"\nSAM 3.1 checkpoint ready at: {checkpoint}")
    print("\nRun the GUI/API with:")
    print(f"  sam3-gui sam.checkpoint_path={checkpoint}")


def download_sam31_from_huggingface(output_dir=None) -> bool:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Error: huggingface_hub is not installed.")
        print("Install SAM3-GUI with: pip install -e .")
        return False

    output_path = _prepare_output_dir(output_dir)
    print(f"Downloading {SAM31_HF_REPO}/{SAM31_CHECKPOINT_NAME} from Hugging Face")
    print(f"Output directory: {output_path}")

    try:
        checkpoint = hf_hub_download(
            repo_id=SAM31_HF_REPO,
            filename=SAM31_CHECKPOINT_NAME,
            local_dir=str(output_path),
        )
        _print_success(checkpoint)
        return True
    except Exception as exc:
        print(f"\nError downloading SAM 3.1 checkpoint from Hugging Face: {exc}")
        print(
            "Make sure you have accepted access on Hugging Face and run `hf auth login`."
        )
        print(
            "If Hugging Face is unreachable, try: python -m tools.download_model --source modelscope"
        )
        return False


def _copy_modelscope_file_to_output(
    downloaded_path: str | os.PathLike[str], output_path: Path
) -> Path:
    source_path = Path(downloaded_path)
    if source_path.is_dir():
        source_path = source_path / SAM31_CHECKPOINT_NAME
    if not source_path.exists():
        raise FileNotFoundError(
            f"ModelScope download did not produce {SAM31_CHECKPOINT_NAME}: {source_path}"
        )
    checkpoint_path = output_path / SAM31_CHECKPOINT_NAME
    if source_path.resolve() != checkpoint_path.resolve():
        shutil.copy2(source_path, checkpoint_path)
    return checkpoint_path


def _import_modelscope_attr(module_name: str, attr_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def download_sam31_from_modelscope(
    output_dir=None, model_id: str | None = None
) -> bool:
    try:
        model_file_download = _import_modelscope_attr(
            "modelscope.hub.file_download", "model_file_download"
        )
    except ImportError:
        try:
            model_file_download = _import_modelscope_attr(
                "modelscope", "model_file_download"
            )
        except (AttributeError, ImportError):
            model_file_download = None
    except AttributeError:
        model_file_download = None

    if model_file_download is None:
        try:
            snapshot_download = _import_modelscope_attr(
                "modelscope", "snapshot_download"
            )
        except (AttributeError, ImportError):
            try:
                snapshot_download = _import_modelscope_attr(
                    "modelscope.hub.snapshot_download", "snapshot_download"
                )
            except (AttributeError, ImportError):
                print(
                    "Error: modelscope is not installed or does not expose a supported download API."
                )
                print("Install SAM3-GUI with: pip install -e .")
                print("Or install it directly with: pip install modelscope")
                return False
    else:
        snapshot_download = None

    output_path = _prepare_output_dir(output_dir)
    resolved_model_id = (
        model_id or os.environ.get("SAM31_MODELSCOPE_REPO") or SAM31_MODELSCOPE_REPO
    )
    print(f"Downloading {resolved_model_id}/{SAM31_CHECKPOINT_NAME} from ModelScope")
    print(f"Output directory: {output_path}")

    try:
        if model_file_download is not None:
            downloaded = model_file_download(
                model_id=resolved_model_id,
                file_path=SAM31_CHECKPOINT_NAME,
                local_dir=str(output_path),
            )
        else:
            downloaded = snapshot_download(
                model_id=resolved_model_id,
                local_dir=str(output_path),
                allow_file_pattern=SAM31_CHECKPOINT_NAME,
            )
        checkpoint = _copy_modelscope_file_to_output(downloaded, output_path)
        _print_success(checkpoint)
        return True
    except Exception as exc:
        print(f"\nError downloading SAM 3.1 checkpoint from ModelScope: {exc}")
        print("Verify the ModelScope repo contains sam3.1_multiplex.pt.")
        print(
            "If your mirror uses a different id, pass --modelscope_model_id <repo-id>."
        )
        return False


def download_sam31_model(
    output_dir=None,
    source: str = SOURCE_HUGGINGFACE,
    modelscope_model_id: str | None = None,
) -> bool:
    if source == SOURCE_HUGGINGFACE:
        return download_sam31_from_huggingface(output_dir)
    if source == SOURCE_MODELSCOPE:
        return download_sam31_from_modelscope(output_dir, modelscope_model_id)

    print(f"Error: unsupported checkpoint source: {source}")
    print(f"Choose one of: {', '.join(SOURCES)}")
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download the SAM 3.1 checkpoint")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=f"Output directory for the checkpoint (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--source",
        choices=SOURCES,
        default=SOURCE_HUGGINGFACE,
        help="Checkpoint source to use (default: huggingface)",
    )
    parser.add_argument(
        "--modelscope_model_id",
        type=str,
        default=None,
        help=f"ModelScope repo id (default: SAM31_MODELSCOPE_REPO or {SAM31_MODELSCOPE_REPO})",
    )
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    ok = download_sam31_model(
        output_dir=args.output_dir,
        source=args.source,
        modelscope_model_id=args.modelscope_model_id,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
