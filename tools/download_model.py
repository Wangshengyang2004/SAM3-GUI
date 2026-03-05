#!/usr/bin/env python3
"""
Download SAM3 model from ModelScope.

Usage:
    python download_model.py [--output_dir PATH]
"""
import argparse
import os
from pathlib import Path


def download_sam3_model(output_dir=None):
    """Download SAM3 model from ModelScope."""
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        print("Error: modelscope is not installed.")
        print("Please install it with: pip install modelscope")
        return False

    # Default output directory
    if output_dir is None:
        output_dir = os.path.expanduser("~/sam3/model")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading SAM3 model from ModelScope...")
    print(f"Output directory: {output_dir}")

    try:
        # Download the model repository
        model_dir = snapshot_download(
            model_id='facebook/sam3',
            cache_dir=str(output_dir.parent),
        )

        print(f"\nModel downloaded to: {model_dir}")

        # Find the .pt file
        pt_files = list(Path(model_dir).glob("*.pt"))
        if pt_files:
            pt_file = pt_files[0]
            target_path = output_dir / "sam3.pt"

            # Create symlink or copy
            if target_path.exists():
                print(f"\nTarget file already exists: {target_path}")
            else:
                try:
                    target_path.symlink_to(pt_file)
                    print(f"\nCreated symlink: {target_path} -> {pt_file}")
                except OSError:
                    # Fallback to copy if symlink fails
                    import shutil
                    shutil.copy2(pt_file, target_path)
                    print(f"\nCopied model to: {target_path}")

            print(f"\nModel ready at: {target_path}")
            print(f"\nYou can now run the GUI with:")
            print(f"  python cli.py --checkpoint_path {target_path}")
            return True
        else:
            print(f"\nWarning: No .pt file found in {model_dir}")
            print(f"Please check the downloaded files manually.")
            return False

    except Exception as e:
        print(f"\nError downloading model: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download SAM3 model from ModelScope"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for the model (default: ~/sam3/model)"
    )
    args = parser.parse_args()

    success = download_sam3_model(args.output_dir)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
