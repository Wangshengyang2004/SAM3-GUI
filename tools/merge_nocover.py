import torch as th
from dataclasses import dataclass
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import argparse
import glob
import re
from concurrent.futures import ThreadPoolExecutor

from tools.paths import (
    OUTPUT_DIR,
    ensure_output_dir,
    image_sequence_dir,
    mask_sequence_dir,
)

inverse_transparency = False


def detect_frame_count(name):
    """
    Auto-detect the number of frames in a video directory.
    Returns the count of frames (assumes consecutive numbering from 1).
    """
    base_path = image_sequence_dir(name)
    if not os.path.exists(base_path):
        raise ValueError(f"Video directory not found: {base_path}")

    # Find all jpg files
    files = glob.glob(f"{base_path}/*.jpg")
    if not files:
        raise ValueError(f"No .jpg files found in {base_path}")

    # Extract numbers from filenames (format: 00001.jpg)
    numbers = []
    for f in files:
        match = re.search(r"(\d+)\.jpg$", os.path.basename(f))
        if match:
            numbers.append(int(match.group(1)))

    if not numbers:
        raise ValueError(f"Could not parse frame numbers in {base_path}")

    return max(numbers)


def get_video_frames(video_path, indices):
    """
    Load frames from image directory instead of video file
    video_path: path relative to data_root/images/
    """
    full_path = image_sequence_dir(video_path)

    frames = []
    for idx in indices:
        img_path = f"{full_path}/{str(idx).zfill(5)}.jpg"
        frame = cv2.imread(img_path)
        if frame is not None:
            frames.append(frame)
        else:
            print(f"Warning: Could not read {img_path}")

    print(f"Loaded {len(frames)} frames from {full_path}")
    return frames


def get_masks(name, indices):
    """
    Load masks from data_root/masks directory
    name: path relative to data_root/masks/
    """
    base_path = mask_sequence_dir(name)
    masks = []
    for i in indices:
        mask = np.load(os.path.join(base_path, f"{str(i).zfill(5)}.npy"))
        masks.append(mask)

    print(f"Loaded {len(masks)} masks from {base_path}")
    return masks


@dataclass
class Config:
    start: int = 0
    interval: int = 1
    end: int = 10
    name: str = ""
    background: str = None


def merge(cfg: Config):
    # Create output directory
    ensure_output_dir()

    indices = np.arange(cfg.start, cfg.end, cfg.interval, dtype=int)

    base_img_path = image_sequence_dir(cfg.name)
    base_mask_path = mask_sequence_dir(cfg.name)

    image_paths = [f"{base_img_path}/{str(i).zfill(5)}.jpg" for i in indices]
    mask_paths = [f"{base_mask_path}/{str(i).zfill(5)}.npy" for i in indices]

    # Load background if specified, otherwise use None
    background = None
    if cfg.background:
        background = get_background(cfg.name, cfg.background)

    img = merge_masked_images(image_paths, mask_paths, background=background)

    # Save output
    name = cfg.name.split("/")[-1]
    output_path = os.path.join(OUTPUT_DIR, f"{name}_merged_nocover.png")
    cv2.imwrite(output_path, img)
    print(f"Saved merged image to {output_path}")


def get_background(name, back):
    if back is None:
        return None
    # Try loading from images directory
    img = cv2.imread(os.path.join(image_sequence_dir(name), back))
    if img is None:
        print(f"Warning: Could not load background {back}")
    return img


def _load_image(path):
    """Helper to load and preprocess image"""
    img = cv2.imread(path)
    if img is None:
        return None
    if img.shape[-1] == 3:
        img = np.dstack((img, np.full(img.shape[:2], 255)))
    return img.astype(np.float32)


def _load_mask(path):
    """Helper to load mask"""
    if not os.path.exists(path):
        return None
    mask = np.load(path)
    return np.clip(mask, None, 1)


def find_last_valid_mask(masks):
    """
    Find the last index where mask has valid (non-zero) content.
    Returns the index of last valid mask.
    """
    for i in range(len(masks) - 1, -1, -1):
        if (
            masks[i] is not None and masks[i].max() > 0.01
        ):  # Has significant mask content
            return i
    return len(masks) - 1  # Default to last if all are empty


def merge_masked_images(image_paths, mask_paths, background=None, chunk_size=32):
    """
    GPU-accelerated image merging with chunked loading to manage memory.
    Transparency gradient: first valid mask frame at 80%, last at 100%.
    """
    if len(image_paths) != len(mask_paths):
        raise ValueError("Number of images and masks must be the same")

    if inverse_transparency:
        image_paths = list(reversed(image_paths))
        mask_paths = list(reversed(mask_paths))

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # Load last image for background (single read)
    last_img = _load_image(image_paths[-1])
    if last_img is None:
        raise ValueError(f"Could not read {image_paths[-1]}")

    if background is None:
        background = last_img
    else:
        if background.shape[-1] == 3:
            background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)
        background = background.astype(np.float32)

    background_t = th.from_numpy(background).to(device)
    combined = background_t.clone()

    mix_factor = 1
    total_frames = len(image_paths)

    print(f"Processing {total_frames} frames on GPU (chunk size: {chunk_size})...")

    # First pass: find mask range (first and last valid mask)
    # We need to check a sample of masks to find the range
    print("Detecting mask range...")
    sample_indices = list(range(0, total_frames, max(1, total_frames // 100)))
    sample_masks = [_load_mask(mask_paths[i]) for i in sample_indices]
    valid_sample_indices = [
        i
        for i, m in zip(sample_indices, sample_masks)
        if m is not None and m.max() > 0.01
    ]

    if valid_sample_indices:
        first_mask_idx = min(valid_sample_indices)
        last_mask_idx = max(valid_sample_indices)
    else:
        first_mask_idx = 0
        last_mask_idx = total_frames - 1

    print(f"Mask range: frame {first_mask_idx} to {last_mask_idx}")

    # Pre-compute transparency factors (80% to 100% over mask range)
    mask_range = last_mask_idx - first_mask_idx
    transparent_factors = []
    for idx in range(total_frames):
        if idx < first_mask_idx:
            trans = 0.8  # Before first mask
        elif idx > last_mask_idx:
            trans = 1.0  # After last mask
        else:
            # Linear interpolation from 0.8 to 1.0
            progress = (idx - first_mask_idx) / mask_range if mask_range > 0 else 1.0
            trans = 0.8 + 0.2 * progress
        transparent_factors.append(trans)

    # Process in chunks to manage memory
    for chunk_start in range(0, total_frames, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_frames)

        # Load chunk in parallel
        chunk_img_paths = image_paths[chunk_start:chunk_end]
        chunk_mask_paths = mask_paths[chunk_start:chunk_end]

        with ThreadPoolExecutor(max_workers=4) as executor:
            frames = list(executor.map(_load_image, chunk_img_paths))
            masks = list(executor.map(_load_mask, chunk_mask_paths))

        # Process chunk on GPU
        for i, (img, mask) in enumerate(zip(frames, masks)):
            if img is None or mask is None:
                continue

            global_idx = chunk_start + i
            img_t = th.from_numpy(img).to(device)
            mask_t = th.from_numpy(mask).to(device).unsqueeze(-1)

            # Compute mask: pixels where |bg*mask - img*mask| > 10 in any channel
            diff = th.abs(background_t * mask_t - img_t * mask_t)
            mask_binary = 1 - (diff <= 10).all(dim=-1, keepdim=True).float()
            mask_binary = mask_binary.expand(-1, -1, 4)

            # Blend with transparency gradient
            trans = transparent_factors[global_idx]
            new = mask_binary * mix_factor
            remain = 1 - new
            combined = (
                combined * remain
                + img_t * new * trans
                + background_t * new * (1 - trans)
            )

        if chunk_end % 50 == 0 or chunk_end == total_frames:
            print(f"Merged {chunk_end}/{total_frames} frames")

    # Synchronize and return
    if th.cuda.is_available():
        th.cuda.synchronize()

    result = combined.cpu().numpy()
    return np.clip(result, 0, 255).astype(np.uint8)


def save_frames(cfg):
    indices = np.arange(cfg.start, cfg.end, cfg.interval, dtype=int)
    frames = get_video_frames(f"{cfg.name}_frames", indices)
    # save frames in a folder
    frame_output_dir = ensure_output_dir("videos", f"{cfg.name}_frames")
    # remove all images end with png in the folder
    for f in os.listdir(frame_output_dir):
        if f.endswith(".png"):
            os.remove(os.path.join(frame_output_dir, f))
    for i, frame in zip(indices, frames):
        cv2.imwrite(os.path.join(frame_output_dir, f"{str(i).zfill(5)}.png"), frame)
    print(f"Saved frames to {frame_output_dir}/")
    return frames


def main():
    global inverse_transparency

    parser = argparse.ArgumentParser(
        description="Merge masked video frames into a single composite image"
    )
    parser.add_argument(
        "name",
        nargs="?",
        default="Sky_color",
        help="Video name (subdirectory in data_root/images and data_root/masks)",
    )
    parser.add_argument(
        "-s", "--start", type=int, default=1, help="Start frame index (default: 1)"
    )
    parser.add_argument(
        "-i", "--interval", type=int, default=1, help="Frame interval/step (default: 1)"
    )
    parser.add_argument(
        "-e",
        "--end",
        type=int,
        default=None,
        help="End frame index (default: auto-detect)",
    )
    parser.add_argument(
        "-b",
        "--background",
        type=str,
        default=None,
        help="Background image filename (optional)",
    )
    parser.add_argument(
        "-r",
        "--reverse",
        action="store_true",
        help="Reverse the order of frames (inverse_transparency)",
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Display the result with matplotlib after saving",
    )

    args = parser.parse_args()

    # Auto-detect end if not specified
    end = args.end
    if end is None:
        try:
            end = detect_frame_count(args.name)
            print(f"Auto-detected {end} frames for '{args.name}'")
        except ValueError as e:
            print(f"Error: {e}")
            print("Please specify --end manually")
            return

    inverse_transparency = args.reverse

    # Configure
    cfg = Config(
        start=args.start,
        interval=args.interval,
        end=end + 1,  # +1 because range is exclusive at end
        name=args.name,
        background=args.background,
    )

    print(f"Processing frames {cfg.start} to {end} with interval {cfg.interval}")
    merge(cfg)

    # Plot if requested
    if args.plot:
        output_path = os.path.join(
            OUTPUT_DIR, f"{args.name.split('/')[-1]}_merged_nocover.png"
        )
        if os.path.exists(output_path):
            img = cv2.imread(output_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.title(f"Merged: {args.name}")
            plt.axis("off")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
