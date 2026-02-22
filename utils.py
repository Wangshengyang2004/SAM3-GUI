import colorsys
import json
import os
import subprocess

import cv2
import imageio.v2 as iio
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def isimage(filename):
    return os.path.splitext(filename.lower())[1] in IMAGE_EXTS


def listdir(path):
    if not path or not os.path.isdir(path):
        return []
    return sorted(os.listdir(path))


def first_or_none(items):
    return items[0] if items else None


def list_video_files(video_dir):
    return [
        name
        for name in listdir(video_dir)
        if os.path.isfile(os.path.join(video_dir, name))
        and os.path.splitext(name.lower())[1] in VIDEO_EXTS
    ]


def list_image_files(image_dir):
    return [
        name
        for name in listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, name)) and isimage(name)
    ]


def list_image_folders(img_root):
    return [
        name
        for name in listdir(img_root)
        if os.path.isdir(os.path.join(img_root, name))
        and list_image_files(os.path.join(img_root, name))
    ]


def frame_dir_path(root_dir, img_name, seq_name):
    if not seq_name:
        return None
    return os.path.join(root_dir, img_name, seq_name)


def mask_dir_path(root_dir, mask_name, seq_name):
    if not seq_name:
        return None
    return os.path.join(root_dir, mask_name, seq_name)


def image_file_path(root_dir, img_name, folder_name, file_name):
    if not folder_name or not file_name:
        return None
    return os.path.join(root_dir, img_name, folder_name, file_name)


def load_rgb_image(path):
    img = iio.imread(path)
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        return img[:, :, :3]
    return img


def to_numpy(data, as_float: bool = False):
    if hasattr(data, "detach"):
        data = data.detach()
    if as_float and hasattr(data, "float"):
        data = data.float()
    if hasattr(data, "cpu"):
        data = data.cpu()
    if hasattr(data, "numpy"):
        return data.numpy()
    if as_float:
        return np.asarray(data, dtype=np.float32)
    return np.asarray(data)


def draw_points(img, points, labels, point_frames=None, current_frame=None):
    """Draw points on image.

    Args:
        img: Image to draw on
        points: List of (x, y) coordinates
        labels: List of labels (1.0 for positive, 0.0 for negative)
        point_frames: Optional list of frame indices for each point
        current_frame: If provided with point_frames, only draw points for this frame
    """
    out = img.copy()
    for i, (p, label) in enumerate(zip(points, labels)):
        # Filter by frame if frame information is provided
        if point_frames is not None and current_frame is not None:
            if i < len(point_frames) and point_frames[i] != current_frame:
                continue  # Skip points from other frames
        x, y = int(p[0]), int(p[1])
        color = (0, 255, 0) if label == 1.0 else (255, 0, 0)
        out = cv2.circle(out, (x, y), 10, color, -1)
    return out


def get_hls_palette(n_colors: int, lightness: float = 0.5, saturation: float = 0.7) -> np.ndarray:
    hues = np.linspace(0, 1, int(n_colors) + 1)[1:-1]
    palette = [(0.0, 0.0, 0.0)] + [
        colorsys.hls_to_rgb(h_i, lightness, saturation) for h_i in hues
    ]
    return (255 * np.asarray(palette)).astype("uint8")


def compose_img_mask(img, color_mask, fac: float = 0.5):
    out_f = fac * img / 255 + (1 - fac) * color_mask / 255
    return (255 * out_f).astype("uint8")


def colorize_masks(images, index_masks, fac: float = 0.5):
    if not images or not index_masks:
        return [], []
    max_idx = max(mask.max() for mask in index_masks)
    palette = get_hls_palette(max_idx + 1)
    color_masks = []
    out_frames = []
    for img, mask in zip(images, index_masks):
        clr_mask = palette[mask.astype("int")]
        color_masks.append(clr_mask)
        out_f = fac * img / 255 + (1 - fac) * clr_mask / 255
        out_frames.append((255 * out_f).astype("uint8"))
    return out_frames, color_masks


def get_video_resolution(video_path):
    """Get the original resolution of a video file.

    Args:
        video_path: Path to the video file

    Returns:
        tuple: (width, height) or None if detection fails
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        if data.get("streams"):
            width = data["streams"][0]["width"]
            height = data["streams"][0]["height"]
            return (width, height)
    except Exception as e:
        print(f"Failed to get video resolution: {e}")
    return None


def get_video_duration(video_path):
    """Get the duration of a video file in seconds.

    Args:
        video_path: Path to the video file

    Returns:
        float: Duration in seconds, or None if detection fails
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        return duration
    except Exception as e:
        print(f"Failed to get video duration: {e}")
        return None


def get_downsampling_choices(resolution):
    """Generate downsampling choices for a video.

    Args:
        resolution: Tuple of (width, height) or None

    Returns:
        tuple: (choices_list, default_value)
    """
    if resolution:
        width, height = resolution
        original_label = f"Original ({width} * {height})"
    else:
        original_label = "Original (auto-detect)"

    choices = [
        original_label,
        "Half",
        "Quarter",
        "Sixth",
        "Eighth",
        "Sixteenth"
    ]

    return choices, original_label
