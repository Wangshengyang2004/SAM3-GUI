import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import gradio as gr

from sam3_gui.utils import (
    first_or_none,
    frame_dir_path,
    get_downsampling_choices,
    get_video_duration,
    get_video_resolution,
    image_file_path,
    list_image_files,
    list_image_folders,
    list_video_files,
    load_rgb_image,
    mask_dir_path,
)


def toggle_video_prompt_type(prompt_type):
    return (
        gr.update(visible=(prompt_type == "Text")),
        gr.update(visible=(prompt_type == "Points")),
        gr.update(visible=(prompt_type == "Box")),
    )


def refresh_video_sources(root_dir, vid_name, img_name):
    video_root = os.path.join(root_dir, vid_name)
    image_root = os.path.join(root_dir, img_name)
    videos = list_video_files(video_root)
    frame_folders = list_image_folders(image_root)
    selected_video = first_or_none(videos)
    selected_seq = first_or_none(frame_folders)
    video_path = os.path.join(video_root, selected_video) if selected_video else None
    selected_path = frame_dir_path(root_dir, img_name, selected_seq)
    message = f"Found {len(videos)} video(s) and {len(frame_folders)} frame folder(s)."
    return (
        gr.update(choices=videos, value=selected_video),
        video_path,
        gr.update(choices=frame_folders, value=selected_seq),
        selected_seq,
        selected_path,
        message,
    )


def select_video(root_dir, seq_file, vid_name, img_name):
    if not seq_file:
        return None, None, gr.update(value=None)
    seq_name = os.path.splitext(seq_file)[0]
    vid_path = os.path.join(root_dir, vid_name, seq_file)
    frame_folders = list_image_folders(os.path.join(root_dir, img_name))
    selected_folder = seq_name if seq_name in frame_folders else None
    return seq_name, vid_path, gr.update(value=selected_folder)


def select_video_with_metadata(root_dir, seq_file, vid_name, img_name):
    """Select video and return metadata for UI updates."""
    if not seq_file:
        return (None, None, gr.update(value=None), 10.0, ["Original"], "Original")

    seq_name = os.path.splitext(seq_file)[0]
    vid_path = os.path.join(root_dir, vid_name, seq_file)

    frame_folders = list_image_folders(os.path.join(root_dir, img_name))
    selected_folder = seq_name if seq_name in frame_folders else None

    duration = get_video_duration(vid_path)
    if duration is None:
        duration = 10.0

    resolution = get_video_resolution(vid_path)
    choices, default_choice = get_downsampling_choices(resolution)

    return (
        seq_name,
        vid_path,
        gr.update(value=selected_folder),
        duration,
        choices,
        default_choice,
    )


def _is_within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _as_finite_number(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _target_frame_height(resolution, downsampling_choice):
    if not isinstance(resolution, (tuple, list)) or len(resolution) != 2:
        raise ValueError("Video resolution could not be detected")

    width = _as_finite_number(resolution[0], "Video width")
    original_height = _as_finite_number(resolution[1], "Video height")
    if (
        width <= 0
        or original_height <= 0
        or not width.is_integer()
        or not original_height.is_integer()
    ):
        raise ValueError("Video resolution must contain positive integer dimensions")

    divisors = {
        "Half": 2,
        "Quarter": 4,
        "Sixth": 6,
        "Eighth": 8,
        "Sixteenth": 16,
    }
    if downsampling_choice == "Original" or (
        isinstance(downsampling_choice, str)
        and downsampling_choice.startswith("Original (")
    ):
        divisor = 1
    else:
        divisor = divisors.get(downsampling_choice)
    if divisor is None:
        raise ValueError(f"Invalid downsampling choice: {downsampling_choice}")

    height = int(original_height) // divisor
    if height <= 0:
        raise ValueError("Output resolution must be at least one pixel high")
    return height


def _replace_frame_directory(temp_dir, out_dir, img_root):
    backup_container = None
    backup_dir = None
    if out_dir.exists():
        backup_container = Path(
            tempfile.mkdtemp(prefix=f".{out_dir.name}.backup-", dir=img_root)
        )
        backup_dir = backup_container / "previous"
        try:
            os.replace(out_dir, backup_dir)
        except Exception:
            shutil.rmtree(backup_container)
            raise

    try:
        os.replace(temp_dir, out_dir)
    except Exception:
        if backup_dir is not None and backup_dir.exists():
            os.replace(backup_dir, out_dir)
        raise
    finally:
        if backup_container is not None and backup_container.exists():
            shutil.rmtree(backup_container)


def extract_video_frames(
    root_dir, vid_file, start, end, fps, downsampling_choice, vid_name, img_name
):
    if not vid_file:
        return None, None, gr.update(), "Please select a video first"

    if (
        not isinstance(vid_file, str)
        or vid_file in {".", ".."}
        or "\x00" in vid_file
        or "/" in vid_file
        or "\\" in vid_file
        or os.path.basename(vid_file) != vid_file
    ):
        return None, None, gr.update(), "Invalid video file: expected a basename"

    seq_name = os.path.splitext(vid_file)[0]
    vid_root = (Path(root_dir).expanduser() / vid_name).resolve()
    img_root = (Path(root_dir).expanduser() / img_name).resolve()
    vid_path = (vid_root / vid_file).resolve()
    out_dir = img_root / seq_name
    resolved_out_dir = out_dir.resolve()

    if not _is_within(vid_path, vid_root):
        return (
            seq_name,
            None,
            gr.update(),
            "Invalid video path: outside configured video root",
        )
    if not _is_within(resolved_out_dir, img_root):
        return (
            seq_name,
            None,
            gr.update(),
            "Invalid output path: outside configured image root",
        )
    if out_dir.is_symlink():
        return (
            seq_name,
            None,
            gr.update(),
            "Invalid output path: symbolic links are not allowed",
        )
    if not vid_path.is_file():
        return seq_name, None, gr.update(), f"Video file not found: {vid_path}"
    if out_dir.exists() and not out_dir.is_dir():
        return seq_name, None, gr.update(), f"Output path is not a directory: {out_dir}"

    try:
        start_value = _as_finite_number(start, "Start")
        end_value = _as_finite_number(end, "End")
        fps_value = _as_finite_number(fps, "FPS")
        if start_value < 0:
            raise ValueError("Start must be non-negative")
        if end_value <= start_value:
            raise ValueError("End must be greater than start")
        if fps_value <= 0:
            raise ValueError("FPS must be positive")
        height = _target_frame_height(
            get_video_resolution(str(vid_path)), downsampling_choice
        )
    except (OSError, ValueError) as exc:
        return seq_name, None, gr.update(), f"Invalid extraction parameters: {exc}"

    img_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{seq_name}.tmp-", dir=img_root))

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_value),
        "-to",
        str(end_value),
        "-i",
        str(vid_path),
        "-vf",
        f"scale=-1:{height},fps={fps_value}",
        "-q:v",
        "2",  # High quality JPEG (1-31, lower is better)
        str(temp_dir / "%05d.jpg"),
    ]

    try:
        subprocess.run(cmd, check=True)
        if not any(
            path.is_file() and path.suffix.lower() == ".jpg"
            for path in temp_dir.iterdir()
        ):
            raise RuntimeError("ffmpeg produced no frames")
        _replace_frame_directory(temp_dir, out_dir, img_root)
    except FileNotFoundError:
        return (
            seq_name,
            None,
            gr.update(),
            "Failed to extract frames: ffmpeg is not installed or not on PATH",
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        return seq_name, None, gr.update(), f"Failed to extract frames: {exc}"
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    new_dirs = list_image_folders(str(img_root))
    return (
        seq_name,
        str(out_dir),
        gr.update(choices=new_dirs, value=seq_name if seq_name in new_dirs else None),
        f"Extracted frames to {out_dir}",
    )


def load_video_frames(root_dir, seq_name, img_name, video_handler, mask_name=None):
    # Gradio 6.x requires maximum > minimum for Slider initialization.
    empty_slider = gr.Slider(minimum=0, maximum=1, value=0, step=1)
    if not seq_name:
        return None, None, empty_slider, None, None, "Please select a frame folder"

    img_dir = frame_dir_path(root_dir, img_name, seq_name)
    if not os.path.isdir(img_dir):
        return (
            seq_name,
            None,
            empty_slider,
            None,
            None,
            f"Frame folder not found: {img_dir}",
        )

    try:
        num_imgs = video_handler.set_img_dir(img_dir)
    except Exception as e:
        return (
            seq_name,
            img_dir,
            empty_slider,
            None,
            None,
            f"Failed to load frames: {e}",
        )
    first_frame = video_handler.set_input_image(0) if num_imgs > 0 else None
    slider = gr.Slider(minimum=0, maximum=max(0, num_imgs - 1), value=0, step=1)
    # Also calculate and return the mask save path
    mask_path = mask_dir_path(root_dir, mask_name, seq_name) if mask_name else None
    return (
        seq_name,
        img_dir,
        slider,
        first_frame,
        mask_path,
        f"Loaded {num_imgs} frames from {seq_name}. Ready!",
    )


def update_mask_save_path(root_dir, seq_name, mask_name):
    return mask_dir_path(root_dir, mask_name, seq_name)


def refresh_image_lists(root_dir, img_name):
    image_root = os.path.join(root_dir, img_name)
    folders = list_image_folders(image_root)
    first_folder = first_or_none(folders)
    files = (
        list_image_files(os.path.join(image_root, first_folder)) if first_folder else []
    )
    first_file = first_or_none(files)
    selected_path = image_file_path(root_dir, img_name, first_folder, first_file)
    message = f"Found {len(folders)} image folder(s)."
    return (
        gr.update(choices=folders, value=first_folder),
        gr.update(choices=files, value=first_file),
        selected_path,
        message,
    )


def select_image_folder(root_dir, folder_name, img_name):
    if not folder_name:
        return gr.update(choices=[], value=None), None
    files = list_image_files(os.path.join(root_dir, img_name, folder_name))
    first_file = first_or_none(files)
    selected_path = image_file_path(root_dir, img_name, folder_name, first_file)
    return gr.update(choices=files, value=first_file), selected_path


def load_image_from_folder(root_dir, folder_name, file_name, img_name, image_handler):
    file_path = image_file_path(root_dir, img_name, folder_name, file_name)
    if not file_path:
        return None, None, "Please select a folder and image file"
    if not os.path.exists(file_path):
        return None, None, f"File not found: {file_path}"
    img = load_rgb_image(file_path)
    result_img, message = image_handler.set_image(img, resource_path=file_path)
    return result_img, result_img, message


def update_selected_image_path(root_dir, folder_name, file_name, img_name):
    return image_file_path(root_dir, img_name, folder_name, file_name)


def toggle_image_mode(mode):
    return (
        gr.update(visible=(mode == "Find All")),
        gr.update(visible=(mode == "Box")),
        gr.update(visible=(mode == "Point")),
    )
