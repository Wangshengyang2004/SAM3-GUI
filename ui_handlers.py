import datetime
import os
import shutil
import subprocess

import gradio as gr
from loguru import logger as guru

from utils import (
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
        return (
            None, None,
            gr.update(value=None),
            10.0,
            ["Original (auto-detect)"],
            "Original (auto-detect)"
        )

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
        default_choice
    )


def extract_video_frames(root_dir, vid_file, start, end, fps, downsampling_choice, vid_name, img_name):
    if not vid_file:
        return None, None, gr.update(), "Please select a video first"

    seq_name = os.path.splitext(vid_file)[0]
    vid_path = os.path.join(root_dir, vid_name, vid_file)
    out_dir = frame_dir_path(root_dir, img_name, seq_name)

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Calculate target height based on downsampling choice
    resolution = get_video_resolution(vid_path)
    if resolution:
        width, original_height = resolution
    else:
        width, original_height = None, 1024
        guru.warning("Failed to detect resolution, using default 1024")

    # Parse downsampling choice and calculate height
    if downsampling_choice.startswith("Original"):
        height = original_height
    elif downsampling_choice == "Half":
        height = original_height // 2
    elif downsampling_choice == "Quarter":
        height = original_height // 4
    elif downsampling_choice == "Sixth":
        height = original_height // 6
    elif downsampling_choice == "Eighth":
        height = original_height // 8
    elif downsampling_choice == "Sixteenth":
        height = original_height // 16
    else:
        height = original_height

    def to_hms(seconds):
        t = int(seconds)
        return datetime.time(t // 3600, (t % 3600) // 60, t % 60).strftime("%H:%M:%S")

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        to_hms(start),
        "-to",
        to_hms(end),
        "-i",
        vid_path,
        "-vf",
        f"scale=-1:{int(height)},fps={int(fps)}",
        "-q:v", "2",  # High quality JPEG (1-31, lower is better)
        f"{out_dir}/%05d.jpg",
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        return seq_name, None, gr.update(), f"Failed to extract frames: {exc}"

    new_dirs = list_image_folders(os.path.join(root_dir, img_name))
    return (
        seq_name,
        out_dir,
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
        return seq_name, None, empty_slider, None, None, f"Frame folder not found: {img_dir}"

    try:
        num_imgs = video_handler.set_img_dir(img_dir)
    except Exception as e:
        return seq_name, img_dir, empty_slider, None, None, f"Failed to load frames: {e}"
    first_frame = video_handler.set_input_image(0) if num_imgs > 0 else None
    slider = gr.Slider(minimum=0, maximum=max(0, num_imgs - 1), value=0, step=1)
    # Also calculate and return the mask save path
    mask_path = mask_dir_path(root_dir, mask_name, seq_name) if mask_name else None
    return seq_name, img_dir, slider, first_frame, mask_path, f"Loaded {num_imgs} frames from {seq_name}. Ready!"


def update_mask_save_path(root_dir, seq_name, mask_name):
    return mask_dir_path(root_dir, mask_name, seq_name)


def refresh_image_lists(root_dir, img_name):
    image_root = os.path.join(root_dir, img_name)
    folders = list_image_folders(image_root)
    first_folder = first_or_none(folders)
    files = list_image_files(os.path.join(image_root, first_folder)) if first_folder else []
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
    result_img, message = image_handler.set_image(img)
    return result_img, result_img, message


def update_selected_image_path(root_dir, folder_name, file_name, img_name):
    return image_file_path(root_dir, img_name, folder_name, file_name)


def toggle_image_mode(mode):
    return (
        gr.update(visible=(mode == "Find All")),
        gr.update(visible=(mode == "Box")),
        gr.update(visible=(mode == "Point")),
    )
