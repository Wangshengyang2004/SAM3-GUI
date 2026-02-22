import torch

# Enable tf32 for Ampere GPUs if any CUDA device supports it
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    for i in range(torch.cuda.device_count()):
        if torch.cuda.get_device_properties(i).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            break

import os
from functools import partial

import cv2
import gradio as gr
from loguru import logger as guru

from config import (
    DEFAULT_IMG_NAME,
    DEFAULT_MASK_NAME,
    DEFAULT_VID_NAME,
)

from image_handler import ImageModeHandler
from ui_handlers import (
    extract_video_frames,
    load_image_from_folder,
    load_video_frames,
    refresh_image_lists,
    refresh_video_sources,
    select_image_folder,
    select_video,
    toggle_image_mode,
    toggle_video_prompt_type,
    update_mask_save_path,
    update_selected_image_path,
)
from video_handler import VideoModeHandler
from utils import (
    compose_img_mask,
    draw_points,
    first_or_none,
    frame_dir_path,
    get_hls_palette,
    image_file_path,
    list_image_files,
    list_image_folders,
    list_video_files,
)

# Backward compatibility for older tests/integrations that import PromptGUI.
PromptGUI = VideoModeHandler


# =============================================================================
# Main Demo with Tabs
# =============================================================================

def make_demo(
    root_dir,
    checkpoint_path=None,
    gpus_to_use=None,
    vid_name: str = DEFAULT_VID_NAME,
    img_name: str = DEFAULT_IMG_NAME,
    mask_name: str = DEFAULT_MASK_NAME,
):
    # Initialize handlers (models are lazy-loaded)
    video_handler = VideoModeHandler(checkpoint_path=checkpoint_path, gpus_to_use=gpus_to_use)
    image_handler = ImageModeHandler(checkpoint_path=checkpoint_path)
    
    vid_root = os.path.join(root_dir, vid_name)
    img_root = os.path.join(root_dir, img_name)
    initial_videos = list_video_files(vid_root)
    initial_video = first_or_none(initial_videos)
    initial_frame_dirs = list_image_folders(img_root)
    initial_frame_dir = first_or_none(initial_frame_dirs)
    initial_image_folders = initial_frame_dirs
    initial_image_folder = first_or_none(initial_image_folders)
    initial_image_files = (
        list_image_files(os.path.join(img_root, initial_image_folder))
        if initial_image_folder else []
    )
    initial_image_file = first_or_none(initial_image_files)
    initial_image_path = image_file_path(root_dir, img_name, initial_image_folder, initial_image_file)
    
    with gr.Blocks(title="SAM3 Segmentation") as demo:
        gr.Markdown("# SAM3 Segmentation Tool")
        instruction = gr.Textbox("Select a mode (Video or Image) to get started.", label="Status", interactive=False)
        
        with gr.Tabs():
            # =================================================================
            # VIDEO TAB
            # =================================================================
            with gr.TabItem("Video Mode", id="video_tab"):
                gr.Markdown("### Video/Frame Sequence Segmentation with Tracking")
                
                with gr.Row():
                    vid_root_dir = gr.Text(root_dir, label="Root Directory")
                    vid_refresh_btn = gr.Button("Refresh Lists")
                    vid_seq_name = gr.Text(initial_frame_dir, label="Sequence Name", interactive=False)
                
                with gr.Row():
                    # Left: Source selection
                    with gr.Column(scale=1):
                        gr.Markdown("#### Source")
                        vid_files_field = gr.Dropdown(
                            label="Video Files",
                            choices=initial_videos,
                            value=initial_video,
                        )
                        vid_preview = gr.Video(
                            label="Video Preview",
                            value=os.path.join(vid_root, initial_video) if initial_video else None,
                            sources=[],  # Disable upload and webcam
                        )

                        with gr.Row():
                            vid_start = gr.Number(0, label="Start (s)")
                            vid_end = gr.Number(10, label="End (s)")
                            vid_fps = gr.Number(30, label="FPS")
                            vid_height = gr.Number(0, label="Height (0=auto)")
                        vid_extract_btn = gr.Button("Extract Frames")

                        gr.Markdown("---")
                        vid_img_dirs = gr.Dropdown(
                            label="Frame Folders",
                            choices=initial_frame_dirs,
                            value=initial_frame_dir,
                        )
                        vid_img_dir_path = gr.Text(
                            frame_dir_path(root_dir, img_name, initial_frame_dir),
                            label="Frame Directory",
                            interactive=False,
                        )
                        vid_load_frames_btn = gr.Button("Load Selected Frames", variant="primary")
                    
                    # Middle: Frame display and prompts
                    with gr.Column(scale=2):
                        gr.Markdown("#### Frame & Prompts")
                        vid_frame_slider = gr.Slider(label="Frame Index", minimum=0, maximum=0, value=0, step=1)
                        vid_input_frame = gr.Image(label="Input Frame", interactive=True, type="numpy", sources=[])
                        
                        vid_prompt_type = gr.Radio(
                            choices=["Text", "Points", "Box"],
                            value="Text",
                            label="Prompt Type",
                            interactive=True
                        )
                        
                        # Text prompt controls
                        with gr.Group(visible=True) as vid_text_group:
                            vid_text_input = gr.Textbox(label="Text Prompt", placeholder="e.g., 'person', 'car', 'dog'")
                            vid_text_btn = gr.Button("Detect with Text", variant="primary")
                        
                        # Point prompt controls
                        with gr.Group(visible=False) as vid_point_group:
                            with gr.Row():
                                vid_pos_btn = gr.Button("+ Positive")
                                vid_neg_btn = gr.Button("- Negative")
                                vid_clear_pts_btn = gr.Button("Clear Points")
                            vid_point_table = gr.DataFrame(
                                headers=["Frame", "X", "Y", "Type", "Obj ID", "Index"],
                                label="Added Points",
                                interactive=False,
                                value=[]
                            )
                            with gr.Row():
                                vid_remove_point_idx = gr.Number(value=0, label="Point Index to Remove", minimum=0, step=1)
                                vid_remove_point_btn = gr.Button("Remove Point by Index")

                        # Box prompt controls
                        with gr.Group(visible=False) as vid_box_group:
                            gr.Markdown("Draw a box on the frame above, then click 'Segment Box'")
                            vid_box_btn = gr.Button("Segment Box", variant="primary")
                    
                    # Right: Output
                    with gr.Column(scale=1):
                        gr.Markdown("#### Output")
                        vid_output_img = gr.Image(label="Segmentation Preview", interactive=True, type="numpy", sources=[])

                        gr.Markdown("---")
                        gr.Markdown("#### Object Management")
                        vid_obj_dropdown = gr.Dropdown(
                            choices=[],
                            label="Tracked Objects",
                            interactive=True
                        )
                        vid_remove_obj_btn = gr.Button("Remove Selected Object")

                        gr.Markdown("---")
                        gr.Markdown("#### Tracking")
                        vid_prop_direction = gr.Radio(
                            choices=["Forward", "Backward", "Both"],
                            value="Both",
                            label="Propagation Direction"
                        )
                        vid_add_mask_btn = gr.Button("Add New Mask")
                        vid_track_btn = gr.Button("Track All Frames", variant="primary")
                        vid_output_video = gr.Video(label="Tracked Video", sources=[])

                        vid_mask_dir = gr.Text(None, label="Mask Save Path", interactive=False)
                        vid_save_btn = gr.Button("Save Masks")
                        vid_reset_btn = gr.Button("Reset")
                
                # --- Video Tab Event Handlers ---
                vid_prompt_type.change(
                    toggle_video_prompt_type,
                    [vid_prompt_type],
                    [vid_text_group, vid_point_group, vid_box_group]
                )
                vid_prompt_type.change(video_handler.set_prompt_type, [vid_prompt_type], [instruction])

                refresh_video = partial(refresh_video_sources, vid_name=vid_name, img_name=img_name)
                select_video_cb = partial(select_video, vid_name=vid_name, img_name=img_name)
                extract_frames = partial(extract_video_frames, vid_name=vid_name, img_name=img_name)
                load_frames = partial(load_video_frames, img_name=img_name, video_handler=video_handler, mask_name=mask_name)
                update_mask_path = partial(update_mask_save_path, mask_name=mask_name)

                vid_refresh_btn.click(
                    refresh_video,
                    [vid_root_dir],
                    [vid_files_field, vid_preview, vid_img_dirs, vid_seq_name, vid_img_dir_path, instruction]
                )

                vid_files_field.select(
                    select_video_cb,
                    [vid_root_dir, vid_files_field],
                    [vid_seq_name, vid_preview, vid_img_dirs]
                )

                vid_extract_btn.click(
                    extract_frames,
                    [vid_root_dir, vid_files_field, vid_start, vid_end, vid_fps, vid_height],
                    [vid_seq_name, vid_img_dir_path, vid_img_dirs, instruction]
                )

                vid_img_dirs.change(
                    load_frames,
                    [vid_root_dir, vid_img_dirs],
                    [vid_seq_name, vid_img_dir_path, vid_frame_slider, vid_input_frame, vid_mask_dir, instruction]
                )

                vid_load_frames_btn.click(
                    load_frames,
                    [vid_root_dir, vid_img_dirs],
                    [vid_seq_name, vid_img_dir_path, vid_frame_slider, vid_input_frame, vid_mask_dir, instruction]
                )
                
                vid_seq_name.change(
                    update_mask_path,
                    [vid_root_dir, vid_seq_name],
                    [vid_mask_dir]
                )
                
                def on_frame_change(frame_idx):
                    img = video_handler.set_input_image(int(frame_idx))
                    # If tracking has been run, show the result for this frame
                    if video_handler.index_masks_all and 0 <= int(frame_idx) < len(video_handler.index_masks_all):
                        idx_mask = video_handler.index_masks_all[int(frame_idx)]
                        if idx_mask.max() > 0:
                            palette = get_hls_palette(idx_mask.max() + 1)
                            color_mask = palette[idx_mask]
                            from utils import compose_img_mask
                            out_img = compose_img_mask(img, color_mask)
                            return img, out_img
                    # Otherwise, clear output to avoid showing misleading points from other frames
                    return img, None

                vid_frame_slider.change(
                    on_frame_change,
                    [vid_frame_slider],
                    [vid_input_frame, vid_output_img]
                )

                vid_box_start_state = gr.State(None)
                vid_current_prompt_type = gr.State("Text")

                def update_point_table():
                    rows = []
                    for idx, (pt, label, frame, obj_id) in enumerate(zip(
                        video_handler.selected_points,
                        video_handler.selected_labels,
                        video_handler.selected_point_frames,
                        video_handler.selected_point_obj_ids
                    )):
                        label_str = "Positive" if label == 1.0 else "Negative"
                        rows.append([frame, int(pt[0]), int(pt[1]), label_str, obj_id, idx])
                    guru.debug(f"update_point_table: returning {len(rows)} rows: {rows}")
                    return gr.update(value=rows)

                def vid_handle_image_click(prompt_type, frame_idx, img, box_start, evt: gr.SelectData):
                    base_img = video_handler.image if video_handler.image is not None else img
                    if base_img is None:
                        return None, "Please load frames first", box_start, []

                    if prompt_type == "Box":
                        x, y = evt.index[0], evt.index[1]
                        if box_start is None:
                            preview = base_img.copy()
                            cv2.circle(preview, (x, y), 8, (255, 255, 255), 2)
                            cv2.circle(preview, (x, y), 6, (0, 255, 0), -1)
                            return preview, "First corner set. Click second corner.", (x, y), [], gr.Dropdown(choices=[])
                        x1, y1 = box_start
                        x_min, x_max = min(x1, x), max(x1, x)
                        y_min, y_max = min(y1, y), max(y1, y)
                        box_coords = (x_min, y_min, x_max, y_max)
                        # Draw box on preview before processing
                        preview = base_img.copy()
                        cv2.rectangle(preview, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.circle(preview, (x1, y1), 6, (255, 255, 255), -1)
                        cv2.circle(preview, (x, y), 6, (255, 255, 255), -1)
                        # Now process the box prompt
                        index_mask, msg = video_handler.add_box_prompt(frame_idx, box_coords)
                        obj_ids = list(video_handler.cur_masks.keys())
                        dropdown = gr.Dropdown(choices=[str(i) for i in obj_ids])
                        palette = get_hls_palette(index_mask.max() + 1)
                        color_mask = palette[index_mask]
                        out_u = compose_img_mask(base_img, color_mask)
                        # Blend the box outline with the segmentation result
                        out_u = cv2.addWeighted(out_u, 0.9, preview, 0.1, 0)
                        cv2.rectangle(out_u, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        return out_u, msg, None, [], dropdown

                    if prompt_type == "Points":
                        i, j = evt.index[1], evt.index[0]
                        index_mask = video_handler.add_point(int(frame_idx), i, j)
                        obj_ids = list(video_handler.cur_masks.keys())
                        dropdown = gr.Dropdown(choices=[str(i) for i in obj_ids])
                        palette = get_hls_palette(index_mask.max() + 1)
                        color_mask = palette[index_mask]
                        out_u = compose_img_mask(base_img, color_mask)
                        # Only draw points for the current frame
                        out = draw_points(
                            out_u,
                            video_handler.selected_points,
                            video_handler.selected_labels,
                            video_handler.selected_point_frames,
                            int(frame_idx)
                        )
                        return out, f"Point added ({len(video_handler.selected_points)} total).", box_start, update_point_table(), dropdown

                    return base_img, "Select a prompt type first", box_start, [], gr.Dropdown(choices=[])

                vid_input_frame.select(
                    vid_handle_image_click,
                    [vid_current_prompt_type, vid_frame_slider, vid_input_frame, vid_box_start_state],
                    [vid_output_img, instruction, vid_box_start_state, vid_point_table, vid_obj_dropdown],
                )

                vid_output_img.select(
                    vid_handle_image_click,
                    [vid_current_prompt_type, vid_frame_slider, vid_input_frame, vid_box_start_state],
                    [vid_output_img, instruction, vid_box_start_state, vid_point_table, vid_obj_dropdown],
                )

                vid_prompt_type.change(lambda pt: pt, [vid_prompt_type], [vid_current_prompt_type])

                vid_box_btn.click(
                    lambda: (video_handler.image, "Box mode: click first corner, then second corner.", None),
                    outputs=[vid_output_img, instruction, vid_box_start_state]
                )
                
                def vid_handle_text_prompt(text, frame_idx, img):
                    base_img = video_handler.image if video_handler.image is not None else img
                    if base_img is None:
                        return None, "Please load frames first", gr.Dropdown(choices=[])
                    if not text or not text.strip():
                        return base_img, "Please enter a text prompt", gr.Dropdown(choices=[])
                    index_mask, msg = video_handler.add_text_prompt(text, int(frame_idx))
                    obj_ids = list(video_handler.cur_masks.keys())
                    dropdown = gr.Dropdown(choices=[str(i) for i in obj_ids])
                    if index_mask is not None:
                        palette = get_hls_palette(index_mask.max() + 1)
                        color_mask = palette[index_mask]
                        out_u = compose_img_mask(base_img, color_mask)
                        return out_u, msg, dropdown
                    return base_img, msg, dropdown

                vid_text_btn.click(
                    vid_handle_text_prompt,
                    [vid_text_input, vid_frame_slider, vid_input_frame],
                    [vid_output_img, instruction, vid_obj_dropdown]
                )
                
                vid_pos_btn.click(video_handler.set_positive, outputs=[instruction])
                vid_neg_btn.click(video_handler.set_negative, outputs=[instruction])
                vid_clear_pts_btn.click(video_handler.clear_points, outputs=[vid_output_img, vid_output_video, instruction])
                
                def update_obj_dropdown():
                    obj_ids = list(video_handler.cur_masks.keys())
                    return gr.Dropdown(choices=[str(i) for i in obj_ids])

                vid_add_mask_btn.click(video_handler.add_new_mask, outputs=[vid_output_img, instruction])
                vid_add_mask_btn.click(update_obj_dropdown, outputs=[vid_obj_dropdown])

                def vid_remove_obj(obj_id_str):
                    if not obj_id_str:
                        return None, "No object selected", gr.Dropdown(choices=[])
                    try:
                        obj_id = int(obj_id_str)
                        guru.debug(f"Removing object {obj_id}, cur_masks keys: {list(video_handler.cur_masks.keys())}")
                        result = video_handler.remove_object(obj_id)
                        index_mask, msg = result
                        obj_ids = list(video_handler.cur_masks.keys())
                        guru.debug(f"After removal, cur_masks keys: {obj_ids}")

                        # Visualize the remaining masks
                        base_img = video_handler.image
                        if index_mask is not None and base_img is not None and index_mask.max() > 0:
                            palette = get_hls_palette(index_mask.max() + 1)
                            color_mask = palette[index_mask]
                            out_u = compose_img_mask(base_img, color_mask)
                            return out_u, msg, gr.Dropdown(choices=[str(i) for i in obj_ids])
                        return None, msg, gr.Dropdown(choices=[str(i) for i in obj_ids])
                    except ValueError:
                        return None, "Invalid object ID", gr.Dropdown(choices=[])

                vid_remove_obj_btn.click(
                    vid_remove_obj,
                    [vid_obj_dropdown],
                    [vid_output_img, instruction, vid_obj_dropdown]
                )

                def vid_track_with_direction(direction):
                    direction_map = {"Forward": "forward", "Backward": "backward", "Both": "both"}
                    return video_handler.run_tracker(propagation_direction=direction_map.get(direction, "both"))

                vid_track_btn.click(vid_track_with_direction, [vid_prop_direction], outputs=[vid_output_video, instruction])
                vid_save_btn.click(video_handler.save_masks_to_dir, [vid_mask_dir], outputs=[instruction])
                vid_reset_btn.click(video_handler.reset)

                def vid_remove_point_by_idx(idx):
                    index_mask, msg = video_handler.remove_selected_point(int(idx))
                    if index_mask is not None:
                        base_img = video_handler.image
                        palette = get_hls_palette(index_mask.max() + 1)
                        color_mask = palette[index_mask]
                        out_u = compose_img_mask(base_img, color_mask)
                        # Only draw points for the current frame
                        out = draw_points(
                            out_u,
                            video_handler.selected_points,
                            video_handler.selected_labels,
                            video_handler.selected_point_frames,
                            video_handler.frame_index
                        )
                        return out, update_point_table(), msg
                    return None, update_point_table(), msg

                vid_remove_point_btn.click(
                    vid_remove_point_by_idx,
                    [vid_remove_point_idx],
                    [vid_output_img, vid_point_table, instruction]
                )
            
            # =================================================================
            # IMAGE TAB
            # =================================================================
            with gr.TabItem("Image Mode", id="image_tab"):
                gr.Markdown("### Single Image Segmentation")
                
                with gr.Row():
                    # Left: Image source
                    with gr.Column(scale=1):
                        gr.Markdown("#### Source (Folder Only)")
                        img_folder_root = gr.Text(root_dir, label="Root Directory")
                        img_refresh_btn = gr.Button("Refresh Image Lists")
                        img_folder_list = gr.Dropdown(
                            label="Image Folders",
                            choices=initial_image_folders,
                            value=initial_image_folder,
                        )
                        img_file_list = gr.Dropdown(
                            label="Image Files",
                            choices=initial_image_files,
                            value=initial_image_file,
                        )
                        img_selected_path = gr.Text(initial_image_path, label="Selected Image Path", interactive=False)
                        img_load_btn = gr.Button("Load Image", variant="primary")
                    
                    # Middle: Segmentation mode and prompts
                    with gr.Column(scale=2):
                        gr.Markdown("#### Segmentation")
                        img_display = gr.Image(label="Image", interactive=False)
                        
                        img_mode = gr.Radio(
                            choices=["Find All", "Box", "Point"],
                            value="Find All",
                            label="Segmentation Mode"
                        )

                        img_confidence_slider = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.3, step=0.05,
                            label="Confidence Threshold"
                        )
                        
                        # Find All mode
                        with gr.Group(visible=True) as img_findall_group:
                            img_text_input = gr.Textbox(label="Text Prompt", placeholder="e.g., 'shoe', 'person', 'car'")
                            img_findall_btn = gr.Button("Find All", variant="primary")
                        
                        # Box mode
                        with gr.Group(visible=False) as img_box_group:
                            gr.Markdown("Draw a box on the image above, then click 'Segment Box'")
                            img_box_btn = gr.Button("Segment Box", variant="primary")
                        
                        # Point mode
                        with gr.Group(visible=False) as img_point_group:
                            with gr.Row():
                                img_pos_btn = gr.Button("+ Positive")
                                img_neg_btn = gr.Button("- Negative")
                            gr.Markdown("Click on the image to add points")
                            img_point_table = gr.DataFrame(
                                headers=["X", "Y", "Type", "Index"],
                                label="Added Points",
                                interactive=False,
                                value=[]
                            )
                            with gr.Row():
                                img_remove_point_idx = gr.Number(value=0, label="Point Index to Remove", minimum=0, step=1)
                                img_remove_point_btn = gr.Button("Remove Point by Index")
                    
                    # Right: Output
                    with gr.Column(scale=1):
                        gr.Markdown("#### Output")
                        img_output = gr.Image(label="Segmentation Result")
                        img_clear_btn = gr.Button("Clear Prompts")
                        img_save_path = gr.Textbox(label="Save Path", value="mask_output.npy")
                        img_save_btn = gr.Button("Save Mask")
                box_start_state = gr.State(None)
                
                # --- Image Tab Event Handlers ---
                refresh_images = partial(refresh_image_lists, img_name=img_name)
                select_folder = partial(select_image_folder, img_name=img_name)
                update_selected_path = partial(update_selected_image_path, img_name=img_name)
                load_image = partial(load_image_from_folder, img_name=img_name, image_handler=image_handler)

                img_refresh_btn.click(
                    refresh_images,
                    [img_folder_root],
                    [img_folder_list, img_file_list, img_selected_path, instruction]
                )
                
                img_folder_root.submit(
                    refresh_images,
                    [img_folder_root],
                    [img_folder_list, img_file_list, img_selected_path, instruction]
                )

                img_folder_list.change(
                    select_folder,
                    [img_folder_root, img_folder_list],
                    [img_file_list, img_selected_path]
                )

                img_file_list.change(
                    update_selected_path,
                    [img_folder_root, img_folder_list, img_file_list],
                    [img_selected_path]
                )

                img_mode.change(
                    lambda mode: (*toggle_image_mode(mode), None),
                    [img_mode],
                    [img_findall_group, img_box_group, img_point_group, box_start_state]
                )
                
                def load_image_and_reset(root, folder_name, file_name):
                    display, output, msg = load_image(root, folder_name, file_name)
                    return display, output, msg, None

                img_load_btn.click(
                    load_image_and_reset,
                    [img_folder_root, img_folder_list, img_file_list],
                    [img_display, img_output, instruction, box_start_state]
                )
                
                # Auto-load when file is selected from folder
                img_file_list.change(
                    load_image_and_reset,
                    [img_folder_root, img_folder_list, img_file_list],
                    [img_display, img_output, instruction, box_start_state]
                )
                
                def img_handle_findall(text):
                    result, msg = image_handler.find_all_with_text(text)
                    return result, msg
                
                img_findall_btn.click(
                    img_handle_findall,
                    [img_text_input],
                    [img_output, instruction]
                )
                
                def show_box_hint():
                    return image_handler.current_image, "Box mode: click first corner, then second corner.", None

                img_box_btn.click(
                    show_box_hint,
                    outputs=[img_output, instruction, box_start_state]
                )
                
                img_pos_btn.click(image_handler.set_positive, outputs=[instruction])
                img_neg_btn.click(image_handler.set_negative, outputs=[instruction])

                img_confidence_slider.change(
                    image_handler.set_confidence_threshold,
                    [img_confidence_slider],
                    [instruction]
                )
                
                def update_img_point_table():
                    rows = []
                    for idx, (pt, label) in enumerate(zip(
                        image_handler.selected_points,
                        image_handler.selected_labels
                    )):
                        label_str = "Positive" if label == 1 else "Negative"
                        rows.append([int(pt[0]), int(pt[1]), label_str, idx])
                    return gr.update(value=rows)

                def img_handle_click(mode, box_start, evt: gr.SelectData):
                    x, y = evt.index[0], evt.index[1]
                    if mode == "Point":
                        result, msg = image_handler.add_point(x, y)
                        return result, msg, box_start, update_img_point_table()
                    if mode == "Box":
                        if image_handler.current_image is None:
                            return None, "Please load an image first", None, []
                        if box_start is None:
                            preview = image_handler.current_image.copy()
                            cv2.circle(preview, (x, y), 8, (255, 255, 255), 2)
                            cv2.circle(preview, (x, y), 6, (0, 255, 0), -1)
                            return preview, "First corner set. Click second corner.", (x, y), []
                        x1, y1 = box_start
                        x_min, x_max = min(x1, x), max(x1, x)
                        y_min, y_max = min(y1, y), max(y1, y)
                        box_coords = (x_min, y_min, x_max, y_max)
                        # Draw box on the image before segmentation
                        preview = image_handler.current_image.copy()
                        cv2.rectangle(preview, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.circle(preview, (x1, y1), 6, (255, 255, 255), -1)
                        cv2.circle(preview, (x, y), 6, (255, 255, 255), -1)
                        result, msg = image_handler.segment_with_box(box_coords)
                        # If segmentation succeeded, overlay the box on the result
                        if result is not None:
                            result = cv2.addWeighted(result, 0.9, preview, 0.1, 0)
                            cv2.rectangle(result, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        return result, msg, None, []
                    return image_handler.current_image, "Use 'Find All' button in text mode.", box_start, []

                img_display.select(
                    img_handle_click,
                    [img_mode, box_start_state],
                    [img_output, instruction, box_start_state, img_point_table],
                )

                def clear_prompts_and_reset_box():
                    output, msg = image_handler.clear_prompts()
                    return output, msg, None, []

                img_clear_btn.click(clear_prompts_and_reset_box, outputs=[img_output, instruction, box_start_state, img_point_table])

                def img_remove_point_by_idx(idx):
                    result, msg = image_handler.remove_point(int(idx))
                    return result, update_img_point_table(), msg

                img_remove_point_btn.click(
                    img_remove_point_by_idx,
                    [img_remove_point_idx],
                    [img_output, img_point_table, instruction]
                )

                img_save_btn.click(image_handler.save_mask, [img_save_path], outputs=[instruction])
    
    return demo


if __name__ == "__main__":
    raise SystemExit("Use `python cli.py` to launch the app.")
