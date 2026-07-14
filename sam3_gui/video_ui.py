import os
from functools import partial

import cv2
import gradio as gr
from loguru import logger as guru

from sam3_gui.common_ui import object_dropdown, overlay_index_mask, video_point_rows
from sam3_gui.ui_handlers import (
    extract_video_frames,
    load_video_frames,
    refresh_video_sources,
    select_video_with_metadata,
    toggle_video_prompt_type,
    update_mask_save_path,
)
from sam3_gui.utils import draw_points, frame_dir_path


def build_video_tab(
    root_dir,
    vid_name,
    img_name,
    mask_name,
    initial_videos,
    initial_video,
    initial_frame_dirs,
    initial_frame_dir,
    video_handler_provider,
    instruction,
):
    vid_root = os.path.join(root_dir, vid_name)

    with gr.TabItem("Video Mode", id="video_tab"):
        gr.Markdown("### Video/Frame Sequence Segmentation with Tracking")

        with gr.Row():
            vid_root_dir = gr.Text(root_dir, label="Root Directory")
            vid_refresh_btn = gr.Button("Refresh Lists")
            vid_seq_name = gr.Text(
                initial_frame_dir, label="Sequence Name", interactive=False
            )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### Source")
                vid_files_field = gr.Dropdown(
                    label="Video Files",
                    choices=initial_videos,
                    value=initial_video,
                )
                vid_preview = gr.Video(
                    label="Video Preview",
                    value=os.path.join(vid_root, initial_video)
                    if initial_video
                    else None,
                    sources=[],
                )

                with gr.Row():
                    vid_start = gr.Number(0, label="Start (s)")
                    vid_end = gr.Number(10, label="End (s)")
                    vid_fps = gr.Number(30, label="FPS")
                    vid_downsample = gr.Radio(
                        choices=[
                            "Original",
                            "Half",
                            "Quarter",
                            "Sixth",
                            "Eighth",
                            "Sixteenth",
                        ],
                        value="Original",
                        label="Downsampling",
                    )
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
                vid_load_frames_btn = gr.Button(
                    "Load Selected Frames", variant="primary"
                )

            with gr.Column(scale=2):
                gr.Markdown("#### Frame & Prompts")
                vid_frame_slider = gr.Slider(
                    label="Frame Index", minimum=0, maximum=1, value=0, step=1
                )
                vid_input_frame = gr.Image(
                    label="Input Frame", interactive=True, type="numpy", sources=[]
                )

                vid_prompt_type = gr.Radio(
                    choices=["Text", "Points", "Box"],
                    value="Text",
                    label="Prompt Type",
                    interactive=True,
                )

                with gr.Group(visible=True) as vid_text_group:
                    vid_text_input = gr.Textbox(
                        label="Text Prompt", placeholder="e.g., 'person', 'car', 'dog'"
                    )
                    vid_text_btn = gr.Button("Detect with Text", variant="primary")

                with gr.Group(visible=False) as vid_point_group:
                    with gr.Row():
                        vid_pos_btn = gr.Button("+ Positive")
                        vid_neg_btn = gr.Button("- Negative")
                        vid_clear_pts_btn = gr.Button("Clear Points")
                    vid_point_table = gr.DataFrame(
                        headers=["Frame", "X", "Y", "Type", "Obj ID", "Index"],
                        label="Added Points",
                        interactive=False,
                        value=[],
                    )
                    with gr.Row():
                        vid_remove_point_idx = gr.Number(
                            value=0, label="Point Index to Remove", minimum=0, step=1
                        )
                        vid_remove_point_btn = gr.Button("Remove Point by Index")

                with gr.Group(visible=False) as vid_box_group:
                    gr.Markdown(
                        "Draw a box on the frame above, then click 'Segment Box'"
                    )
                    vid_box_btn = gr.Button("Segment Box", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("#### Output")
                vid_output_img = gr.Image(
                    label="Segmentation Preview",
                    interactive=True,
                    type="numpy",
                    sources=[],
                )

                gr.Markdown("---")
                gr.Markdown("#### Object Management")
                vid_obj_dropdown = gr.Dropdown(
                    choices=[], label="Tracked Objects", interactive=True
                )
                vid_remove_obj_btn = gr.Button("Remove Selected Object")

                gr.Markdown("---")
                gr.Markdown("#### Tracking")
                vid_prop_direction = gr.Radio(
                    choices=["Forward", "Backward", "Both"],
                    value="Both",
                    label="Propagation Direction",
                )
                vid_add_mask_btn = gr.Button("Add New Mask")
                vid_track_btn = gr.Button("Track All Frames", variant="primary")
                vid_output_video = gr.Video(label="Tracked Video", sources=[])

                vid_mask_dir = gr.Text(None, label="Mask Save Path", interactive=False)
                vid_save_btn = gr.Button("Save Masks")
                vid_reset_btn = gr.Button("Reset")

        vid_prompt_type.change(
            toggle_video_prompt_type,
            [vid_prompt_type],
            [vid_text_group, vid_point_group, vid_box_group],
        )

        def set_prompt_type(prompt_type, request: gr.Request):
            return video_handler_provider(request).set_prompt_type(prompt_type)

        vid_prompt_type.change(set_prompt_type, [vid_prompt_type], [instruction])

        refresh_video = partial(
            refresh_video_sources, vid_name=vid_name, img_name=img_name
        )
        select_video_cb = partial(
            select_video_with_metadata, vid_name=vid_name, img_name=img_name
        )
        extract_frames = partial(
            extract_video_frames, vid_name=vid_name, img_name=img_name
        )
        update_mask_path = partial(update_mask_save_path, mask_name=mask_name)

        def load_frames(root, seq_name, request: gr.Request):
            return load_video_frames(
                root,
                seq_name,
                img_name=img_name,
                video_handler=video_handler_provider(request),
                mask_name=mask_name,
            )

        vid_refresh_btn.click(
            refresh_video,
            [vid_root_dir],
            [
                vid_files_field,
                vid_preview,
                vid_img_dirs,
                vid_seq_name,
                vid_img_dir_path,
                instruction,
            ],
        )

        vid_files_field.select(
            select_video_cb,
            [vid_root_dir, vid_files_field],
            [vid_seq_name, vid_preview, vid_img_dirs, vid_end, vid_downsample],
        )

        vid_extract_btn.click(
            extract_frames,
            [
                vid_root_dir,
                vid_files_field,
                vid_start,
                vid_end,
                vid_fps,
                vid_downsample,
            ],
            [vid_seq_name, vid_img_dir_path, vid_img_dirs, instruction],
        )

        vid_img_dirs.change(
            load_frames,
            [vid_root_dir, vid_img_dirs],
            [
                vid_seq_name,
                vid_img_dir_path,
                vid_frame_slider,
                vid_input_frame,
                vid_mask_dir,
                instruction,
            ],
        )

        vid_load_frames_btn.click(
            load_frames,
            [vid_root_dir, vid_img_dirs],
            [
                vid_seq_name,
                vid_img_dir_path,
                vid_frame_slider,
                vid_input_frame,
                vid_mask_dir,
                instruction,
            ],
        )

        vid_seq_name.change(
            update_mask_path, [vid_root_dir, vid_seq_name], [vid_mask_dir]
        )

        def on_frame_change(frame_idx, request: gr.Request):
            video_handler = video_handler_provider(request)
            img = video_handler.set_input_image(int(frame_idx))
            if video_handler.index_masks_all and 0 <= int(frame_idx) < len(
                video_handler.index_masks_all
            ):
                idx_mask = video_handler.index_masks_all[int(frame_idx)]
                out_img = overlay_index_mask(img, idx_mask)
                if out_img is not None:
                    return img, out_img
            return img, None

        vid_frame_slider.change(
            on_frame_change, [vid_frame_slider], [vid_input_frame, vid_output_img]
        )

        vid_box_start_state = gr.State(None)
        vid_current_prompt_type = gr.State("Text")

        def update_point_table(video_handler):
            rows = video_point_rows(video_handler)
            guru.debug(f"update_point_table: returning {len(rows)} rows: {rows}")
            return gr.update(value=rows)

        def vid_handle_image_click(
            prompt_type,
            frame_idx,
            img,
            box_start,
            evt: gr.SelectData,
            request: gr.Request,
        ):
            video_handler = video_handler_provider(request)
            base_img = video_handler.image if video_handler.image is not None else img
            if base_img is None:
                return None, "Please load frames first", box_start, []

            if prompt_type == "Box":
                x, y = evt.index[0], evt.index[1]
                if box_start is None:
                    preview = base_img.copy()
                    cv2.circle(preview, (x, y), 8, (255, 255, 255), 2)
                    cv2.circle(preview, (x, y), 6, (0, 255, 0), -1)
                    return (
                        preview,
                        "First corner set. Click second corner.",
                        (x, y),
                        [],
                        object_dropdown([]),
                    )
                x1, y1 = box_start
                x_min, x_max = min(x1, x), max(x1, x)
                y_min, y_max = min(y1, y), max(y1, y)
                box_coords = (x_min, y_min, x_max, y_max)
                preview = base_img.copy()
                cv2.rectangle(preview, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.circle(preview, (x1, y1), 6, (255, 255, 255), -1)
                cv2.circle(preview, (x, y), 6, (255, 255, 255), -1)
                index_mask, msg = video_handler.add_box_prompt(frame_idx, box_coords)
                obj_ids = list(video_handler.cur_masks.keys())
                dropdown = object_dropdown(obj_ids)
                if index_mask is None:
                    return preview, msg, None, [], dropdown
                out_u = overlay_index_mask(base_img, index_mask)
                if out_u is None:
                    return preview, msg, None, [], dropdown
                out_u = cv2.addWeighted(out_u, 0.9, preview, 0.1, 0)
                cv2.rectangle(out_u, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                return out_u, msg, None, [], dropdown

            if prompt_type == "Points":
                i, j = evt.index[1], evt.index[0]
                index_mask = video_handler.add_point(int(frame_idx), i, j)
                obj_ids = list(video_handler.cur_masks.keys())
                dropdown = object_dropdown(obj_ids)
                if index_mask is None:
                    return (
                        base_img,
                        "No mask generated from point.",
                        box_start,
                        update_point_table(video_handler),
                        dropdown,
                    )
                out_u = overlay_index_mask(base_img, index_mask)
                if out_u is None:
                    return (
                        base_img,
                        "No mask generated from point.",
                        box_start,
                        update_point_table(video_handler),
                        dropdown,
                    )
                out = draw_points(
                    out_u,
                    video_handler.selected_points,
                    video_handler.selected_labels,
                    video_handler.selected_point_frames,
                    int(frame_idx),
                )
                return (
                    out,
                    f"Point added ({len(video_handler.selected_points)} total).",
                    box_start,
                    update_point_table(video_handler),
                    dropdown,
                )

            return (
                base_img,
                "Select a prompt type first",
                box_start,
                [],
                object_dropdown([]),
            )

        vid_input_frame.select(
            vid_handle_image_click,
            [
                vid_current_prompt_type,
                vid_frame_slider,
                vid_input_frame,
                vid_box_start_state,
            ],
            [
                vid_output_img,
                instruction,
                vid_box_start_state,
                vid_point_table,
                vid_obj_dropdown,
            ],
        )

        vid_output_img.select(
            vid_handle_image_click,
            [
                vid_current_prompt_type,
                vid_frame_slider,
                vid_input_frame,
                vid_box_start_state,
            ],
            [
                vid_output_img,
                instruction,
                vid_box_start_state,
                vid_point_table,
                vid_obj_dropdown,
            ],
        )

        vid_prompt_type.change(
            lambda pt: pt, [vid_prompt_type], [vid_current_prompt_type]
        )

        def show_box_hint(request: gr.Request):
            video_handler = video_handler_provider(request)
            return (
                video_handler.image,
                "Box mode: click first corner, then second corner.",
                None,
            )

        vid_box_btn.click(
            show_box_hint, outputs=[vid_output_img, instruction, vid_box_start_state]
        )

        def vid_handle_text_prompt(text, frame_idx, img, request: gr.Request):
            video_handler = video_handler_provider(request)
            base_img = video_handler.image if video_handler.image is not None else img
            if base_img is None:
                return None, "Please load frames first", object_dropdown([])
            if not text or not text.strip():
                return base_img, "Please enter a text prompt", object_dropdown([])
            index_mask, msg = video_handler.add_text_prompt(text, int(frame_idx))
            obj_ids = list(video_handler.cur_masks.keys())
            dropdown = object_dropdown(obj_ids)
            if index_mask is not None:
                out_u = overlay_index_mask(base_img, index_mask)
                return out_u if out_u is not None else base_img, msg, dropdown
            return base_img, msg, dropdown

        vid_text_btn.click(
            vid_handle_text_prompt,
            [vid_text_input, vid_frame_slider, vid_input_frame],
            [vid_output_img, instruction, vid_obj_dropdown],
        )

        def set_positive(request: gr.Request):
            return video_handler_provider(request).set_positive()

        def set_negative(request: gr.Request):
            return video_handler_provider(request).set_negative()

        def clear_points(request: gr.Request):
            return video_handler_provider(request).clear_points()

        vid_pos_btn.click(set_positive, outputs=[instruction])
        vid_neg_btn.click(set_negative, outputs=[instruction])
        vid_clear_pts_btn.click(
            clear_points, outputs=[vid_output_img, vid_output_video, instruction]
        )

        def update_obj_dropdown(request: gr.Request):
            video_handler = video_handler_provider(request)
            obj_ids = list(video_handler.cur_masks.keys())
            return object_dropdown(obj_ids)

        def add_new_mask(request: gr.Request):
            return video_handler_provider(request).add_new_mask()

        vid_add_mask_btn.click(add_new_mask, outputs=[vid_output_img, instruction])
        vid_add_mask_btn.click(update_obj_dropdown, outputs=[vid_obj_dropdown])

        def vid_remove_obj(obj_id_str, request: gr.Request):
            video_handler = video_handler_provider(request)
            if not obj_id_str:
                return None, "No object selected", object_dropdown([])
            try:
                obj_id = int(obj_id_str)
                guru.debug(
                    f"Removing object {obj_id}, cur_masks keys: {list(video_handler.cur_masks.keys())}"
                )
                index_mask, msg = video_handler.remove_object(obj_id)
                obj_ids = list(video_handler.cur_masks.keys())
                guru.debug(f"After removal, cur_masks keys: {obj_ids}")

                base_img = video_handler.image
                out_u = overlay_index_mask(base_img, index_mask)
                return out_u, msg, object_dropdown(obj_ids)
            except ValueError:
                return None, "Invalid object ID", object_dropdown([])

        vid_remove_obj_btn.click(
            vid_remove_obj,
            [vid_obj_dropdown],
            [vid_output_img, instruction, vid_obj_dropdown],
        )

        def vid_track_with_direction(direction, request: gr.Request):
            direction_map = {
                "Forward": "forward",
                "Backward": "backward",
                "Both": "both",
            }
            return video_handler_provider(request).run_tracker(
                propagation_direction=direction_map.get(direction, "both")
            )

        vid_track_btn.click(
            vid_track_with_direction,
            [vid_prop_direction],
            outputs=[vid_output_video, instruction],
        )

        def save_masks_to_dir(path, request: gr.Request):
            return video_handler_provider(request).save_masks_to_dir(path)

        def reset(request: gr.Request):
            return video_handler_provider(request).reset()

        vid_save_btn.click(save_masks_to_dir, [vid_mask_dir], outputs=[instruction])
        vid_reset_btn.click(reset)

        def vid_remove_point_by_idx(idx, request: gr.Request):
            video_handler = video_handler_provider(request)
            index_mask, msg = video_handler.remove_selected_point(int(idx))
            if index_mask is not None:
                base_img = video_handler.image
                out_u = overlay_index_mask(base_img, index_mask)
                if out_u is None:
                    return None, update_point_table(video_handler), msg
                out = draw_points(
                    out_u,
                    video_handler.selected_points,
                    video_handler.selected_labels,
                    video_handler.selected_point_frames,
                    video_handler.frame_index,
                )
                return out, update_point_table(video_handler), msg
            return None, update_point_table(video_handler), msg

        vid_remove_point_btn.click(
            vid_remove_point_by_idx,
            [vid_remove_point_idx],
            [vid_output_img, vid_point_table, instruction],
        )
