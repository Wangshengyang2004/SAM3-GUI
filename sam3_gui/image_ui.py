from functools import partial

import cv2
import gradio as gr

from sam3_gui.common_ui import image_point_rows
from sam3_gui.ui_handlers import (
    load_image_from_folder,
    refresh_image_lists,
    select_image_folder,
    toggle_image_mode,
    update_selected_image_path,
)


def build_image_tab(
    root_dir,
    img_name,
    initial_image_folders,
    initial_image_folder,
    initial_image_files,
    initial_image_file,
    initial_image_path,
    image_handler_provider,
    instruction,
):
    with gr.TabItem("Image Mode", id="image_tab"):
        gr.Markdown("### Single Image Segmentation")

        with gr.Row():
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
                img_selected_path = gr.Text(
                    initial_image_path, label="Selected Image Path", interactive=False
                )
                img_load_btn = gr.Button("Load Image", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("#### Segmentation")
                img_display = gr.Image(label="Image", interactive=False)

                img_mode = gr.Radio(
                    choices=["Find All", "Box", "Point"],
                    value="Find All",
                    label="Segmentation Mode",
                )

                img_confidence_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.3,
                    step=0.05,
                    label="Confidence Threshold",
                )

                with gr.Group(visible=True) as img_findall_group:
                    img_text_input = gr.Textbox(
                        label="Text Prompt", placeholder="e.g., 'shoe', 'person', 'car'"
                    )
                    img_findall_btn = gr.Button("Find All", variant="primary")

                with gr.Group(visible=False) as img_box_group:
                    gr.Markdown(
                        "Draw a box on the image above, then click 'Segment Box'"
                    )
                    img_box_btn = gr.Button("Segment Box", variant="primary")

                with gr.Group(visible=False) as img_point_group:
                    with gr.Row():
                        img_pos_btn = gr.Button("+ Positive")
                        img_neg_btn = gr.Button("- Negative")
                    gr.Markdown("Click on the image to add points")
                    img_point_table = gr.DataFrame(
                        headers=["X", "Y", "Type", "Index"],
                        label="Added Points",
                        interactive=False,
                        value=[],
                    )
                    with gr.Row():
                        img_remove_point_idx = gr.Number(
                            value=0, label="Point Index to Remove", minimum=0, step=1
                        )
                        img_remove_point_btn = gr.Button("Remove Point by Index")

            with gr.Column(scale=1):
                gr.Markdown("#### Output")
                img_output = gr.Image(label="Segmentation Result")
                img_clear_btn = gr.Button("Clear Prompts")
                img_save_path = gr.Textbox(label="Save Path", value="mask_output.npy")
                img_save_btn = gr.Button("Save Mask")
        box_start_state = gr.State(None)

        refresh_images = partial(refresh_image_lists, img_name=img_name)
        select_folder = partial(select_image_folder, img_name=img_name)
        update_selected_path = partial(update_selected_image_path, img_name=img_name)

        img_refresh_btn.click(
            refresh_images,
            [img_folder_root],
            [img_folder_list, img_file_list, img_selected_path, instruction],
        )

        img_folder_root.submit(
            refresh_images,
            [img_folder_root],
            [img_folder_list, img_file_list, img_selected_path, instruction],
        )

        img_folder_list.change(
            select_folder,
            [img_folder_root, img_folder_list],
            [img_file_list, img_selected_path],
        )

        img_file_list.change(
            update_selected_path,
            [img_folder_root, img_folder_list, img_file_list],
            [img_selected_path],
        )

        img_mode.change(
            lambda mode: (*toggle_image_mode(mode), None),
            [img_mode],
            [img_findall_group, img_box_group, img_point_group, box_start_state],
        )

        def load_image_and_reset(root, folder_name, file_name, request: gr.Request):
            image_handler = image_handler_provider(request)
            display, output, msg = load_image_from_folder(
                root,
                folder_name,
                file_name,
                img_name=img_name,
                image_handler=image_handler,
            )
            return display, output, msg, None

        img_load_btn.click(
            load_image_and_reset,
            [img_folder_root, img_folder_list, img_file_list],
            [img_display, img_output, instruction, box_start_state],
        )

        img_file_list.change(
            load_image_and_reset,
            [img_folder_root, img_folder_list, img_file_list],
            [img_display, img_output, instruction, box_start_state],
        )

        def img_handle_findall(text, request: gr.Request):
            image_handler = image_handler_provider(request)
            result, msg = image_handler.find_all_with_text(text)
            return result, msg

        img_findall_btn.click(
            img_handle_findall, [img_text_input], [img_output, instruction]
        )

        def show_box_hint(request: gr.Request):
            image_handler = image_handler_provider(request)
            return (
                image_handler.current_image,
                "Box mode: click first corner, then second corner.",
                None,
            )

        img_box_btn.click(
            show_box_hint, outputs=[img_output, instruction, box_start_state]
        )

        def set_positive(request: gr.Request):
            return image_handler_provider(request).set_positive()

        def set_negative(request: gr.Request):
            return image_handler_provider(request).set_negative()

        img_pos_btn.click(set_positive, outputs=[instruction])
        img_neg_btn.click(set_negative, outputs=[instruction])

        def set_confidence_threshold(value, request: gr.Request):
            return image_handler_provider(request).set_confidence_threshold(value)

        img_confidence_slider.change(
            set_confidence_threshold, [img_confidence_slider], [instruction]
        )

        def update_img_point_table(image_handler):
            return gr.update(value=image_point_rows(image_handler))

        def img_handle_click(mode, box_start, evt: gr.SelectData, request: gr.Request):
            image_handler = image_handler_provider(request)
            x, y = evt.index[0], evt.index[1]
            if mode == "Point":
                result, msg = image_handler.add_point(x, y)
                return result, msg, box_start, update_img_point_table(image_handler)
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
                preview = image_handler.current_image.copy()
                cv2.rectangle(preview, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.circle(preview, (x1, y1), 6, (255, 255, 255), -1)
                cv2.circle(preview, (x, y), 6, (255, 255, 255), -1)
                result, msg = image_handler.segment_with_box(box_coords)
                if result is not None:
                    result = cv2.addWeighted(result, 0.9, preview, 0.1, 0)
                    cv2.rectangle(
                        result, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2
                    )
                return result, msg, None, []
            return (
                image_handler.current_image,
                "Use 'Find All' button in text mode.",
                box_start,
                [],
            )

        img_display.select(
            img_handle_click,
            [img_mode, box_start_state],
            [img_output, instruction, box_start_state, img_point_table],
        )

        def clear_prompts_and_reset_box(request: gr.Request):
            image_handler = image_handler_provider(request)
            output, msg = image_handler.clear_prompts()
            return output, msg, None, []

        img_clear_btn.click(
            clear_prompts_and_reset_box,
            outputs=[img_output, instruction, box_start_state, img_point_table],
        )

        def img_remove_point_by_idx(idx, request: gr.Request):
            image_handler = image_handler_provider(request)
            result, msg = image_handler.remove_point(int(idx))
            return result, update_img_point_table(image_handler), msg

        img_remove_point_btn.click(
            img_remove_point_by_idx,
            [img_remove_point_idx],
            [img_output, img_point_table, instruction],
        )

        def save_mask(path, request: gr.Request):
            return image_handler_provider(request).save_mask(path)

        img_save_btn.click(save_mask, [img_save_path], outputs=[instruction])
