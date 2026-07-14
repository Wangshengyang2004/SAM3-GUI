import gradio as gr

from sam3_gui.utils import compose_img_mask, get_hls_palette


def object_dropdown(obj_ids):
    return gr.Dropdown(choices=[str(obj_id) for obj_id in obj_ids])


def overlay_index_mask(base_img, index_mask):
    if base_img is None or index_mask is None or index_mask.max() <= 0:
        return None
    palette = get_hls_palette(index_mask.max() + 1)
    return compose_img_mask(base_img, palette[index_mask])


def video_point_rows(video_handler):
    rows = []
    for idx, (pt, label, frame, obj_id) in enumerate(
        zip(
            video_handler.selected_points,
            video_handler.selected_labels,
            video_handler.selected_point_frames,
            video_handler.selected_point_obj_ids,
        )
    ):
        label_str = "Positive" if label == 1.0 else "Negative"
        rows.append([frame, int(pt[0]), int(pt[1]), label_str, obj_id, idx])
    return rows


def image_point_rows(image_handler):
    rows = []
    for idx, (pt, label) in enumerate(
        zip(image_handler.selected_points, image_handler.selected_labels)
    ):
        label_str = "Positive" if label == 1 else "Negative"
        rows.append([int(pt[0]), int(pt[1]), label_str, idx])
    return rows
