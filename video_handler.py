import os

import imageio.v2 as iio
import numpy as np
from loguru import logger as guru

from sam31_backend import (
    Sam31Backend,
    index_mask_from_obj_masks,
    normalize_points,
    normalize_xyxy_box,
    output_masks_by_obj,
    validate_sam31_checkpoint_path,
)
from utils import colorize_masks, isimage


class VideoModeHandler:
    """Native SAM 3.1 video segmentation and tracking handler."""

    def __init__(self, checkpoint_path=None, gpus_to_use=None, backend: Sam31Backend | None = None):
        self.backend = backend or Sam31Backend()
        if checkpoint_path is not None:
            self.backend.config.checkpoint_path = validate_sam31_checkpoint_path(checkpoint_path)
        if gpus_to_use:
            self.backend.config.device_id = gpus_to_use[0]

        self.session_id = None
        self.selected_points = []
        self.selected_labels = []
        self.selected_point_frames = []
        self.selected_point_obj_ids = []
        self.cur_label_val = 1.0

        self.frame_index = 0
        self.image = None
        self.cur_mask_idx = 0
        self.cur_masks: dict[int, np.ndarray] = {}
        self.index_masks_all = []
        self.color_masks_all = []

        self.img_dir = ""
        self.img_paths = []
        self.current_text_prompt = None
        self.text_prompt_frame_idx = 0

    @property
    def inference_state(self):
        return self.session_id

    def init_model(self):
        self.backend.ensure_predictor()

    def _resize_mask(self, mask, target_h, target_w):
        from sam31_backend import resize_mask

        return resize_mask(mask, target_h, target_w)

    def _normalize_points(self, points):
        if self.image is None:
            return points
        h, w = self.image.shape[:2]
        return normalize_points(points, w, h)

    def make_index_mask(self, masks):
        fallback = self.image.shape[:2] if self.image is not None else (1, 1)
        return index_mask_from_obj_masks(masks, fallback)

    def clear_points(self):
        self.selected_points.clear()
        self.selected_labels.clear()
        self.selected_point_frames.clear()
        self.selected_point_obj_ids.clear()
        return None, None, "Cleared points"

    def set_positive(self):
        self.cur_label_val = 1.0
        return "Selecting positive points"

    def set_negative(self):
        self.cur_label_val = 0.0
        return "Selecting negative points"

    def set_prompt_type(self, prompt_type):
        return f"{prompt_type} mode selected."

    def add_new_mask(self):
        existing = set(self.cur_masks)
        self.cur_mask_idx = max(existing | {self.cur_mask_idx}) + 1
        self.clear_points()
        return None, f"Creating new object with id {self.cur_mask_idx}"

    def reset(self):
        if self.session_id is not None:
            try:
                self.backend.close_session(self.session_id)
            except Exception as exc:
                guru.debug(f"Ignoring session close error: {exc}")
        self.session_id = None
        self.image = None
        self.cur_mask_idx = 0
        self.frame_index = 0
        self.cur_masks = {}
        self.index_masks_all = []
        self.color_masks_all = []
        self.selected_points.clear()
        self.selected_labels.clear()
        self.selected_point_frames.clear()
        self.selected_point_obj_ids.clear()
        self.current_text_prompt = None
        self.text_prompt_frame_idx = 0

    def set_img_dir(self, img_dir: str) -> int:
        self.reset()
        self.img_dir = img_dir
        self.img_paths = [f"{img_dir}/{p}" for p in sorted(os.listdir(img_dir)) if isimage(p)]
        self.session_id = self.backend.start_session(self.img_dir)
        guru.debug(f"Started SAM 3.1 session: {self.session_id}")
        return len(self.img_paths)

    def set_input_image(self, i: int = 0):
        if i < 0 or i >= len(self.img_paths):
            return self.image
        self.frame_index = i
        self.image = iio.imread(self.img_paths[i])[:, :, :3]
        return self.image

    def _update_masks_from_outputs(self, outputs):
        if self.image is None:
            return {}
        target_shape = self.image.shape[:2]
        masks = output_masks_by_obj(outputs, target_shape=target_shape)
        self.cur_masks.update(masks)
        return masks

    def add_text_prompt(self, text_prompt: str, frame_idx: int):
        if self.session_id is None:
            return None, "Please select an image directory first"
        if self.image is None:
            return None, "Please select a frame first"
        if not text_prompt or not text_prompt.strip():
            return None, "Please enter a text prompt"

        try:
            self.backend.reset_session(self.session_id)
            self.cur_masks.clear()
            self.current_text_prompt = text_prompt.strip()
            self.text_prompt_frame_idx = int(frame_idx)
            response = self.backend.add_prompt(
                self.session_id,
                frame_idx,
                text=self.current_text_prompt,
            )
            self._update_masks_from_outputs(response.get("outputs"))
            if self.cur_masks:
                index_mask = self.make_index_mask(self.cur_masks)
                return index_mask, f"Detected {len(self.cur_masks)} object(s)"
            return None, f"No objects detected for '{text_prompt}'"
        except Exception as exc:
            guru.exception("Text prompt failed")
            return None, f"Error: {exc}"

    def add_point(self, frame_idx, i, j):
        self.selected_points.append([j, i])
        self.selected_labels.append(self.cur_label_val)
        self.selected_point_frames.append(int(frame_idx))
        self.selected_point_obj_ids.append(self.cur_mask_idx)

        current_points = [
            (pt, lbl)
            for pt, lbl, frm, obj_id in zip(
                self.selected_points,
                self.selected_labels,
                self.selected_point_frames,
                self.selected_point_obj_ids,
            )
            if frm == int(frame_idx) and obj_id == self.cur_mask_idx
        ]
        points_array = np.array([pt for pt, _ in current_points], dtype=np.float32)
        labels_array = np.array([lbl for _, lbl in current_points], dtype=np.int32)
        masks = self.get_sam_mask(int(frame_idx), points_array, labels_array)
        self.cur_masks.update(masks)
        index_mask = self.make_index_mask(self.cur_masks)
        if self.index_masks_all and 0 <= int(frame_idx) < len(self.index_masks_all):
            self.index_masks_all[int(frame_idx)] = index_mask
        return index_mask

    def remove_point(self, index: int):
        if 0 <= index < len(self.selected_points):
            del self.selected_points[index]
            del self.selected_labels[index]
            del self.selected_point_frames[index]
            del self.selected_point_obj_ids[index]
            return True
        return False

    def remove_selected_point(self, selected_index: int):
        if selected_index is None or selected_index < 0:
            return None, "No point selected"
        if selected_index >= len(self.selected_points):
            return None, "Invalid point index"

        self.remove_point(selected_index)
        current_points = [
            (pt, lbl)
            for pt, lbl, frm, obj_id in zip(
                self.selected_points,
                self.selected_labels,
                self.selected_point_frames,
                self.selected_point_obj_ids,
            )
            if frm == self.frame_index and obj_id == self.cur_mask_idx
        ]
        if current_points:
            points_array = np.array([pt for pt, _ in current_points], dtype=np.float32)
            labels_array = np.array([lbl for _, lbl in current_points], dtype=np.int32)
            self.cur_masks.update(self.get_sam_mask(self.frame_index, points_array, labels_array))
        else:
            self.cur_masks.pop(self.cur_mask_idx, None)

        index_mask = self.make_index_mask(self.cur_masks)
        return index_mask, f"Removed point. {len(self.selected_points)} points remaining."

    def add_box_prompt(self, frame_idx: int, box_coords: tuple):
        if self.session_id is None:
            return None, "Please select an image directory first"
        if self.image is None:
            return None, "Please select a frame first"

        try:
            h, w = self.image.shape[:2]
            box_xywh = normalize_xyxy_box(box_coords, w, h)
            response = self.backend.add_prompt(
                self.session_id,
                frame_idx,
                bounding_boxes=[box_xywh],
                bounding_box_labels=[1],
            )
            masks = self._update_masks_from_outputs(response.get("outputs"))
            if masks:
                index_mask = self.make_index_mask(self.cur_masks)
                return index_mask, f"Box prompt detected {len(masks)} object(s)"
            return None, "No objects detected in box region"
        except Exception as exc:
            guru.exception("Box prompt failed")
            return None, f"Error: {exc}"

    def get_sam_mask(self, frame_idx, input_points, input_labels):
        if self.session_id is None or self.image is None:
            return {}
        h, w = self.image.shape[:2]
        rel_points = normalize_points(input_points, w, h)
        response = self.backend.add_prompt(
            self.session_id,
            frame_idx,
            obj_id=self.cur_mask_idx,
            points=rel_points,
            point_labels=np.asarray(input_labels, dtype=np.int32).tolist(),
            clear_old_points=True,
            rel_coordinates=True,
        )
        return output_masks_by_obj(response.get("outputs"), target_shape=(h, w))

    def remove_object(self, obj_id: int):
        if self.session_id is None:
            return None, "No active session"
        try:
            response = self.backend.remove_object(self.session_id, obj_id, self.frame_index)
            self.cur_masks.pop(int(obj_id), None)
            masks = self._update_masks_from_outputs(response.get("outputs"))
            if not masks and obj_id in self.cur_masks:
                self.cur_masks.pop(int(obj_id), None)
            index_mask = self.make_index_mask(self.cur_masks)
            if self.cur_masks:
                return index_mask, f"Removed object {obj_id}. {len(self.cur_masks)} object(s) remaining."
            return index_mask, f"Removed object {obj_id}. No objects left."
        except Exception as exc:
            guru.exception("Remove object failed")
            return None, f"Error removing object: {exc}"

    def run_tracker(self, propagation_direction: str = "both"):
        if self.session_id is None:
            return None, "Please load frames first."
        if not self.cur_masks:
            return None, "No objects detected yet."

        images = [iio.imread(p)[:, :, :3] for p in self.img_paths]
        video_segments = {}
        frames_with_masks = 0
        try:
            for result in self.backend.propagate(
                self.session_id,
                propagation_direction=propagation_direction,
            ):
                frame_idx = result.get("frame_index")
                if frame_idx is None:
                    continue
                frame_idx = int(frame_idx)
                if not (0 <= frame_idx < len(images)):
                    continue
                target_shape = images[frame_idx].shape[:2]
                masks = output_masks_by_obj(result.get("outputs"), target_shape=target_shape)
                video_segments[frame_idx] = masks
                if masks:
                    frames_with_masks += 1
        except Exception as exc:
            guru.exception("Tracking failed")
            return None, f"Tracking failed: {exc}"

        if not video_segments:
            return None, "Tracking returned no frames."
        if frames_with_masks == 0:
            return None, "No masks generated. Add prompts first."

        self.index_masks_all = []
        for frame_idx, img in enumerate(images):
            idx_mask = index_mask_from_obj_masks(video_segments.get(frame_idx, {}), img.shape[:2])
            self.index_masks_all.append(idx_mask)

        out_frames, self.color_masks_all = colorize_masks(images, self.index_masks_all)
        out_vidpath = "tracked_colors.mp4"
        iio.mimwrite(out_vidpath, out_frames)
        return out_vidpath, f"Tracked {len(out_frames)} frames. Save masks if it looks good."

    def save_masks_to_dir(self, output_dir: str):
        if not self.color_masks_all:
            return "No masks to save. Run tracking first."
        if not output_dir or not output_dir.strip():
            return "Error: Mask save path is empty. Please load frames first to set the save path."
        os.makedirs(output_dir, exist_ok=True)
        for img_path, clr_mask, id_mask in zip(self.img_paths, self.color_masks_all, self.index_masks_all):
            name = os.path.basename(img_path)
            iio.imwrite(f"{output_dir}/{name}", clr_mask)
            np.save(f"{output_dir}/{name[:-4]}.npy", id_mask)
        return f"Saved masks to {output_dir}."
