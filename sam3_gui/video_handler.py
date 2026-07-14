import os
from pathlib import Path
from uuid import uuid4

import imageio.v2 as iio
import numpy as np
from loguru import logger as guru

from sam3_gui.sam31_backend import (
    Sam31Backend,
    index_mask_from_obj_masks,
    normalize_points,
    normalize_xyxy_box,
    output_masks_by_obj,
    validate_sam31_checkpoint_path,
)
from sam3_gui.paths import runtime_output_path
from sam3_gui.utils import colorize_masks, isimage


class VideoModeHandler:
    """Native SAM 3.1 video segmentation and tracking handler."""

    def __init__(
        self,
        checkpoint_path=None,
        gpus_to_use=None,
        backend: Sam31Backend | None = None,
    ):
        self.backend = backend or Sam31Backend()
        if checkpoint_path is not None:
            self.backend.config.checkpoint_path = validate_sam31_checkpoint_path(
                checkpoint_path
            )
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
        self.box_prompts = []
        self.removed_object_ids = set()
        self.tracking_output_path = None

    @property
    def inference_state(self):
        return self.session_id

    def init_model(self):
        self.backend.ensure_predictor()

    def _resize_mask(self, mask, target_h, target_w):
        from sam3_gui.sam31_backend import resize_mask

        return resize_mask(mask, target_h, target_w)

    def _normalize_points(self, points):
        if self.image is None:
            return points
        h, w = self.image.shape[:2]
        return normalize_points(points, w, h)

    def make_index_mask(self, masks):
        fallback = self.image.shape[:2] if self.image is not None else (1, 1)
        return index_mask_from_obj_masks(masks, fallback)

    def _invalidate_tracking_results(self):
        self.index_masks_all = []
        self.color_masks_all = []
        if self.tracking_output_path:
            try:
                Path(self.tracking_output_path).unlink(missing_ok=True)
            except OSError as exc:
                guru.debug(f"Ignoring tracked video cleanup error: {exc}")
        self.tracking_output_path = None

    def _frame_shape(self, frame_idx: int):
        if frame_idx == self.frame_index and self.image is not None:
            return self.image.shape[:2]
        if 0 <= frame_idx < len(self.img_paths):
            return iio.imread(self.img_paths[frame_idx]).shape[:2]
        if self.image is not None:
            return self.image.shape[:2]
        return 1, 1

    def _record_outputs(self, response, frame_idx: int):
        target_shape = self._frame_shape(frame_idx)
        masks = output_masks_by_obj(response.get("outputs"), target_shape=target_shape)
        self.cur_masks.update(masks)
        return masks

    def _replay_prompt_state(self):
        if self.session_id is None:
            self.cur_masks.clear()
            return {}

        self.backend.reset_session(self.session_id)
        self.cur_masks.clear()
        latest_masks = {}

        if self.current_text_prompt:
            response = self.backend.add_prompt(
                self.session_id,
                self.text_prompt_frame_idx,
                text=self.current_text_prompt,
            )
            latest_masks = self._record_outputs(response, self.text_prompt_frame_idx)

        for prompt in self.box_prompts:
            response = self.backend.add_prompt(
                self.session_id,
                prompt["frame_idx"],
                bounding_boxes=[prompt["box_xywh"]],
                bounding_box_labels=[1],
            )
            masks = self._record_outputs(response, prompt["frame_idx"])
            prompt["obj_ids"] = set(masks)
            latest_masks = masks

        point_groups = {}
        for point, label, frame_idx, obj_id in zip(
            self.selected_points,
            self.selected_labels,
            self.selected_point_frames,
            self.selected_point_obj_ids,
        ):
            point_groups.setdefault((frame_idx, obj_id), []).append((point, label))

        for (frame_idx, obj_id), points_and_labels in point_groups.items():
            height, width = self._frame_shape(frame_idx)
            points = np.asarray(
                [point for point, _ in points_and_labels], dtype=np.float32
            )
            labels = np.asarray(
                [label for _, label in points_and_labels], dtype=np.int32
            )
            response = self.backend.add_prompt(
                self.session_id,
                frame_idx,
                obj_id=obj_id,
                points=normalize_points(points, width, height),
                point_labels=labels.tolist(),
                clear_old_points=True,
                rel_coordinates=True,
            )
            latest_masks = self._record_outputs(response, frame_idx)

        for obj_id in sorted(self.removed_object_ids):
            if obj_id not in self.cur_masks:
                continue
            response = self.backend.remove_object(
                self.session_id, obj_id, self.frame_index
            )
            self.cur_masks.pop(obj_id, None)
            self._record_outputs(response, self.frame_index)

        return latest_masks

    def clear_points(self):
        self.selected_points.clear()
        self.selected_labels.clear()
        self.selected_point_frames.clear()
        self.selected_point_obj_ids.clear()
        self._invalidate_tracking_results()
        try:
            self._replay_prompt_state()
        except Exception as exc:
            guru.exception("Clearing points failed")
            return None, None, f"Error clearing points: {exc}"
        index_mask = self.make_index_mask(self.cur_masks) if self.cur_masks else None
        return index_mask, None, "Cleared points"

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
        self._invalidate_tracking_results()
        try:
            self._replay_prompt_state()
        except Exception as exc:
            guru.exception("Creating object failed")
            return None, f"Error creating object: {exc}"
        return None, f"Creating new object with id {self.cur_mask_idx}"

    def reset(self):
        self._invalidate_tracking_results()
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
        self.box_prompts.clear()
        self.removed_object_ids.clear()
        self.img_dir = ""
        self.img_paths = []

    def set_img_dir(self, img_dir: str) -> int:
        self.reset()
        self.img_dir = img_dir
        self.img_paths = [
            f"{img_dir}/{p}" for p in sorted(os.listdir(img_dir)) if isimage(p)
        ]
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
            self.current_text_prompt = text_prompt.strip()
            self.text_prompt_frame_idx = int(frame_idx)
            self.removed_object_ids.clear()
            self._invalidate_tracking_results()
            self._replay_prompt_state()
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

        self.removed_object_ids.discard(self.cur_mask_idx)
        self._invalidate_tracking_results()
        self._replay_prompt_state()
        index_mask = self.make_index_mask(self.cur_masks)
        return index_mask

    def remove_point(self, index: int):
        if 0 <= index < len(self.selected_points):
            del self.selected_points[index]
            del self.selected_labels[index]
            del self.selected_point_frames[index]
            del self.selected_point_obj_ids[index]
            self._invalidate_tracking_results()
            self._replay_prompt_state()
            return True
        return False

    def remove_selected_point(self, selected_index: int):
        if selected_index is None or selected_index < 0:
            return None, "No point selected"
        if selected_index >= len(self.selected_points):
            return None, "Invalid point index"

        try:
            self.remove_point(selected_index)
        except Exception as exc:
            guru.exception("Remove point failed")
            return None, f"Error removing point: {exc}"

        index_mask = self.make_index_mask(self.cur_masks) if self.cur_masks else None
        return (
            index_mask,
            f"Removed point. {len(self.selected_points)} points remaining.",
        )

    def add_box_prompt(self, frame_idx: int, box_coords: tuple):
        if self.session_id is None:
            return None, "Please select an image directory first"
        if self.image is None:
            return None, "Please select a frame first"

        try:
            h, w = self.image.shape[:2]
            box_xywh = normalize_xyxy_box(box_coords, w, h)
            prompt = {
                "frame_idx": int(frame_idx),
                "box_xywh": box_xywh,
                "obj_ids": set(),
            }
            self.box_prompts.append(prompt)
            self._invalidate_tracking_results()
            masks = self._replay_prompt_state()
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
        h, w = self._frame_shape(int(frame_idx))
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
            obj_id = int(obj_id)
            retained_points = [
                (point, label, frame_idx, point_obj_id)
                for point, label, frame_idx, point_obj_id in zip(
                    self.selected_points,
                    self.selected_labels,
                    self.selected_point_frames,
                    self.selected_point_obj_ids,
                )
                if point_obj_id != obj_id
            ]
            self.selected_points = [record[0] for record in retained_points]
            self.selected_labels = [record[1] for record in retained_points]
            self.selected_point_frames = [record[2] for record in retained_points]
            self.selected_point_obj_ids = [record[3] for record in retained_points]
            self.box_prompts = [
                prompt for prompt in self.box_prompts if obj_id not in prompt["obj_ids"]
            ]
            self.removed_object_ids.add(obj_id)
            self._invalidate_tracking_results()
            self._replay_prompt_state()
            self.cur_masks.pop(obj_id, None)
            index_mask = self.make_index_mask(self.cur_masks)
            if self.cur_masks:
                return (
                    index_mask,
                    f"Removed object {obj_id}. {len(self.cur_masks)} object(s) remaining.",
                )
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
                masks = output_masks_by_obj(
                    result.get("outputs"), target_shape=target_shape
                )
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
            idx_mask = index_mask_from_obj_masks(
                video_segments.get(frame_idx, {}), img.shape[:2]
            )
            self.index_masks_all.append(idx_mask)

        out_frames, self.color_masks_all = colorize_masks(images, self.index_masks_all)
        output_id = uuid4().hex
        out_vidpath = Path(runtime_output_path(f"tracked_colors_{output_id}.mp4"))
        temp_vidpath = out_vidpath.with_name(
            f".{out_vidpath.stem}.{uuid4().hex}.tmp{out_vidpath.suffix}"
        )
        try:
            iio.mimwrite(str(temp_vidpath), out_frames)
            os.replace(temp_vidpath, out_vidpath)
        except Exception as exc:
            temp_vidpath.unlink(missing_ok=True)
            guru.exception("Writing tracked video failed")
            return None, f"Tracking failed: {exc}"
        self.tracking_output_path = str(out_vidpath)
        return (
            self.tracking_output_path,
            f"Tracked {len(out_frames)} frames. Save masks if it looks good.",
        )

    def save_masks_to_dir(self, output_dir: str):
        if not self.color_masks_all:
            return "No masks to save. Run tracking first."
        if not output_dir or not output_dir.strip():
            return "Error: Mask save path is empty. Please load frames first to set the save path."
        os.makedirs(output_dir, exist_ok=True)
        output_path = Path(output_dir)
        for img_path, clr_mask, id_mask in zip(
            self.img_paths, self.color_masks_all, self.index_masks_all
        ):
            source_path = Path(img_path)
            iio.imwrite(
                output_path / f"{source_path.stem}{source_path.suffix}", clr_mask
            )
            np.save(output_path / f"{source_path.stem}.npy", id_mask)
        return f"Saved masks to {output_dir}."
