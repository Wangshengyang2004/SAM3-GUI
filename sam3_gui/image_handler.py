import os
import tempfile

import cv2
import numpy as np
from PIL import Image
from loguru import logger as guru

from sam3_gui.sam31_backend import (
    Sam31Backend,
    normalize_points,
    normalize_xyxy_box,
    output_masks_by_obj,
    validate_sam31_checkpoint_path,
)
from sam3_gui.utils import get_hls_palette


class ImageModeHandler:
    """Native SAM 3.1 single-image segmentation handler."""

    def __init__(self, checkpoint_path=None, backend: Sam31Backend | None = None):
        self.backend = backend or Sam31Backend()
        if checkpoint_path is not None:
            self.backend.config.checkpoint_path = validate_sam31_checkpoint_path(
                checkpoint_path
            )
        self.session_id = None
        self.current_image = None
        self.current_masks = []
        self.current_scores = []
        self.selected_points = []
        self.selected_labels = []
        self.cur_label_val = 1
        self.drawn_box = None
        self.confidence_threshold = 0.3
        self.point_obj_id = 0

    @property
    def inference_state(self):
        return self.session_id

    def init_model(self):
        self.backend.ensure_predictor()

    def _clear_prompt_state(self):
        self.current_masks = []
        self.current_scores = []
        self.selected_points = []
        self.selected_labels = []
        self.cur_label_val = 1
        self.drawn_box = None
        self.point_obj_id = 0

    def _clear_state(self):
        self.session_id = None
        self.current_image = None
        self._clear_prompt_state()

    def set_image(self, image, resource_path=None):
        if image is None and resource_path is None:
            return None, "No image provided"

        self.init_model()

        if resource_path is not None:
            with Image.open(resource_path) as source_image:
                pil_image = source_image.convert("RGB")
            new_image = np.array(pil_image)
            session_resource = resource_path
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                new_image = np.stack([image] * 3, axis=-1)
            elif image.ndim == 3 and image.shape[2] >= 3:
                new_image = image[:, :, :3]
            elif image.ndim == 3 and image.shape[2] == 1:
                new_image = np.repeat(image, 3, axis=2)
            else:
                raise ValueError("Image must be a 2D grayscale or 3D color array")
            pil_image = Image.fromarray(new_image).convert("RGB")
            new_image = np.array(pil_image)
            session_resource = [pil_image]
        else:
            pil_image = (
                image.convert("RGB")
                if isinstance(image, Image.Image)
                else Image.fromarray(np.array(image))
            )
            pil_image = pil_image.convert("RGB")
            new_image = np.array(pil_image)
            session_resource = [pil_image]

        new_session_id = self.backend.start_session(session_resource)
        old_session_id = self.session_id

        self.session_id = new_session_id
        self.current_image = new_image
        self._clear_prompt_state()

        if old_session_id is not None and old_session_id != new_session_id:
            try:
                self.backend.close_session(old_session_id)
            except Exception as exc:
                guru.debug(f"Ignoring image session close error: {exc}")

        return self.current_image, "Image loaded. Choose a mode and add prompts."

    def set_positive(self):
        self.cur_label_val = 1
        return "Selecting positive points"

    def set_negative(self):
        self.cur_label_val = 0
        return "Selecting negative points"

    def clear_prompts(self):
        self.selected_points = []
        self.selected_labels = []
        self.drawn_box = None
        self.current_masks = []
        self.current_scores = []
        if self.session_id is not None:
            self.backend.reset_session(self.session_id)
        return self.current_image, "Cleared all prompts"

    def reset(self):
        session_id = self.session_id
        self._clear_state()
        if session_id is not None:
            try:
                self.backend.close_session(session_id)
            except Exception as exc:
                guru.debug(f"Ignoring image session close error: {exc}")

    def close(self):
        self.reset()

    def _set_outputs(self, outputs):
        if self.current_image is None:
            self.current_masks = []
            self.current_scores = []
            return
        masks_by_obj = output_masks_by_obj(
            outputs, target_shape=self.current_image.shape[:2]
        )
        self.current_masks = [masks_by_obj[obj_id] for obj_id in sorted(masks_by_obj)]
        probs = outputs.get("out_probs", []) if outputs else []
        if hasattr(probs, "detach"):
            probs = probs.detach().cpu().numpy()
        self.current_scores = list(np.asarray(probs, dtype=np.float32).reshape(-1))

    def segment_with_box(self, box_coords, text_label=""):
        if self.session_id is None:
            return None, "Please load an image first"
        if box_coords is None:
            return self.current_image, "Please draw a box on the image"

        try:
            h, w = self.current_image.shape[:2]
            box_xywh = normalize_xyxy_box(box_coords, w, h)
            self.backend.reset_session(self.session_id)
            response = self.backend.add_prompt(
                self.session_id,
                0,
                bounding_boxes=[box_xywh],
                bounding_box_labels=[1],
                output_prob_thresh=self.confidence_threshold,
            )
            self._set_outputs(response.get("outputs"))
            if self.current_masks:
                return self._visualize_masks(
                    show_all=True
                ), f"Segmented with box, {len(self.current_masks)} mask(s)"
            return self.current_image, "No objects found in box region"
        except Exception as exc:
            guru.exception("Box segmentation failed")
            return self.current_image, f"Error: {exc}"

    def find_all_with_text(self, text_prompt: str):
        if self.session_id is None:
            return None, "Please load an image first"
        if not text_prompt or not text_prompt.strip():
            return self.current_image, "Please enter a text prompt"

        try:
            self.backend.reset_session(self.session_id)
            response = self.backend.add_prompt(
                self.session_id,
                0,
                text=text_prompt.strip(),
                output_prob_thresh=self.confidence_threshold,
            )
            self._set_outputs(response.get("outputs"))
            if self.current_masks:
                return self._visualize_masks(
                    show_all=True
                ), f"Found {len(self.current_masks)} instance(s) of '{text_prompt}'"
            return self.current_image, f"No '{text_prompt}' found in image"
        except Exception as exc:
            guru.exception("Text search failed")
            return self.current_image, f"Error: {exc}"

    def add_point(self, x, y):
        self.selected_points.append([x, y])
        self.selected_labels.append(self.cur_label_val)
        return self._segment_with_points()

    def remove_point(self, index: int):
        if index < 0 or index >= len(self.selected_points):
            return self.current_image, "Invalid point index"
        del self.selected_points[index]
        del self.selected_labels[index]
        if self.selected_points:
            return self._segment_with_points()
        self.current_masks = []
        self.current_scores = []
        if self.session_id is not None:
            self.backend.reset_session(self.session_id)
        return self.current_image, "All points removed"

    def _segment_with_points(self):
        if self.session_id is None:
            return self.current_image, "Please load an image first"
        if not self.selected_points:
            return self.current_image, "Click on the image to add points"

        try:
            h, w = self.current_image.shape[:2]
            response = self.backend.add_prompt(
                self.session_id,
                0,
                obj_id=self.point_obj_id,
                points=normalize_points(self.selected_points, w, h),
                point_labels=np.asarray(self.selected_labels, dtype=np.int32).tolist(),
                clear_old_points=True,
                rel_coordinates=True,
                output_prob_thresh=self.confidence_threshold,
            )
            self._set_outputs(response.get("outputs"))
            if self.current_masks:
                score_msg = (
                    f", best score: {float(self.current_scores[0]):.2f}"
                    if self.current_scores
                    else ""
                )
                return self._visualize_masks(
                    show_points=True
                ), f"Generated {len(self.current_masks)} mask(s){score_msg}"
            return self.current_image, "No mask generated from points"
        except Exception as exc:
            guru.exception("Point segmentation failed")
            return self.current_image, f"Error: {exc}"

    def _visualize_masks(self, show_points=False, mask_idx=0, show_all=False):
        if self.current_image is None:
            return None

        out_img = self.current_image.copy()
        if self.current_masks:
            mask_indices = range(len(self.current_masks)) if show_all else [mask_idx]
            mask_indices = [
                idx for idx in mask_indices if idx < len(self.current_masks)
            ]
            palette = get_hls_palette(len(mask_indices) + 1)[1:]
            overlay = np.zeros_like(out_img)
            h, w = out_img.shape[:2]

            for color_idx, idx in enumerate(mask_indices):
                mask = np.squeeze(self.current_masks[idx])
                if mask.shape != (h, w):
                    mask = cv2.resize(
                        mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                    )
                bin_mask = mask > 0
                overlay[bin_mask] = palette[color_idx]
                contours, _ = cv2.findContours(
                    bin_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(out_img, contours, -1, (255, 255, 255), 2)
            out_img = cv2.addWeighted(out_img, 0.65, overlay, 0.35, 0)

        if show_points:
            for pt, label in zip(self.selected_points, self.selected_labels):
                color = (0, 255, 0) if label == 1 else (255, 0, 0)
                cv2.circle(out_img, (int(pt[0]), int(pt[1])), 8, color, -1)
                cv2.circle(out_img, (int(pt[0]), int(pt[1])), 8, (255, 255, 255), 2)
        return out_img

    def save_mask(self, output_path: str):
        if not self.current_masks:
            return "No mask to save. Run segmentation first."
        masks = np.stack([np.squeeze(mask) for mask in self.current_masks])
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        temp_fd, temp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(output_path)}.",
            suffix=".tmp",
            dir=output_dir,
        )
        try:
            with os.fdopen(temp_fd, "wb") as temp_file:
                np.save(temp_file, masks)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_path, output_path)
        except Exception:
            try:
                os.close(temp_fd)
            except OSError:
                pass
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            raise
        return f"Saved {len(masks)} mask(s) to {output_path}"

    def set_confidence_threshold(self, threshold: float):
        self.confidence_threshold = float(threshold)
        return f"Confidence threshold set to {self.confidence_threshold:.2f}"
