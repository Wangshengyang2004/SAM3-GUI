import os

import cv2
import numpy as np
import torch
from PIL import Image
from loguru import logger as guru

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from utils import get_hls_palette, to_numpy


class ImageModeHandler:
    """Handler for single image segmentation (no tracking)."""

    def __init__(self, checkpoint_path=None):
        self.checkpoint_path = checkpoint_path
        self.image_model = None
        self.processor = None
        self.inference_state = None
        self.current_image = None
        self.current_masks = []
        self.current_scores = []

        self.selected_points = []
        self.selected_labels = []
        self.cur_label_val = 1
        self.drawn_box = None

    def init_model(self):
        if self.image_model is None:
            import sam3

            sam3_root = os.path.dirname(sam3.__file__)
            bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"

            self.image_model = build_sam3_image_model(
                bpe_path=bpe_path,
                checkpoint_path=self.checkpoint_path,
                enable_inst_interactivity=True,
            )
            self.processor = Sam3Processor(self.image_model, confidence_threshold=0.3)
            guru.info("Loaded SAM3 image model")

    def set_image(self, image):
        self.init_model()
        if image is None:
            return None, "No image provided"

        if isinstance(image, np.ndarray):
            self.current_image = image
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
            self.current_image = np.array(image)

        self.inference_state = self.processor.set_image(pil_image)
        self.current_masks = []
        self.current_scores = []
        self.selected_points = []
        self.selected_labels = []
        self.drawn_box = None

        return image, "Image loaded. Choose a mode and add prompts."

    def set_positive(self):
        self.cur_label_val = 1
        return "Selecting positive points (foreground)"

    def set_negative(self):
        self.cur_label_val = 0
        return "Selecting negative points (background)"

    def clear_prompts(self):
        self.selected_points = []
        self.selected_labels = []
        self.drawn_box = None
        self.current_masks = []
        self.current_scores = []
        if self.inference_state is not None:
            self.processor.reset_all_prompts(self.inference_state)
        return self.current_image, "Cleared all prompts"

    def reset(self):
        self.current_image = None
        self.inference_state = None
        self.current_masks = []
        self.current_scores = []
        self.selected_points = []
        self.selected_labels = []
        self.drawn_box = None

    def segment_with_box(self, box_coords, text_label=""):
        if self.inference_state is None:
            return None, "Please load an image first"
        if box_coords is None:
            return self.current_image, "Please draw a box on the image"

        try:
            x1, y1, x2, y2 = box_coords
            h, w = self.current_image.shape[:2]

            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = abs(x2 - x1) / w
            bh = abs(y2 - y1) / h
            norm_box = [cx, cy, bw, bh]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                self.processor.reset_all_prompts(self.inference_state)
                self.inference_state = self.processor.add_geometric_prompt(
                    state=self.inference_state,
                    box=norm_box,
                    label=True,
                )

            masks = self.inference_state.get("masks", [])
            scores = self.inference_state.get("scores", [])

            masks = to_numpy(masks, as_float=True)
            scores = to_numpy(scores, as_float=True)

            has_masks = False
            if isinstance(masks, (list, tuple)):
                has_masks = len(masks) > 0
            elif hasattr(masks, "shape"):
                has_masks = masks.size > 0

            if has_masks:
                self.current_masks = list(masks) if hasattr(masks, "__iter__") else [masks]
                self.current_scores = list(scores) if hasattr(scores, "__iter__") else [scores]
                return self._visualize_masks(show_all=True), f"Segmented with box, {len(self.current_masks)} mask(s)"

            return self.current_image, "No objects found in box region"
        except Exception as e:
            guru.error(f"Box segmentation error: {e}")
            return self.current_image, f"Error: {str(e)}"

    def find_all_with_text(self, text_prompt: str):
        if self.inference_state is None:
            return None, "Please load an image first"
        if not text_prompt or not text_prompt.strip():
            return self.current_image, "Please enter a text prompt"

        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                self.processor.reset_all_prompts(self.inference_state)
                self.inference_state = self.processor.set_text_prompt(
                    state=self.inference_state,
                    prompt=text_prompt.strip(),
                )

            masks = self.inference_state.get("masks", [])
            scores = self.inference_state.get("scores", [])

            masks = to_numpy(masks, as_float=True)
            scores = to_numpy(scores, as_float=True)

            has_masks = False
            if isinstance(masks, (list, tuple)):
                has_masks = len(masks) > 0
            elif hasattr(masks, "shape"):
                has_masks = masks.size > 0

            if has_masks:
                self.current_masks = list(masks) if hasattr(masks, "__iter__") else [masks]
                self.current_scores = list(scores) if hasattr(scores, "__iter__") else [scores]
                return self._visualize_masks(show_all=True), f"Found {len(self.current_masks)} instance(s) of '{text_prompt}'"

            return self.current_image, f"No '{text_prompt}' found in image"
        except Exception as e:
            guru.error(f"Text search error: {e}")
            return self.current_image, f"Error: {str(e)}"

    def add_point(self, x, y):
        self.selected_points.append([x, y])
        self.selected_labels.append(self.cur_label_val)
        return self._segment_with_points()

    def remove_point(self, index: int):
        """Remove a specific point by index."""
        if index < 0 or index >= len(self.selected_points):
            return self.current_image, "Invalid point index"

        del self.selected_points[index]
        del self.selected_labels[index]

        # Re-generate mask with remaining points
        if self.selected_points:
            return self._segment_with_points()
        return self.current_image, "All points removed"

    def _segment_with_points(self):
        if self.inference_state is None:
            return self.current_image, "Please load an image first"
        if not self.selected_points:
            return self.current_image, "Click on the image to add points"

        try:
            input_points = np.array(self.selected_points)
            input_labels = np.array(self.selected_labels)

            masks, scores, _ = self.image_model.predict_inst(
                self.inference_state,
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )

            sorted_ind = np.argsort(scores)[::-1]
            self.current_masks = [masks[i] for i in sorted_ind]
            self.current_scores = [scores[i] for i in sorted_ind]
            return (
                self._visualize_masks(show_points=True),
                f"Generated {len(masks)} mask(s), best score: {self.current_scores[0]:.2f}",
            )
        except Exception as e:
            guru.error(f"Point segmentation error: {e}")
            return self.current_image, f"Error: {str(e)}"

    def _visualize_masks(self, show_points=False, mask_idx=0, show_all=False):
        if self.current_image is None:
            return None

        out_img = self.current_image.copy()

        if self.current_masks:
            if show_all:
                mask_indices = range(len(self.current_masks))
            else:
                mask_indices = [mask_idx] if mask_idx < len(self.current_masks) else []

            if mask_indices:
                palette = get_hls_palette(len(list(mask_indices)) + 1)[1:]
                overlay = np.zeros_like(out_img)
                h, w = out_img.shape[:2]

                for color_idx, idx in enumerate(mask_indices):
                    mask = self.current_masks[idx]
                    if mask.ndim == 3:
                        mask = mask.squeeze()
                    if mask.shape[0] != h or mask.shape[1] != w:
                        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    bin_mask = mask > 0
                    overlay[bin_mask] = palette[color_idx]
                    contours, _ = cv2.findContours(bin_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

        mask = self.current_masks[0]
        if mask.ndim == 3:
            mask = mask.squeeze()

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        np.save(output_path, mask)
        return f"Saved mask to {output_path}"

    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold for mask generation."""
        if self.processor is not None:
            self.processor.confidence_threshold = threshold
            return f"Confidence threshold set to {threshold:.2f}"
        return "Model not loaded yet. Threshold will apply on next image load."
