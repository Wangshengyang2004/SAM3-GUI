import os

import cv2
import imageio.v2 as iio
import numpy as np
import torch
from loguru import logger as guru

from sam3.model_builder import build_sam3_video_predictor
from utils import colorize_masks, isimage


class VideoModeHandler:
    """Handler for video segmentation with tracking."""

    def __init__(self, checkpoint_path=None, gpus_to_use=None):
        self.checkpoint_path = checkpoint_path
        self.gpus_to_use = gpus_to_use
        self.video_predictor = None
        self.tracker = None
        self.tracker_state = None
        self.tracker_num_frames = 0
        self.inference_state = None
        self.use_text_mode = False

        self.selected_points = []
        self.selected_labels = []
        self.selected_point_frames = []
        self.selected_point_obj_ids = []  # Track which object each point belongs to
        self.cur_label_val = 1.0

        self.frame_index = 0
        self.image = None
        self.cur_mask_idx = 0
        self.cur_masks = {}
        self.index_masks_all = []
        self.color_masks_all = []

        self.img_dir = ""
        self.img_paths = []

        # Store text prompt info for re-applying before tracking
        self.current_text_prompt = None
        self.text_prompt_frame_idx = 0
        # Store box prompts for multiple objects: {obj_id: {"box": [xmin, ymin, w, h], "frame_idx": int}}
        self.box_prompts = {}

    def _init_tracker_state(self):
        """Initialize point-tracking state lazily (only when points mode is used)."""
        if self.tracker_state is not None:
            return True
        try:
            # Use autocast to match the dtype used in SAM3 model (BFloat16)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                self.tracker_state = self.tracker.init_state(video_path=self.img_dir)
            self.tracker_num_frames = self.tracker_state.get("num_frames", 0)
            guru.debug(f"Initialized tracker state for points mode ({self.tracker_num_frames} frames)")
            return True
        except (ValueError, RuntimeError) as e:
            guru.warning(f"Points mode unavailable: {e}. Text mode still works.")
            self.tracker_state = None
            self.tracker_num_frames = 0
            return False

    def init_model(self):
        if self.video_predictor is None:
            if self.gpus_to_use is None:
                num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
                gpus_to_use = list(range(num_gpus)) if num_gpus > 0 else []
            else:
                gpus_to_use = self.gpus_to_use

            self.video_predictor = build_sam3_video_predictor(
                checkpoint_path=self.checkpoint_path,
                gpus_to_use=gpus_to_use,
            )
            self.tracker = self.video_predictor.model.tracker
            self.tracker.backbone = self.video_predictor.model.detector.backbone
            guru.info(f"Loaded SAM3 video predictor, using GPUs: {gpus_to_use}")

    def _resize_mask(self, mask, target_h, target_w):
        if mask.shape[0] == target_h and mask.shape[1] == target_w:
            return mask
        return cv2.resize(
            mask.astype(np.uint8),
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    def _normalize_points(self, points):
        if self.image is None:
            return points
        h, w = self.image.shape[:2]
        if hasattr(points, "tolist"):
            points = points.tolist()
        return [[x / w, y / h] for x, y in points]

    def make_index_mask(self, masks):
        if not masks:
            if self.image is not None:
                return np.zeros((self.image.shape[0], self.image.shape[1]), dtype="uint8")
            return np.zeros((1, 1), dtype="uint8")

        idcs = sorted(masks.keys())  # Sort for consistent ordering
        # Get shape from first mask, ensuring we have a valid shape
        first_mask = masks[idcs[0]]
        if hasattr(first_mask, 'shape'):
            mask_shape = first_mask.shape
        else:
            # Fallback if mask doesn't have shape (shouldn't happen)
            mask_shape = (self.image.shape[0], self.image.shape[1]) if self.image is not None else (1, 1)

        idx_mask = np.zeros(mask_shape, dtype="uint8")
        for i in idcs:
            mask = masks[i]
            # Ensure mask is boolean or can be used for indexing
            if mask.dtype != bool:
                mask = mask > 0
            idx_mask[mask] = int(i) + 1  # Ensure i is int
        guru.debug(f"make_index_mask: {len(idcs)} objects, mask max={idx_mask.max()}, shape={idx_mask.shape}")
        return idx_mask

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
        """Switch interaction mode explicitly from the UI."""
        if prompt_type == "Points":
            self.use_text_mode = False
            self.cur_masks = {}
            return "Points mode selected."
        return "Text mode selected."

    def add_new_mask(self):
        self.cur_mask_idx += 1
        self.clear_points()
        return None, f"Creating new mask with index {self.cur_mask_idx}"

    def reset(self):
        self.image = None
        self.cur_mask_idx = 0
        self.frame_index = 0
        self.cur_masks = {}
        self.index_masks_all = []
        self.color_masks_all = []
        self.use_text_mode = False
        self.selected_points.clear()
        self.selected_labels.clear()
        self.selected_point_frames.clear()
        self.selected_point_obj_ids.clear()
        self.current_text_prompt = None
        self.text_prompt_frame_idx = 0
        self.box_prompts = {}

        if self.inference_state is not None:
            try:
                self.video_predictor.handle_request(
                    request=dict(type="close_session", session_id=self.inference_state)
                )
            except Exception:
                pass
        self.inference_state = None

        if self.tracker_state is not None:
            try:
                self.tracker.clear_all_points_in_video(self.tracker_state)
            except Exception:
                pass
        self.tracker_state = None

    def set_img_dir(self, img_dir: str) -> int:
        self.reset()
        self.init_model()
        self.img_dir = img_dir
        self.img_paths = [f"{img_dir}/{p}" for p in sorted(os.listdir(img_dir)) if isimage(p)]

        response = self.video_predictor.handle_request(
            request=dict(type="start_session", resource_path=self.img_dir)
        )
        self.inference_state = response["session_id"]
        guru.debug(f"Started SAM3 session: {self.inference_state}")

        return len(self.img_paths)

    def set_input_image(self, i: int = 0):
        if i < 0 or i >= len(self.img_paths):
            return self.image
        # Don't clear points when changing frames - points should persist across frames
        self.frame_index = i
        self.image = iio.imread(self.img_paths[i])
        return self.image

    def add_text_prompt(self, text_prompt: str, frame_idx: int):
        if self.inference_state is None:
            return None, "Please select an image directory first"
        if self.image is None:
            return None, "Please select a frame first"

        try:
            self.use_text_mode = True
            # Save prompt info for re-applying before tracking
            self.current_text_prompt = text_prompt
            self.text_prompt_frame_idx = frame_idx

            self.video_predictor.handle_request(
                request=dict(type="reset_session", session_id=self.inference_state)
            )
            self.cur_masks.clear()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                response = self.video_predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=self.inference_state,
                        frame_index=frame_idx,
                        text=text_prompt,
                    )
                )

            outputs = response.get("outputs", {})
            if outputs:
                masks = outputs.get("out_binary_masks", [])
                scores = outputs.get("out_probs", [])
                obj_ids = outputs.get("out_obj_ids", [])

                # Convert to lists if numpy arrays (keep masks as arrays for _resize_mask)
                if hasattr(scores, 'tolist'):
                    scores = scores.tolist()
                if hasattr(obj_ids, 'tolist'):
                    obj_ids = obj_ids.tolist()

                target_h, target_w = self.image.shape[:2]
                for i, obj_id in enumerate(obj_ids):
                    if i < len(masks):
                        mask_array = np.array(masks[i])
                        self.cur_masks[int(obj_id)] = self._resize_mask(mask_array, target_h, target_w).copy()

                if self.cur_masks:
                    index_mask = self.make_index_mask(self.cur_masks)
                    avg_score = np.mean(scores) if scores else 0.0
                    return index_mask, f"Detected {len(masks)} object(s), avg confidence: {avg_score:.2f}"

            return None, f"No objects detected for '{text_prompt}'"
        except Exception as e:
            guru.error(f"Text prompt error: {e}")
            self.use_text_mode = False
            self.current_text_prompt = None
            return None, f"Error: {str(e)}"

    def add_point(self, frame_idx, i, j):
        self.selected_points.append([j, i])
        self.selected_labels.append(self.cur_label_val)
        self.selected_point_frames.append(frame_idx)
        self.selected_point_obj_ids.append(self.cur_mask_idx)  # Track which object this point belongs to
        guru.debug(
            f"Added point at frame={frame_idx}, xy=({j},{i}), "
            f"label={self.cur_label_val}, obj_id={self.cur_mask_idx}, total_points={len(self.selected_points)}"
        )
        # Only pass points that belong to the current frame and current object
        # SAM3 stores points per-frame in point_inputs_per_frame[frame_idx]
        # Passing points from other frames would incorrectly add them to the current frame
        current_frame_points = [
            (pt, lbl) for pt, lbl, frm, obj_id in zip(
                self.selected_points,
                self.selected_labels,
                self.selected_point_frames,
                self.selected_point_obj_ids
            ) if frm == frame_idx and obj_id == self.cur_mask_idx
        ]

        if current_frame_points:
            points_array = np.array([pt for pt, _ in current_frame_points], dtype=np.float32)
            labels_array = np.array([lbl for _, lbl in current_frame_points], dtype=np.int32)
        else:
            points_array = np.array([], dtype=np.float32).reshape(0, 2)
            labels_array = np.array([], dtype=np.int32)

        masks = self.get_sam_mask(frame_idx, points_array, labels_array)

        # Update cur_masks with the new mask for current object
        # This preserves masks from other objects
        for obj_id, mask in masks.items():
            self.cur_masks[obj_id] = mask

        # Generate index mask from all objects in cur_masks
        index_mask = self.make_index_mask(self.cur_masks)

        # Update the tracked masks for this frame if we have tracked results
        if self.index_masks_all and 0 <= frame_idx < len(self.index_masks_all):
            self.index_masks_all[frame_idx] = index_mask
            guru.debug(f"Updated index_masks_all for frame {frame_idx}")

        return index_mask

    def remove_point(self, index: int):
        """Remove a specific point by index."""
        if 0 <= index < len(self.selected_points):
            del self.selected_points[index]
            del self.selected_labels[index]
            del self.selected_point_frames[index]
            del self.selected_point_obj_ids[index]
            return True
        return False

    def remove_selected_point(self, selected_index: int):
        """Remove point based on index in the points list."""
        if selected_index is None or selected_index < 0:
            return None, "No point selected"
        if selected_index >= len(self.selected_points):
            return None, "Invalid point index"

        del self.selected_points[selected_index]
        del self.selected_labels[selected_index]
        del self.selected_point_frames[selected_index]
        del self.selected_point_obj_ids[selected_index]

        # Re-generate mask with remaining points from the current frame and current object only
        if self.selected_points and self.image is not None:
            current_frame_points = [
                (pt, lbl) for pt, lbl, frm, obj_id in zip(
                    self.selected_points,
                    self.selected_labels,
                    self.selected_point_frames,
                    self.selected_point_obj_ids
                ) if frm == self.frame_index and obj_id == self.cur_mask_idx
            ]

            if current_frame_points:
                points_array = np.array([pt for pt, _ in current_frame_points], dtype=np.float32)
                labels_array = np.array([lbl for _, lbl in current_frame_points], dtype=np.int32)
            else:
                points_array = np.array([], dtype=np.float32).reshape(0, 2)
                labels_array = np.array([], dtype=np.int32)

            masks = self.get_sam_mask(self.frame_index, points_array, labels_array)

            # Update cur_masks with the new mask for current object
            # This preserves masks from other objects
            for obj_id, mask in masks.items():
                self.cur_masks[obj_id] = mask

            # Generate index mask from all objects in cur_masks
            index_mask = self.make_index_mask(self.cur_masks)

            # Update the tracked masks for this frame if we have tracked results
            if self.index_masks_all and 0 <= self.frame_index < len(self.index_masks_all):
                self.index_masks_all[self.frame_index] = index_mask
                guru.debug(f"Updated index_masks_all for frame {self.frame_index}")

            return index_mask, f"Removed point. {len(self.selected_points)} points remaining."
        return None, f"Removed point. No points remaining."

    def add_box_prompt(self, frame_idx: int, box_coords: tuple):
        """Add a box prompt for video segmentation.

        Args:
            frame_idx: Frame index where box is drawn
            box_coords: (x1, y1, x2, y2) in pixel coordinates

        Returns:
            (index_mask, message)
        """
        if self.inference_state is None:
            return None, "Please select an image directory first"
        if self.image is None:
            return None, "Please select a frame first"

        try:
            # Only reset session if this is the first box prompt (switching from points mode)
            first_box = self.cur_mask_idx == 0 and not self.box_prompts
            if first_box:
                self.use_text_mode = True
                self.video_predictor.handle_request(
                    request=dict(type="reset_session", session_id=self.inference_state)
                )
                self.cur_masks.clear()

            x1, y1, x2, y2 = box_coords
            h, w = self.image.shape[:2]

            # Normalize to [xmin, ymin, width, height] in 0-1 range
            xmin = min(x1, x2) / w
            ymin = min(y1, y2) / h
            box_width = abs(x2 - x1) / w
            box_height = abs(y2 - y1) / h

            # Save box prompt info for this object
            self.box_prompts[self.cur_mask_idx] = {
                "box": [xmin, ymin, box_width, box_height],
                "frame_idx": frame_idx
            }

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                response = self.video_predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=self.inference_state,
                        frame_index=frame_idx,
                        bounding_boxes=[[xmin, ymin, box_width, box_height]],
                        bounding_box_labels=[1],
                        obj_id=self.cur_mask_idx,
                    )
                )

            outputs = response.get("outputs", {})
            if outputs:
                masks = outputs.get("out_binary_masks", [])
                obj_ids = outputs.get("out_obj_ids", [])
                guru.debug(f"Box prompt response: {len(masks)} masks, obj_ids={obj_ids}")

                target_h, target_w = self.image.shape[:2]
                for i, obj_id in enumerate(obj_ids):
                    if i < len(masks):
                        mask_array = np.array(masks[i])
                        self.cur_masks[int(obj_id)] = self._resize_mask(mask_array, target_h, target_w).copy()

                guru.debug(f"After box prompt: cur_masks keys={list(self.cur_masks.keys())}, use_text_mode={self.use_text_mode}")

                if self.cur_masks:
                    index_mask = self.make_index_mask(self.cur_masks)
                    return index_mask, f"Box prompt for object {self.cur_mask_idx}: detected {len(masks)} object(s)"

            return None, "No objects detected in box region"
        except Exception as e:
            guru.error(f"Box prompt error: {e}")
            self.use_text_mode = False
            return None, f"Error: {str(e)}"

    def get_sam_mask(self, frame_idx, input_points, input_labels):
        h, w = self.image.shape[:2]

        if self.use_text_mode:
            normalized_points = self._normalize_points(input_points)
            try:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    response = self.video_predictor.handle_request(
                        request=dict(
                            type="add_prompt",
                            session_id=self.inference_state,
                            frame_index=frame_idx,
                            obj_id=self.cur_mask_idx,
                            points=normalized_points,
                            point_labels=input_labels.tolist()
                            if hasattr(input_labels, "tolist")
                            else input_labels,
                        )
                    )
                outputs = response.get("outputs", {})
                if outputs:
                    masks = outputs.get("out_binary_masks", [])
                    obj_ids = outputs.get("out_obj_ids", [self.cur_mask_idx])
                    result = {}
                    for i, obj_id in enumerate(obj_ids):
                        if i < len(masks):
                            result[obj_id] = self._resize_mask(masks[i], h, w)
                    if result:
                        return result
            except Exception as e:
                guru.error(f"Point refinement error: {e}")
            # If text refinement path is unavailable, fall back to points mode.
            self.use_text_mode = False

        if not self._init_tracker_state():
            guru.warning("Points mode unavailable.")
            return {self.cur_mask_idx: np.zeros((h, w), dtype=bool)}

        if frame_idx >= self.tracker_num_frames:
            guru.warning(
                f"Frame {frame_idx} is out of range for points mode "
                f"(max: {self.tracker_num_frames - 1})."
            )
            return {self.cur_mask_idx: np.zeros((h, w), dtype=bool)}

        try:
            rel_points = [[x / w, y / h] for x, y in input_points]
            points_tensor = torch.tensor(rel_points, dtype=torch.float32)
            labels_tensor = torch.tensor(
                input_labels.tolist() if hasattr(input_labels, "tolist") else input_labels,
                dtype=torch.int32,
            )

            # Use autocast to match the dtype used in SAM3 model (BFloat16)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, _, _, video_res_masks = self.tracker.add_new_points(
                    inference_state=self.tracker_state,
                    frame_idx=frame_idx,
                    obj_id=self.cur_mask_idx,
                    points=points_tensor,
                    labels=labels_tensor,
                    clear_old_points=False,
                    rel_coordinates=True,
                )

            if video_res_masks is not None and len(video_res_masks) > 0:
                mask = (video_res_masks[0] > 0.0).cpu().numpy().squeeze()
                return {self.cur_mask_idx: self._resize_mask(mask, h, w)}
        except Exception as e:
            guru.error(f"Points mode error: {e}")

        return {self.cur_mask_idx: np.zeros((h, w), dtype=bool)}

    def remove_object(self, obj_id: int):
        """Remove a tracked object from the session."""
        if self.inference_state is None:
            return None, "No active session"
        try:
            self.video_predictor.handle_request(
                request=dict(
                    type="remove_object",
                    session_id=self.inference_state,
                    obj_id=obj_id,
                )
            )
            guru.debug(f"Removed object {obj_id} from session, cur_masks before: {list(self.cur_masks.keys())}")
            if obj_id in self.cur_masks:
                del self.cur_masks[obj_id]
            guru.debug(f"cur_masks after: {list(self.cur_masks.keys())}")

            if not self.cur_masks:
                # Return empty mask if no objects left
                if self.image is not None:
                    empty_mask = np.zeros((self.image.shape[0], self.image.shape[1]), dtype="uint8")
                    return empty_mask, f"Removed object {obj_id}. No objects left."
                return None, f"Removed object {obj_id}. No objects left."

            # Generate index mask from remaining objects
            index_mask = self.make_index_mask(self.cur_masks)
            guru.debug(f"Generated index_mask: shape={index_mask.shape}, max={index_mask.max()}, unique={np.unique(index_mask)}")
            return index_mask, f"Removed object {obj_id}. {len(self.cur_masks)} object(s) remaining."
        except Exception as e:
            guru.error(f"Remove object error: {e}")
            import traceback
            traceback.print_exc()
            return None, f"Error removing object: {e}"

    def run_tracker(self, propagation_direction: str = "both"):
        guru.debug(f"run_tracker called: use_text_mode={self.use_text_mode}, cur_masks={list(self.cur_masks.keys()) if self.cur_masks else 'empty'}")

        if not self.use_text_mode and not self._init_tracker_state():
            return None, "Points mode unavailable right now."
        if self.use_text_mode and not self.cur_masks:
            guru.warning("use_text_mode=True but cur_masks is empty!")
            return None, "No objects detected yet."

        if not self.use_text_mode:
            # Group points by object and frame for multi-object tracking
            prompts_by_obj_and_frame = {}  # {obj_id: {frame_idx: {"points": [], "labels": []}}}
            for pt, label, frame_idx, obj_id in zip(
                self.selected_points, self.selected_labels, self.selected_point_frames, self.selected_point_obj_ids
            ):
                if obj_id not in prompts_by_obj_and_frame:
                    prompts_by_obj_and_frame[obj_id] = {}
                if frame_idx not in prompts_by_obj_and_frame[obj_id]:
                    prompts_by_obj_and_frame[obj_id][frame_idx] = {"points": [], "labels": []}
                prompts_by_obj_and_frame[obj_id][frame_idx]["points"].append(pt)
                prompts_by_obj_and_frame[obj_id][frame_idx]["labels"].append(label)

            if not prompts_by_obj_and_frame:
                return None, "No point prompt was applied. Click on the frame to add a point, then track."

            guru.debug(f"Multi-object tracking: {len(prompts_by_obj_and_frame)} objects with prompts")
            for obj_id, frames in prompts_by_obj_and_frame.items():
                guru.debug(f"  Object {obj_id}: {len(frames)} frames with points")

            self.tracker.clear_all_points_in_video(self.tracker_state)

            # Add points for each object separately
            for obj_id, frames in prompts_by_obj_and_frame.items():
                for frame_idx, prompts in frames.items():
                    # Temporarily set cur_mask_idx to the target object
                    original_mask_idx = self.cur_mask_idx
                    self.cur_mask_idx = obj_id
                    try:
                        self.get_sam_mask(
                            frame_idx,
                            np.array(prompts["points"], dtype=np.float32),
                            np.array(prompts["labels"], dtype=np.int32),
                        )
                    finally:
                        self.cur_mask_idx = original_mask_idx

            point_inputs = self.tracker_state.get("point_inputs_per_obj", {})
            num_prompted_frames = sum(len(v) for v in point_inputs.values())
            guru.debug(
                f"Run tracker in points mode: selected_points={len(self.selected_points)}, "
                f"prompted_frames={num_prompted_frames}"
            )

        if self.use_text_mode:
            images = [iio.imread(p)[:, :, :3] for p in self.img_paths]
            num_frames_to_track = len(self.img_paths)
        else:
            num_frames_to_track = self.tracker_num_frames
            images = [
                iio.imread(self.img_paths[i])[:, :, :3]
                for i in range(min(num_frames_to_track, len(self.img_paths)))
            ]

        video_segments = {}
        frames_with_masks = 0

        if self.use_text_mode:
            try:
                # Re-initialize SAM3 state with text prompt to ensure proper action_history
                # For box prompts, we DON'T re-add them because the initial detection
                # already set up the tracking state with all detected objects
                # We'll filter out removed objects when collecting results
                kept_obj_ids = set(self.cur_masks.keys()) if self.cur_masks else set()

                if self.current_text_prompt:
                    guru.debug(f"Re-initializing SAM3 with text prompt before tracking (keeping obj_ids: {kept_obj_ids})")
                    self.video_predictor.handle_request(
                        request=dict(type="reset_session", session_id=self.inference_state)
                    )

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        self.video_predictor.handle_request(
                            request=dict(
                                type="add_prompt",
                                session_id=self.inference_state,
                                frame_index=self.text_prompt_frame_idx,
                                text=self.current_text_prompt,
                            )
                        )
                        guru.debug(f"Re-initialized with text prompt")
                elif self.box_prompts:
                    # Re-initialize session and re-add box prompts to ensure proper action_history
                    guru.debug(f"Re-initializing SAM3 with {len(self.box_prompts)} box prompt(s) before tracking (keeping obj_ids: {kept_obj_ids})")
                    self.video_predictor.handle_request(
                        request=dict(type="reset_session", session_id=self.inference_state)
                    )

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        # Re-add all box prompts to establish action_history
                        for obj_id, prompt_info in self.box_prompts.items():
                            box = prompt_info["box"]
                            frame_idx = prompt_info["frame_idx"]
                            self.video_predictor.handle_request(
                                request=dict(
                                    type="add_prompt",
                                    session_id=self.inference_state,
                                    frame_index=frame_idx,
                                    bounding_boxes=[[box[0], box[1], box[2], box[3]]],
                                    bounding_box_labels=[1],
                                    obj_id=obj_id,
                                )
                            )
                        guru.debug(f"Re-initialized with {len(self.box_prompts)} box prompt(s)")

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    guru.debug(f"Starting stream request: session={self.inference_state}, direction={propagation_direction}")
                    response_stream = self.video_predictor.handle_stream_request(
                        request=dict(
                            type="propagate_in_video",
                            session_id=self.inference_state,
                            propagation_direction=propagation_direction,
                        )
                    )
                    frame_count = 0
                    null_output_count = 0
                    for result in response_stream:
                        frame_count += 1
                        out_frame_idx = result.get("frame_index")
                        outputs = result.get("outputs")
                        if outputs is None:
                            null_output_count += 1
                            continue
                        masks_list = outputs.get("out_binary_masks", [])
                        out_obj_ids = outputs.get("out_obj_ids", [])
                        if frame_count <= 5 or frame_count % 50 == 0:
                            guru.debug(f"Stream frame {frame_count}: idx={out_frame_idx}, masks={len(masks_list)}, obj_ids={list(out_obj_ids)}")
                        masks = {}
                        for i, obj_id in enumerate(out_obj_ids):
                            # Filter: only keep objects that user hasn't removed
                            if kept_obj_ids and obj_id not in kept_obj_ids:
                                continue
                            if i < len(masks_list):
                                masks[obj_id] = masks_list[i]
                        if out_frame_idx is None:
                            continue
                        out_frame_idx = int(out_frame_idx)
                        if 0 <= out_frame_idx < len(images):
                            video_segments[out_frame_idx] = masks
                            if masks:
                                frames_with_masks += 1
                    guru.debug(f"Stream complete: {frame_count} frames received, {null_output_count} null outputs, {len(video_segments)} segments")
            except Exception as e:
                guru.error(f"Text mode tracking error: {e}")
                import traceback
                traceback.print_exc()
                return None, f"Tracking failed: {e}"
        else:
            try:
                # Handle propagation direction for points mode
                directions_to_run = []
                if propagation_direction == "forward":
                    directions_to_run = [False]
                elif propagation_direction == "backward":
                    directions_to_run = [True]
                else:  # both
                    directions_to_run = [False, True]

                # Use autocast to match the dtype used in SAM3 model initialization
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    for reverse in directions_to_run:
                        for frame_idx, obj_ids, _, video_res_masks, _ in self.tracker.propagate_in_video(
                            self.tracker_state,
                            start_frame_idx=0,
                            max_frame_num_to_track=num_frames_to_track,
                            reverse=reverse,
                            propagate_preflight=True,
                        ):
                            masks = {}
                            for i, obj_id in enumerate(obj_ids):
                                if video_res_masks is not None and i < len(video_res_masks):
                                    masks[obj_id] = (video_res_masks[i] > 0.0).cpu().numpy().squeeze()
                            if frame_idx is None:
                                continue
                            frame_idx = int(frame_idx)
                            if 0 <= frame_idx < len(images):
                                video_segments[frame_idx] = masks
                                if masks:
                                    frames_with_masks += 1
            except RuntimeError as e:
                msg = str(e)
                if "No points are provided" in msg:
                    return None, "No point prompt was applied. Click on the frame to add a point, then track."
                guru.error(f"Tracker runtime error: {msg}")
                import traceback
                traceback.print_exc()
                return None, f"Tracking failed: {msg}"

        if not video_segments:
            return None, "Tracking returned no frames."
        if frames_with_masks == 0:
            return None, "No masks generated. Add prompts first."

        guru.debug(
            f"Video segments: {len(video_segments)} frames, "
            f"frames_with_masks: {frames_with_masks}, images: {len(images)}"
        )

        self.index_masks_all = []
        for frame_idx, img in enumerate(images):
            target_h, target_w = img.shape[:2]
            masks_dict = video_segments.get(frame_idx, {})
            if masks_dict:
                resized_masks = {
                    obj_id: self._resize_mask(mask, target_h, target_w)
                    for obj_id, mask in masks_dict.items()
                }
                idx_mask = self.make_index_mask(resized_masks)
            else:
                idx_mask = np.zeros((target_h, target_w), dtype="uint8")
            self.index_masks_all.append(idx_mask)
            if masks_dict:
                guru.debug(f"Frame {frame_idx}: mask shape {idx_mask.shape}, max {idx_mask.max()}")

        guru.debug(f"Total index_masks: {len(self.index_masks_all)}, images: {len(images)}")
        out_frames, self.color_masks_all = colorize_masks(images, self.index_masks_all)
        guru.debug(f"Colorized {len(out_frames)} frames")
        out_vidpath = "tracked_colors.mp4"
        iio.mimwrite(out_vidpath, out_frames)
        return out_vidpath, f"Tracked {len(out_frames)} frames. Save masks if it looks good!"

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
        return f"Saved masks to {output_dir}!"
