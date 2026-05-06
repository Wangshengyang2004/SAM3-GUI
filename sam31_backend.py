import os
import subprocess
from functools import wraps
from inspect import Parameter, signature
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
from loguru import logger as guru
from pycocotools import mask as mask_utils


SAM31_VERSION = "sam3.1"
SAM31_HF_REPO = "facebook/sam3.1"
SAM31_MODELSCOPE_REPO = "facebook/sam3.1"
SAM31_CHECKPOINT_NAME = "sam3.1_multiplex.pt"


class Sam31ModelLoadError(RuntimeError):
    """Raised when the native SAM 3.1 predictor cannot be constructed."""


def validate_sam31_checkpoint_path(checkpoint_path: str | None) -> str | None:
    if not checkpoint_path:
        return None
    checkpoint_name = os.path.basename(checkpoint_path).lower()
    if "sam3.1" not in checkpoint_name or "multiplex" not in checkpoint_name or not checkpoint_name.endswith(".pt"):
        raise ValueError(
            "SAM3-GUI only supports SAM 3.1 Object Multiplex checkpoints. "
            f"Use {SAM31_HF_REPO}/{SAM31_CHECKPOINT_NAME} or "
            f"ModelScope {SAM31_MODELSCOPE_REPO}/{SAM31_CHECKPOINT_NAME}."
        )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    return checkpoint_path


def assert_sam31_available() -> None:
    try:
        from sam3.model_builder import build_sam3_predictor  # noqa: F401
        from sam3.model_builder import build_sam3_multiplex_video_predictor  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "The installed sam3 package does not expose SAM 3.1 Object Multiplex. "
            "Update /home/wsy/sam3 and reinstall it with `pip install -e /home/wsy/sam3`."
        ) from exc


def make_predictor_init_state_compatible(predictor: Any) -> None:
    """Bridge SAM3 base predictor kwargs to older SAM 3.1 multiplex init_state signatures."""
    model = getattr(predictor, "model", None)
    init_state = getattr(model, "init_state", None)
    if init_state is None:
        return

    init_signature = signature(init_state)
    if any(param.kind == Parameter.VAR_KEYWORD for param in init_signature.parameters.values()):
        return
    if "offload_state_to_cpu" in init_signature.parameters:
        return

    @wraps(init_state)
    def compatible_init_state(*args, **kwargs):
        offload_state_to_cpu = kwargs.pop("offload_state_to_cpu", False)
        if offload_state_to_cpu:
            raise TypeError("SAM 3.1 multiplex init_state does not support offload_state_to_cpu=True")
        return init_state(*args, **kwargs)

    model.init_state = compatible_init_state


def sam31_model_load_error(exc: Exception) -> Sam31ModelLoadError:
    checkpoint_hint = (
        f"Place {SAM31_CHECKPOINT_NAME} under ~/sam3/model, pass --checkpoint_path, "
        f"authenticate with `hf auth login` for {SAM31_HF_REPO}, "
        "or run `python download_model.py --source modelscope`."
    )
    return Sam31ModelLoadError(
        "Failed to load native SAM 3.1 Object Multiplex. "
        f"{checkpoint_hint} Original error: {exc}"
    )


def torch_autocast_context():
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


def to_numpy(data: Any, dtype: Any | None = None) -> np.ndarray:
    if data is None:
        return np.asarray([])
    if hasattr(data, "detach"):
        data = data.detach()
    if hasattr(data, "cpu"):
        data = data.cpu()
    if hasattr(data, "numpy"):
        data = data.numpy()
    arr = np.asarray(data)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def resize_mask(mask: Any, target_h: int, target_w: int) -> np.ndarray:
    arr = to_numpy(mask)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {arr.shape}")
    if arr.shape == (target_h, target_w):
        return arr.astype(bool, copy=False)
    resized = cv2.resize(
        arr.astype(np.uint8, copy=False),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool, copy=False)


def output_masks_by_obj(outputs: dict[str, Any] | None, target_shape: tuple[int, int] | None = None) -> dict[int, np.ndarray]:
    if not outputs:
        return {}

    obj_ids = to_numpy(outputs.get("out_obj_ids", []), dtype=np.int64).reshape(-1)
    masks = outputs.get("out_binary_masks", [])
    if hasattr(masks, "detach") or hasattr(masks, "numpy"):
        masks = to_numpy(masks)

    result: dict[int, np.ndarray] = {}
    if len(obj_ids) == 0:
        return result
    for idx, obj_id in enumerate(obj_ids):
        if idx >= len(masks):
            continue
        mask = masks[idx]
        if target_shape is not None:
            mask = resize_mask(mask, target_shape[0], target_shape[1])
        else:
            mask = np.squeeze(to_numpy(mask)).astype(bool, copy=False)
        result[int(obj_id)] = mask
    return result


def index_mask_from_obj_masks(
    masks: dict[int, np.ndarray],
    fallback_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    if not masks:
        shape = fallback_shape or (1, 1)
        return np.zeros(shape, dtype=np.uint8)

    first = next(iter(masks.values()))
    idx_mask = np.zeros(first.shape[:2], dtype=np.uint8)
    for obj_id in sorted(masks):
        mask = np.squeeze(masks[obj_id])
        idx_mask[mask > 0] = int(obj_id) + 1
    return idx_mask


def normalize_points(points: list[list[float]] | np.ndarray, width: int, height: int) -> list[list[float]]:
    arr = to_numpy(points, dtype=np.float32).reshape(-1, 2)
    return [[float(x) / width, float(y) / height] for x, y in arr]


def normalize_xyxy_box(box_coords: tuple[float, float, float, float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box_coords
    xmin = min(x1, x2) / width
    ymin = min(y1, y2) / height
    box_width = abs(x2 - x1) / width
    box_height = abs(y2 - y1) / height
    return [float(xmin), float(ymin), float(box_width), float(box_height)]


def mask_to_rle(mask: Any) -> dict[str, Any]:
    arr = np.asfortranarray(np.squeeze(to_numpy(mask)).astype(np.uint8, copy=False))
    encoded = mask_utils.encode(arr)
    return {
        "size": [int(v) for v in encoded["size"]],
        "counts": encoded["counts"].decode("ascii"),
    }


def serialize_outputs(outputs: dict[str, Any] | None, include_masks: bool = True) -> dict[str, Any]:
    outputs = outputs or {}
    obj_ids = to_numpy(outputs.get("out_obj_ids", []), dtype=np.int64).reshape(-1)
    probs = to_numpy(outputs.get("out_probs", []), dtype=np.float32).reshape(-1)
    boxes = to_numpy(outputs.get("out_boxes_xywh", []), dtype=np.float32)
    masks = outputs.get("out_binary_masks", [])
    if hasattr(masks, "detach") or hasattr(masks, "numpy"):
        masks = to_numpy(masks)

    payload: dict[str, Any] = {
        "object_ids": [int(v) for v in obj_ids.tolist()],
        "probabilities": [float(v) for v in probs.tolist()],
        "boxes_xywh": boxes.tolist() if boxes.size else [],
        "mask_count": int(len(masks)),
    }
    if include_masks:
        payload["masks_rle"] = [mask_to_rle(mask) for mask in masks]
    return payload


def current_sam3_commit() -> str | None:
    try:
        import sam3

        repo = os.path.abspath(os.path.join(os.path.dirname(sam3.__file__), ".."))
        return subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


@dataclass
class Sam31BackendConfig:
    checkpoint_path: str | None = None
    device_id: int | None = None
    max_num_objects: int = 16
    multiplex_count: int = 16
    output_prob_thresh: float = 0.5
    compile_model: bool = False
    warm_up: bool = False
    use_fa3: bool = False
    use_rope_real: bool = True
    async_loading_frames: bool = False


class Sam31Backend:
    """Lazy native SAM 3.1 Object Multiplex backend shared by UI and API."""

    def __init__(self, config: Sam31BackendConfig | None = None):
        self.config = config or Sam31BackendConfig()
        self.config.checkpoint_path = validate_sam31_checkpoint_path(self.config.checkpoint_path)
        self.predictor = None
        self._session_object_counts: dict[str, int] = {}

    @property
    def version(self) -> str:
        return SAM31_VERSION

    def status(self) -> dict[str, Any]:
        checkpoint_path = self.config.checkpoint_path
        return {
            "version": SAM31_VERSION,
            "checkpoint_repo": SAM31_HF_REPO,
            "checkpoint_repos": {
                "huggingface": SAM31_HF_REPO,
                "modelscope": SAM31_MODELSCOPE_REPO,
            },
            "checkpoint_name": SAM31_CHECKPOINT_NAME,
            "checkpoint_path": checkpoint_path,
            "checkpoint_available": bool(checkpoint_path and os.path.exists(checkpoint_path)),
            "will_download_checkpoint": checkpoint_path is None,
            "model_loaded": self.predictor is not None,
        }

    def ensure_predictor(self):
        if self.predictor is not None:
            return self.predictor

        self.config.checkpoint_path = validate_sam31_checkpoint_path(self.config.checkpoint_path)
        assert_sam31_available()
        if self.config.device_id is not None and torch.cuda.is_available():
            torch.cuda.set_device(self.config.device_id)
            guru.info(f"Using CUDA device {self.config.device_id} for SAM 3.1")

        from sam3.model_builder import build_sam3_predictor

        guru.info("Loading native SAM 3.1 Object Multiplex predictor")
        try:
            self.predictor = build_sam3_predictor(
                version=SAM31_VERSION,
                checkpoint_path=self.config.checkpoint_path,
                max_num_objects=self.config.max_num_objects,
                multiplex_count=self.config.multiplex_count,
                compile=self.config.compile_model,
                warm_up=self.config.warm_up,
                use_fa3=self.config.use_fa3,
                use_rope_real=self.config.use_rope_real,
                async_loading_frames=self.config.async_loading_frames,
            )
            make_predictor_init_state_compatible(self.predictor)
        except Exception as exc:
            raise sam31_model_load_error(exc) from exc
        return self.predictor

    def start_session(self, resource_path: Any, **kwargs) -> str:
        with torch_autocast_context():
            response = self.ensure_predictor().handle_request(
                request={
                    "type": "start_session",
                    "resource_path": resource_path,
                    **kwargs,
                }
            )
        return response["session_id"]

    def reset_session(self, session_id: str) -> dict[str, Any]:
        self._session_object_counts.pop(session_id, None)
        with torch_autocast_context():
            return self.ensure_predictor().handle_request(
                request={"type": "reset_session", "session_id": session_id}
            )

    def close_session(self, session_id: str) -> dict[str, Any]:
        self._session_object_counts.pop(session_id, None)
        with torch_autocast_context():
            return self.ensure_predictor().handle_request(
                request={"type": "close_session", "session_id": session_id}
            )

    def add_prompt(self, session_id: str, frame_index: int, **kwargs) -> dict[str, Any]:
        request = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": int(frame_index),
            "output_prob_thresh": kwargs.pop("output_prob_thresh", self.config.output_prob_thresh),
            **kwargs,
        }
        with torch_autocast_context():
            response = self.ensure_predictor().handle_request(request=request)
        outputs = response.get("outputs") or {}
        obj_ids = to_numpy(outputs.get("out_obj_ids", []), dtype=np.int64).reshape(-1)
        self._session_object_counts[session_id] = int(len(obj_ids))
        return response

    def remove_object(self, session_id: str, obj_id: int, frame_index: int = 0) -> dict[str, Any]:
        with torch_autocast_context():
            response = self.ensure_predictor().handle_request(
                request={
                    "type": "remove_object",
                    "session_id": session_id,
                    "frame_index": int(frame_index),
                    "obj_id": int(obj_id),
                }
            )
        outputs = response.get("outputs") or {}
        obj_ids = to_numpy(outputs.get("out_obj_ids", []), dtype=np.int64).reshape(-1)
        self._session_object_counts[session_id] = int(len(obj_ids))
        return response

    def propagate(
        self,
        session_id: str,
        propagation_direction: str = "both",
        start_frame_index: int | None = None,
        max_frame_num_to_track: int | None = None,
        output_prob_thresh: float | None = None,
    ):
        request = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": propagation_direction,
            "output_prob_thresh": self.config.output_prob_thresh
            if output_prob_thresh is None
            else float(output_prob_thresh),
        }
        if start_frame_index is not None:
            request["start_frame_index"] = int(start_frame_index)
        if max_frame_num_to_track is not None:
            request["max_frame_num_to_track"] = int(max_frame_num_to_track)
        if self._session_object_counts.get(session_id) == 0:
            return
        with torch_autocast_context():
            yield from self.ensure_predictor().handle_stream_request(request=request)
