import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from gradio.routes import mount_gradio_app
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, model_validator

from sam3_gui.mask_app import make_demo
from sam3_gui.sam31_backend import (
    Sam31Backend,
    Sam31BackendConfig,
    Sam31ModelLoadError,
    Sam31ObjectNotFound,
    Sam31SessionNotFound,
    current_sam3_commit,
    output_masks_by_obj,
    serialize_instances,
    serialize_outputs,
)
from sam3_gui.sam31_constants import (
    SAM31_CHECKPOINT_NAME,
    SAM31_HF_REPO,
    SAM31_MODELSCOPE_REPO,
)
from sam3_gui.utils import compose_img_mask, get_hls_palette


logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_PROPAGATION_FRAMES = 1000
ALLOWED_IMAGE_FORMATS = {
    "BMP": ".bmp",
    "JPEG": ".jpg",
    "PNG": ".png",
    "TIFF": ".tiff",
    "WEBP": ".webp",
}


class StartSessionRequest(BaseModel):
    resource_path: str = Field(
        ..., description="Image file, video file, or frame directory."
    )


class SessionResponse(BaseModel):
    session_id: str


class AddPromptRequest(BaseModel):
    session_id: str
    frame_index: int = Field(default=0, ge=0)
    text: str | None = None
    points: list[list[float]] | None = None
    point_labels: list[int] | None = None
    bounding_boxes: list[list[float]] | None = None
    bounding_box_labels: list[int] | None = None
    obj_id: int | None = Field(default=None, ge=0)
    clear_old_points: bool = True
    clear_old_boxes: bool = True
    rel_coordinates: bool = True
    output_prob_thresh: float | None = Field(default=None, ge=0, le=1)
    include_masks: bool = True

    @model_validator(mode="after")
    def validate_prompt(self):
        self.text = self.text.strip() if self.text and self.text.strip() else None
        if self.points is not None and self.point_labels is None:
            self.point_labels = [1] * len(self.points)
        if self.bounding_boxes is not None and self.bounding_box_labels is None:
            self.bounding_box_labels = [1] * len(self.bounding_boxes)
        _validate_prompt_fields(
            self.text,
            self.points,
            self.point_labels,
            self.bounding_boxes,
            self.bounding_box_labels,
        )
        return self


class PropagateRequest(BaseModel):
    session_id: str
    propagation_direction: Literal["both", "forward", "backward"] = "both"
    start_frame_index: int | None = Field(default=None, ge=0)
    max_frame_num_to_track: int = Field(
        default=MAX_PROPAGATION_FRAMES, gt=0, le=MAX_PROPAGATION_FRAMES
    )
    output_prob_thresh: float | None = Field(default=None, ge=0, le=1)
    include_masks: bool = False


class RemoveObjectRequest(BaseModel):
    session_id: str
    obj_id: int = Field(ge=0)
    frame_index: int = Field(default=0, ge=0)
    include_masks: bool = True


class SegmentImageResponse(BaseModel):
    outputs: dict[str, Any]
    image_size: list[int] | None = None
    prompt: str | None = None
    instances: list[dict[str, Any]] | None = None


def _validate_prompt_fields(
    text: str | None,
    points: list[Any] | None,
    point_labels: list[Any] | None,
    bounding_boxes: list[Any] | None,
    bounding_box_labels: list[Any] | None,
) -> None:
    if text is None and points is None and bounding_boxes is None:
        raise ValueError(
            "At least one of 'text', 'points', or 'bounding_boxes' is required."
        )

    if points is not None:
        if (
            not isinstance(points, list)
            or not points
            or any(
                not isinstance(point, (list, tuple)) or len(point) != 2
                for point in points
            )
        ):
            raise ValueError("Each point must contain exactly two coordinates.")
        if not isinstance(point_labels, list) or len(point_labels) != len(points):
            raise ValueError("'point_labels' must match 'points' length.")
    elif point_labels is not None:
        raise ValueError("'point_labels' requires 'points'.")

    if bounding_boxes is not None:
        if (
            not isinstance(bounding_boxes, list)
            or not bounding_boxes
            or any(
                not isinstance(box, (list, tuple)) or len(box) != 4
                for box in bounding_boxes
            )
        ):
            raise ValueError("Each bounding box must contain exactly four coordinates.")
        if not isinstance(bounding_box_labels, list) or len(bounding_box_labels) != len(
            bounding_boxes
        ):
            raise ValueError(
                "'bounding_box_labels' must match 'bounding_boxes' length."
            )
    elif bounding_box_labels is not None:
        raise ValueError("'bounding_box_labels' requires 'bounding_boxes'.")


def _parse_json_form_field(raw: str | None, field_name: str) -> Any:
    if raw is None or not str(raw).strip():
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400, detail=f"Field '{field_name}' must be valid JSON."
        ) from exc


def _parse_segment_prompt_form(
    text: str | None,
    points_raw: str | None,
    point_labels_raw: str | None,
    bounding_boxes_raw: str | None,
    bounding_box_labels_raw: str | None,
    rel_coordinates: bool,
) -> dict[str, Any]:
    prompt_text = text.strip() if text and text.strip() else None
    points = _parse_json_form_field(points_raw, "points")
    point_labels = _parse_json_form_field(point_labels_raw, "point_labels")
    bounding_boxes = _parse_json_form_field(bounding_boxes_raw, "bounding_boxes")
    bounding_box_labels = _parse_json_form_field(
        bounding_box_labels_raw, "bounding_box_labels"
    )

    if isinstance(points, list):
        if point_labels is None:
            point_labels = [1] * len(points)

    if isinstance(bounding_boxes, list):
        if bounding_box_labels is None:
            bounding_box_labels = [1] * len(bounding_boxes)

    try:
        _validate_prompt_fields(
            prompt_text,
            points,
            point_labels,
            bounding_boxes,
            bounding_box_labels,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    kwargs: dict[str, Any] = {
        "rel_coordinates": rel_coordinates,
        "clear_old_points": True,
        "clear_old_boxes": True,
    }
    if prompt_text is not None:
        kwargs["text"] = prompt_text
    if points is not None:
        kwargs["points"] = points
        kwargs["point_labels"] = point_labels
    if bounding_boxes is not None:
        kwargs["bounding_boxes"] = bounding_boxes
        kwargs["bounding_box_labels"] = bounding_box_labels
    return kwargs


def _resolve_resource_path(root_path: Path, resource_path: str) -> Path:
    requested_path = Path(resource_path).expanduser()
    candidate = (
        requested_path if requested_path.is_absolute() else root_path / requested_path
    )
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise HTTPException(
            status_code=403,
            detail="Resource path must stay within the configured root directory.",
        ) from exc
    if not resolved.exists():
        raise HTTPException(
            status_code=404, detail=f"Resource not found: {resource_path}"
        )
    return resolved


async def _write_upload_with_limit(file: UploadFile, path: str) -> None:
    total_bytes = 0
    with open(path, "wb") as output:
        while chunk := await file.read(UPLOAD_CHUNK_BYTES):
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Image upload exceeds the {MAX_UPLOAD_BYTES}-byte limit.",
                )
            output.write(chunk)


def _load_static_rgb(path: str) -> tuple[np.ndarray, str]:
    try:
        with Image.open(path) as image:
            image_format = image.format
            if image_format not in ALLOWED_IMAGE_FORMATS:
                raise HTTPException(
                    status_code=415, detail="Unsupported static image format."
                )
            if getattr(image, "n_frames", 1) != 1:
                raise HTTPException(
                    status_code=415, detail="Animated images are not supported."
                )
            width, height = image.size
            if width <= 0 or height <= 0:
                raise HTTPException(status_code=415, detail="Invalid image dimensions.")
            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail=f"Image exceeds the {MAX_IMAGE_PIXELS}-pixel limit.",
                )
            return np.asarray(image.convert("RGB")), ALLOWED_IMAGE_FORMATS[image_format]
    except HTTPException:
        raise
    except (
        Image.DecompressionBombError,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=415, detail="Uploaded file is not a supported static image."
        ) from exc


def _api_error(exc: Exception) -> HTTPException:
    if isinstance(exc, (Sam31SessionNotFound, Sam31ObjectNotFound)):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, FileNotFoundError):
        return HTTPException(
            status_code=404, detail="Requested resource was not found."
        )
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, Sam31ModelLoadError):
        logger.exception("SAM 3.1 model is unavailable")
        return HTTPException(status_code=503, detail="SAM 3.1 model is unavailable.")
    logger.exception("Unhandled SAM3-GUI API error")
    return HTTPException(status_code=500, detail="Internal server error.")


def create_app(
    root_dir: str,
    checkpoint_path: str | None = None,
    gpus_to_use: list[int] | None = None,
    vid_name: str = "videos",
    img_name: str = "images",
    mask_name: str = "masks",
    use_fa3: bool = False,
) -> FastAPI:
    root_path = Path(root_dir).expanduser().resolve()
    config = Sam31BackendConfig(
        checkpoint_path=checkpoint_path,
        device_id=gpus_to_use[0] if gpus_to_use else None,
        use_fa3=use_fa3,
    )
    backend = Sam31Backend(config)
    app = FastAPI(
        title="SAM3-GUI API",
        version="3.1",
        description="Native SAM 3.1 Object Multiplex API for segmentation and video tracking.",
    )
    app.state.sam31_backend = backend

    def get_backend() -> Sam31Backend:
        return app.state.sam31_backend

    @app.get("/api/health")
    def health():
        backend_status = get_backend().status()
        return {
            "status": "ok",
            "sam_version": "sam3.1",
            "checkpoint_repo": SAM31_HF_REPO,
            "checkpoint_repos": {
                "huggingface": SAM31_HF_REPO,
                "modelscope": SAM31_MODELSCOPE_REPO,
            },
            "checkpoint_name": SAM31_CHECKPOINT_NAME,
            "checkpoint_available": backend_status["checkpoint_available"],
            "will_download_checkpoint": backend_status["will_download_checkpoint"],
            "use_fa3": backend_status["use_fa3"],
            "sam3_commit": current_sam3_commit(),
            "model_loaded": backend_status["model_loaded"],
        }

    @app.post("/api/sessions", response_model=SessionResponse)
    def start_session(payload: StartSessionRequest):
        resource_path = _resolve_resource_path(root_path, payload.resource_path)
        try:
            return {"session_id": get_backend().start_session(str(resource_path))}
        except Exception as exc:
            raise _api_error(exc) from exc

    @app.delete("/api/sessions/{session_id}")
    def close_session(session_id: str):
        try:
            return get_backend().close_session(session_id)
        except Exception as exc:
            raise _api_error(exc) from exc

    @app.post("/api/prompts")
    def add_prompt(payload: AddPromptRequest):
        try:
            kwargs = payload.model_dump(
                exclude={"session_id", "frame_index", "include_masks"},
                exclude_none=True,
            )
            response = get_backend().add_prompt(
                payload.session_id, payload.frame_index, **kwargs
            )
            return {
                "frame_index": response["frame_index"],
                "outputs": serialize_outputs(
                    response.get("outputs"), include_masks=payload.include_masks
                ),
            }
        except Exception as exc:
            raise _api_error(exc) from exc

    @app.post("/api/propagate")
    def propagate(payload: PropagateRequest):
        try:
            frames = []
            for result in get_backend().propagate(
                payload.session_id,
                propagation_direction=payload.propagation_direction,
                start_frame_index=payload.start_frame_index,
                max_frame_num_to_track=payload.max_frame_num_to_track,
                output_prob_thresh=payload.output_prob_thresh,
            ):
                frames.append(
                    {
                        "frame_index": int(result["frame_index"]),
                        "outputs": serialize_outputs(
                            result.get("outputs"), include_masks=payload.include_masks
                        ),
                    }
                )
            return {"frames": frames, "frame_count": len(frames)}
        except Exception as exc:
            raise _api_error(exc) from exc

    @app.post("/api/objects/remove")
    def remove_object(payload: RemoveObjectRequest):
        try:
            response = get_backend().remove_object(
                payload.session_id, payload.obj_id, payload.frame_index
            )
            return {
                "frame_index": response.get("frame_index", payload.frame_index),
                "outputs": serialize_outputs(
                    response.get("outputs"), include_masks=payload.include_masks
                ),
            }
        except Exception as exc:
            raise _api_error(exc) from exc

    @app.post("/api/images/segment")
    async def segment_image(
        file: UploadFile = File(...),
        text: str | None = Form(None),
        points: str | None = Form(None),
        point_labels: str | None = Form(None),
        bounding_boxes: str | None = Form(None),
        bounding_box_labels: str | None = Form(None),
        rel_coordinates: bool = Form(True),
        output_prob_thresh: float = Form(0.5, ge=0, le=1),
        response_format: Literal["legacy", "aspire"] = Form("legacy"),
        box_format: Literal["xywh_normalized", "xywh_pixel"] = Form("xywh_normalized"),
        include_overlay: bool = Form(False),
    ):
        prompt_kwargs = _parse_segment_prompt_form(
            text,
            points,
            point_labels,
            bounding_boxes,
            bounding_box_labels,
            rel_coordinates,
        )
        fd, path = tempfile.mkstemp(prefix="sam31-image-")
        os.close(fd)
        session_id = None
        active_backend = None
        try:
            await _write_upload_with_limit(file, path)
            img, suffix = _load_static_rgb(path)
            image_path = f"{path}{suffix}"
            os.replace(path, image_path)
            path = image_path
            image_size = (int(img.shape[0]), int(img.shape[1]))
            active_backend = get_backend()
            session_id = active_backend.start_session(path)
            response = active_backend.add_prompt(
                session_id,
                0,
                output_prob_thresh=output_prob_thresh,
                **prompt_kwargs,
            )
            raw_outputs = response.get("outputs")
            outputs = serialize_outputs(raw_outputs, include_masks=True)
            if include_overlay:
                masks = output_masks_by_obj(raw_outputs, image_size)
                idx_mask = np.zeros(image_size, dtype=np.uint8)
                for obj_id, mask in masks.items():
                    idx_mask[mask] = obj_id + 1
                palette = get_hls_palette(int(idx_mask.max()) + 1)
                outputs["overlay_shape"] = list(
                    compose_img_mask(img, palette[idx_mask]).shape
                )

            prompt_text = prompt_kwargs.get("text")
            payload: dict[str, Any] = {
                "outputs": outputs,
                "image_size": [image_size[0], image_size[1]],
            }
            if response_format == "aspire":
                payload["prompt"] = prompt_text
                payload["instances"] = serialize_instances(
                    raw_outputs,
                    label=prompt_text,
                    include_masks=True,
                    box_format=box_format,
                    image_size=image_size,
                )
            return payload
        except HTTPException:
            raise
        except Exception as exc:
            raise _api_error(exc) from exc
        finally:
            if session_id is not None:
                try:
                    active_backend.close_session(session_id)
                except Exception:
                    logger.exception(
                        "Failed to close stateless image session %s", session_id
                    )
            try:
                os.remove(path)
            except OSError:
                pass
            await file.close()

    demo = make_demo(
        root_dir,
        checkpoint_path=checkpoint_path,
        gpus_to_use=gpus_to_use,
        vid_name=vid_name,
        img_name=img_name,
        mask_name=mask_name,
        backend=backend,
    )
    return mount_gradio_app(app, demo, path="/")
