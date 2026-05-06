import os
import tempfile
from typing import Any, Literal

import imageio.v2 as iio
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from gradio.routes import mount_gradio_app
from pydantic import BaseModel, Field

from mask_app import make_demo
from sam31_backend import (
    SAM31_CHECKPOINT_NAME,
    SAM31_HF_REPO,
    SAM31_MODELSCOPE_REPO,
    Sam31Backend,
    Sam31BackendConfig,
    current_sam3_commit,
    output_masks_by_obj,
    serialize_outputs,
)
from utils import compose_img_mask, get_hls_palette


class StartSessionRequest(BaseModel):
    resource_path: str = Field(..., description="Image file, video file, or frame directory.")


class SessionResponse(BaseModel):
    session_id: str


class AddPromptRequest(BaseModel):
    session_id: str
    frame_index: int = 0
    text: str | None = None
    points: list[list[float]] | None = None
    point_labels: list[int] | None = None
    bounding_boxes: list[list[float]] | None = None
    bounding_box_labels: list[int] | None = None
    obj_id: int | None = None
    clear_old_points: bool = True
    clear_old_boxes: bool = True
    rel_coordinates: bool = True
    output_prob_thresh: float | None = None
    include_masks: bool = True


class PropagateRequest(BaseModel):
    session_id: str
    propagation_direction: Literal["both", "forward", "backward"] = "both"
    start_frame_index: int | None = None
    max_frame_num_to_track: int | None = None
    output_prob_thresh: float | None = None
    include_masks: bool = False


class RemoveObjectRequest(BaseModel):
    session_id: str
    obj_id: int
    frame_index: int = 0
    include_masks: bool = True


class SegmentImageResponse(BaseModel):
    outputs: dict[str, Any]


def _ensure_rgb(path: str) -> np.ndarray:
    img = iio.imread(path)
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)
    return img[:, :, :3]


def create_app(
    root_dir: str,
    checkpoint_path: str | None = None,
    gpus_to_use: list[int] | None = None,
    vid_name: str = "videos",
    img_name: str = "images",
    mask_name: str = "masks",
) -> FastAPI:
    config = Sam31BackendConfig(
        checkpoint_path=checkpoint_path,
        device_id=gpus_to_use[0] if gpus_to_use else None,
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

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
            "checkpoint_path": backend_status["checkpoint_path"],
            "checkpoint_available": backend_status["checkpoint_available"],
            "will_download_checkpoint": backend_status["will_download_checkpoint"],
            "sam3_commit": current_sam3_commit(),
            "model_loaded": backend_status["model_loaded"],
        }

    @app.post("/api/sessions", response_model=SessionResponse)
    def start_session(payload: StartSessionRequest):
        if not os.path.exists(payload.resource_path):
            raise HTTPException(status_code=404, detail=f"Resource not found: {payload.resource_path}")
        try:
            return {"session_id": get_backend().start_session(payload.resource_path)}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.delete("/api/sessions/{session_id}")
    def close_session(session_id: str):
        try:
            return get_backend().close_session(session_id)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/prompts")
    def add_prompt(payload: AddPromptRequest):
        try:
            kwargs = payload.model_dump(exclude={"session_id", "frame_index", "include_masks"}, exclude_none=True)
            response = get_backend().add_prompt(payload.session_id, payload.frame_index, **kwargs)
            return {
                "frame_index": response["frame_index"],
                "outputs": serialize_outputs(response.get("outputs"), include_masks=payload.include_masks),
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

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
                        "outputs": serialize_outputs(result.get("outputs"), include_masks=payload.include_masks),
                    }
                )
            return {"frames": frames, "frame_count": len(frames)}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/objects/remove")
    def remove_object(payload: RemoveObjectRequest):
        try:
            response = get_backend().remove_object(payload.session_id, payload.obj_id, payload.frame_index)
            return {
                "frame_index": response.get("frame_index", payload.frame_index),
                "outputs": serialize_outputs(response.get("outputs"), include_masks=payload.include_masks),
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/images/segment", response_model=SegmentImageResponse)
    async def segment_image(
        file: UploadFile = File(...),
        text: str | None = Form(None),
        output_prob_thresh: float = Form(0.5),
        include_overlay: bool = Form(False),
    ):
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Field 'text' is required.")
        suffix = os.path.splitext(file.filename or "image.png")[1] or ".png"
        fd, path = tempfile.mkstemp(prefix="sam31-image-", suffix=suffix)
        os.close(fd)
        session_id = None
        try:
            with open(path, "wb") as f:
                f.write(await file.read())
            active_backend = get_backend()
            session_id = active_backend.start_session(path)
            response = active_backend.add_prompt(
                session_id,
                0,
                text=text.strip(),
                output_prob_thresh=output_prob_thresh,
            )
            outputs = serialize_outputs(response.get("outputs"), include_masks=True)
            if include_overlay:
                img = _ensure_rgb(path)
                masks = output_masks_by_obj(response.get("outputs"), img.shape[:2])
                idx_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                for obj_id, mask in masks.items():
                    idx_mask[mask] = obj_id + 1
                palette = get_hls_palette(int(idx_mask.max()) + 1)
                outputs["overlay_shape"] = list(compose_img_mask(img, palette[idx_mask]).shape)
            return {"outputs": outputs}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if session_id is not None:
                try:
                    get_backend().close_session(session_id)
                except Exception:
                    pass
            try:
                os.remove(path)
            except OSError:
                pass

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
