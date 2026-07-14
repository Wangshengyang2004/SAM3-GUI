import os

import pytest
from fastapi.testclient import TestClient

from sam3_gui.api_app import create_app
from sam3_gui.image_handler import ImageModeHandler
from sam3_gui.video_handler import VideoModeHandler

pytestmark = pytest.mark.integration


def test_native_sam31_api_and_gui_smoke(
    require_sam31_checkpoint, test_img_dir, test_images
):
    checkpoint_path = require_sam31_checkpoint
    root_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_root")
    app = create_app(root_dir, checkpoint_path=checkpoint_path)
    client = TestClient(app)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["sam_version"] == "sam3.1"

    started = client.post("/api/sessions", json={"resource_path": test_img_dir})
    assert started.status_code == 200
    session_id = started.json()["session_id"]
    assert session_id

    try:
        prompt = client.post(
            "/api/prompts",
            json={
                "session_id": session_id,
                "frame_index": 0,
                "text": "wheel",
                "output_prob_thresh": 0.2,
                "include_masks": False,
            },
        )
        assert prompt.status_code == 200
        assert "outputs" in prompt.json()

        propagated = client.post(
            "/api/propagate",
            json={
                "session_id": session_id,
                "propagation_direction": "forward",
                "max_frame_num_to_track": 2,
                "include_masks": False,
            },
        )
        assert propagated.status_code == 200
        assert propagated.json()["frame_count"] >= 1
    finally:
        closed = client.delete(f"/api/sessions/{session_id}")
        assert closed.status_code == 200

    backend = app.state.sam31_backend
    video_handler = VideoModeHandler(backend=backend)
    assert video_handler.set_img_dir(test_img_dir) > 0
    assert video_handler.set_input_image(0) is not None

    image_handler = ImageModeHandler(backend=backend)
    image, message = image_handler.set_image(None, resource_path=test_images[0])
    assert image is not None
    assert "loaded" in message.lower()


def test_native_sam31_real_mp4_text_box_prompt_smoke(
    require_sam31_checkpoint, test_video_path
):
    checkpoint_path = require_sam31_checkpoint
    root_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_root")
    app = create_app(root_dir, checkpoint_path=checkpoint_path)
    client = TestClient(app)

    started = client.post("/api/sessions", json={"resource_path": test_video_path})
    assert started.status_code == 200
    session_id = started.json()["session_id"]

    try:
        prompt = client.post(
            "/api/prompts",
            json={
                "session_id": session_id,
                "frame_index": 0,
                "text": "vehicle",
                "bounding_boxes": [[0.4094, 0.8184, 0.3274, 0.1816]],
                "bounding_box_labels": [1],
                "output_prob_thresh": 0.01,
                "include_masks": False,
            },
        )
        assert prompt.status_code == 200
        assert prompt.json()["outputs"]["object_ids"]

        propagated = client.post(
            "/api/propagate",
            json={
                "session_id": session_id,
                "propagation_direction": "forward",
                "start_frame_index": 0,
                "max_frame_num_to_track": 2,
                "output_prob_thresh": 0.01,
                "include_masks": False,
            },
        )
        assert propagated.status_code == 200
        frames = propagated.json()["frames"]
        assert propagated.json()["frame_count"] >= 2
        assert any(frame["outputs"]["object_ids"] for frame in frames)
    finally:
        closed = client.delete(f"/api/sessions/{session_id}")
        assert closed.status_code == 200
