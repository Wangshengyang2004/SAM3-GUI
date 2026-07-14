import imageio.v2 as iio
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pathlib import Path

import sam3_gui.api_app as api_app
from sam3_gui.api_app import create_app
from sam3_gui.sam31_backend import (
    Sam31ObjectNotFound,
    Sam31SessionNotFound,
    serialize_instances,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeBackend:
    version = "sam3.1"

    def __init__(self):
        self.predictor = None
        self.closed = []
        self.prompts = []
        self.propagation_requests = []

    def status(self):
        return {
            "checkpoint_path": "/secret/models/sam3.pt",
            "checkpoint_available": True,
            "will_download_checkpoint": False,
            "use_fa3": False,
            "model_loaded": self.predictor is not None,
        }

    def start_session(self, resource_path):
        self.resource_path = resource_path
        return "session-1"

    def close_session(self, session_id):
        if session_id == "missing":
            raise Sam31SessionNotFound("Session not found: missing")
        self.closed.append(session_id)
        return {"is_success": True}

    def add_prompt(self, session_id, frame_index, **kwargs):
        if session_id == "missing":
            raise Sam31SessionNotFound("Session not found: missing")
        self.prompts.append((session_id, frame_index, kwargs))
        if kwargs.get("text") == "bowl":
            return {
                "frame_index": frame_index,
                "outputs": {
                    "out_obj_ids": [0, 1],
                    "out_probs": [0.95, 0.88],
                    "out_boxes_xywh": [[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]],
                    "out_binary_masks": [[[1, 0], [0, 0]], [[0, 1], [1, 0]]],
                },
            }
        return {
            "frame_index": frame_index,
            "outputs": {
                "out_obj_ids": [0],
                "out_probs": [0.95],
                "out_boxes_xywh": [[0.1, 0.1, 0.5, 0.5]],
                "out_binary_masks": [[[1, 0], [0, 1]]],
            },
        }

    def propagate(self, session_id, **kwargs):
        if session_id == "missing":
            raise Sam31SessionNotFound("Session not found: missing")
        self.propagation_requests.append((session_id, kwargs))
        yield {
            "frame_index": 0,
            "outputs": {
                "out_obj_ids": [0],
                "out_probs": [0.95],
                "out_boxes_xywh": [[0.1, 0.1, 0.5, 0.5]],
                "out_binary_masks": [[[1, 0], [0, 1]]],
            },
        }

    def remove_object(self, session_id, obj_id, frame_index=0):
        if session_id == "missing":
            raise Sam31SessionNotFound("Session not found: missing")
        if obj_id == 7:
            raise Sam31ObjectNotFound("Object 7 not found in session session-1")
        return {
            "frame_index": frame_index,
            "outputs": {"out_obj_ids": [], "out_binary_masks": []},
        }


def _write_test_png(path, height=12, width=16):
    iio.imwrite(path, np.zeros((height, width, 3), dtype=np.uint8))


def _client_with_fake_backend(root_dir=None):
    app = create_app(str(root_dir or REPO_ROOT / "data_root"))
    fake = FakeBackend()
    app.state.sam31_backend = fake
    return TestClient(app), fake


def test_api_health_and_docs_do_not_load_model():
    client, fake = _client_with_fake_backend()

    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["sam_version"] == "sam3.1"
    assert response.json()["model_loaded"] is False
    assert response.json()["checkpoint_available"] is True
    assert response.json()["will_download_checkpoint"] is False
    assert response.json()["use_fa3"] is False
    assert "checkpoint_path" not in response.json()
    assert fake.predictor is None
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_api_session_prompt_propagate_remove_close(tmp_path):
    resource = tmp_path / "frame.png"
    resource.write_bytes(b"not-used-by-fake-backend")
    client, fake = _client_with_fake_backend(tmp_path)

    started = client.post("/api/sessions", json={"resource_path": str(resource)})
    assert started.status_code == 200
    assert started.json() == {"session_id": "session-1"}

    prompt = client.post(
        "/api/prompts",
        json={
            "session_id": "session-1",
            "frame_index": 3,
            "text": "person",
            "include_masks": False,
        },
    )
    assert prompt.status_code == 200
    assert prompt.json()["outputs"]["object_ids"] == [0]
    assert "masks_rle" not in prompt.json()["outputs"]
    assert fake.prompts[0][2]["text"] == "person"

    propagated = client.post(
        "/api/propagate",
        json={"session_id": "session-1", "propagation_direction": "forward"},
    )
    assert propagated.status_code == 200
    assert propagated.json()["frame_count"] == 1
    assert fake.propagation_requests[0][1]["propagation_direction"] == "forward"
    assert (
        fake.propagation_requests[0][1]["max_frame_num_to_track"]
        == api_app.MAX_PROPAGATION_FRAMES
    )

    removed = client.post(
        "/api/objects/remove",
        json={"session_id": "session-1", "obj_id": 0, "frame_index": 3},
    )
    assert removed.status_code == 200
    assert removed.json()["frame_index"] == 3

    closed = client.delete("/api/sessions/session-1")
    assert closed.status_code == 200
    assert fake.closed == ["session-1"]


def test_api_unknown_session_and_object_return_404(tmp_path):
    resource = tmp_path / "frame.png"
    resource.write_bytes(b"not-used-by-fake-backend")
    client, _ = _client_with_fake_backend(tmp_path)

    assert client.delete("/api/sessions/missing").status_code == 404

    prompt = client.post(
        "/api/prompts",
        json={"session_id": "missing", "frame_index": 0, "text": "person"},
    )
    assert prompt.status_code == 404

    propagated = client.post(
        "/api/propagate", json={"session_id": "missing", "max_frame_num_to_track": 2}
    )
    assert propagated.status_code == 404

    client.post("/api/sessions", json={"resource_path": str(resource)})
    removed = client.post(
        "/api/objects/remove", json={"session_id": "session-1", "obj_id": 7}
    )
    assert removed.status_code == 404


def test_api_internal_errors_are_not_leaked(tmp_path):
    resource = tmp_path / "frame.png"
    resource.write_bytes(b"not-used-by-fake-backend")
    client, fake = _client_with_fake_backend(tmp_path)

    def fail_start_session(resource_path):
        raise RuntimeError("private path: /srv/customer/secret.pt")

    fake.start_session = fail_start_session
    response = client.post("/api/sessions", json={"resource_path": str(resource)})

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error."}


def test_session_resource_path_is_confined_to_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    inside = root / "inside.png"
    inside.write_bytes(b"inside")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    inside_link = root / "inside-link.png"
    inside_link.symlink_to(inside)
    escape_link = root / "escape-link.png"
    escape_link.symlink_to(outside)
    client, fake = _client_with_fake_backend(root)

    relative = client.post("/api/sessions", json={"resource_path": "inside.png"})
    assert relative.status_code == 200
    assert fake.resource_path == str(inside.resolve())

    absolute_inside = client.post("/api/sessions", json={"resource_path": str(inside)})
    assert absolute_inside.status_code == 200

    linked_inside = client.post(
        "/api/sessions", json={"resource_path": "inside-link.png"}
    )
    assert linked_inside.status_code == 200
    assert fake.resource_path == str(inside.resolve())

    assert (
        client.post("/api/sessions", json={"resource_path": str(outside)}).status_code
        == 403
    )
    assert (
        client.post(
            "/api/sessions", json={"resource_path": "../outside.png"}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/sessions", json={"resource_path": "escape-link.png"}
        ).status_code
        == 403
    )
    assert (
        client.post("/api/sessions", json={"resource_path": "missing.png"}).status_code
        == 404
    )


@pytest.mark.parametrize(
    "path,payload",
    [
        (
            "/api/prompts",
            {"session_id": "session-1", "frame_index": -1, "text": "person"},
        ),
        ("/api/prompts", {"session_id": "session-1", "obj_id": -1, "text": "person"}),
        (
            "/api/prompts",
            {"session_id": "session-1", "output_prob_thresh": 1.1, "text": "person"},
        ),
        ("/api/prompts", {"session_id": "session-1"}),
        (
            "/api/prompts",
            {"session_id": "session-1", "points": [[0.1]], "point_labels": [1]},
        ),
        (
            "/api/prompts",
            {"session_id": "session-1", "points": [[0.1, 0.2]], "point_labels": [1, 0]},
        ),
        (
            "/api/prompts",
            {"session_id": "session-1", "bounding_boxes": [[0.1, 0.2, 0.3]]},
        ),
        ("/api/propagate", {"session_id": "session-1", "start_frame_index": -1}),
        ("/api/propagate", {"session_id": "session-1", "max_frame_num_to_track": 0}),
        (
            "/api/propagate",
            {
                "session_id": "session-1",
                "max_frame_num_to_track": api_app.MAX_PROPAGATION_FRAMES + 1,
            },
        ),
        ("/api/propagate", {"session_id": "session-1", "output_prob_thresh": -0.1}),
        ("/api/objects/remove", {"session_id": "session-1", "obj_id": -1}),
    ],
)
def test_api_request_validation_rejects_invalid_bounds(path, payload):
    client, _ = _client_with_fake_backend()
    assert client.post(path, json=payload).status_code == 422


def test_segment_image_requires_any_prompt(tmp_path):
    image_path = tmp_path / "image.png"
    _write_test_png(image_path)
    client, _ = _client_with_fake_backend()

    response = client.post(
        "/api/images/segment",
        files={"file": ("image.png", image_path.read_bytes(), "image/png")},
    )
    assert response.status_code == 400
    assert "At least one of" in response.json()["detail"]


def test_api_does_not_enable_wildcard_cors():
    client, _ = _client_with_fake_backend()
    response = client.options(
        "/api/health",
        headers={
            "Origin": "https://attacker.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.headers.get("access-control-allow-origin") != "*"


def test_segment_image_rejects_oversized_upload(monkeypatch):
    monkeypatch.setattr(api_app, "MAX_UPLOAD_BYTES", 8)
    client, _ = _client_with_fake_backend()

    response = client.post(
        "/api/images/segment",
        files={"file": ("image.png", b"more-than-eight-bytes", "image/png")},
        data={"text": "person"},
    )
    assert response.status_code == 413


def test_segment_image_rejects_unsupported_content():
    client, _ = _client_with_fake_backend()
    response = client.post(
        "/api/images/segment",
        files={"file": ("image.png", b"not-an-image", "image/png")},
        data={"text": "person"},
    )
    assert response.status_code == 415


def test_segment_image_rejects_excessive_pixels(tmp_path, monkeypatch):
    image_path = tmp_path / "image.png"
    _write_test_png(image_path, height=4, width=4)
    monkeypatch.setattr(api_app, "MAX_IMAGE_PIXELS", 15)
    client, _ = _client_with_fake_backend()

    response = client.post(
        "/api/images/segment",
        files={"file": ("image.png", image_path.read_bytes(), "image/png")},
        data={"text": "person"},
    )
    assert response.status_code == 413


@pytest.mark.parametrize(
    "data",
    [
        {"points": "[[0.1]]", "point_labels": "[1]"},
        {"points": "[[0.1, 0.2]]", "point_labels": "[1, 0]"},
        {"bounding_boxes": "[[0.1, 0.2, 0.3]]"},
    ],
)
def test_segment_image_rejects_invalid_prompt_dimensions(tmp_path, data):
    image_path = tmp_path / "image.png"
    _write_test_png(image_path)
    client, _ = _client_with_fake_backend()

    response = client.post(
        "/api/images/segment",
        files={"file": ("image.png", image_path.read_bytes(), "image/png")},
        data=data,
    )
    assert response.status_code == 400


def test_segment_image_cleanup_uses_session_backend(tmp_path):
    image_path = tmp_path / "image.png"
    _write_test_png(image_path)
    app = create_app(str(REPO_ROOT / "data_root"))
    replacement = FakeBackend()

    class SwitchingBackend(FakeBackend):
        def start_session(self, resource_path):
            session_id = super().start_session(resource_path)
            app.state.sam31_backend = replacement
            return session_id

    original = SwitchingBackend()
    app.state.sam31_backend = original
    client = TestClient(app)

    response = client.post(
        "/api/images/segment",
        files={"file": ("image.png", image_path.read_bytes(), "image/png")},
        data={"text": "person"},
    )
    assert response.status_code == 200
    assert original.closed == ["session-1"]
    assert replacement.closed == []


def test_segment_image_cleanup_failure_is_logged(tmp_path, caplog):
    image_path = tmp_path / "image.png"
    _write_test_png(image_path)
    app = create_app(str(REPO_ROOT / "data_root"))

    class FailingCloseBackend(FakeBackend):
        def close_session(self, session_id):
            raise RuntimeError("cleanup failed")

    app.state.sam31_backend = FailingCloseBackend()
    client = TestClient(app)

    with caplog.at_level("ERROR", logger="sam3_gui.api_app"):
        response = client.post(
            "/api/images/segment",
            files={"file": ("image.png", image_path.read_bytes(), "image/png")},
            data={"text": "person"},
        )
    assert response.status_code == 200
    assert "Failed to close stateless image session session-1" in caplog.text


def test_api_segment_image_with_text_uses_single_frame_session(tmp_path):
    image_path = tmp_path / "image.png"
    _write_test_png(image_path)
    client, fake = _client_with_fake_backend()

    response = client.post(
        "/api/images/segment",
        files={"file": ("image.png", image_path.read_bytes(), "image/png")},
        data={"text": "person"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["outputs"]["object_ids"] == [0]
    assert body["image_size"] == [12, 16]
    assert "instances" not in body
    assert fake.prompts[0][0] == "session-1"
    assert fake.prompts[0][1] == 0
    assert fake.prompts[0][2]["text"] == "person"
    assert fake.closed == ["session-1"]


def test_segment_image_point_prompt(tmp_path):
    image_path = tmp_path / "image.png"
    _write_test_png(image_path)
    client, fake = _client_with_fake_backend()

    response = client.post(
        "/api/images/segment",
        files={"file": ("image.png", image_path.read_bytes(), "image/png")},
        data={
            "points": "[[100, 200]]",
            "point_labels": "[1]",
            "rel_coordinates": "false",
        },
    )

    assert response.status_code == 200
    kwargs = fake.prompts[0][2]
    assert kwargs["points"] == [[100, 200]]
    assert kwargs["point_labels"] == [1]
    assert kwargs["rel_coordinates"] is False
    assert "text" not in kwargs


def test_segment_image_text_plus_box(tmp_path):
    image_path = tmp_path / "image.png"
    _write_test_png(image_path)
    client, fake = _client_with_fake_backend()

    response = client.post(
        "/api/images/segment",
        files={"file": ("image.png", image_path.read_bytes(), "image/png")},
        data={
            "text": "vehicle",
            "bounding_boxes": "[[0.2, 0.2, 0.4, 0.5]]",
            "bounding_box_labels": "[1]",
        },
    )

    assert response.status_code == 200
    kwargs = fake.prompts[0][2]
    assert kwargs["text"] == "vehicle"
    assert kwargs["bounding_boxes"] == [[0.2, 0.2, 0.4, 0.5]]
    assert kwargs["bounding_box_labels"] == [1]


def test_segment_image_aspire_format(tmp_path):
    image_path = tmp_path / "image.png"
    _write_test_png(image_path, height=100, width=200)
    client, _ = _client_with_fake_backend()

    response = client.post(
        "/api/images/segment",
        files={"file": ("image.png", image_path.read_bytes(), "image/png")},
        data={
            "text": "bowl",
            "response_format": "aspire",
            "box_format": "xywh_pixel",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["prompt"] == "bowl"
    assert body["image_size"] == [100, 200]
    assert len(body["instances"]) == 2
    assert body["instances"][0]["score"] == pytest.approx(0.95)
    assert body["instances"][0]["label"] == "bowl"
    assert body["instances"][0]["box_xywh"] == pytest.approx([20.0, 10.0, 40.0, 20.0])
    assert "mask_rle" in body["instances"][0]
    assert body["outputs"]["mask_count"] == 2


def test_segment_image_legacy_unchanged(tmp_path):
    image_path = tmp_path / "image.png"
    _write_test_png(image_path)
    client, _ = _client_with_fake_backend()

    response = client.post(
        "/api/images/segment",
        files={"file": ("image.png", image_path.read_bytes(), "image/png")},
        data={"text": "person", "response_format": "legacy"},
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"outputs", "image_size"}
    assert body["outputs"]["object_ids"] == [0]
    assert "masks_rle" in body["outputs"]


def test_serialize_instances_pixel_boxes():
    outputs = {
        "out_obj_ids": [0],
        "out_probs": [0.9],
        "out_boxes_xywh": [[0.1, 0.2, 0.5, 0.25]],
        "out_binary_masks": [[[1, 0], [0, 1]]],
    }
    instances = serialize_instances(
        outputs,
        label="bowl",
        box_format="xywh_pixel",
        image_size=(100, 200),
    )
    assert instances[0]["box_xywh"] == pytest.approx([20.0, 20.0, 100.0, 25.0])
    assert instances[0]["label"] == "bowl"
