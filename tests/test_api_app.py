from fastapi.testclient import TestClient

from api_app import create_app


class FakeBackend:
    version = "sam3.1"

    def __init__(self):
        self.predictor = None
        self.closed = []
        self.prompts = []
        self.propagation_requests = []

    def status(self):
        return {
            "checkpoint_path": None,
            "checkpoint_available": False,
            "will_download_checkpoint": True,
            "model_loaded": self.predictor is not None,
        }

    def start_session(self, resource_path):
        self.resource_path = resource_path
        return "session-1"

    def close_session(self, session_id):
        self.closed.append(session_id)
        return {"is_success": True}

    def add_prompt(self, session_id, frame_index, **kwargs):
        self.prompts.append((session_id, frame_index, kwargs))
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
        return {"frame_index": frame_index, "outputs": {"out_obj_ids": [], "out_binary_masks": []}}


def _client_with_fake_backend():
    app = create_app("/home/wsy/SAM3-GUI/data_root")
    fake = FakeBackend()
    app.state.sam31_backend = fake
    return TestClient(app), fake


def test_api_health_and_docs_do_not_load_model():
    client, fake = _client_with_fake_backend()

    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["sam_version"] == "sam3.1"
    assert response.json()["model_loaded"] is False
    assert response.json()["checkpoint_available"] is False
    assert response.json()["will_download_checkpoint"] is True
    assert fake.predictor is None
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_api_session_prompt_propagate_remove_close(tmp_path):
    resource = tmp_path / "frame.png"
    resource.write_bytes(b"not-used-by-fake-backend")
    client, fake = _client_with_fake_backend()

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

    removed = client.post(
        "/api/objects/remove",
        json={"session_id": "session-1", "obj_id": 0, "frame_index": 3},
    )
    assert removed.status_code == 200
    assert removed.json()["frame_index"] == 3

    closed = client.delete("/api/sessions/session-1")
    assert closed.status_code == 200
    assert fake.closed == ["session-1"]


def test_api_segment_image_requires_text(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"not-used")
    client, _ = _client_with_fake_backend()

    response = client.post(
        "/api/images/segment",
        files={"file": ("image.png", image_path.read_bytes(), "image/png")},
    )
    assert response.status_code == 400


def test_api_segment_image_with_text_uses_single_frame_session(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"not-a-real-image-but-not-rendered")
    client, fake = _client_with_fake_backend()

    response = client.post(
        "/api/images/segment",
        files={"file": ("image.png", image_path.read_bytes(), "image/png")},
        data={"text": "person"},
    )

    assert response.status_code == 200
    assert response.json()["outputs"]["object_ids"] == [0]
    assert fake.prompts[0][0] == "session-1"
    assert fake.prompts[0][1] == 0
    assert fake.prompts[0][2]["text"] == "person"
    assert fake.closed == ["session-1"]
