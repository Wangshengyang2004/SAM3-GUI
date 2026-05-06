import numpy as np
import pytest
from PIL import Image

from image_handler import ImageModeHandler


class RecordingBackend:
    def __init__(self):
        self.loaded = False
        self.started = []
        self.closed = []
        self.requests = []

    def ensure_predictor(self):
        self.loaded = True

    def start_session(self, resource_path):
        self.started.append(resource_path)
        return "image-session"

    def close_session(self, session_id):
        self.closed.append(session_id)
        return {"is_success": True}

    def reset_session(self, session_id):
        self.requests.append(("reset_session", session_id))
        return {"is_success": True}

    def add_prompt(self, session_id, frame_index, **kwargs):
        self.requests.append(("add_prompt", session_id, frame_index, kwargs))
        return {
            "frame_index": frame_index,
            "outputs": {
                "out_obj_ids": [0],
                "out_probs": [0.8],
                "out_boxes_xywh": [[0.0, 0.0, 1.0, 1.0]],
                "out_binary_masks": [np.ones((4, 6), dtype=bool)],
            },
        }


def test_set_image_without_image_does_not_load_model():
    backend = RecordingBackend()
    handler = ImageModeHandler(backend=backend)

    image, msg = handler.set_image(None)

    assert image is None
    assert msg == "No image provided"
    assert backend.loaded is False


def test_image_handler_rejects_non_sam31_checkpoint(tmp_path):
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"")

    with pytest.raises(ValueError):
        ImageModeHandler(checkpoint_path=str(checkpoint), backend=RecordingBackend())


def test_set_image_prefers_resource_path_for_native_session(tmp_path):
    path = tmp_path / "image.png"
    Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8)).save(path)
    backend = RecordingBackend()
    handler = ImageModeHandler(backend=backend)

    image, msg = handler.set_image(np.ones((2, 2, 3), dtype=np.uint8), resource_path=str(path))

    assert image.shape == (4, 6, 3)
    assert "loaded" in msg.lower()
    assert backend.started == [str(path)]


def test_image_handler_prompts_use_native_sam31_request_shapes(tmp_path):
    path = tmp_path / "image.png"
    Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8)).save(path)
    backend = RecordingBackend()
    handler = ImageModeHandler(backend=backend)
    handler.set_image(None, resource_path=str(path))

    result, msg = handler.find_all_with_text("truck")
    assert result is not None
    assert "Found" in msg
    assert backend.requests[-1][3]["text"] == "truck"

    result, msg = handler.segment_with_box((1, 1, 5, 3))
    assert result is not None
    assert "Segmented with box" in msg
    box_kwargs = backend.requests[-1][3]
    assert box_kwargs["bounding_boxes"] == [[1 / 6, 1 / 4, 4 / 6, 2 / 4]]
    assert box_kwargs["bounding_box_labels"] == [1]

    result, msg = handler.add_point(3, 2)
    assert result is not None
    assert "Generated" in msg
    point_kwargs = backend.requests[-1][3]
    assert point_kwargs["points"] == [[3 / 6, 2 / 4]]
    assert point_kwargs["point_labels"] == [1]
    assert point_kwargs["rel_coordinates"] is True
