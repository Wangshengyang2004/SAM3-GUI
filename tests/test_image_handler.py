import os

import numpy as np
import pytest
from PIL import Image

from sam3_gui.image_handler import ImageModeHandler


class RecordingBackend:
    def __init__(self):
        self.loaded = False
        self.started = []
        self.closed = []
        self.requests = []
        self.fail_next_start = False

    def ensure_predictor(self):
        self.loaded = True

    def start_session(self, resource_path):
        self.started.append(resource_path)
        if self.fail_next_start:
            self.fail_next_start = False
            raise RuntimeError("session startup failed")
        return f"image-session-{len(self.started)}"

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

    image, msg = handler.set_image(
        np.ones((2, 2, 3), dtype=np.uint8), resource_path=str(path)
    )

    assert image.shape == (4, 6, 3)
    assert "loaded" in msg.lower()
    assert backend.started == [str(path)]


def test_failed_image_switch_preserves_old_session_and_state():
    backend = RecordingBackend()
    handler = ImageModeHandler(backend=backend)
    old_image = np.zeros((4, 6, 3), dtype=np.uint8)
    handler.set_image(old_image)
    old_session_id = handler.session_id
    handler.current_masks = [np.ones((4, 6), dtype=bool)]
    handler.current_scores = [0.8]
    handler.selected_points = [[2, 3]]
    handler.selected_labels = [1]
    handler.cur_label_val = 0
    handler.drawn_box = (1, 1, 4, 3)
    handler.point_obj_id = 2
    backend.fail_next_start = True

    with pytest.raises(RuntimeError, match="session startup failed"):
        handler.set_image(np.full((3, 5, 3), 255, dtype=np.uint8))

    assert handler.session_id == old_session_id
    assert np.array_equal(handler.current_image, old_image)
    assert len(handler.current_masks) == 1
    assert handler.current_scores == [0.8]
    assert handler.selected_points == [[2, 3]]
    assert handler.selected_labels == [1]
    assert handler.drawn_box == (1, 1, 4, 3)
    assert handler.point_obj_id == 2
    assert backend.closed == []


def test_successful_image_switch_closes_old_session_after_state_reset():
    backend = RecordingBackend()
    handler = ImageModeHandler(backend=backend)
    handler.set_image(np.zeros((4, 6, 3), dtype=np.uint8))
    old_session_id = handler.session_id
    handler.current_masks = [np.ones((4, 6), dtype=bool)]
    handler.selected_points = [[2, 3]]
    handler.cur_label_val = 0

    handler.set_image(np.ones((3, 5, 3), dtype=np.uint8))

    assert handler.session_id != old_session_id
    assert backend.closed == [old_session_id]
    assert handler.current_image.shape == (3, 5, 3)
    assert handler.current_masks == []
    assert handler.selected_points == []
    assert handler.cur_label_val == 1


@pytest.mark.parametrize("method_name", ["reset", "close"])
def test_reset_and_close_clear_all_session_state(method_name):
    backend = RecordingBackend()
    handler = ImageModeHandler(backend=backend)
    handler.set_image(np.zeros((4, 6, 3), dtype=np.uint8))
    session_id = handler.session_id
    handler.current_masks = [np.ones((4, 6), dtype=bool)]
    handler.current_scores = [0.8]
    handler.selected_points = [[2, 3]]
    handler.selected_labels = [1]
    handler.cur_label_val = 0
    handler.drawn_box = (1, 1, 4, 3)
    handler.point_obj_id = 2

    getattr(handler, method_name)()

    assert handler.session_id is None
    assert handler.current_image is None
    assert handler.current_masks == []
    assert handler.current_scores == []
    assert handler.selected_points == []
    assert handler.selected_labels == []
    assert handler.cur_label_val == 1
    assert handler.drawn_box is None
    assert handler.point_obj_id == 0
    assert backend.closed == [session_id]


def test_save_mask_writes_all_instances(tmp_path):
    handler = ImageModeHandler(backend=RecordingBackend())
    first_mask = np.zeros((4, 6), dtype=bool)
    first_mask[0, 0] = True
    second_mask = np.zeros((4, 6), dtype=bool)
    second_mask[1, 1] = True
    handler.current_masks = [first_mask, second_mask]
    output_path = tmp_path / "masks.npy"

    message = handler.save_mask(str(output_path))

    saved_masks = np.load(output_path)
    assert saved_masks.shape == (2, 4, 6)
    assert np.array_equal(saved_masks[0], first_mask)
    assert np.array_equal(saved_masks[1], second_mask)
    assert "2 mask(s)" in message


def test_save_mask_atomically_replaces_existing_file(monkeypatch, tmp_path):
    handler = ImageModeHandler(backend=RecordingBackend())
    handler.current_masks = [np.ones((4, 6), dtype=bool)]
    output_path = tmp_path / "masks.npy"
    output_path.write_bytes(b"old mask data")
    replace_calls = []
    real_replace = os.replace

    def record_replace(source, destination):
        assert os.path.dirname(source) == str(tmp_path)
        assert output_path.read_bytes() == b"old mask data"
        assert np.load(source).shape == (1, 4, 6)
        replace_calls.append((source, destination))
        real_replace(source, destination)

    monkeypatch.setattr("sam3_gui.image_handler.os.replace", record_replace)

    handler.save_mask(str(output_path))

    assert len(replace_calls) == 1
    assert replace_calls[0][1] == str(output_path)
    assert np.load(output_path).shape == (1, 4, 6)
    assert list(tmp_path.glob(".masks.npy.*.tmp")) == []


def test_default_confidence_matches_image_ui():
    handler = ImageModeHandler(backend=RecordingBackend())

    assert handler.confidence_threshold == 0.3


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
