import importlib
import sys
import types

import imageio.v2 as iio
import numpy as np
import pytest


class _FakePredictor:
    def __init__(self, responses):
        self._responses = responses

    def handle_request(self, request):
        pass

    def handle_stream_request(self, request):
        assert request["type"] == "propagate_in_video"
        for response in self._responses:
            yield response


class _FakeBackend:
    def __init__(self, responses):
        self._responses = responses

    def propagate(self, *args, **kwargs):
        yield from self._responses


class _RecordingBackend:
    def __init__(self, outputs=None):
        self.outputs = outputs or {
            "out_obj_ids": [0],
            "out_binary_masks": [np.ones((4, 6), dtype=bool)],
            "out_probs": [0.9],
        }
        self.requests = []
        self.closed = []

    def start_session(self, resource_path):
        self.requests.append(("start_session", resource_path))
        return "session-1"

    def reset_session(self, session_id):
        self.requests.append(("reset_session", session_id))
        return {"is_success": True}

    def close_session(self, session_id):
        self.closed.append(session_id)
        return {"is_success": True}

    def add_prompt(self, session_id, frame_index, **kwargs):
        self.requests.append(("add_prompt", session_id, frame_index, kwargs))
        return {"frame_index": frame_index, "outputs": self.outputs}

    def remove_object(self, session_id, obj_id, frame_index=0):
        self.requests.append(("remove_object", session_id, obj_id, frame_index))
        return {"frame_index": frame_index, "outputs": {"out_obj_ids": [], "out_binary_masks": []}}


def _write_test_frames(tmp_path, count=5, h=4, w=6):
    img_paths = []
    for i in range(count):
        path = tmp_path / f"{i:04d}.png"
        iio.imwrite(path, np.zeros((h, w, 3), dtype=np.uint8))
        img_paths.append(str(path))
    return img_paths


def _load_video_handler(monkeypatch):
    fake_model_builder = types.ModuleType("sam3.model_builder")
    fake_model_builder.build_sam3_predictor = lambda *args, **kwargs: None
    fake_model_builder.build_sam3_multiplex_video_predictor = lambda *args, **kwargs: None
    fake_sam3 = types.ModuleType("sam3")
    fake_sam3.model_builder = fake_model_builder
    monkeypatch.setitem(sys.modules, "sam3", fake_sam3)
    monkeypatch.setitem(sys.modules, "sam3.model_builder", fake_model_builder)
    module = importlib.import_module("video_handler")
    return importlib.reload(module)


def test_text_mode_tracker_keeps_video_length_when_most_frames_are_empty(monkeypatch, tmp_path):
    video_handler_module = _load_video_handler(monkeypatch)
    img_paths = _write_test_frames(tmp_path, count=5, h=4, w=6)
    responses = [
        {
            "frame_index": 0,
            "outputs": {
                "out_obj_ids": [0],
                "out_binary_masks": [np.ones((4, 6), dtype=bool)],
            },
        },
        {"frame_index": 1, "outputs": {"out_obj_ids": [], "out_binary_masks": []}},
        {"frame_index": 2, "outputs": {"out_obj_ids": [], "out_binary_masks": []}},
        {"frame_index": 3, "outputs": {"out_obj_ids": [], "out_binary_masks": []}},
        {"frame_index": 4, "outputs": {"out_obj_ids": [], "out_binary_masks": []}},
    ]

    handler = video_handler_module.VideoModeHandler()
    handler.current_text_prompt = "a person"
    handler.text_prompt_frame_idx = 0
    handler.cur_masks = {0: np.ones((4, 6), dtype=bool)}
    handler.session_id = "session-1"
    handler.img_paths = img_paths
    handler.backend = _FakeBackend(responses)

    written = {}
    monkeypatch.setattr(
        video_handler_module.iio,
        "mimwrite",
        lambda path, frames: written.update(path=path, frame_count=len(frames)),
    )

    out_path, msg = handler.run_tracker(propagation_direction="both")

    assert out_path == "tracked_colors.mp4"
    assert "Tracked 5 frames" in msg
    assert written["path"] == "tracked_colors.mp4"
    assert written["frame_count"] == 5
    assert len(handler.index_masks_all) == 5
    assert handler.index_masks_all[0].max() == 1
    assert all(mask.max() == 0 for mask in handler.index_masks_all[1:])


def test_text_mode_tracker_returns_error_when_no_masks_exist(monkeypatch, tmp_path):
    video_handler_module = _load_video_handler(monkeypatch)
    img_paths = _write_test_frames(tmp_path, count=3, h=4, w=6)
    responses = [
        {"frame_index": 0, "outputs": {"out_obj_ids": [], "out_binary_masks": []}},
        {"frame_index": 1, "outputs": {"out_obj_ids": [], "out_binary_masks": []}},
        {"frame_index": 2, "outputs": {"out_obj_ids": [], "out_binary_masks": []}},
    ]

    handler = video_handler_module.VideoModeHandler()
    handler.current_text_prompt = "a person"
    handler.text_prompt_frame_idx = 0
    handler.cur_masks = {0: np.ones((4, 6), dtype=bool)}
    handler.session_id = "session-2"
    handler.img_paths = img_paths
    handler.backend = _FakeBackend(responses)

    out_path, msg = handler.run_tracker(propagation_direction="both")

    assert out_path is None
    assert msg == "No masks generated. Add prompts first."


def test_video_handler_rejects_non_sam31_checkpoint(monkeypatch, tmp_path):
    video_handler_module = _load_video_handler(monkeypatch)
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"")

    with pytest.raises(ValueError):
        video_handler_module.VideoModeHandler(checkpoint_path=str(checkpoint))


def test_video_handler_uses_native_sam31_prompt_requests(monkeypatch, tmp_path):
    video_handler_module = _load_video_handler(monkeypatch)
    _write_test_frames(tmp_path, count=2, h=4, w=6)
    backend = _RecordingBackend()

    handler = video_handler_module.VideoModeHandler(backend=backend)
    assert handler.set_img_dir(str(tmp_path)) == 2
    handler.set_input_image(0)

    mask, msg = handler.add_text_prompt("truck", 0)
    assert mask is not None
    assert msg == "Detected 1 object(s)"
    assert backend.requests[-2] == ("reset_session", "session-1")
    assert backend.requests[-1][3] == {"text": "truck"}

    box_mask, box_msg = handler.add_box_prompt(0, (1, 1, 5, 3))
    assert box_mask is not None
    assert "Box prompt detected" in box_msg
    box_kwargs = backend.requests[-1][3]
    assert box_kwargs["bounding_boxes"] == [[1 / 6, 1 / 4, 4 / 6, 2 / 4]]
    assert box_kwargs["bounding_box_labels"] == [1]

    point_mask = handler.add_point(0, 2, 3)
    assert point_mask is not None
    point_kwargs = backend.requests[-1][3]
    assert point_kwargs["points"] == [[3 / 6, 2 / 4]]
    assert point_kwargs["point_labels"] == [1]
    assert point_kwargs["rel_coordinates"] is True
