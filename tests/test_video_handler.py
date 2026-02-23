import contextlib
import importlib
import sys
import types

import imageio.v2 as iio
import numpy as np


class _FakePredictor:
    def __init__(self, responses):
        self._responses = responses

    def handle_request(self, request):
        pass

    def handle_stream_request(self, request):
        assert request["type"] == "propagate_in_video"
        for response in self._responses:
            yield response


def _write_test_frames(tmp_path, count=5, h=4, w=6):
    img_paths = []
    for i in range(count):
        path = tmp_path / f"{i:04d}.png"
        iio.imwrite(path, np.zeros((h, w, 3), dtype=np.uint8))
        img_paths.append(str(path))
    return img_paths


def _load_video_handler(monkeypatch):
    fake_model_builder = types.ModuleType("sam3.model_builder")
    fake_model_builder.build_sam3_video_predictor = lambda *args, **kwargs: None
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
    handler.use_text_mode = True
    handler.current_text_prompt = "a person"
    handler.text_prompt_frame_idx = 0
    handler.cur_masks = {0: np.ones((4, 6), dtype=bool)}
    handler.inference_state = "session-1"
    handler.img_paths = img_paths
    handler.video_predictor = _FakePredictor(responses)

    written = {}
    monkeypatch.setattr(
        video_handler_module.torch,
        "autocast",
        lambda *args, **kwargs: contextlib.nullcontext(),
    )
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
    handler.use_text_mode = True
    handler.current_text_prompt = "a person"
    handler.text_prompt_frame_idx = 0
    handler.cur_masks = {0: np.ones((4, 6), dtype=bool)}
    handler.inference_state = "session-2"
    handler.img_paths = img_paths
    handler.video_predictor = _FakePredictor(responses)

    monkeypatch.setattr(
        video_handler_module.torch,
        "autocast",
        lambda *args, **kwargs: contextlib.nullcontext(),
    )

    out_path, msg = handler.run_tracker(propagation_direction="both")

    assert out_path is None
    assert msg == "No masks generated. Add prompts first."
