import importlib
import sys
import types
from pathlib import Path

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
        return {
            "frame_index": frame_index,
            "outputs": {"out_obj_ids": [], "out_binary_masks": []},
        }


def _write_test_frames(tmp_path, count=5, h=4, w=6):
    tmp_path.mkdir(parents=True, exist_ok=True)
    img_paths = []
    for i in range(count):
        path = tmp_path / f"{i:04d}.png"
        iio.imwrite(path, np.zeros((h, w, 3), dtype=np.uint8))
        img_paths.append(str(path))
    return img_paths


def _load_video_handler(monkeypatch):
    fake_model_builder = types.ModuleType("sam3.model_builder")
    fake_model_builder.build_sam3_predictor = lambda *args, **kwargs: None
    fake_sam3 = types.ModuleType("sam3")
    fake_sam3.model_builder = fake_model_builder
    monkeypatch.setitem(sys.modules, "sam3", fake_sam3)
    monkeypatch.setitem(sys.modules, "sam3.model_builder", fake_model_builder)
    module = importlib.import_module("sam3_gui.video_handler")
    return importlib.reload(module)


def test_text_mode_tracker_keeps_video_length_when_most_frames_are_empty(
    monkeypatch, tmp_path
):
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
        video_handler_module,
        "runtime_output_path",
        lambda name: str(tmp_path / name),
    )

    def write_video(path, frames):
        Path(path).write_bytes(b"video")
        written.update(path=path, frame_count=len(frames))

    monkeypatch.setattr(
        video_handler_module.iio,
        "mimwrite",
        write_video,
    )

    out_path, msg = handler.run_tracker(propagation_direction="both")

    assert Path(out_path).parent == tmp_path
    assert Path(out_path).name.startswith("tracked_colors_")
    assert Path(out_path).suffix == ".mp4"
    assert "Tracked 5 frames" in msg
    assert written["path"] != out_path
    assert not Path(written["path"]).exists()
    assert Path(out_path).read_bytes() == b"video"
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


def test_remove_point_replays_original_frames_and_objects(monkeypatch, tmp_path):
    video_handler_module = _load_video_handler(monkeypatch)
    _write_test_frames(tmp_path, count=3, h=4, w=6)
    backend = _RecordingBackend()
    handler = video_handler_module.VideoModeHandler(backend=backend)
    handler.set_img_dir(str(tmp_path))
    handler.set_input_image(2)
    handler.cur_mask_idx = 0
    handler.selected_points = [[1, 1], [2, 2], [3, 3]]
    handler.selected_labels = [1.0, 0.0, 1.0]
    handler.selected_point_frames = [0, 1, 2]
    handler.selected_point_obj_ids = [0, 1, 0]
    backend.requests.clear()

    index_mask, msg = handler.remove_selected_point(1)

    assert index_mask is not None
    assert msg == "Removed point. 2 points remaining."
    assert backend.requests[0] == ("reset_session", "session-1")
    replayed = [request for request in backend.requests if request[0] == "add_prompt"]
    assert [(request[2], request[3]["obj_id"]) for request in replayed] == [
        (0, 0),
        (2, 0),
    ]
    assert all(
        request[2] != handler.frame_index or request[3]["points"] == [[3 / 6, 3 / 4]]
        for request in replayed
    )


def test_clear_points_resets_backend_and_invalidates_tracking(monkeypatch, tmp_path):
    video_handler_module = _load_video_handler(monkeypatch)
    _write_test_frames(tmp_path, count=2, h=4, w=6)
    backend = _RecordingBackend()
    handler = video_handler_module.VideoModeHandler(backend=backend)
    handler.set_img_dir(str(tmp_path))
    handler.set_input_image(0)
    handler.selected_points = [[2, 1]]
    handler.selected_labels = [1.0]
    handler.selected_point_frames = [0]
    handler.selected_point_obj_ids = [0]
    handler.cur_masks = {0: np.ones((4, 6), dtype=bool)}
    handler.index_masks_all = [np.ones((4, 6), dtype=np.uint8)]
    handler.color_masks_all = [np.ones((4, 6, 3), dtype=np.uint8)]
    old_video = tmp_path / "old.mp4"
    old_video.write_bytes(b"old")
    handler.tracking_output_path = str(old_video)
    backend.requests.clear()

    index_mask, video_path, msg = handler.clear_points()

    assert (index_mask, video_path, msg) == (None, None, "Cleared points")
    assert backend.requests == [("reset_session", "session-1")]
    assert handler.cur_masks == {}
    assert handler.index_masks_all == []
    assert handler.color_masks_all == []
    assert handler.tracking_output_path is None
    assert not old_video.exists()
    assert handler.run_tracker() == (None, "No objects detected yet.")


def test_remove_object_replays_only_other_objects_prompts(monkeypatch, tmp_path):
    video_handler_module = _load_video_handler(monkeypatch)
    _write_test_frames(tmp_path, count=3, h=4, w=6)
    backend = _RecordingBackend()
    handler = video_handler_module.VideoModeHandler(backend=backend)
    handler.set_img_dir(str(tmp_path))
    handler.set_input_image(2)
    handler.cur_masks = {
        0: np.ones((4, 6), dtype=bool),
        1: np.ones((4, 6), dtype=bool),
    }
    handler.selected_points = [[1, 1], [4, 2]]
    handler.selected_labels = [1.0, 0.0]
    handler.selected_point_frames = [0, 1]
    handler.selected_point_obj_ids = [0, 1]
    backend.requests.clear()

    handler.remove_object(0)

    assert handler.selected_points == [[4, 2]]
    assert handler.selected_labels == [0.0]
    assert handler.selected_point_frames == [1]
    assert handler.selected_point_obj_ids == [1]
    assert backend.requests[0] == ("reset_session", "session-1")
    assert backend.requests[1][0:3] == ("add_prompt", "session-1", 1)
    assert backend.requests[1][3]["obj_id"] == 1
    assert backend.requests[2] == ("remove_object", "session-1", 0, 2)


def test_prompt_change_invalidates_cached_results_and_old_video(monkeypatch, tmp_path):
    video_handler_module = _load_video_handler(monkeypatch)
    _write_test_frames(tmp_path, count=1, h=4, w=6)
    backend = _RecordingBackend()
    handler = video_handler_module.VideoModeHandler(backend=backend)
    handler.set_img_dir(str(tmp_path))
    handler.set_input_image(0)
    handler.index_masks_all = [np.ones((4, 6), dtype=np.uint8)]
    handler.color_masks_all = [np.ones((4, 6, 3), dtype=np.uint8)]
    old_video = tmp_path / "tracked.mp4"
    old_video.write_bytes(b"old")
    handler.tracking_output_path = str(old_video)

    mask = handler.add_point(0, 1, 2)

    assert mask is not None
    assert handler.index_masks_all == []
    assert handler.color_masks_all == []
    assert handler.tracking_output_path is None
    assert not old_video.exists()
    assert backend.requests[-2] == ("reset_session", "session-1")
    assert backend.requests[-1][2:4] == (
        0,
        {
            "obj_id": 0,
            "points": [[2 / 6, 1 / 4]],
            "point_labels": [1],
            "clear_old_points": True,
            "rel_coordinates": True,
        },
    )


def test_tracker_uses_unique_atomic_output_paths(monkeypatch, tmp_path):
    video_handler_module = _load_video_handler(monkeypatch)
    img_paths = _write_test_frames(tmp_path / "frames", count=1, h=4, w=6)
    responses = [
        {
            "frame_index": 0,
            "outputs": {
                "out_obj_ids": [0],
                "out_binary_masks": [np.ones((4, 6), dtype=bool)],
            },
        }
    ]
    output_dir = tmp_path / "output"
    handler = video_handler_module.VideoModeHandler()
    handler.session_id = "session-1"
    handler.img_paths = img_paths
    handler.cur_masks = {0: np.ones((4, 6), dtype=bool)}
    handler.backend = _FakeBackend(responses)
    monkeypatch.setattr(
        video_handler_module,
        "runtime_output_path",
        lambda name: str(output_dir / name),
    )

    temp_paths = []

    def write_video(path, frames):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"video")
        temp_paths.append(path)

    monkeypatch.setattr(video_handler_module.iio, "mimwrite", write_video)

    first_path, _ = handler.run_tracker()
    second_path, _ = handler.run_tracker()

    assert first_path != second_path
    assert Path(first_path).exists()
    assert Path(second_path).exists()
    assert all(path.name.startswith(".tracked_colors_") for path in temp_paths)
    assert all(not path.exists() for path in temp_paths)


def test_reset_clears_loaded_frames_and_tracking_output(monkeypatch, tmp_path):
    video_handler_module = _load_video_handler(monkeypatch)
    _write_test_frames(tmp_path, count=1)
    backend = _RecordingBackend()
    handler = video_handler_module.VideoModeHandler(backend=backend)
    handler.set_img_dir(str(tmp_path))
    handler.set_input_image(0)
    old_video = tmp_path / "tracked.mp4"
    old_video.write_bytes(b"old")
    handler.tracking_output_path = str(old_video)

    handler.reset()

    assert handler.session_id is None
    assert handler.img_dir == ""
    assert handler.img_paths == []
    assert handler.image is None
    assert handler.tracking_output_path is None
    assert not old_video.exists()
    assert backend.closed == ["session-1"]


def test_save_masks_uses_path_stem(monkeypatch, tmp_path):
    video_handler_module = _load_video_handler(monkeypatch)
    handler = video_handler_module.VideoModeHandler()
    handler.img_paths = [str(tmp_path / "frame.final.jpeg")]
    handler.color_masks_all = [np.zeros((4, 6, 3), dtype=np.uint8)]
    handler.index_masks_all = [np.zeros((4, 6), dtype=np.uint8)]
    output_dir = tmp_path / "masks"
    written_images = []
    monkeypatch.setattr(
        video_handler_module.iio,
        "imwrite",
        lambda path, image: written_images.append(Path(path)),
    )

    msg = handler.save_masks_to_dir(str(output_dir))

    assert msg == f"Saved masks to {output_dir}."
    assert written_images == [output_dir / "frame.final.jpeg"]
    assert (output_dir / "frame.final.npy").exists()
