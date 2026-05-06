import numpy as np
import pytest

from sam31_backend import (
    SAM31_CHECKPOINT_NAME,
    SAM31_HF_REPO,
    Sam31Backend,
    Sam31BackendConfig,
    Sam31ModelLoadError,
    index_mask_from_obj_masks,
    make_predictor_init_state_compatible,
    normalize_points,
    normalize_xyxy_box,
    output_masks_by_obj,
    serialize_outputs,
    validate_sam31_checkpoint_path,
)


def test_checkpoint_validation_rejects_non_multiplex_names(tmp_path):
    for name in ("sam3.pt", "sam3_multiplex.pt", "sam3.1.pt"):
        path = tmp_path / name
        path.write_bytes(b"")
        with pytest.raises(ValueError):
            validate_sam31_checkpoint_path(str(path))


def test_checkpoint_validation_accepts_sam31_multiplex_name(tmp_path):
    path = tmp_path / "sam3.1_multiplex.pt"
    path.write_bytes(b"")
    assert validate_sam31_checkpoint_path(str(path)) == str(path)


def test_backend_status_reports_checkpoint_state(tmp_path):
    lazy_backend = Sam31Backend()
    assert lazy_backend.status()["checkpoint_available"] is False
    assert lazy_backend.status()["will_download_checkpoint"] is True

    checkpoint = tmp_path / "sam3.1_multiplex.pt"
    checkpoint.write_bytes(b"")
    backend = Sam31Backend()
    backend.config.checkpoint_path = str(checkpoint)

    status = backend.status()
    assert status["checkpoint_path"] == str(checkpoint)
    assert status["checkpoint_available"] is True
    assert status["will_download_checkpoint"] is False
    assert status["model_loaded"] is False


def test_backend_defaults_to_synchronous_frame_loading():
    assert Sam31BackendConfig().async_loading_frames is False


def test_propagate_skips_native_call_when_prompt_finds_no_objects():
    class Predictor:
        def __init__(self):
            self.stream_called = False

        def handle_request(self, request):
            assert request["type"] == "add_prompt"
            return {
                "frame_index": request["frame_index"],
                "outputs": {"out_obj_ids": [], "out_binary_masks": []},
            }

        def handle_stream_request(self, request):
            self.stream_called = True
            yield {"frame_index": 0, "outputs": {}}

    backend = Sam31Backend()
    predictor = Predictor()
    backend.predictor = predictor

    backend.add_prompt("session-1", 0, text="missing object")
    assert list(backend.propagate("session-1", propagation_direction="forward")) == []
    assert predictor.stream_called is False


def test_mask_helpers_map_outputs_by_object_and_serialize_rle():
    outputs = {
        "out_obj_ids": np.array([0, 2], dtype=np.int64),
        "out_probs": np.array([0.9, 0.7], dtype=np.float32),
        "out_boxes_xywh": np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]], dtype=np.float32),
        "out_binary_masks": np.array(
            [
                [[1, 0], [0, 0]],
                [[0, 0], [0, 1]],
            ],
            dtype=bool,
        ),
    }

    masks = output_masks_by_obj(outputs, target_shape=(4, 4))
    assert sorted(masks) == [0, 2]
    assert masks[0].shape == (4, 4)

    index_mask = index_mask_from_obj_masks(masks)
    assert index_mask.dtype == np.uint8
    assert set(np.unique(index_mask)) == {0, 1, 3}

    payload = serialize_outputs(outputs, include_masks=True)
    assert payload["object_ids"] == [0, 2]
    assert payload["mask_count"] == 2
    assert len(payload["masks_rle"]) == 2


def test_prompt_coordinate_normalization():
    assert normalize_points([[50, 25]], width=100, height=50) == [[0.5, 0.5]]
    assert normalize_xyxy_box((80, 40, 20, 10), width=100, height=50) == [0.2, 0.2, 0.6, 0.6]


def test_model_load_failure_has_actionable_message(monkeypatch):
    def fail_build_sam3_predictor(**kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr("sam3.model_builder.build_sam3_predictor", fail_build_sam3_predictor)

    with pytest.raises(Sam31ModelLoadError) as exc_info:
        Sam31Backend().ensure_predictor()

    message = str(exc_info.value)
    assert "Failed to load native SAM 3.1 Object Multiplex" in message
    assert SAM31_CHECKPOINT_NAME in message
    assert SAM31_HF_REPO in message
    assert "hf auth login" in message
    assert "network unavailable" in message


def test_predictor_init_state_compatibility_ignores_default_offload_state():
    class Model:
        def init_state(self, resource_path, offload_video_to_cpu=False, async_loading_frames=False):
            return {
                "resource_path": resource_path,
                "offload_video_to_cpu": offload_video_to_cpu,
                "async_loading_frames": async_loading_frames,
            }

    class Predictor:
        model = Model()

    predictor = Predictor()
    make_predictor_init_state_compatible(predictor)

    assert predictor.model.init_state(
        resource_path="images",
        offload_video_to_cpu=True,
        offload_state_to_cpu=False,
        async_loading_frames=True,
    ) == {
        "resource_path": "images",
        "offload_video_to_cpu": True,
        "async_loading_frames": True,
    }

    with pytest.raises(TypeError, match="offload_state_to_cpu=True"):
        predictor.model.init_state(resource_path="images", offload_state_to_cpu=True)
