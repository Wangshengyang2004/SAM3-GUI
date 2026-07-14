import subprocess
from pathlib import Path

import pytest

from sam3_gui import ui_handlers


def _create_video_tree(tmp_path):
    video_root = tmp_path / "videos"
    image_root = tmp_path / "images"
    video_root.mkdir()
    image_root.mkdir()
    video_path = video_root / "clip.mp4"
    video_path.write_bytes(b"video")
    return video_path, image_root


def _extract(tmp_path, **overrides):
    arguments = {
        "root_dir": str(tmp_path),
        "vid_file": "clip.mp4",
        "start": 0,
        "end": 2,
        "fps": 10,
        "downsampling_choice": "Half",
        "vid_name": "videos",
        "img_name": "images",
    }
    arguments.update(overrides)
    return ui_handlers.extract_video_frames(**arguments)


@pytest.mark.parametrize(
    "vid_file", ["../clip.mp4", "nested/clip.mp4", r"nested\clip.mp4"]
)
def test_extract_rejects_video_paths_that_are_not_basenames(
    tmp_path, monkeypatch, vid_file
):
    _create_video_tree(tmp_path)
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(ui_handlers.subprocess, "run", fail_if_called)

    result = _extract(tmp_path, vid_file=vid_file)

    assert result[1] is None
    assert "basename" in result[3]
    assert called is False


def test_extract_rejects_resolved_paths_outside_configured_roots(tmp_path, monkeypatch):
    _, image_root = _create_video_tree(tmp_path)
    outside_video = tmp_path / "outside.mp4"
    outside_video.write_bytes(b"outside")
    (tmp_path / "videos" / "escape.mp4").symlink_to(outside_video)
    outside_frames = tmp_path / "outside-frames"
    outside_frames.mkdir()
    (image_root / "clip").symlink_to(outside_frames, target_is_directory=True)
    monkeypatch.setattr(ui_handlers, "get_video_resolution", lambda path: (1920, 1080))

    video_result = _extract(tmp_path, vid_file="escape.mp4")
    output_result = _extract(tmp_path)

    assert "outside configured video root" in video_result[3]
    assert "outside configured image root" in output_result[3]
    assert list(outside_frames.iterdir()) == []


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"start": -1}, "non-negative"),
        ({"start": 2, "end": 2}, "greater than start"),
        ({"fps": 0}, "positive"),
        ({"downsampling_choice": "Unknown"}, "downsampling"),
    ],
)
def test_extract_validates_parameters_before_overwriting(
    tmp_path, monkeypatch, overrides, message
):
    _, image_root = _create_video_tree(tmp_path)
    output_dir = image_root / "clip"
    output_dir.mkdir()
    old_frame = output_dir / "old.jpg"
    old_frame.write_bytes(b"old")
    monkeypatch.setattr(ui_handlers, "get_video_resolution", lambda path: (1920, 1080))
    monkeypatch.setattr(
        ui_handlers.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "ffmpeg must not run for invalid parameters"
        ),
    )

    result = _extract(tmp_path, **overrides)

    assert message in result[3]
    assert old_frame.read_bytes() == b"old"


@pytest.mark.parametrize("resolution", [None, (0, 1080), (1920, -1), (1920, 8)])
def test_extract_validates_resolution_before_overwriting(
    tmp_path, monkeypatch, resolution
):
    _, image_root = _create_video_tree(tmp_path)
    output_dir = image_root / "clip"
    output_dir.mkdir()
    old_frame = output_dir / "old.jpg"
    old_frame.write_bytes(b"old")
    monkeypatch.setattr(ui_handlers, "get_video_resolution", lambda path: resolution)
    monkeypatch.setattr(
        ui_handlers.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "ffmpeg must not run for invalid resolution"
        ),
    )

    result = _extract(tmp_path, downsampling_choice="Sixteenth")

    assert "resolution" in result[3].lower()
    assert old_frame.read_bytes() == b"old"


def test_extract_failure_preserves_old_directory_and_cleans_temp(tmp_path, monkeypatch):
    _, image_root = _create_video_tree(tmp_path)
    output_dir = image_root / "clip"
    output_dir.mkdir()
    old_frame = output_dir / "old.jpg"
    old_frame.write_bytes(b"old")
    monkeypatch.setattr(ui_handlers, "get_video_resolution", lambda path: (1920, 1080))

    def fail_ffmpeg(command, check):
        Path(command[-1]).parent.joinpath("00001.jpg").write_bytes(b"partial")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(ui_handlers.subprocess, "run", fail_ffmpeg)

    result = _extract(tmp_path)

    assert "Failed to extract frames" in result[3]
    assert old_frame.read_bytes() == b"old"
    assert sorted(path.name for path in image_root.iterdir()) == ["clip"]


def test_extract_handles_missing_ffmpeg_and_preserves_old_directory(
    tmp_path, monkeypatch
):
    _, image_root = _create_video_tree(tmp_path)
    output_dir = image_root / "clip"
    output_dir.mkdir()
    old_frame = output_dir / "old.jpg"
    old_frame.write_bytes(b"old")
    monkeypatch.setattr(ui_handlers, "get_video_resolution", lambda path: (1920, 1080))
    monkeypatch.setattr(
        ui_handlers.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("ffmpeg")),
    )

    result = _extract(tmp_path)

    assert "ffmpeg is not installed" in result[3]
    assert old_frame.read_bytes() == b"old"
    assert sorted(path.name for path in image_root.iterdir()) == ["clip"]


def test_extract_requires_at_least_one_frame_before_replacing(tmp_path, monkeypatch):
    _, image_root = _create_video_tree(tmp_path)
    output_dir = image_root / "clip"
    output_dir.mkdir()
    old_frame = output_dir / "old.jpg"
    old_frame.write_bytes(b"old")
    monkeypatch.setattr(ui_handlers, "get_video_resolution", lambda path: (1920, 1080))
    monkeypatch.setattr(ui_handlers.subprocess, "run", lambda command, check: None)

    result = _extract(tmp_path)

    assert "produced no frames" in result[3]
    assert old_frame.read_bytes() == b"old"
    assert sorted(path.name for path in image_root.iterdir()) == ["clip"]


def test_extract_replace_failure_restores_old_directory(tmp_path, monkeypatch):
    _, image_root = _create_video_tree(tmp_path)
    output_dir = image_root / "clip"
    output_dir.mkdir()
    old_frame = output_dir / "old.jpg"
    old_frame.write_bytes(b"old")
    monkeypatch.setattr(ui_handlers, "get_video_resolution", lambda path: (1920, 1080))

    def successful_ffmpeg(command, check):
        Path(command[-1]).parent.joinpath("00001.jpg").write_bytes(b"new")

    real_replace = ui_handlers.os.replace

    def fail_new_directory_install(source, destination):
        source_path = Path(source)
        if (
            source_path.name.startswith(".clip.tmp-")
            and Path(destination) == output_dir
        ):
            raise OSError("replace failed")
        real_replace(source, destination)

    monkeypatch.setattr(ui_handlers.subprocess, "run", successful_ffmpeg)
    monkeypatch.setattr(ui_handlers.os, "replace", fail_new_directory_install)

    result = _extract(tmp_path)

    assert "replace failed" in result[3]
    assert old_frame.read_bytes() == b"old"
    assert sorted(path.name for path in image_root.iterdir()) == ["clip"]


def test_extract_successfully_replaces_old_directory(tmp_path, monkeypatch):
    _, image_root = _create_video_tree(tmp_path)
    output_dir = image_root / "clip"
    output_dir.mkdir()
    (output_dir / "old.jpg").write_bytes(b"old")
    monkeypatch.setattr(ui_handlers, "get_video_resolution", lambda path: (1920, 1080))

    def successful_ffmpeg(command, check):
        output_pattern = Path(command[-1])
        assert output_pattern.parent.parent == image_root
        output_pattern.parent.joinpath("00001.jpg").write_bytes(b"new")

    monkeypatch.setattr(ui_handlers.subprocess, "run", successful_ffmpeg)

    result = _extract(tmp_path)

    assert result[0] == "clip"
    assert result[1] == str(output_dir)
    assert "Extracted frames" in result[3]
    assert not (output_dir / "old.jpg").exists()
    assert (output_dir / "00001.jpg").read_bytes() == b"new"
    assert sorted(path.name for path in image_root.iterdir()) == ["clip"]
