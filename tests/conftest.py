"""
Shared pytest fixtures for SAM3-GUI tests.
"""

import os
import sys

import numpy as np
import pytest

GUI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM3_REPO_ROOT = os.path.expanduser("~/sam3")

# Add the GUI and SAM3 package roots without shadowing the editable sam3 package.
sys.path.insert(0, GUI_ROOT)
sys.path.insert(0, SAM3_REPO_ROOT)


# Configuration
DEFAULT_CHECKPOINT_PATH = os.environ.get(
    "SAM3_CHECKPOINT_PATH", 
    os.path.expanduser("~/sam3/model/sam3.1_multiplex.pt")
)
ALLOW_HF_DOWNLOAD = os.environ.get("SAM3_ALLOW_HF_DOWNLOAD", "").lower() in {"1", "true", "yes"}
DEFAULT_TEST_IMG_DIR = os.environ.get(
    "SAM3_TEST_IMG_DIR",
    os.path.join(GUI_ROOT, "data_root/images/Cam1_color")
)
DEFAULT_TEST_VIDEO_PATH = os.environ.get(
    "SAM3_TEST_VIDEO_PATH",
    os.path.join(GUI_ROOT, "data_root/videos/Cam1_color.mp4"),
)


def _resolve_sam31_checkpoint() -> str | None:
    checkpoint_path = DEFAULT_CHECKPOINT_PATH
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    if ALLOW_HF_DOWNLOAD:
        return None
    pytest.skip(
        "SAM 3.1 checkpoint not found at "
        f"{checkpoint_path}. Run download_model.py, set SAM3_CHECKPOINT_PATH, "
        "or set SAM3_ALLOW_HF_DOWNLOAD=1 for online Hugging Face download."
    )


@pytest.fixture(scope="session")
def require_sam31_checkpoint() -> str | None:
    """Require a local/online SAM 3.1 checkpoint for real integration tests."""
    return _resolve_sam31_checkpoint()


@pytest.fixture(scope="session")
def sam3_model():
    """Load SAM3 video predictor once for all tests."""
    from sam3.model_builder import build_sam3_predictor

    model = build_sam3_predictor(
        version="sam3.1",
        checkpoint_path=_resolve_sam31_checkpoint(),
    )
    
    yield model
    
    try:
        model.shutdown()
    except Exception:
        pass


@pytest.fixture(scope="session")
def test_img_dir():
    """Path to test image directory."""
    img_dir = DEFAULT_TEST_IMG_DIR
    if not os.path.exists(img_dir):
        pytest.skip(f"Test image directory not found: {img_dir}")
    return img_dir


@pytest.fixture(scope="session")
def test_images(test_img_dir):
    """List of test image paths."""
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    images = [
        os.path.join(test_img_dir, f)
        for f in sorted(os.listdir(test_img_dir))
        if os.path.splitext(f.lower())[1] in valid_extensions
    ]
    if not images:
        pytest.skip(f"No test images found in {test_img_dir}")
    return images


@pytest.fixture(scope="session")
def test_video_path():
    """Path to a real test video file."""
    if not os.path.exists(DEFAULT_TEST_VIDEO_PATH):
        pytest.skip(f"Test video not found: {DEFAULT_TEST_VIDEO_PATH}")
    return DEFAULT_TEST_VIDEO_PATH


@pytest.fixture
def session_id(sam3_model, test_img_dir):
    """Create a fresh SAM3 session for each test."""
    response = sam3_model.handle_request(
        request=dict(
            type="start_session",
            resource_path=test_img_dir,
        )
    )
    session_id = response["session_id"]
    
    yield session_id
    
    try:
        sam3_model.handle_request(
            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )
    except Exception:
        pass


@pytest.fixture
def sample_point():
    """Sample point for testing (normalized 0-1 coordinates)."""
    return [[0.5, 0.5]]


@pytest.fixture
def sample_point_label():
    """Sample point label (1 = positive)."""
    return [1]


@pytest.fixture
def sample_text_prompt():
    """Sample text prompt for testing."""
    return "truck"


def normalize_points(points, img_width, img_height):
    """Convert absolute pixel coordinates to relative (0-1) coordinates."""
    return [[x / img_width, y / img_height] for x, y in points]
