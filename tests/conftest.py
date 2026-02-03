"""
Shared pytest fixtures for SAM3-GUI tests.
"""

import os
import sys

import numpy as np
import pytest

# Add parent and sam3 root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# Configuration
DEFAULT_CHECKPOINT_PATH = os.environ.get(
    "SAM3_CHECKPOINT_PATH", 
    os.path.expanduser("~/sam3/model/sam3.pt")
)
DEFAULT_TEST_IMG_DIR = os.environ.get(
    "SAM3_TEST_IMG_DIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_root/images/test_seq")
)


@pytest.fixture(scope="session")
def sam3_model():
    """Load SAM3 video predictor once for all tests."""
    import torch
    from sam3.model_builder import build_sam3_video_predictor
    
    gpus_to_use = [0] if torch.cuda.is_available() else []
    
    checkpoint_path = DEFAULT_CHECKPOINT_PATH
    if not os.path.exists(checkpoint_path):
        checkpoint_path = None
    
    model = build_sam3_video_predictor(
        checkpoint_path=checkpoint_path,
        gpus_to_use=gpus_to_use
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
