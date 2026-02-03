"""
Tests for mask utility functions.

Corresponds to: mask_app.py -> make_index_mask(), colorize_masks(), compose_img_mask()
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mask_app import (
    get_hls_palette,
    colorize_masks,
    compose_img_mask,
    PromptGUI,
)


class TestMakeIndexMask:
    """Test make_index_mask functionality."""

    def test_make_index_mask_with_single_mask(self):
        """Test make_index_mask with a single mask."""
        gui = PromptGUI.__new__(PromptGUI)
        gui.image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        masks = {0: np.zeros((100, 100), dtype=bool)}
        masks[0][20:40, 20:40] = True
        
        result = gui.make_index_mask(masks)
        
        assert result.shape == (100, 100)
        assert result.dtype == np.uint8
        assert result[30, 30] == 1  # Inside mask region
        assert result[0, 0] == 0  # Outside mask region

    def test_make_index_mask_with_multiple_masks(self):
        """Test make_index_mask with multiple masks."""
        gui = PromptGUI.__new__(PromptGUI)
        gui.image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        masks = {
            0: np.zeros((100, 100), dtype=bool),
            1: np.zeros((100, 100), dtype=bool),
        }
        masks[0][10:30, 10:30] = True
        masks[1][50:70, 50:70] = True
        
        result = gui.make_index_mask(masks)
        
        assert result.shape == (100, 100)
        assert result[20, 20] == 1  # First mask
        assert result[60, 60] == 2  # Second mask
        assert result[0, 0] == 0  # Background

    def test_make_index_mask_empty(self):
        """Test make_index_mask with empty masks dict."""
        gui = PromptGUI.__new__(PromptGUI)
        gui.image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        masks = {}
        result = gui.make_index_mask(masks)
        
        assert result.shape == (100, 100)
        assert result.max() == 0  # All background

    def test_make_index_mask_empty_no_image(self):
        """Test make_index_mask with empty masks and no image."""
        gui = PromptGUI.__new__(PromptGUI)
        gui.image = None
        
        masks = {}
        result = gui.make_index_mask(masks)
        
        assert result.shape == (1, 1)
        assert result.max() == 0


class TestColorPalette:
    """Test color palette generation."""

    def test_get_hls_palette_size(self):
        """Test that palette has correct number of colors."""
        palette = get_hls_palette(5)
        assert len(palette) == 5
        
    def test_get_hls_palette_first_is_black(self):
        """Test that first color is black (background)."""
        palette = get_hls_palette(5)
        assert tuple(palette[0]) == (0, 0, 0)

    def test_get_hls_palette_dtype(self):
        """Test that palette has uint8 dtype."""
        palette = get_hls_palette(5)
        assert palette.dtype == np.uint8


class TestColorize:
    """Test mask colorization."""

    def test_colorize_masks_output_shape(self):
        """Test that colorize_masks returns correct shapes."""
        images = [np.zeros((100, 100, 3), dtype=np.uint8)]
        index_masks = [np.zeros((100, 100), dtype=np.uint8)]
        
        out_frames, color_masks = colorize_masks(images, index_masks)
        
        assert len(out_frames) == 1
        assert len(color_masks) == 1
        assert out_frames[0].shape == (100, 100, 3)
        assert color_masks[0].shape == (100, 100, 3)

    def test_compose_img_mask(self):
        """Test image-mask composition."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        color_mask = np.zeros((100, 100, 3), dtype=np.uint8)
        color_mask[:, :, 0] = 255  # Red mask
        
        result = compose_img_mask(img, color_mask, fac=0.5)
        
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8


class TestNormalizePoints:
    """Test point normalization."""

    def test_normalize_points(self):
        """Test _normalize_points method."""
        gui = PromptGUI.__new__(PromptGUI)
        gui.image = np.zeros((480, 640, 3), dtype=np.uint8)  # 640x480 image
        
        points = [[320, 240]]  # Center of image in pixels
        normalized = gui._normalize_points(points)
        
        assert len(normalized) == 1
        assert abs(normalized[0][0] - 0.5) < 0.01  # x should be ~0.5
        assert abs(normalized[0][1] - 0.5) < 0.01  # y should be ~0.5

    def test_normalize_points_no_image(self):
        """Test _normalize_points with no image set."""
        gui = PromptGUI.__new__(PromptGUI)
        gui.image = None
        
        points = [[100, 100]]
        result = gui._normalize_points(points)
        
        # Should return unchanged
        assert result == points

    def test_normalize_multiple_points(self):
        """Test _normalize_points with multiple points."""
        gui = PromptGUI.__new__(PromptGUI)
        gui.image = np.zeros((100, 200, 3), dtype=np.uint8)  # 200x100 image
        
        points = [[0, 0], [200, 100], [100, 50]]
        normalized = gui._normalize_points(points)
        
        assert len(normalized) == 3
        assert normalized[0] == [0.0, 0.0]
        assert normalized[1] == [1.0, 1.0]
        assert normalized[2] == [0.5, 0.5]
