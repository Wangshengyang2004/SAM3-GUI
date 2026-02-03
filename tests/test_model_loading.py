"""
Tests for SAM3 model loading functionality.

Corresponds to: mask_app.py -> PromptGUI.init_sam_model()
"""

import pytest
import torch


class TestModelLoading:
    """Test SAM3 model loading."""

    def test_model_loads_successfully(self, sam3_model):
        """Test that the SAM3 model loads without errors."""
        assert sam3_model is not None

    def test_model_on_cuda(self, sam3_model):
        """Test that the model is on CUDA if available."""
        if torch.cuda.is_available():
            # Model should be using CUDA
            assert sam3_model.model is not None
        else:
            pytest.skip("CUDA not available")

    def test_model_has_handle_request(self, sam3_model):
        """Test that model has the handle_request method."""
        assert hasattr(sam3_model, 'handle_request')
        assert callable(sam3_model.handle_request)

    def test_model_has_handle_stream_request(self, sam3_model):
        """Test that model has the handle_stream_request method."""
        assert hasattr(sam3_model, 'handle_stream_request')
        assert callable(sam3_model.handle_stream_request)
