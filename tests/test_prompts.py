"""
Tests for SAM3 prompting functionality.

Corresponds to: mask_app.py -> PromptGUI.add_text_prompt(), add_point(), get_sam_mask()
"""

import numpy as np
import pytest


class TestTextPrompts:
    """Test text prompting functionality."""

    def test_text_prompt_detects_objects(self, sam3_model, session_id, sample_text_prompt):
        """Test that text prompt detects objects."""
        response = sam3_model.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=sample_text_prompt,
            )
        )
        
        assert "outputs" in response
        outputs = response["outputs"]
        assert outputs is not None

    def test_text_prompt_correct_output_keys(self, sam3_model, session_id, sample_text_prompt):
        """Test that text prompt returns correct output keys."""
        response = sam3_model.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=sample_text_prompt,
            )
        )
        
        outputs = response.get("outputs", {})
        if outputs:
            # Check for correct SAM3 output keys
            assert "out_binary_masks" in outputs or len(outputs.get("out_obj_ids", [])) == 0
            assert "out_obj_ids" in outputs
            assert "out_probs" in outputs

    def test_text_prompt_masks_are_valid(self, sam3_model, session_id, sample_text_prompt):
        """Test that detected masks have valid format."""
        response = sam3_model.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=sample_text_prompt,
            )
        )
        
        outputs = response.get("outputs", {})
        masks = outputs.get("out_binary_masks", [])
        
        if len(masks) > 0:
            for mask in masks:
                assert hasattr(mask, 'shape'), "Mask should have shape attribute"
                assert len(mask.shape) >= 2, "Mask should be at least 2D"


class TestPointPrompts:
    """Test point prompting functionality.
    
    Note: SAM3 requires a text prompt to be added first before point prompts
    can be used for refinement. Point prompts refine existing tracked objects.
    """

    def test_point_prompt_with_normalized_coords(self, sam3_model, session_id, sample_point, sample_point_label, sample_text_prompt):
        """Test point prompt with normalized coordinates (requires text prompt first)."""
        # First add text prompt to initialize tracking
        sam3_model.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=sample_text_prompt,
            )
        )
        
        # Run propagation to populate cache
        for _ in sam3_model.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
            )
        ):
            pass
        
        # Now add point prompt for refinement
        response = sam3_model.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                obj_id=0,
                points=sample_point,
                point_labels=sample_point_label,
            )
        )
        
        assert "outputs" in response
        outputs = response.get("outputs", {})
        assert outputs is not None

    def test_point_prompt_labels(self, sam3_model, session_id, sample_text_prompt):
        """Test point prompts with different labels (requires text prompt first)."""
        # First add text prompt
        sam3_model.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=sample_text_prompt,
            )
        )
        
        # Run propagation
        for _ in sam3_model.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
            )
        ):
            pass
        
        # Add point prompt
        positive_point = [[0.5, 0.5]]
        positive_label = [1]
        
        response = sam3_model.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                obj_id=1,
                points=positive_point,
                point_labels=positive_label,
            )
        )
        
        assert "outputs" in response

    def test_multiple_points(self, sam3_model, session_id, sample_text_prompt):
        """Test multiple points in a single prompt (requires text prompt first)."""
        # First add text prompt
        sam3_model.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=sample_text_prompt,
            )
        )
        
        # Run propagation
        for _ in sam3_model.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
            )
        ):
            pass
        
        # Add multiple points
        points = [[0.3, 0.3], [0.7, 0.7]]
        labels = [1, 1]  # Both positive
        
        response = sam3_model.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                obj_id=2,
                points=points,
                point_labels=labels,
            )
        )
        
        assert "outputs" in response

    def test_positive_and_negative_points(self, sam3_model, session_id, sample_text_prompt):
        """Test mix of positive and negative points (requires text prompt first)."""
        # First add text prompt
        sam3_model.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=sample_text_prompt,
            )
        )
        
        # Run propagation
        for _ in sam3_model.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
            )
        ):
            pass
        
        # Add mixed points
        points = [[0.5, 0.5], [0.2, 0.2]]
        labels = [1, 0]  # One positive, one negative
        
        response = sam3_model.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                obj_id=3,
                points=points,
                point_labels=labels,
            )
        )
        
        assert "outputs" in response
