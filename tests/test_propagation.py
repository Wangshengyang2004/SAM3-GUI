"""
Tests for SAM3 video propagation functionality.

Corresponds to: mask_app.py -> PromptGUI.run_tracker()
"""

import pytest


class TestPropagation:
    """Test video propagation functionality."""

    def test_propagate_after_text_prompt(self, sam3_model, session_id, sample_text_prompt):
        """Test propagation after adding a text prompt."""
        # First add a text prompt
        sam3_model.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=sample_text_prompt,
            )
        )
        
        # Then propagate
        frame_count = 0
        for response in sam3_model.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
            )
        ):
            frame_count += 1
            assert "frame_index" in response
            assert "outputs" in response
        
        assert frame_count > 0, "Should propagate through at least one frame"

    def test_propagate_both_directions(self, sam3_model, session_id, sample_text_prompt):
        """Test propagation in both directions."""
        # First add a text prompt
        sam3_model.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=sample_text_prompt,
            )
        )
        
        # Propagate both directions
        frame_indices = set()
        for response in sam3_model.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                propagation_direction="both",
            )
        ):
            frame_indices.add(response["frame_index"])
        
        assert len(frame_indices) > 0

    def test_propagate_output_format(self, sam3_model, session_id, sample_text_prompt):
        """Test that propagation returns correct output format."""
        # First add a text prompt
        sam3_model.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=sample_text_prompt,
            )
        )
        
        # Check output format
        for response in sam3_model.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
            )
        ):
            outputs = response.get("outputs", {})
            if outputs:
                # Should use correct SAM3 keys
                assert "out_obj_ids" in outputs
                # If there are objects, should have masks
                if len(outputs.get("out_obj_ids", [])) > 0:
                    assert "out_binary_masks" in outputs
            break  # Only check first frame
