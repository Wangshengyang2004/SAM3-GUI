"""
Tests for SAM3 session management.

Corresponds to: mask_app.py -> PromptGUI.get_sam_features(), reset()
"""

import pytest


class TestSessionManagement:
    """Test SAM3 session lifecycle."""

    def test_start_session(self, sam3_model, test_img_dir):
        """Test starting a new session."""
        response = sam3_model.handle_request(
            request=dict(
                type="start_session",
                resource_path=test_img_dir,
            )
        )
        
        assert "session_id" in response
        session_id = response["session_id"]
        assert session_id is not None
        assert isinstance(session_id, str)
        
        # Cleanup
        sam3_model.handle_request(
            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )

    def test_reset_session(self, sam3_model, session_id):
        """Test resetting a session."""
        response = sam3_model.handle_request(
            request=dict(
                type="reset_session",
                session_id=session_id,
            )
        )
        
        assert response.get("is_success") is True

    def test_close_session(self, sam3_model, test_img_dir):
        """Test closing a session."""
        # Create a new session to close
        start_response = sam3_model.handle_request(
            request=dict(
                type="start_session",
                resource_path=test_img_dir,
            )
        )
        session_id = start_response["session_id"]
        
        # Close the session
        close_response = sam3_model.handle_request(
            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )
        
        assert close_response.get("is_success") is True

    def test_close_nonexistent_session(self, sam3_model):
        """Test closing a session that doesn't exist."""
        # This should not raise an error
        response = sam3_model.handle_request(
            request=dict(
                type="close_session",
                session_id="nonexistent-session-id",
            )
        )
        # Should still return success (idempotent)
        assert response.get("is_success") is True
