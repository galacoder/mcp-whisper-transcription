#!/usr/bin/env python3
"""
Basic test to verify resource endpoints don't exist yet
Following TDD - these should fail until we implement them
"""

import pytest
import sys
import importlib


def test_resources_implemented():
    """Test that resource functions are now implemented"""
    # Import the server module
    import src.whisper_mcp_server as server

    # These functions should exist now
    assert hasattr(server, "get_transcription_history")
    assert hasattr(server, "get_transcription_details")
    assert hasattr(server, "get_models_resource")
    assert hasattr(server, "get_config_resource")
    assert hasattr(server, "get_formats_resource")
    assert hasattr(server, "get_performance_stats")
    assert hasattr(server, "record_transcription")


def test_server_start_time_tracked():
    """Test that server start time is being tracked"""
    import src.whisper_mcp_server as server

    # This should exist now
    assert hasattr(server, "server_start_time")
    assert isinstance(server.server_start_time, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
