"""
Tests for Whisper Transcription MCP Server
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_import_transcriber():
    """Test that we can import the WhisperTranscriber class"""
    from transcribe_mlx import WhisperTranscriber

    assert WhisperTranscriber is not None


def test_import_utils():
    """Test that we can import utility functions"""
    from whisper_utils import (
        TranscriptionStats,
        PerformanceReport,
        OutputFormatter,
        setup_logger,
    )

    assert TranscriptionStats is not None
    assert PerformanceReport is not None
    assert OutputFormatter is not None
    assert setup_logger is not None


def test_mcp_server_import():
    """Test that we can import the MCP server"""
    from src.whisper_mcp_server import mcp

    assert mcp is not None
    assert mcp.name == "Whisper Transcription MCP"


# TODO: Add more comprehensive tests in future tasks
# - Test transcription functionality
# - Test MCP tool implementations
# - Test resource endpoints
# - Test error handling
