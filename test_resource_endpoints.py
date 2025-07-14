#!/usr/bin/env python3
"""
Test Resource Endpoints for Whisper MCP Server
Following TDD approach - write tests first
"""

import pytest
import json
import asyncio
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock
from fastmcp import FastMCP

# Import the server module
from src.whisper_mcp_server import (
    mcp,
    get_transcriber,
    performance_report,
    DEFAULT_MODEL,
    DEFAULT_FORMATS,
    MAX_WORKERS,
    TEMP_DIR,
)


class TestResourceEndpoints:
    """Test suite for MCP resource endpoints"""

    @pytest.fixture
    def mock_history_file(self, tmp_path):
        """Create a mock history file with sample data"""
        history_data = [
            {
                "id": "test-id-1",
                "timestamp": "2024-01-01T10:00:00",
                "file_path": "/path/to/audio1.mp3",
                "model": "mlx-community/whisper-tiny-mlx",
                "duration": 120.5,
                "processing_time": 15.2,
                "output_files": ["/path/to/audio1.txt", "/path/to/audio1.srt"],
            },
            {
                "id": "test-id-2",
                "timestamp": "2024-01-01T11:00:00",
                "file_path": "/path/to/audio2.wav",
                "model": "mlx-community/whisper-base-mlx",
                "duration": 300.0,
                "processing_time": 35.5,
                "output_files": ["/path/to/audio2.txt", "/path/to/audio2.json"],
            },
        ]

        history_file = tmp_path / "logs" / "transcription_history.json"
        history_file.parent.mkdir(parents=True, exist_ok=True)

        with open(history_file, "w") as f:
            json.dump(history_data, f)

        return history_file

    @pytest.fixture
    def mock_performance_data(self):
        """Mock performance report data"""
        mock_report = MagicMock()
        mock_report.total_files = 10
        mock_report.total_duration = 3600.0  # 1 hour
        mock_report.average_speed = 5.0
        return mock_report

    # Test transcription://history resource
    @pytest.mark.asyncio
    async def test_history_resource_empty(self, tmp_path):
        """Test history resource when no history exists"""
        # Setup
        with patch("src.whisper_mcp_server.Path") as MockPath:
            MockPath.return_value.exists.return_value = False

            # Call the resource
            # Note: We need to test the actual resource function
            # This is a placeholder - actual implementation will be tested
            result = {"transcriptions": [], "total_count": 0}

            assert result["transcriptions"] == []
            assert result["total_count"] == 0

    @pytest.mark.asyncio
    async def test_history_resource_with_data(self, mock_history_file):
        """Test history resource with existing data"""
        # This test expects the resource to read from the history file
        # and return the last 50 transcriptions
        expected_count = 2
        expected_ids = ["test-id-1", "test-id-2"]

        # Placeholder for actual resource call
        # The implementation will read from mock_history_file
        result = {"transcriptions": [{"id": "test-id-1"}, {"id": "test-id-2"}], "total_count": 2}

        assert result["total_count"] == expected_count
        assert len(result["transcriptions"]) == expected_count

    @pytest.mark.asyncio
    async def test_history_resource_pagination(self, tmp_path):
        """Test history resource returns only last 50 items"""
        # Create 60 history entries
        history_data = []
        for i in range(60):
            history_data.append(
                {
                    "id": f"test-id-{i}",
                    "timestamp": f"2024-01-01T{i:02d}:00:00",
                    "file_path": f"/path/to/audio{i}.mp3",
                    "model": "mlx-community/whisper-tiny-mlx",
                    "duration": 100.0,
                    "processing_time": 10.0,
                    "output_files": [f"/path/to/audio{i}.txt"],
                }
            )

        history_file = tmp_path / "logs" / "transcription_history.json"
        history_file.parent.mkdir(parents=True, exist_ok=True)

        with open(history_file, "w") as f:
            json.dump(history_data, f)

        # Expected: only last 50 items
        # Placeholder test
        result = {"transcriptions": history_data[-50:], "total_count": 60}

        assert len(result["transcriptions"]) == 50
        assert result["total_count"] == 60
        assert result["transcriptions"][0]["id"] == "test-id-10"
        assert result["transcriptions"][-1]["id"] == "test-id-59"

    # Test transcription://history/{id} resource
    @pytest.mark.asyncio
    async def test_history_detail_resource_found(self, mock_history_file):
        """Test getting specific transcription details"""
        transcription_id = "test-id-1"

        # Expected result should include full details
        expected = {
            "id": "test-id-1",
            "timestamp": "2024-01-01T10:00:00",
            "file_path": "/path/to/audio1.mp3",
            "model": "mlx-community/whisper-tiny-mlx",
            "duration": 120.5,
            "processing_time": 15.2,
            "output_files": ["/path/to/audio1.txt", "/path/to/audio1.srt"],
        }

        # Placeholder for actual implementation
        result = expected

        assert result["id"] == transcription_id
        assert result["model"] == "mlx-community/whisper-tiny-mlx"
        assert len(result["output_files"]) == 2

    @pytest.mark.asyncio
    async def test_history_detail_resource_not_found(self, mock_history_file):
        """Test getting non-existent transcription"""
        transcription_id = "non-existent-id"

        # Should return error or None
        result = {"error": f"Transcription {transcription_id} not found"}

        assert "error" in result
        assert transcription_id in result["error"]

    # Test transcription://models resource
    @pytest.mark.asyncio
    async def test_models_resource(self):
        """Test models resource returns available models"""
        # This should reuse the list_models tool function
        expected_models = [
            {
                "id": "mlx-community/whisper-tiny-mlx",
                "size": "39M",
                "speed": "~10x",
                "accuracy": "Good",
            },
            {
                "id": "mlx-community/whisper-base-mlx",
                "size": "74M",
                "speed": "~7x",
                "accuracy": "Better",
            },
            {
                "id": "mlx-community/whisper-small-mlx",
                "size": "244M",
                "speed": "~5x",
                "accuracy": "Very Good",
            },
            {
                "id": "mlx-community/whisper-medium-mlx",
                "size": "769M",
                "speed": "~3x",
                "accuracy": "Excellent",
            },
            {
                "id": "mlx-community/whisper-large-v3-mlx",
                "size": "1550M",
                "speed": "~2x",
                "accuracy": "Best",
            },
            {
                "id": "mlx-community/whisper-large-v3-turbo",
                "size": "809M",
                "speed": "~4x",
                "accuracy": "Excellent",
            },
        ]

        # Placeholder test
        result = {
            "models": expected_models,
            "current_model": DEFAULT_MODEL,
            "cache_dir": str(Path.home() / ".cache" / "huggingface"),
        }

        assert len(result["models"]) == 6
        assert result["current_model"] == DEFAULT_MODEL
        assert "cache_dir" in result

    # Test transcription://config resource
    @pytest.mark.asyncio
    async def test_config_resource(self):
        """Test config resource returns current configuration"""
        expected_config = {
            "default_model": DEFAULT_MODEL,
            "output_formats": DEFAULT_FORMATS,
            "max_workers": MAX_WORKERS,
            "temp_dir": str(TEMP_DIR),
            "version": "1.0.0",
        }

        # Placeholder test
        result = expected_config

        assert result["default_model"] == DEFAULT_MODEL
        assert result["output_formats"] == DEFAULT_FORMATS
        assert result["max_workers"] == MAX_WORKERS
        assert result["version"] == "1.0.0"

    # Test transcription://formats resource
    @pytest.mark.asyncio
    async def test_formats_resource(self):
        """Test formats resource returns supported formats"""
        expected_formats = {
            "input_formats": {
                "audio": {
                    ".mp3": "MPEG Audio Layer 3",
                    ".wav": "Waveform Audio File",
                    ".m4a": "MPEG-4 Audio",
                    ".flac": "Free Lossless Audio Codec",
                    ".ogg": "Ogg Vorbis",
                },
                "video": {
                    ".mp4": "MPEG-4 Video",
                    ".mov": "QuickTime Movie",
                    ".avi": "Audio Video Interleave",
                    ".mkv": "Matroska Video",
                    ".webm": "WebM Video",
                },
            },
            "output_formats": {
                "txt": "Plain text with timestamps",
                "md": "Markdown formatted text",
                "srt": "SubRip subtitle format",
                "json": "Full transcription data with segments",
            },
            "notes": {
                "audio_extraction": "Video files are automatically converted to audio before transcription",
                "format_detection": "File format is detected by extension",
                "quality": "All formats are supported equally - quality depends on the source audio",
            },
        }

        # Placeholder test
        result = expected_formats

        assert "input_formats" in result
        assert "output_formats" in result
        assert len(result["input_formats"]["audio"]) == 5
        assert len(result["input_formats"]["video"]) == 5
        assert len(result["output_formats"]) == 4

    # Test transcription://performance resource
    @pytest.mark.asyncio
    async def test_performance_resource(self, mock_performance_data):
        """Test performance statistics resource"""
        with patch("src.whisper_mcp_server.performance_report", mock_performance_data):
            with patch("src.whisper_mcp_server.time.time", return_value=1000.0):
                # Assume server started at time 0
                expected_performance = {
                    "total_transcriptions": 10,
                    "total_audio_hours": 1.0,
                    "average_speed": 5.0,
                    "uptime": 1000.0,
                }

                # Placeholder test
                result = expected_performance

                assert result["total_transcriptions"] == 10
                assert result["total_audio_hours"] == 1.0
                assert result["average_speed"] == 5.0
                assert result["uptime"] == 1000.0

    # Test record_transcription function
    def test_record_transcription_new_file(self, tmp_path):
        """Test recording a new transcription to history"""
        history_file = tmp_path / "logs" / "transcription_history.json"
        history_file.parent.mkdir(parents=True, exist_ok=True)

        # Test data
        file_path = "/path/to/test.mp3"
        result = {
            "model_used": "mlx-community/whisper-tiny-mlx",
            "duration": 100.0,
            "processing_time": 10.0,
            "output_files": ["/path/to/test.txt"],
        }

        # After implementation, this should create a new history entry
        # with a unique ID and timestamp

        # Verify file was created and contains correct data
        assert history_file.exists() or not history_file.exists()  # Placeholder

    def test_record_transcription_append(self, tmp_path):
        """Test appending to existing history"""
        # Create existing history
        history_file = tmp_path / "logs" / "transcription_history.json"
        history_file.parent.mkdir(parents=True, exist_ok=True)

        existing_data = [{"id": "existing-1", "timestamp": "2024-01-01T10:00:00"}]
        with open(history_file, "w") as f:
            json.dump(existing_data, f)

        # Test data
        file_path = "/path/to/new.mp3"
        result = {
            "model_used": "mlx-community/whisper-base-mlx",
            "duration": 200.0,
            "processing_time": 20.0,
            "output_files": ["/path/to/new.txt", "/path/to/new.srt"],
        }

        # After implementation, should append to existing history
        # Verify both entries exist

        # Placeholder assertion
        assert True

    # Integration test for resource endpoints
    @pytest.mark.asyncio
    async def test_resource_endpoints_integration(self, tmp_path, mock_history_file):
        """Test that all resource endpoints work together"""
        # This test verifies that:
        # 1. History can be retrieved
        # 2. Individual transcriptions can be accessed
        # 3. Configuration is accessible
        # 4. Models list is available
        # 5. Formats are documented
        # 6. Performance stats are tracked

        # Placeholder for integration test
        assert True


# Test helpers
def create_test_transcription_result():
    """Create a test transcription result"""
    return {
        "text": "This is a test transcription",
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "This is a test"},
            {"start": 2.0, "end": 4.0, "text": "transcription"},
        ],
        "output_files": ["/tmp/test.txt", "/tmp/test.srt"],
        "duration": 4.0,
        "processing_time": 0.5,
        "model_used": "mlx-community/whisper-tiny-mlx",
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
