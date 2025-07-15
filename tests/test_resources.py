#!/usr/bin/env python3
"""
Tests for MCP resource endpoints.
Tests the resource endpoints that provide access to transcription data.
"""

import pytest
import asyncio
import json
import tempfile
import shutil
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from fastmcp import Client
from src.whisper_mcp_server import mcp


class TestMCPResources:
    """Tests for MCP resource endpoints."""
    
    @pytest.fixture
    def test_audio_path(self):
        """Get path to test audio file."""
        return str(Path(__file__).parent.parent / "examples" / "test_short.wav")
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_transcription_history_resource(self):
        """Test transcription history resource endpoint."""
        async with Client(mcp) as client:
            # First list resources to ensure the endpoint exists
            resources = await client.list_resources()
            resource_uris = [str(r.uri) for r in resources]
            
            assert "transcription://history" in resource_uris
            
            # Read the history resource
            resource = await client.read_resource("transcription://history")
            history_data = json.loads(resource[0].text)
            
            # Check structure
            assert "transcriptions" in history_data
            assert "total_count" in history_data
            assert isinstance(history_data["transcriptions"], list)
            assert isinstance(history_data["total_count"], int)
    
    @pytest.mark.asyncio
    async def test_models_resource(self):
        """Test models resource endpoint."""
        async with Client(mcp) as client:
            # Read the models resource
            resource = await client.read_resource("transcription://models")
            models_data = json.loads(resource[0].text)
            
            # Check structure matches list_models tool
            assert "models" in models_data
            assert "current_model" in models_data
            assert "cache_dir" in models_data
            
            # Check models list
            models = models_data["models"]
            assert len(models) > 0
            assert all("id" in m for m in models)
            assert all("size" in m for m in models)
            assert all("speed" in m for m in models)
            assert all("accuracy" in m for m in models)
    
    @pytest.mark.asyncio
    async def test_config_resource(self):
        """Test configuration resource endpoint."""
        async with Client(mcp) as client:
            # Read the config resource
            resource = await client.read_resource("transcription://config")
            config_data = json.loads(resource[0].text)
            
            # Check structure
            assert "default_model" in config_data
            assert "output_formats" in config_data
            assert "max_workers" in config_data
            assert "temp_dir" in config_data
            assert "version" in config_data
            
            # Check values
            assert isinstance(config_data["max_workers"], int)
            assert config_data["max_workers"] > 0
            assert "mlx-community" in config_data["default_model"]
    
    @pytest.mark.asyncio
    async def test_formats_resource(self):
        """Test formats resource endpoint."""
        async with Client(mcp) as client:
            # Read the formats resource
            resource = await client.read_resource("transcription://formats")
            formats_data = json.loads(resource[0].text)
            
            # Check structure matches get_supported_formats tool
            assert "input_formats" in formats_data
            assert "output_formats" in formats_data
            assert "notes" in formats_data
            
            # Check input formats
            assert "audio" in formats_data["input_formats"]
            assert "video" in formats_data["input_formats"]
            assert ".mp3" in formats_data["input_formats"]["audio"]
            assert ".mp4" in formats_data["input_formats"]["video"]
            
            # Check output formats
            assert "txt" in formats_data["output_formats"]
            assert "md" in formats_data["output_formats"]
            assert "srt" in formats_data["output_formats"]
            assert "json" in formats_data["output_formats"]
    
    @pytest.mark.asyncio
    async def test_performance_resource(self):
        """Test performance statistics resource endpoint."""
        async with Client(mcp) as client:
            # Read the performance resource
            resource = await client.read_resource("transcription://performance")
            perf_data = json.loads(resource[0].text)
            
            # Check structure
            assert "total_transcriptions" in perf_data
            assert "total_audio_hours" in perf_data
            assert "average_speed" in perf_data
            assert "uptime" in perf_data
            
            # Check types
            assert isinstance(perf_data["total_transcriptions"], (int, float))
            assert isinstance(perf_data["total_audio_hours"], (int, float))
            assert isinstance(perf_data["average_speed"], (int, float))
            assert isinstance(perf_data["uptime"], (int, float))
            
            # Uptime should be positive
            assert perf_data["uptime"] > 0
    
    @pytest.mark.asyncio
    async def test_transcription_details_resource(self, test_audio_path, temp_output_dir):
        """Test transcription details resource endpoint after creating a transcription."""
        async with Client(mcp) as client:
            # First create a transcription to ensure we have history
            response = await client.call_tool("transcribe_file", {
                "file_path": test_audio_path,
                "output_dir": temp_output_dir
            })
            transcription_result = json.loads(response[0].text)
            
            # Read the history to get a transcription ID
            resource = await client.read_resource("transcription://history")
            history_data = json.loads(resource[0].text)
            
            if history_data["total_count"] > 0:
                # Get the most recent transcription ID
                latest_transcription = history_data["transcriptions"][-1]
                transcription_id = latest_transcription["id"]
                
                # Read the specific transcription details
                details_resource = await client.read_resource(f"transcription://history/{transcription_id}")
                details_data = json.loads(details_resource[0].text)
                
                # Check structure
                assert "id" in details_data
                assert "timestamp" in details_data
                assert "file_path" in details_data
                assert "model" in details_data
                assert "duration" in details_data
                assert "processing_time" in details_data
                assert "output_files" in details_data
                
                # Verify it's the correct transcription
                assert details_data["id"] == transcription_id
                assert details_data["file_path"] == test_audio_path
    
    @pytest.mark.asyncio
    async def test_invalid_transcription_details_resource(self):
        """Test transcription details resource with invalid ID."""
        async with Client(mcp) as client:
            # Try to read details for non-existent transcription
            resource = await client.read_resource("transcription://history/invalid-id-12345")
            details_data = json.loads(resource[0].text)
            
            # Should return error
            assert "error" in details_data
            assert "invalid-id-12345" in details_data["error"]
    
    @pytest.mark.asyncio
    async def test_resource_list_contains_all_endpoints(self):
        """Test that all expected resource endpoints are registered."""
        async with Client(mcp) as client:
            resources = await client.list_resources()
            resource_uris = [str(r.uri) for r in resources]
            
            # Check all expected resources are present
            expected_resources = [
                "transcription://history",
                "transcription://models",
                "transcription://config", 
                "transcription://formats",
                "transcription://performance"
            ]
            
            for expected in expected_resources:
                assert expected in resource_uris, f"Resource {expected} not found in {resource_uris}"
    
    @pytest.mark.asyncio
    async def test_resource_consistency_with_tools(self):
        """Test that resource endpoints return consistent data with corresponding tools."""
        async with Client(mcp) as client:
            # Test models resource vs list_models tool
            models_resource = await client.read_resource("transcription://models")
            models_from_resource = json.loads(models_resource[0].text)
            
            models_tool_response = await client.call_tool("list_models", {})
            models_from_tool = json.loads(models_tool_response[0].text)
            
            # Should be identical
            assert models_from_resource == models_from_tool
            
            # Test formats resource vs get_supported_formats tool
            formats_resource = await client.read_resource("transcription://formats")
            formats_from_resource = json.loads(formats_resource[0].text)
            
            formats_tool_response = await client.call_tool("get_supported_formats", {})
            formats_from_tool = json.loads(formats_tool_response[0].text)
            
            # Should be identical
            assert formats_from_resource == formats_from_tool


if __name__ == "__main__":
    pytest.main([__file__, "-v"])