#!/usr/bin/env python3
"""
Integration tests for MCP tools.
Tests actual tool functionality with real audio files using FastMCP Client.
"""

import pytest
import asyncio
from pathlib import Path
import sys
import json
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from fastmcp import Client
from src.whisper_mcp_server import mcp


class TestMCPTools:
    """Integration tests for MCP tool functions."""
    
    @pytest.fixture
    def test_audio_path(self):
        """Get path to test audio file."""
        return str(Path(__file__).parent.parent / "examples" / "test_short.wav")
    
    @pytest.fixture
    def test_audio_dir(self):
        """Get path to test audio directory."""
        return str(Path(__file__).parent.parent / "examples")
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Test transcribe_file tool
    @pytest.mark.asyncio
    async def test_transcribe_file_basic(self, test_audio_path):
        """Test basic single file transcription."""
        async with Client(mcp) as client:
            response = await client.call_tool("transcribe_file", {"file_path": test_audio_path})
            result = json.loads(response[0].text)
        
        # Check result structure
        assert "text" in result
        assert "segments" in result
        assert "output_files" in result
        assert "duration" in result
        assert "processing_time" in result
        assert "model_used" in result
        
        # Check content
        assert len(result["text"]) > 0
        assert isinstance(result["segments"], list)
        assert result["duration"] > 0
        assert result["processing_time"] > 0
    
    @pytest.mark.asyncio
    async def test_transcribe_file_with_output_dir(self, test_audio_path, temp_output_dir):
        """Test transcription with custom output directory."""
        async with Client(mcp) as client:
            response = await client.call_tool("transcribe_file", {
                "file_path": test_audio_path,
                "output_dir": temp_output_dir
            })
            result = json.loads(response[0].text)
        
        # Check output files were created in temp dir
        assert len(result["output_files"]) > 0
        for file_path in result["output_files"]:
            assert temp_output_dir in file_path
            assert Path(file_path).exists()
    
    @pytest.mark.asyncio
    async def test_transcribe_file_custom_formats(self, test_audio_path, temp_output_dir):
        """Test transcription with custom output formats."""
        async with Client(mcp) as client:
            response = await client.call_tool("transcribe_file", {
                "file_path": test_audio_path,
                "output_formats": "txt,json",
                "output_dir": temp_output_dir
            })
            result = json.loads(response[0].text)
        
        # Check only requested formats were created
        output_files = result["output_files"]
        assert any(f.endswith(".txt") for f in output_files)
        assert any(f.endswith(".json") for f in output_files)
        assert not any(f.endswith(".srt") for f in output_files)
        assert not any(f.endswith(".md") for f in output_files)
    
    @pytest.mark.asyncio
    async def test_transcribe_file_different_model(self, test_audio_path):
        """Test transcription with different model."""
        async with Client(mcp) as client:
            response = await client.call_tool("transcribe_file", {
                "file_path": test_audio_path,
                "model": "mlx-community/whisper-tiny-mlx"
            })
            result = json.loads(response[0].text)
        
        assert result["model_used"] == "mlx-community/whisper-tiny-mlx"
    
    # Test batch_transcribe tool
    @pytest.mark.asyncio
    async def test_batch_transcribe_basic(self, test_audio_dir):
        """Test basic batch transcription."""
        async with Client(mcp) as client:
            response = await client.call_tool("batch_transcribe", {
                "directory": test_audio_dir,
                "pattern": "test_*.wav"
            })
            result = json.loads(response[0].text)
        
        # Check result structure
        assert "total_files" in result
        assert "processed" in result
        assert "skipped" in result
        assert "failed" in result
        assert "results" in result
        assert "performance_report" in result
        
        # Should find and process test files
        assert result["total_files"] >= 1  # At least one test file
        assert result["processed"] >= 0
    
    @pytest.mark.asyncio
    async def test_batch_transcribe_skip_existing(self, test_audio_dir, temp_output_dir):
        """Test batch transcription skips existing files."""
        async with Client(mcp) as client:
            # First run
            response1 = await client.call_tool("batch_transcribe", {
                "directory": test_audio_dir,
                "pattern": "test_*.wav",
                "output_dir": temp_output_dir
            })
            result1 = json.loads(response1[0].text)
            
            processed_first = result1["processed"]
            if processed_first > 0:
                # Second run should skip
                response2 = await client.call_tool("batch_transcribe", {
                    "directory": test_audio_dir,
                    "pattern": "test_*.wav",
                    "output_dir": temp_output_dir,
                    "skip_existing": True
                })
                result2 = json.loads(response2[0].text)
                
                assert result2["processed"] == 0
                assert result2["skipped"] >= processed_first
    
    @pytest.mark.asyncio
    async def test_batch_transcribe_max_workers(self, test_audio_dir):
        """Test batch transcription with limited workers."""
        async with Client(mcp) as client:
            response = await client.call_tool("batch_transcribe", {
                "directory": test_audio_dir,
                "pattern": "test_*.wav",
                "max_workers": 1
            })
            result = json.loads(response[0].text)
        
        # Should still process files, just sequentially
        assert result["processed"] >= 0
    
    # Test list_models tool
    @pytest.mark.asyncio
    async def test_list_models(self):
        """Test listing available models."""
        async with Client(mcp) as client:
            response = await client.call_tool("list_models", {})
            result = json.loads(response[0].text)
        
        # Check result structure
        assert "models" in result
        assert "current_model" in result
        assert "cache_dir" in result
        
        # Check models list
        models = result["models"]
        assert len(models) > 0
        assert all("id" in m for m in models)
        assert all("size" in m for m in models)
        assert all("speed" in m for m in models)
        assert all("accuracy" in m for m in models)
    
    # Test get_model_info tool
    @pytest.mark.asyncio
    async def test_get_model_info_valid(self):
        """Test getting info for valid model."""
        async with Client(mcp) as client:
            response = await client.call_tool("get_model_info", {"model_id": "mlx-community/whisper-tiny-mlx"})
            result = json.loads(response[0].text)
        
        # Check result structure
        assert "id" in result
        assert "size" in result
        assert "speed" in result
        assert "accuracy" in result
        assert "parameters" in result
        assert "memory_required" in result
        assert "recommended_for" in result
        assert "is_current" in result
        assert "is_cached" in result
    
    @pytest.mark.asyncio
    async def test_get_model_info_invalid(self):
        """Test getting info for invalid model."""
        async with Client(mcp) as client:
            response = await client.call_tool("get_model_info", {"model_id": "invalid-model-name"})
            result = json.loads(response[0].text)
        
        # Should return error
        assert "error" in result
        assert "available_models" in result
    
    # Test estimate_processing_time tool
    @pytest.mark.asyncio
    async def test_estimate_processing_time(self, test_audio_path):
        """Test processing time estimation."""
        async with Client(mcp) as client:
            response = await client.call_tool("estimate_processing_time", {"file_path": test_audio_path})
            result = json.loads(response[0].text)
        
        # Check result structure
        assert "file" in result
        assert "duration" in result
        assert "duration_formatted" in result
        assert "model" in result
        assert "model_speed" in result
        assert "estimated_time" in result
        assert "estimated_time_formatted" in result
        assert "includes_overhead" in result
        assert "overhead_seconds" in result
        
        # Check values
        assert result["duration"] > 0
        assert result["estimated_time"] > 0
        assert result["overhead_seconds"] >= 2.0
    
    @pytest.mark.asyncio
    async def test_estimate_processing_time_invalid_file(self):
        """Test estimation with invalid file."""
        async with Client(mcp) as client:
            response = await client.call_tool("estimate_processing_time", {"file_path": "nonexistent.mp3"})
            result = json.loads(response[0].text)
        
        # Should return error
        assert "error" in result
    
    # Test validate_media_file tool
    @pytest.mark.asyncio
    async def test_validate_media_file_valid(self, test_audio_path):
        """Test validation of valid media file."""
        async with Client(mcp) as client:
            response = await client.call_tool("validate_media_file", {"file_path": test_audio_path})
            result = json.loads(response[0].text)
        
        # Check result structure
        assert "is_valid" in result
        assert "file" in result
        assert "format" in result
        assert "file_size" in result
        assert "issues" in result
        
        # Should be valid
        assert result["is_valid"] is True
        assert len(result["issues"]) == 0
    
    @pytest.mark.asyncio
    async def test_validate_media_file_invalid(self):
        """Test validation of invalid file."""
        async with Client(mcp) as client:
            response = await client.call_tool("validate_media_file", {"file_path": "nonexistent.mp3"})
            result = json.loads(response[0].text)
        
        # Should be invalid
        assert result["is_valid"] is False
        assert "error" in result or "issues" in result
        if "issues" in result:
            assert len(result["issues"]) > 0
    
    # Test get_supported_formats tool
    @pytest.mark.asyncio
    async def test_get_supported_formats(self):
        """Test getting supported formats."""
        async with Client(mcp) as client:
            response = await client.call_tool("get_supported_formats", {})
            result = json.loads(response[0].text)
        
        # Check result structure
        assert "input_formats" in result
        assert "output_formats" in result
        assert "notes" in result
        
        # Check input formats
        assert "audio" in result["input_formats"]
        assert "video" in result["input_formats"]
        assert ".mp3" in result["input_formats"]["audio"]
        assert ".mp4" in result["input_formats"]["video"]
        
        # Check output formats
        assert "txt" in result["output_formats"]
        assert "md" in result["output_formats"]
        assert "srt" in result["output_formats"]
        assert "json" in result["output_formats"]
    
    # Test clear_cache tool
    @pytest.mark.asyncio
    async def test_clear_cache_specific_model(self):
        """Test clearing cache for specific model."""
        async with Client(mcp) as client:
            response = await client.call_tool("clear_cache", {"model_id": "mlx-community/whisper-tiny-mlx"})
            result = json.loads(response[0].text)
        
        # Check result structure
        if "error" in result:
            assert "cache_dir" in result
        else:
            assert "message" in result
            assert "cleared_models" in result
            assert "freed_space" in result
    
    @pytest.mark.asyncio
    async def test_clear_cache_all_models(self):
        """Test clearing all model caches."""
        async with Client(mcp) as client:
            response = await client.call_tool("clear_cache", {})
            result = json.loads(response[0].text)
        
        # Check result structure
        assert "message" in result
        assert "cleared_models" in result
        assert "freed_space" in result
        assert isinstance(result["cleared_models"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])