#!/usr/bin/env python3
"""
Error handling tests for MCP Whisper Transcription Server.
Tests various error conditions and edge cases to ensure robust error handling.
"""

import pytest
import asyncio
import tempfile
import shutil
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from fastmcp import Client
from src.whisper_mcp_server import mcp


class TestErrorHandling:
    """Tests for error handling in the MCP server."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_transcribe_nonexistent_file(self):
        """Test transcription of non-existent file."""
        async with Client(mcp) as client:
            with pytest.raises(Exception) as exc_info:
                response = await client.call_tool("transcribe_file", {
                    "file_path": "/path/to/nonexistent/file.wav"
                })
            
            # Should raise an error about file not found
            assert "not found" in str(exc_info.value).lower() or "no such file" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_transcribe_invalid_file_format(self, temp_output_dir):
        """Test transcription of invalid file format."""
        # Create a text file with audio extension
        fake_audio = Path(temp_output_dir) / "fake_audio.wav"
        fake_audio.write_text("This is not audio data")
        
        async with Client(mcp) as client:
            with pytest.raises(Exception) as exc_info:
                response = await client.call_tool("transcribe_file", {
                    "file_path": str(fake_audio)
                })
            
            # Should raise an error about invalid format or transcription failure
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in [
                "transcription failed", "invalid", "error", "format"
            ])
    
    @pytest.mark.asyncio
    async def test_transcribe_empty_file(self, temp_output_dir):
        """Test transcription of empty file."""
        # Create an empty file
        empty_file = Path(temp_output_dir) / "empty.wav"
        empty_file.touch()
        
        async with Client(mcp) as client:
            with pytest.raises(Exception) as exc_info:
                response = await client.call_tool("transcribe_file", {
                    "file_path": str(empty_file)
                })
            
            # Should raise an error about empty or invalid file
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in [
                "transcription failed", "empty", "invalid", "error"
            ])
    
    @pytest.mark.asyncio
    async def test_transcribe_invalid_output_directory(self):
        """Test transcription with invalid output directory."""
        test_audio = Path(__file__).parent.parent / "examples" / "test_short.wav"
        
        async with Client(mcp) as client:
            with pytest.raises(Exception) as exc_info:
                response = await client.call_tool("transcribe_file", {
                    "file_path": str(test_audio),
                    "output_dir": "/invalid/readonly/path"
                })
            
            # Should raise an error about directory access
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in [
                "permission", "directory", "access", "error", "failed"
            ])
    
    @pytest.mark.asyncio
    async def test_transcribe_invalid_model(self):
        """Test transcription with invalid model."""
        test_audio = Path(__file__).parent.parent / "examples" / "test_short.wav"
        
        async with Client(mcp) as client:
            with pytest.raises(Exception) as exc_info:
                response = await client.call_tool("transcribe_file", {
                    "file_path": str(test_audio),
                    "model": "invalid-model-name"
                })
            
            # Should raise an error about invalid model
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in [
                "model", "invalid", "not found", "error"
            ])
    
    @pytest.mark.asyncio
    async def test_batch_transcribe_nonexistent_directory(self):
        """Test batch transcription of non-existent directory."""
        async with Client(mcp) as client:
            with pytest.raises(Exception) as exc_info:
                response = await client.call_tool("batch_transcribe", {
                    "directory": "/path/to/nonexistent/directory"
                })
            
            # Should raise an error about directory not found
            assert "directory not found" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_batch_transcribe_empty_directory(self, temp_output_dir):
        """Test batch transcription of empty directory."""
        async with Client(mcp) as client:
            # This should not raise an error, but return empty results
            response = await client.call_tool("batch_transcribe", {
                "directory": temp_output_dir
            })
            result = json.loads(response[0].text)
            
            # Should return valid structure with zero files
            assert "total_files" in result
            assert result["total_files"] == 0
            assert result["processed"] == 0
            assert result["performance_report"] == "No media files found"
    
    @pytest.mark.asyncio
    async def test_batch_transcribe_invalid_pattern(self):
        """Test batch transcription with invalid pattern."""
        test_audio_dir = Path(__file__).parent.parent / "examples"
        
        async with Client(mcp) as client:
            # Invalid pattern should return no files
            response = await client.call_tool("batch_transcribe", {
                "directory": str(test_audio_dir),
                "pattern": "*.invalid_extension"
            })
            result = json.loads(response[0].text)
            
            # Should return valid structure with zero files
            assert result["total_files"] == 0
            assert result["processed"] == 0
    
    @pytest.mark.asyncio
    async def test_batch_transcribe_invalid_max_workers(self):
        """Test batch transcription with invalid max_workers."""
        test_audio_dir = Path(__file__).parent.parent / "examples"
        
        async with Client(mcp) as client:
            # Zero workers should be handled gracefully
            response = await client.call_tool("batch_transcribe", {
                "directory": str(test_audio_dir),
                "pattern": "test_*.wav",
                "max_workers": 0
            })
            result = json.loads(response[0].text)
            
            # Should still process files (fallback to default workers)
            assert result["total_files"] > 0
    
    @pytest.mark.asyncio
    async def test_get_model_info_invalid_model(self):
        """Test getting model info for invalid model."""
        async with Client(mcp) as client:
            response = await client.call_tool("get_model_info", {
                "model_id": "totally-invalid-model-name"
            })
            result = json.loads(response[0].text)
            
            # Should return error structure
            assert "error" in result
            assert "available_models" in result
            assert "not found" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_estimate_processing_time_invalid_file(self):
        """Test processing time estimation for invalid file."""
        async with Client(mcp) as client:
            response = await client.call_tool("estimate_processing_time", {
                "file_path": "/path/to/nonexistent/file.wav"
            })
            result = json.loads(response[0].text)
            
            # Should return error structure
            assert "error" in result
            assert "not found" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_validate_media_file_nonexistent(self):
        """Test media file validation for non-existent file."""
        async with Client(mcp) as client:
            response = await client.call_tool("validate_media_file", {
                "file_path": "/path/to/nonexistent/file.mp3"
            })
            result = json.loads(response[0].text)
            
            # Should return invalid with error
            assert result["is_valid"] is False
            assert "error" in result or "issues" in result
            if "error" in result:
                assert "not found" in result["error"].lower()
            if "issues" in result:
                assert len(result["issues"]) > 0
    
    @pytest.mark.asyncio
    async def test_validate_media_file_invalid_format(self, temp_output_dir):
        """Test media file validation for invalid format."""
        # Create a text file with media extension
        fake_media = Path(temp_output_dir) / "fake.mp3"
        fake_media.write_text("This is not media data")
        
        async with Client(mcp) as client:
            response = await client.call_tool("validate_media_file", {
                "file_path": str(fake_media)
            })
            result = json.loads(response[0].text)
            
            # Should return invalid with issues
            assert result["is_valid"] is False
            assert "issues" in result
            assert len(result["issues"]) > 0
    
    @pytest.mark.asyncio
    async def test_clear_cache_invalid_model(self):
        """Test cache clearing for invalid model."""
        async with Client(mcp) as client:
            response = await client.call_tool("clear_cache", {
                "model_id": "totally-invalid-model-name"
            })
            result = json.loads(response[0].text)
            
            # Should return error about model not found in cache
            assert "error" in result
            assert "not found" in result["error"].lower()
            assert "cache_dir" in result
    
    @pytest.mark.asyncio
    async def test_resource_invalid_transcription_id(self):
        """Test reading resource with invalid transcription ID."""
        async with Client(mcp) as client:
            resource = await client.read_resource("transcription://history/invalid-uuid-12345")
            data = json.loads(resource[0].text)
            
            # Should return error
            assert "error" in data
            assert "not found" in data["error"].lower()
    
    @pytest.mark.asyncio
    async def test_resource_invalid_uri(self):
        """Test reading invalid resource URI."""
        async with Client(mcp) as client:
            with pytest.raises(Exception) as exc_info:
                resource = await client.read_resource("transcription://invalid-resource")
            
            # Should raise an error about invalid resource
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in [
                "not found", "invalid", "unknown", "error"
            ])
    
    @pytest.mark.asyncio
    async def test_tool_missing_required_parameters(self):
        """Test tools with missing required parameters."""
        async with Client(mcp) as client:
            
            # Test transcribe_file without file_path
            with pytest.raises(Exception) as exc_info:
                response = await client.call_tool("transcribe_file", {})
            
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in [
                "required", "missing", "file_path", "parameter"
            ])
            
            # Test get_model_info without model_id  
            with pytest.raises(Exception) as exc_info:
                response = await client.call_tool("get_model_info", {})
            
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in [
                "required", "missing", "model_id", "parameter"
            ])
    
    @pytest.mark.asyncio
    async def test_tool_invalid_parameter_types(self):
        """Test tools with invalid parameter types."""
        async with Client(mcp) as client:
            test_audio = Path(__file__).parent.parent / "examples" / "test_short.wav"
            
            # Test transcribe_file with invalid temperature type
            with pytest.raises(Exception) as exc_info:
                response = await client.call_tool("transcribe_file", {
                    "file_path": str(test_audio),
                    "temperature": "invalid_string"  # Should be float
                })
            
            # Should raise validation error
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in [
                "validation", "type", "invalid", "temperature"
            ])
    
    @pytest.mark.asyncio
    async def test_concurrent_error_handling(self, temp_output_dir):
        """Test error handling under concurrent load."""
        async with Client(mcp) as client:
            
            async def failing_transcription(index):
                """Attempt transcription that will fail."""
                try:
                    response = await client.call_tool("transcribe_file", {
                        "file_path": f"/nonexistent/file_{index}.wav"
                    })
                    return {"success": True, "index": index}
                except Exception as e:
                    return {"success": False, "index": index, "error": str(e)}
            
            # Run multiple failing tasks concurrently
            tasks = [failing_transcription(i) for i in range(5)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All should fail
            successful = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
            assert successful == 0, "Expected all transcriptions to fail"
            
            # Should get consistent error messages
            error_messages = [
                str(r.get("error", r)) for r in results 
                if isinstance(r, dict) or isinstance(r, Exception)
            ]
            
            # All errors should mention file not found
            for error_msg in error_messages:
                assert any(keyword in error_msg.lower() for keyword in [
                    "not found", "no such file", "file"
                ])
    
    @pytest.mark.asyncio
    async def test_server_resilience_after_errors(self):
        """Test that server remains functional after encountering errors."""
        async with Client(mcp) as client:
            # Cause some errors
            for i in range(3):
                try:
                    await client.call_tool("transcribe_file", {
                        "file_path": f"/nonexistent/error_test_{i}.wav"
                    })
                except Exception:
                    pass  # Expected to fail
            
            # Server should still work for valid requests
            response = await client.call_tool("list_models", {})
            result = json.loads(response[0].text)
            
            # Should return valid response
            assert "models" in result
            assert len(result["models"]) > 0
            
            # Resource access should also work
            resource = await client.read_resource("transcription://config")
            config_data = json.loads(resource[0].text)
            
            assert "default_model" in config_data
            assert "version" in config_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])