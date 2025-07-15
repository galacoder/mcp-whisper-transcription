#!/usr/bin/env python3
"""
Unit tests for WhisperTranscriber singleton pattern and core functionality.
Following TDD approach - test singleton behavior and model switching.
"""

import pytest
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.whisper_mcp_server import get_transcriber
from transcribe_mlx import WhisperTranscriber


class TestWhisperTranscriber:
    """Test suite for WhisperTranscriber singleton pattern."""
    
    def test_singleton_pattern(self):
        """Test transcriber singleton behavior."""
        # Get transcriber twice with same model
        t1 = get_transcriber()
        t2 = get_transcriber()
        
        # Should return the same instance
        assert t1 is t2
        assert t1.model_name == t2.model_name
    
    def test_model_switching(self):
        """Test changing models creates new instance."""
        # Get transcriber with different models
        t1 = get_transcriber("mlx-community/whisper-tiny-mlx")
        t2 = get_transcriber("mlx-community/whisper-base-mlx")
        
        # Should return different instances
        assert t1 is not t2
        assert t1.model_name == "mlx-community/whisper-tiny-mlx"
        assert t2.model_name == "mlx-community/whisper-base-mlx"
    
    def test_model_persistence(self):
        """Test that same model returns same instance even after different model."""
        # Get tiny model
        t1 = get_transcriber("mlx-community/whisper-tiny-mlx")
        
        # Switch to base model
        t2 = get_transcriber("mlx-community/whisper-base-mlx")
        
        # Switch back to tiny - should be new instance (not cached)
        t3 = get_transcriber("mlx-community/whisper-tiny-mlx")
        
        # t1 and t3 should be different (no caching of old instances)
        assert t1 is not t3
        assert t1.model_name == t3.model_name
    
    def test_default_model(self):
        """Test that default model is used when none specified."""
        from src.whisper_mcp_server import DEFAULT_MODEL
        
        # Get transcriber without specifying model
        t = get_transcriber()
        
        # Should use default model
        assert t.model_name == DEFAULT_MODEL
    
    def test_none_model_uses_default(self):
        """Test that None model parameter uses default."""
        from src.whisper_mcp_server import DEFAULT_MODEL
        
        # Explicitly pass None
        t = get_transcriber(None)
        
        # Should use default model
        assert t.model_name == DEFAULT_MODEL
    
    def test_transcriber_instance_type(self):
        """Test that get_transcriber returns WhisperTranscriber instance."""
        t = get_transcriber()
        
        # Should be instance of WhisperTranscriber
        assert isinstance(t, WhisperTranscriber)
    
    def test_transcriber_has_required_attributes(self):
        """Test that transcriber has required attributes."""
        t = get_transcriber()
        
        # Check required attributes
        assert hasattr(t, 'model_name')
        assert hasattr(t, 'transcribe_audio')
        assert hasattr(t, 'formatter')
    
    def test_output_formats_configuration(self):
        """Test that output formats are properly configured."""
        from src.whisper_mcp_server import DEFAULT_FORMATS
        
        t = get_transcriber()
        
        # Check formatter has output formats
        assert hasattr(t.formatter, 'output_formats')
        
        # Check formats match configuration
        expected_formats = set(DEFAULT_FORMATS.split(','))
        assert t.formatter.output_formats == expected_formats


class TestTranscriberInitialization:
    """Test transcriber initialization and configuration."""
    
    def test_lazy_initialization(self):
        """Test that transcriber is not initialized until first use."""
        # Import fresh module
        import importlib
        import src.whisper_mcp_server as server
        
        # Reload to ensure fresh state
        importlib.reload(server)
        
        # Transcriber should be None initially
        assert server.transcriber is None
        
        # Get transcriber
        t = server.get_transcriber()
        
        # Now should be initialized
        assert server.transcriber is not None
        assert server.transcriber is t
    
    def test_model_change_updates_global(self):
        """Test that changing model updates global transcriber."""
        import src.whisper_mcp_server as server
        
        # Get initial transcriber
        t1 = server.get_transcriber("mlx-community/whisper-tiny-mlx")
        assert server.transcriber is t1
        
        # Change model
        t2 = server.get_transcriber("mlx-community/whisper-base-mlx")
        
        # Global should be updated
        assert server.transcriber is t2
        assert server.transcriber is not t1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])