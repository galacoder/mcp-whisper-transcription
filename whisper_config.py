"""
Configuration management for Whisper transcription.

Provides optimized parameter sets for different audio types and durations.
"""

from typing import Dict, Any


class WhisperConfig:
    """Configuration presets for different transcription scenarios."""
    
    # Default configuration for general use
    DEFAULT_CONFIG = {
        "temperature": 0.0,
        "no_speech_threshold": 0.45,
        "logprob_threshold": -1.0,
        "condition_on_previous_text": True,
        "compression_ratio_threshold": 2.4,
        "initial_prompt": None,
    }
    
    # Optimized for short audio (<5 minutes)
    SHORT_AUDIO_CONFIG = {
        "temperature": 0.0,
        "no_speech_threshold": 0.45,
        "logprob_threshold": -1.0,
        "condition_on_previous_text": True,
        "compression_ratio_threshold": 2.4,
        "initial_prompt": "The following is a clear and accurate transcription.",
    }
    
    # Optimized for long audio (>5 minutes) to prevent hallucinations
    LONG_AUDIO_CONFIG = {
        "temperature": 0.0,
        "no_speech_threshold": 0.6,  # Stricter silence detection
        "logprob_threshold": -1.0,  # Reject low-confidence segments
        "condition_on_previous_text": False,  # Prevent error propagation
        "compression_ratio_threshold": 2.4,  # Detect repetitions
        "initial_prompt": "Transcribe the speech accurately. Ignore background noise and music.",
        "suppress_tokens": "-1",  # Suppress special tokens
        "suppress_blank": True,  # Suppress blank outputs
    }
    
    # Optimized for noisy audio
    NOISY_AUDIO_CONFIG = {
        "temperature": 0.2,  # Slight randomness can help with noise
        "no_speech_threshold": 0.7,  # Very strict silence detection
        "logprob_threshold": -1.0,
        "condition_on_previous_text": False,
        "compression_ratio_threshold": 2.0,  # Lower threshold for noisy audio
        "initial_prompt": "Focus on clear speech and ignore background noise.",
    }
    
    # Optimized for meeting/conversation audio
    CONVERSATION_CONFIG = {
        "temperature": 0.0,
        "no_speech_threshold": 0.5,
        "logprob_threshold": -0.8,
        "condition_on_previous_text": True,  # Helps with context
        "compression_ratio_threshold": 2.4,
        "initial_prompt": "The following is a conversation between multiple speakers.",
    }
    
    @classmethod
    def get_config_for_duration(cls, duration_seconds: float) -> Dict[str, Any]:
        """
        Get optimal configuration based on audio duration.
        
        Args:
            duration_seconds: Duration of audio in seconds
            
        Returns:
            Configuration dictionary
        """
        if duration_seconds < 300:  # Less than 5 minutes
            return cls.SHORT_AUDIO_CONFIG.copy()
        else:
            return cls.LONG_AUDIO_CONFIG.copy()
    
    @classmethod
    def get_config_by_type(cls, audio_type: str = "default") -> Dict[str, Any]:
        """
        Get configuration by audio type.
        
        Args:
            audio_type: One of 'default', 'short', 'long', 'noisy', 'conversation'
            
        Returns:
            Configuration dictionary
        """
        configs = {
            "default": cls.DEFAULT_CONFIG,
            "short": cls.SHORT_AUDIO_CONFIG,
            "long": cls.LONG_AUDIO_CONFIG,
            "noisy": cls.NOISY_AUDIO_CONFIG,
            "conversation": cls.CONVERSATION_CONFIG,
        }
        
        return configs.get(audio_type, cls.DEFAULT_CONFIG).copy()
    
    @classmethod
    def merge_configs(cls, base_config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge configuration with user overrides.
        
        Args:
            base_config: Base configuration dictionary
            overrides: User-provided overrides
            
        Returns:
            Merged configuration
        """
        config = base_config.copy()
        
        # Only override non-None values
        for key, value in overrides.items():
            if value is not None:
                config[key] = value
                
        return config