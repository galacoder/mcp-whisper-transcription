"""
Voice Activity Detection (VAD) processor for audio preprocessing.

This module provides VAD functionality to:
1. Detect voice activity in audio with high accuracy
2. Remove silence periods to improve transcription quality
3. Create optimal chunks for Whisper processing
4. Preserve timestamp accuracy for segment alignment
"""

import numpy as np
import torch
import torchaudio
from typing import List, Tuple, Dict, Optional, Any
import logging
from dataclasses import dataclass
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


@dataclass
class VADSegment:
    """Represents a voice activity segment."""
    start: float  # Start time in seconds
    end: float    # End time in seconds
    audio: np.ndarray  # Audio data for this segment


class VADProcessor:
    """Voice Activity Detection processor using Silero VAD."""
    
    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration: float = 0.25,
        min_silence_duration: float = 0.5,
        speech_pad: float = 0.1,
        sample_rate: int = 16000,
        window_size_samples: int = 512,
    ):
        """
        Initialize VAD processor.
        
        Args:
            threshold: VAD threshold (0.0-1.0)
            min_speech_duration: Minimum speech duration to keep (seconds)
            min_silence_duration: Minimum silence duration to split (seconds)
            speech_pad: Padding around speech segments (seconds)
            sample_rate: Audio sample rate
            window_size_samples: Window size for VAD processing
        """
        self.threshold = threshold
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.speech_pad = speech_pad
        self.sample_rate = sample_rate
        self.window_size_samples = window_size_samples
        
        # Load Silero VAD model
        self.model, self.utils = self._load_vad_model()
        
    def _load_vad_model(self) -> Tuple[Any, Dict]:
        """Load Silero VAD model."""
        try:
            # Load Silero VAD
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
                trust_repo=True
            )
            model.eval()
            logger.info("Silero VAD model loaded successfully")
            return model, utils
        except Exception as e:
            logger.error(f"Failed to load VAD model: {e}")
            raise
    
    def process_audio(
        self, 
        audio: np.ndarray, 
        sample_rate: int
    ) -> Tuple[List[VADSegment], Dict[str, Any]]:
        """
        Process audio with VAD to extract voice segments.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Tuple of (voice segments, statistics)
        """
        # Ensure audio is mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
            
        # Resample if necessary
        if sample_rate != self.sample_rate:
            audio = self._resample_audio(audio, sample_rate, self.sample_rate)
            
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # Get speech timestamps
        speech_timestamps = self._get_speech_timestamps(audio_tensor)
        
        # Convert timestamps to segments
        segments = self._timestamps_to_segments(audio, speech_timestamps)
        
        # Calculate statistics
        stats = self._calculate_stats(segments, len(audio) / self.sample_rate)
        
        return segments, stats
    
    def _resample_audio(
        self, 
        audio: np.ndarray, 
        orig_sr: int, 
        target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return audio
            
        # Use torchaudio for resampling
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        resampled = resampler(audio_tensor).squeeze(0).numpy()
        
        return resampled
    
    def _get_speech_timestamps(self, audio: torch.Tensor) -> List[Dict]:
        """Get speech timestamps using VAD model."""
        # Get VAD iterator
        get_speech_timestamps = self.utils[0]
        
        # Process audio
        speech_timestamps = get_speech_timestamps(
            audio,
            self.model,
            threshold=self.threshold,
            min_speech_duration_ms=int(self.min_speech_duration * 1000),
            min_silence_duration_ms=int(self.min_silence_duration * 1000),
            window_size_samples=self.window_size_samples,
            speech_pad_ms=int(self.speech_pad * 1000),
            return_seconds=False,  # Get sample indices
            sampling_rate=self.sample_rate
        )
        
        return speech_timestamps
    
    def _timestamps_to_segments(
        self, 
        audio: np.ndarray, 
        timestamps: List[Dict]
    ) -> List[VADSegment]:
        """Convert timestamps to VADSegment objects."""
        segments = []
        
        for ts in timestamps:
            start_sample = ts['start']
            end_sample = ts['end']
            
            # Extract audio segment
            segment_audio = audio[start_sample:end_sample]
            
            # Convert to seconds
            start_time = start_sample / self.sample_rate
            end_time = end_sample / self.sample_rate
            
            segments.append(VADSegment(
                start=start_time,
                end=end_time,
                audio=segment_audio
            ))
            
        return segments
    
    def _calculate_stats(
        self, 
        segments: List[VADSegment], 
        total_duration: float
    ) -> Dict[str, Any]:
        """Calculate VAD statistics."""
        if not segments:
            return {
                'total_duration': total_duration,
                'speech_duration': 0.0,
                'silence_duration': total_duration,
                'speech_ratio': 0.0,
                'num_segments': 0,
                'average_segment_duration': 0.0
            }
        
        speech_duration = sum(seg.end - seg.start for seg in segments)
        silence_duration = total_duration - speech_duration
        
        return {
            'total_duration': total_duration,
            'speech_duration': speech_duration,
            'silence_duration': silence_duration,
            'speech_ratio': speech_duration / total_duration,
            'num_segments': len(segments),
            'average_segment_duration': speech_duration / len(segments) if segments else 0.0,
            'segment_durations': [seg.end - seg.start for seg in segments]
        }
    
    def merge_segments(
        self, 
        segments: List[VADSegment], 
        max_duration: float = 30.0,
        overlap: float = 0.5
    ) -> List[VADSegment]:
        """
        Merge small segments into larger chunks for efficient processing.
        
        Args:
            segments: List of VAD segments
            max_duration: Maximum duration for merged segments (seconds)
            overlap: Overlap duration between segments (seconds)
            
        Returns:
            List of merged segments
        """
        if not segments:
            return []
            
        merged = []
        current_start = segments[0].start
        current_audio = segments[0].audio.copy()
        
        for i in range(1, len(segments)):
            seg = segments[i]
            current_duration = seg.end - current_start
            
            # Check if we should start a new merged segment
            if current_duration > max_duration:
                # Add overlap from next segment if possible
                if overlap > 0 and i < len(segments):
                    overlap_samples = int(overlap * self.sample_rate)
                    current_audio = np.concatenate([
                        current_audio,
                        seg.audio[:overlap_samples]
                    ])
                
                # Create merged segment
                merged.append(VADSegment(
                    start=current_start,
                    end=current_start + len(current_audio) / self.sample_rate,
                    audio=current_audio
                ))
                
                # Start new segment
                current_start = seg.start
                current_audio = seg.audio.copy()
            else:
                # Add silence gap if needed
                gap_duration = seg.start - (current_start + len(current_audio) / self.sample_rate)
                if gap_duration > 0:
                    gap_samples = int(gap_duration * self.sample_rate)
                    silence = np.zeros(gap_samples)
                    current_audio = np.concatenate([current_audio, silence])
                
                # Append segment audio
                current_audio = np.concatenate([current_audio, seg.audio])
        
        # Add final segment
        if len(current_audio) > 0:
            merged.append(VADSegment(
                start=current_start,
                end=current_start + len(current_audio) / self.sample_rate,
                audio=current_audio
            ))
        
        logger.info(f"Merged {len(segments)} segments into {len(merged)} chunks")
        
        return merged
    
    def create_continuous_audio(
        self, 
        segments: List[VADSegment], 
        original_duration: float
    ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        Create continuous audio from segments with timestamp mapping.
        
        Args:
            segments: List of VAD segments
            original_duration: Original audio duration
            
        Returns:
            Tuple of (continuous audio, timestamp mappings)
        """
        if not segments:
            return np.array([]), []
            
        # Concatenate all segment audio
        continuous_audio = np.concatenate([seg.audio for seg in segments])
        
        # Create timestamp mappings (continuous_time -> original_time)
        mappings = []
        continuous_time = 0.0
        
        for seg in segments:
            seg_duration = seg.end - seg.start
            mappings.append((continuous_time, seg.start))
            continuous_time += seg_duration
            
        # Add final mapping
        mappings.append((continuous_time, segments[-1].end))
        
        return continuous_audio, mappings
    
    def map_continuous_to_original_time(
        self, 
        continuous_time: float, 
        mappings: List[Tuple[float, float]]
    ) -> float:
        """
        Map time from continuous audio back to original audio time.
        
        Args:
            continuous_time: Time in continuous audio
            mappings: Timestamp mappings from create_continuous_audio
            
        Returns:
            Corresponding time in original audio
        """
        if not mappings or continuous_time <= 0:
            return continuous_time
            
        # Binary search for the right mapping
        for i in range(len(mappings) - 1):
            cont_start, orig_start = mappings[i]
            cont_end, orig_end = mappings[i + 1]
            
            if cont_start <= continuous_time <= cont_end:
                # Linear interpolation
                ratio = (continuous_time - cont_start) / (cont_end - cont_start)
                return orig_start + ratio * (orig_end - orig_start)
                
        # If beyond last mapping, extrapolate
        if continuous_time > mappings[-1][0]:
            last_cont, last_orig = mappings[-1]
            return last_orig + (continuous_time - last_cont)
            
        return continuous_time


def apply_vad_to_audio(
    audio_path: str,
    output_path: Optional[str] = None,
    threshold: float = 0.5,
    min_speech_duration: float = 0.25,
    min_silence_duration: float = 0.5,
    merge_chunks: bool = True,
    max_chunk_duration: float = 30.0
) -> Dict[str, Any]:
    """
    Apply VAD to an audio file and optionally save processed audio.
    
    Args:
        audio_path: Path to input audio file
        output_path: Optional path to save processed audio
        threshold: VAD threshold
        min_speech_duration: Minimum speech duration
        min_silence_duration: Minimum silence duration
        merge_chunks: Whether to merge segments into chunks
        max_chunk_duration: Maximum chunk duration if merging
        
    Returns:
        Dictionary with processing results and statistics
    """
    import librosa
    import soundfile as sf
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    
    # Initialize VAD processor
    vad = VADProcessor(
        threshold=threshold,
        min_speech_duration=min_speech_duration,
        min_silence_duration=min_silence_duration
    )
    
    # Process with VAD
    segments, stats = vad.process_audio(audio, sr)
    
    # Merge segments if requested
    if merge_chunks and segments:
        segments = vad.merge_segments(segments, max_chunk_duration)
        
    # Create continuous audio
    continuous_audio, mappings = vad.create_continuous_audio(
        segments, 
        len(audio) / sr
    )
    
    # Save if output path provided
    if output_path and len(continuous_audio) > 0:
        sf.write(output_path, continuous_audio, vad.sample_rate)
        logger.info(f"Saved VAD-processed audio to {output_path}")
        
    return {
        'segments': segments,
        'stats': stats,
        'mappings': mappings,
        'continuous_duration': len(continuous_audio) / vad.sample_rate if len(continuous_audio) > 0 else 0,
        'reduction_ratio': 1 - (stats['speech_duration'] / stats['total_duration']) if stats['total_duration'] > 0 else 0
    }