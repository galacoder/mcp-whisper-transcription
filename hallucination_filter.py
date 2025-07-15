"""
Hallucination detection and filtering for Whisper transcriptions.

This module implements advanced post-processing filters to detect and remove
hallucinations commonly produced by Whisper models on long-form audio.
"""

import re
from typing import List, Dict, Any, Tuple
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class HallucinationFilter:
    """Advanced hallucination detection and filtering for Whisper transcriptions."""
    
    def __init__(self):
        # Common Whisper hallucinations (Bag of Hallucinations - BoH)
        self.common_hallucinations = {
            "Thank you for watching",
            "Please subscribe",
            "Thanks for watching", 
            "Subscribe to my channel",
            "Like and subscribe",
            "See you next time",
            "Bye bye",
            "Thank you",
            "Music",
            "[Music]",
            "[Applause]",
            "you",
        }
        
        # Repetition thresholds
        self.max_phrase_repetitions = 3
        self.max_word_repetitions = 5
        self.min_segment_confidence = 0.6
        
    def filter_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply all filters to segment list.
        
        Args:
            segments: List of segment dictionaries from Whisper
            
        Returns:
            Filtered list of segments with hallucinations removed
        """
        if not segments:
            return segments
            
        # Apply filters in sequence
        segments = self._filter_repetitive_segments(segments)
        segments = self._filter_low_confidence(segments)
        segments = self._filter_common_hallucinations(segments)
        segments = self._merge_adjacent_segments(segments)
        
        return segments
    
    def _filter_repetitive_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove segments with excessive repetition."""
        filtered = []
        
        for i, segment in enumerate(segments):
            text = segment.get("text", "").strip()
            
            # Check for word-level repetitions
            if self._has_excessive_word_repetition(text):
                logger.warning(f"Filtered repetitive segment at {segment.get('start', 0):.1f}s: {text[:50]}...")
                continue
                
            # Check for phrase-level repetitions across segments
            if i > 0 and self._is_repetitive_with_previous(text, filtered):
                logger.warning(f"Filtered repetitive phrase at {segment.get('start', 0):.1f}s: {text[:50]}...")
                continue
                
            filtered.append(segment)
            
        return filtered
    
    def _has_excessive_word_repetition(self, text: str) -> bool:
        """Check if text has excessive word repetitions."""
        words = text.lower().split()
        if len(words) < 3:
            return False
            
        # Check for immediate repetitions (e.g., "we, we, we, we")
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                # Count consecutive repetitions
                count = 1
                j = i + 1
                while j < len(words) and words[j] == words[i]:
                    count += 1
                    j += 1
                if count >= self.max_word_repetitions:
                    return True
                    
        # Check for pattern repetitions
        word_counts = Counter(words)
        total_words = len(words)
        
        # If any single word makes up >50% of segment, it's likely repetitive
        for word, count in word_counts.items():
            if count > total_words * 0.5 and count >= 3:
                return True
                
        return False
    
    def _is_repetitive_with_previous(self, text: str, previous_segments: List[Dict]) -> bool:
        """Check if text repeats recent segments."""
        if not previous_segments:
            return False
            
        # Normalize text for comparison
        normalized_text = text.lower().strip()
        
        # Check last N segments for repetition
        check_count = min(5, len(previous_segments))
        recent_texts = [seg.get("text", "").lower().strip() for seg in previous_segments[-check_count:]]
        
        # Count exact matches
        repetition_count = recent_texts.count(normalized_text)
        
        # Also check for substring repetitions
        for recent in recent_texts:
            if len(normalized_text) > 20 and normalized_text in recent:
                repetition_count += 0.5
            elif len(recent) > 20 and recent in normalized_text:
                repetition_count += 0.5
                
        return repetition_count >= self.max_phrase_repetitions
    
    def _filter_low_confidence(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter segments with low confidence scores."""
        filtered = []
        
        for segment in segments:
            # Check various confidence indicators
            no_speech_prob = segment.get("no_speech_prob", 0)
            avg_logprob = segment.get("avg_logprob", -1)
            compression_ratio = segment.get("compression_ratio", 1)
            
            # Filter based on confidence thresholds
            if no_speech_prob > 0.9:  # High probability of no speech
                logger.debug(f"Filtered no-speech segment at {segment.get('start', 0):.1f}s")
                continue
                
            if avg_logprob < -1.5:  # Very low confidence
                logger.debug(f"Filtered low-confidence segment at {segment.get('start', 0):.1f}s")
                continue
                
            if compression_ratio > 2.4:  # High compression often indicates repetition
                logger.debug(f"Filtered high-compression segment at {segment.get('start', 0):.1f}s")
                continue
                
            filtered.append(segment)
            
        return filtered
    
    def _filter_common_hallucinations(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove segments containing common hallucinations."""
        filtered = []
        
        for segment in segments:
            text = segment.get("text", "").strip()
            
            # Check if entire segment is a known hallucination
            if text in self.common_hallucinations:
                logger.debug(f"Filtered known hallucination at {segment.get('start', 0):.1f}s: {text}")
                continue
                
            # Check if segment contains only punctuation or very short
            if len(text) < 3 or text.replace(".", "").replace(",", "").strip() == "":
                continue
                
            filtered.append(segment)
            
        return filtered
    
    def _merge_adjacent_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge adjacent segments that were split unnecessarily."""
        if len(segments) < 2:
            return segments
            
        merged = []
        current = segments[0].copy()
        
        for next_segment in segments[1:]:
            # Check if segments should be merged
            time_gap = next_segment.get("start", 0) - current.get("end", 0)
            
            if time_gap < 0.5:  # Less than 0.5 second gap
                # Merge segments
                current["text"] = current.get("text", "") + " " + next_segment.get("text", "")
                current["end"] = next_segment.get("end", current.get("end", 0))
                
                # Merge token-level data if available
                if "words" in current and "words" in next_segment:
                    current["words"].extend(next_segment.get("words", []))
            else:
                # Save current and start new
                merged.append(current)
                current = next_segment.copy()
                
        # Don't forget the last segment
        merged.append(current)
        
        return merged
    
    def detect_hallucination_zones(self, segments: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
        """
        Detect time ranges where hallucinations occur.
        
        Returns:
            List of (start_time, end_time) tuples indicating hallucination zones
        """
        zones = []
        
        for i, segment in enumerate(segments):
            text = segment.get("text", "").strip()
            
            # Check various hallucination indicators
            if (self._has_excessive_word_repetition(text) or 
                (i > 0 and self._is_repetitive_with_previous(text, segments[:i])) or
                text in self.common_hallucinations):
                
                start = segment.get("start", 0)
                end = segment.get("end", start)
                
                # Merge with previous zone if overlapping
                if zones and zones[-1][1] >= start - 1.0:
                    zones[-1] = (zones[-1][0], max(zones[-1][1], end))
                else:
                    zones.append((start, end))
                    
        return zones


def post_process_transcription(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply post-processing filters to transcription result.
    
    Args:
        result: Raw transcription result from Whisper
        
    Returns:
        Filtered transcription result
    """
    filter = HallucinationFilter()
    
    # Filter segments
    if "segments" in result and result["segments"]:
        original_count = len(result["segments"])
        result["segments"] = filter.filter_segments(result["segments"])
        filtered_count = len(result["segments"])
        
        if filtered_count < original_count:
            logger.info(f"Filtered {original_count - filtered_count} hallucinated segments")
            
        # Reconstruct text from filtered segments
        result["text"] = " ".join(seg.get("text", "") for seg in result["segments"])
        
    # Detect hallucination zones for reporting
    if "segments" in result:
        zones = filter.detect_hallucination_zones(result["segments"])
        if zones:
            result["hallucination_zones"] = zones
            logger.warning(f"Detected {len(zones)} hallucination zones in transcription")
    
    return result