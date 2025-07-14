#!/usr/bin/env python3

import os
import logging
import json
from pathlib import Path
import ffmpeg
from datetime import datetime, timedelta
import humanize
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Set
import re


# Shared dataclass for transcription statistics
@dataclass
class TranscriptionStats:
    video_name: str
    video_duration: float  # in seconds
    processing_time: float  # in seconds
    model_name: str
    peak_memory: float  # in GB
    peak_gpu_memory: Optional[float] = None  # in GB
    chunks_processed: int = 0
    words_transcribed: int = 0
    processing_speed: float = 0.0  # minutes of audio per minute of processing
    device: str = "cpu"
    batch_size: int = 32  # Default batch size
    chunk_size: int = 30  # Default chunk size in seconds


class PerformanceReport:
    """Shared performance report generator for both transcription versions"""

    def __init__(self):
        self.stats: List[TranscriptionStats] = []
        self.start_time = datetime.now()

    def add_stats(self, stat: TranscriptionStats):
        self.stats.append(stat)

    def generate_report(self) -> str:
        if not self.stats:
            return "\n📊 No files were processed."

        total_duration = sum(stat.video_duration for stat in self.stats)
        total_processing_time = sum(stat.processing_time for stat in self.stats)
        total_words = sum(stat.words_transcribed for stat in self.stats)

        # Calculate overall speed
        try:
            overall_speed = (
                (total_duration / 60) / (total_processing_time / 60)
                if total_processing_time > 0
                else 0
            )
        except ZeroDivisionError:
            overall_speed = 0

        report = [
            "\n📊 Transcription Performance Report",
            "=" * 50,
            f"🕒 Total Runtime: {humanize.naturaldelta(total_processing_time)}",
            f"📼 Files Processed: {len(self.stats)}",
            f"⏱️ Total Audio Duration: {humanize.naturaldelta(total_duration)}",
            f"📝 Total Words Transcribed: {total_words:,}",
            f"⚡️ Overall Speed: {overall_speed:.2f}x realtime",
            "\nPer File Statistics:",
            "-" * 40,
        ]

        for stat in self.stats:
            try:
                speed = stat.processing_speed if stat.processing_speed > 0 else 0
                words_per_sec = (
                    (stat.words_transcribed / stat.video_duration)
                    if stat.video_duration > 0
                    else 0
                )
                chunks_timing = (
                    (stat.video_duration / stat.chunks_processed)
                    if stat.chunks_processed > 0
                    else 0
                )
            except ZeroDivisionError:
                speed = 0
                words_per_sec = 0
                chunks_timing = 0

            report.extend(
                [
                    f"\n📹 {stat.video_name}:",
                    f"  • Duration: {humanize.naturaldelta(stat.video_duration)}",
                    f"  • Processing Time: {humanize.naturaldelta(stat.processing_time)}",
                    f"  • Speed: {speed:.2f}x realtime",
                    f"  • Words: {stat.words_transcribed:,} ({words_per_sec:.1f} words/sec)",
                    f"  • Peak Memory: {stat.peak_memory:.2f}GB",
                ]
            )

            # Add GPU memory info if available
            if stat.peak_gpu_memory:
                report.append(f"  • Peak GPU Memory: {stat.peak_gpu_memory:.2f}GB")

            report.append(
                f"  • Chunks: {stat.chunks_processed} ({chunks_timing:.1f}s per chunk)"
            )

        if self.stats:
            report.extend(
                [
                    "\n🔧 System Configuration:",
                    "-" * 40,
                    f"• Model: {self.stats[0].model_name}",
                    f"• Device: {self.stats[0].device}",
                    f"• Batch Size: {self.stats[0].batch_size}",
                    f"• Chunk Size: {self.stats[0].chunk_size}s",
                ]
            )

        return "\n".join(report)


class OutputFormatter:
    """Handles saving transcription output in different formats"""

    def __init__(self, output_formats: Set[str] = None):
        if output_formats is None:
            output_formats = {"txt", "md", "srt"}
        self.output_formats = output_formats
        self.logger = logging.getLogger("WhisperTranscriber")

    def save_transcription(self, result: Dict[str, Any], output_path: Path):
        """Save transcription in the specified formats"""
        saved_files = []

        # Always save JSON internally for potential future use
        json_path = None
        if "json" in self.output_formats:
            json_path = self._save_json(result, output_path)
            saved_files.append(("JSON data", json_path))

        if "txt" in self.output_formats:
            txt_path = self._save_txt(result, output_path)
            saved_files.append(("Timestamped text", txt_path))

        if "md" in self.output_formats:
            md_path = self._save_markdown(result, output_path)
            saved_files.append(("Clean markdown", md_path))

        if "srt" in self.output_formats:
            srt_path = self._save_srt(result, output_path)
            saved_files.append(("SRT subtitles", srt_path))

        # Log the results
        if saved_files:
            print(f"\n✓ Transcription saved in {len(saved_files)} format(s):")
            for desc, path in saved_files:
                print(f"  - {desc}: {path}")

            self.logger.info(
                f"Transcription saved in {len(saved_files)} format(s): {output_path}"
            )
        else:
            self.logger.warning(f"No output formats were specified for {output_path}")

    def _save_json(self, result: Dict[str, Any], output_path: Path) -> Path:
        """Save raw JSON data"""
        json_path = output_path.with_suffix(".json")
        self.logger.info(f"Saving JSON output to {json_path}")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return json_path

    def _save_txt(self, result: Dict[str, Any], output_path: Path) -> Path:
        """Save timestamped text format with improved cleaning"""
        txt_path = output_path.with_suffix(".txt")
        self.logger.info(f"Saving TXT output to {txt_path}")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(
                f"Transcription completed at: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n"
            )

            # First check if we have populated chunks
            if isinstance(result, dict) and "chunks" in result and result["chunks"]:
                chunks = result["chunks"]
                if all(
                    isinstance(c, dict) and "timestamp" in c and "text" in c
                    for c in chunks
                ):
                    # Process chunks with filtering and cleaning
                    for chunk in chunks:
                        # Skip unwanted content
                        if should_skip_content(chunk["text"]):
                            continue

                        timestamp = f"[{chunk['timestamp'][0]:.2f}s - {chunk['timestamp'][1]:.2f}s]"
                        cleaned_text = clean_text_content(chunk["text"].strip())
                        f.write(f"{timestamp}: {cleaned_text}\n")
                    return txt_path

            # Then check if we have segments
            if isinstance(result, dict) and "segments" in result and result["segments"]:
                segments = result["segments"]
                if all(isinstance(s, dict) and "text" in s for s in segments):
                    # Process segments with filtering and cleaning
                    for segment in segments:
                        # Skip unwanted content
                        if should_skip_content(segment["text"]):
                            continue

                        start = segment.get("start", 0)
                        end = segment.get("end", 0)
                        timestamp = f"[{start:.2f}s - {end:.2f}s]"
                        cleaned_text = clean_text_content(segment["text"].strip())
                        f.write(f"{timestamp}: {cleaned_text}\n")
                    return txt_path

            # Fallback to raw text content
            if isinstance(result, dict) and "text" in result and result["text"]:
                # Clean and process the raw text
                text = result["text"]
                if any(
                    line.strip().startswith("[") and "]:" in line
                    for line in text.split("\n")
                ):
                    # Process timestamped text line by line
                    processed_lines = []
                    for line in text.split("\n"):
                        # Skip unwanted content
                        if should_skip_content(line):
                            continue

                        # Clean lines with timestamps
                        timestamp_match = re.match(
                            r"(\[[\d\.]+s - [\d\.]+s\]:)(.*)", line
                        )
                        if timestamp_match:
                            timestamp = timestamp_match.group(1)
                            content = timestamp_match.group(2).strip()
                            if content and not should_skip_content(content):
                                processed_lines.append(
                                    f"{timestamp} {clean_text_content(content)}"
                                )
                        else:
                            # Include non-timestamp lines that aren't problematic
                            if line.strip():
                                processed_lines.append(line)

                    # Post-process to remove duplicates
                    final_lines = []
                    prev_line = None
                    for line in processed_lines:
                        # Skip duplicate lines
                        if prev_line and line.strip() == prev_line.strip():
                            continue

                        # Compare content after timestamps to catch near-duplicates
                        if prev_line and ":" in prev_line and ":" in line:
                            prev_content = (
                                prev_line[prev_line.find(":") + 1 :].strip().lower()
                            )
                            current_content = line[line.find(":") + 1 :].strip().lower()

                            # Skip if content is nearly identical
                            if (
                                prev_content
                                and current_content
                                and similar_text(prev_content, current_content) > 0.75
                            ):
                                continue

                        final_lines.append(line)
                        prev_line = line

                    # Write the cleaned text
                    f.write("\n".join(final_lines))
                else:
                    # Clean and write the raw text if no timestamps
                    cleaned_text = post_process_transcript(text)
                    f.write(cleaned_text)
            else:
                # Last resort
                f.write("No transcription available")

        return txt_path

    def _save_markdown(self, result: Dict[str, Any], output_path: Path) -> Path:
        """Save clean markdown format with improved cleaning"""
        md_path = output_path.with_suffix(".md")
        self.logger.info(f"Saving Markdown output to {md_path}")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {output_path.stem}\n\n")

            # First try to extract text from chunks or segments
            chunks = []
            full_text = ""

            if isinstance(result, dict):
                # Try to get text from chunks
                if "chunks" in result and result["chunks"]:
                    if all(
                        isinstance(c, dict) and "text" in c for c in result["chunks"]
                    ):
                        # Filter out problematic chunks
                        chunks = [
                            c
                            for c in result["chunks"]
                            if not should_skip_content(c.get("text", ""))
                        ]

                        # Clean text for each chunk
                        for chunk in chunks:
                            if "text" in chunk:
                                chunk["text"] = clean_text_content(
                                    chunk["text"].strip()
                                )

                        full_text = " ".join(
                            chunk["text"] for chunk in chunks if chunk.get("text")
                        )

                # If no text from chunks, try segments
                if not full_text and "segments" in result and result["segments"]:
                    if all(
                        isinstance(s, dict) and "text" in s for s in result["segments"]
                    ):
                        # Filter out problematic segments
                        chunks = [
                            s
                            for s in result["segments"]
                            if not should_skip_content(s.get("text", ""))
                        ]

                        # Clean text for each segment
                        for segment in chunks:
                            if "text" in segment:
                                segment["text"] = clean_text_content(
                                    segment["text"].strip()
                                )

                        full_text = " ".join(
                            segment["text"] for segment in chunks if segment.get("text")
                        )

                # If still no text, try using direct text field
                if not full_text and "text" in result and result["text"]:
                    # If text contains timestamps, try to extract just the text content
                    raw_text = result["text"]
                    if isinstance(raw_text, str):
                        # Process the raw text with our cleaning utilities
                        cleaned_raw = clean_raw_transcription(raw_text)
                        post_processed = post_process_transcript(cleaned_raw)

                        # Extract only the text content after timestamps if they exist
                        clean_lines = []
                        for line in post_processed.split("\n"):
                            if ":" in line and "[" in line and "]" in line:
                                text_part = line.split(":", 1)[1].strip()
                                if text_part and not should_skip_content(text_part):
                                    clean_lines.append(clean_text_content(text_part))
                            elif line.strip() and not (
                                line.startswith("Transcription completed")
                                or should_skip_content(line)
                            ):
                                clean_lines.append(clean_text_content(line.strip()))

                        if clean_lines:
                            full_text = " ".join(clean_lines)
                        else:
                            full_text = post_processed
            else:
                full_text = str(result)

            # Ensure the text is clean
            cleaned_text = clean_text_content(full_text)

            # Split into paragraphs with improved logic
            paragraphs = []
            current_para = []

            if chunks:
                for i, chunk in enumerate(chunks):
                    text = chunk.get("text", "").strip()

                    # Skip empty or problematic text
                    if not text or should_skip_content(text):
                        continue

                    # Start new paragraph on longer pauses or natural breaks
                    if i > 0:
                        prev_chunk = chunks[i - 1]
                        prev_text = prev_chunk.get("text", "").strip()

                        # Skip if previous chunk had problematic text
                        if not prev_text or should_skip_content(prev_text):
                            continue

                        # Calculate time gap based on format
                        if "timestamp" in chunk and "timestamp" in prev_chunk:
                            time_gap = (
                                chunk["timestamp"][0] - prev_chunk["timestamp"][1]
                            )
                        else:
                            time_gap = chunk.get("start", 0) - prev_chunk.get("end", 0)

                        # New paragraph if:
                        # 1. Time gap > 2 seconds OR
                        # 2. Current chunk starts with capital letter and previous chunk ends with sentence-ending punctuation
                        if (time_gap > 2.0) or (
                            text
                            and text[0].isupper()
                            and prev_text.endswith((".", "!", "?"))
                        ):
                            if current_para:
                                paragraphs.append(" ".join(current_para))
                                current_para = []

                    current_para.append(text)

                # Add final paragraph
                if current_para:
                    paragraphs.append(" ".join(current_para))
            else:
                # For plain text, process more thoroughly
                cleaned_text = clean_text_content(cleaned_text)

                # Split on sentence endings for better paragraph formation
                sentences = re.split(r"(?<=[.!?])\s+", cleaned_text)

                for i, sentence in enumerate(sentences):
                    if not sentence.strip() or should_skip_content(sentence):
                        continue

                    current_para.append(sentence)

                    # Start a new paragraph after every 3-4 sentences or on topic shifts
                    if (len(current_para) >= 3) or (
                        i < len(sentences) - 1
                        and len(sentence) > 20
                        and sentence.endswith((".", "!", "?"))
                    ):
                        paragraphs.append(" ".join(current_para).strip())
                        current_para = []

                # Add remaining sentences
                if current_para:
                    paragraphs.append(" ".join(current_para).strip())

            # Write paragraphs with final cleaning
            for para in paragraphs:
                if para.strip() and not should_skip_content(para):
                    # One final cleaning pass
                    clean_para = clean_text_content(para)
                    if clean_para:
                        f.write(f"{clean_para}\n\n")

        return md_path

    def _save_srt(self, result: Dict[str, Any], output_path: Path) -> Path:
        """Save SRT subtitle format with conversation flow preservation"""
        srt_path = output_path.with_suffix(".srt")
        self.logger.info(f"Saving SRT output to {srt_path}")

        # First check if a corresponding txt file exists
        txt_path = output_path.with_suffix(".txt")
        if txt_path.exists():
            try:
                # Use the proven convert_to_srt approach that works well
                with open(txt_path, "r", encoding="utf-8") as f:
                    transcript_content = f.read()

                # Skip header metadata (like "Transcription completed at: ...")
                # Split on the first empty line
                parts = re.split(r"^\s*\n", transcript_content, maxsplit=1)
                if len(parts) > 1:
                    main_content = parts[1]
                else:
                    main_content = parts[0]

                # Extract timestamp chunks
                time_pattern = (
                    r"\[(\d+\.\d+)s - (\d+\.\d+)s\]: (.*?)(?=\n\[\d+\.\d+s|\Z)"
                )
                matches = re.findall(time_pattern, main_content, re.DOTALL)

                # Extract text entries (non-empty only)
                entries = []
                for _, _, text in matches:
                    text = text.strip()
                    if text:
                        entries.append(text)

                # Create SRT content with sequential timestamps based on text length
                srt_content = []
                current_time = 0.0

                for idx, text in enumerate(entries, 1):
                    # Calculate duration based on text length
                    # Approximately 20 characters per second is a reasonable reading speed
                    duration = max(1.0, len(text) / 20)

                    start_time = current_time
                    end_time = start_time + duration

                    srt_timestamp = f"{format_srt_timestamp(start_time)} --> {format_srt_timestamp(end_time)}"

                    srt_content.append(f"{idx}\n{srt_timestamp}\n{text}\n\n")
                    current_time = end_time  # Move to the next timestamp

                # Write SRT file
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write("".join(srt_content))

                self.logger.info(
                    f"Generated SRT directly from txt transcript: {srt_path}"
                )
                return srt_path

            except Exception as e:
                self.logger.warning(
                    f"Error generating SRT from txt: {str(e)}. Falling back to standard method."
                )
                # Continue with the standard method if there's an error

        # If we're here, either there's no TXT file or an error occurred
        # Extract all subtitle entries (fallback method)
        text_entries = []

        # Try to extract text entries from various result formats
        if isinstance(result, dict):
            # Try chunks format first
            if "chunks" in result and result["chunks"]:
                for chunk in result["chunks"]:
                    if "text" in chunk:
                        text = clean_text_content(chunk["text"].strip())
                        if text and not should_skip_content(text):
                            text_entries.append(text)

            # Then try segments format
            elif "segments" in result and result["segments"]:
                for segment in result["segments"]:
                    if "text" in segment:
                        text = clean_text_content(segment["text"].strip())
                        if text and not should_skip_content(text):
                            text_entries.append(text)

            # Try raw text as last resort
            elif "text" in result and result["text"]:
                text = result["text"]
                # If text contains timestamps, try to extract just the text content
                if any(
                    line.strip().startswith("[") and "]:" in line
                    for line in text.split("\n")
                ):
                    for line in text.split("\n"):
                        timestamp_match = re.match(
                            r"\[([\d\.]+)s - ([\d\.]+)s\]:(.*)", line
                        )
                        if timestamp_match:
                            text_content = timestamp_match.group(3).strip()
                            if text_content and not should_skip_content(text_content):
                                text_entries.append(clean_text_content(text_content))
                else:
                    # Just use the whole text as one entry if it doesn't have timestamps
                    text = clean_text_content(text)
                    if text and not should_skip_content(text):
                        text_entries.append(text)

        # If we extracted any text entries, create SRT content with sequential timestamps
        if text_entries:
            srt_content = []
            current_time = 0.0

            for idx, text in enumerate(text_entries, 1):
                # Calculate duration based on text length
                duration = max(1.0, len(text) / 20)  # ~20 chars per second

                start_time = current_time
                end_time = start_time + duration

                srt_timestamp = f"{format_srt_timestamp(start_time)} --> {format_srt_timestamp(end_time)}"

                srt_content.append(f"{idx}\n{srt_timestamp}\n{text}\n\n")
                current_time = end_time  # Move to the next timestamp

            # Write the SRT file
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write("".join(srt_content))
        else:
            # Fallback: write empty SRT file with a message
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(
                    "1\n00:00:00,000 --> 00:00:05,000\nNo transcription available\n\n"
                )

        return srt_path


def format_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def extract_audio(video_path: Path, audio_path: Path, logger=None) -> bool:
    """Extract audio from video file with robust error handling"""
    try:
        # Ensure parent directory exists
        audio_path.parent.mkdir(exist_ok=True)

        if logger:
            logger.info(f"Extracting audio from {video_path} to {audio_path}")

        # First attempt with standard parameters
        try:
            stream = (
                ffmpeg.input(str(video_path))
                .output(
                    str(audio_path),
                    acodec="pcm_s16le",
                    ac=1,
                    ar="16000",
                    loglevel="error",
                )
                .overwrite_output()
            )
            ffmpeg.run(stream, capture_stderr=True)

            # Verify the output
            if audio_path.exists() and audio_path.stat().st_size > 0:
                if logger:
                    logger.info(f"Audio extraction successful: {audio_path}")
                return True

        except ffmpeg.Error as e:
            if logger:
                logger.warning(f"First audio extraction attempt failed: {str(e)}")

        # Fallback attempt with more options
        try:
            stream = (
                ffmpeg.input(str(video_path))
                .output(
                    str(audio_path),
                    **{
                        "vn": None,
                        "acodec": "pcm_s16le",
                        "ac": 1,
                        "ar": "16000",
                        "loglevel": "error",
                        "af": "aresample=async=1:first_pts=0",
                        "max_muxing_queue_size": "1024",
                        "fflags": "+igndts+genpts+discardcorrupt",
                        "threads": "0",
                    },
                )
                .overwrite_output()
            )

            ffmpeg.run(stream, capture_stderr=True)

            # Verify the output again
            if audio_path.exists() and audio_path.stat().st_size > 0:
                try:
                    # Additional validation
                    probe = ffmpeg.probe(str(audio_path))
                    if probe.get("streams"):
                        if logger:
                            logger.info(
                                f"Audio extraction successful (fallback method): {audio_path}"
                            )
                        return True
                except ffmpeg.Error:
                    pass

            if logger:
                logger.error(f"Failed to extract valid audio from {video_path}")
            return False

        except ffmpeg.Error as e:
            if logger:
                logger.error(f"All audio extraction attempts failed: {str(e)}")
            return False

    except Exception as e:
        if logger:
            logger.error(f"Unexpected error during audio extraction: {str(e)}")
        return False
    finally:
        # Cleanup if extraction failed
        if not (audio_path.exists() and audio_path.stat().st_size > 0):
            audio_path.unlink(missing_ok=True)
            return False


def cleanup_temp_files(temp_dir: Path, logger=None):
    """Clean up temporary files and directory with better handling of nested directories"""
    try:
        if not temp_dir.exists():
            return

        if logger:
            logger.info(f"Cleaning up temporary directory: {temp_dir}")

        # First function to handle nested directories recursively
        def clean_directory(directory):
            # First remove all files in the current directory
            for item in directory.iterdir():
                if item.is_file():
                    try:
                        item.unlink()
                        if logger:
                            logger.debug(f"Removed temp file: {item}")
                    except Exception as e:
                        if logger:
                            logger.warning(
                                f"Could not remove temporary file {item}: {str(e)}"
                            )

                # Then recursively clean subdirectories
                elif item.is_dir():
                    try:
                        clean_directory(item)
                        # Try to remove the now-empty directory
                        item.rmdir()
                        if logger:
                            logger.debug(f"Removed temp directory: {item}")
                    except Exception as e:
                        if logger:
                            logger.warning(
                                f"Could not remove directory {item}: {str(e)}"
                            )

        # Start the recursive cleaning
        clean_directory(temp_dir)

        # Finally try to remove the main temp directory
        try:
            temp_dir.rmdir()
            if logger:
                logger.info("Temporary directory cleaned up successfully")
        except Exception as e:
            if logger:
                logger.warning(f"Could not remove main temporary directory: {str(e)}")
    except Exception as e:
        if logger:
            logger.warning(f"Error during cleanup: {str(e)}")


def setup_logger(logs_dir: Path, name="WhisperTranscriber"):
    """Set up a configured logger with file and console output"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # Create handlers
    c_handler = logging.StreamHandler()
    logs_dir.mkdir(exist_ok=True)
    f_handler = logging.FileHandler(
        logs_dir / f"transcription_{datetime.now():%Y%m%d_%H%M%S}.log"
    )

    # Create formatter with timestamps
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


def get_video_duration(file_path: Path, logger=None) -> float:
    """Get duration of video/audio file in seconds"""
    try:
        probe = ffmpeg.probe(str(file_path))
        duration = float(probe["format"]["duration"])
        return duration
    except Exception as e:
        if logger:
            logger.error(f"Error getting duration for {file_path}: {str(e)}")
        return 0


# Text processing utilities
def clean_text_content(text: str) -> str:
    """Apply cleaning to a text segment to fix common issues"""
    if not text:
        return ""

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Fix common transcription errors
    text = text.replace(" ,", ",")
    text = text.replace(" .", ".")
    text = text.replace(" ?", "?")
    text = text.replace(" !", "!")
    text = text.replace(" :", ":")
    text = text.replace(" ;", ";")

    # Remove duplicate words (like "the the", "is is", etc.)
    words = text.split()
    deduped_words = []
    for i, word in enumerate(words):
        if i > 0 and word.lower() == words[i - 1].lower():
            continue
        deduped_words.append(word)

    return " ".join(deduped_words)


def should_skip_content(text: str) -> bool:
    """Check if content should be skipped based on various criteria"""
    # Skip empty content
    if not text or not text.strip():
        return True

    # Skip lines that likely contain noise or errors from the model
    skip_patterns = [
        r"downloading",
        r"^\s*\d{3,}\s*$",  # Just numbers (line numbers)
        r"^\s*\.\.\.\s*$",  # Just dots
        r"^\s*$",  # Empty lines
        r"^\s*\[.*?EMPTY.*?\]\s*$",  # Empty markers
        r"^line\s+\d+\s*$",  # Line markers
        r"^file\s+\d+\s*$",  # File markers
    ]

    for pattern in skip_patterns:
        if re.search(pattern, text.lower()):
            return True

    return False


def similar_text(a: str, b: str) -> float:
    """Calculate similarity between two strings (0-1 scale)"""
    # Quick check for identical strings
    if a == b:
        return 1.0

    # Simple similarity based on word overlap
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())

    if not a_words or not b_words:
        return 0.0

    intersection = len(a_words.intersection(b_words))
    union = len(a_words.union(b_words))

        return intersection / union if union > 0 else 0.0


def dedup_lines(lines: list[str]) -> list[str]:
    """Collapse identical / near-identical lines, preserving single blank gaps."""
    out: list[str] = []
    last_key: str | None = None
    for ln in lines:
        stripped = ln.strip()
        # Blank line handling – keep at most one
        if not stripped:
            if out and out[-1].strip():
                out.append("")
            continue
        if stripped.startswith("[") and "]:" in stripped:
            key = stripped[stripped.find("]: ") + 3 :].lower()
        else:
            key = stripped.lower()
        if last_key and (key == last_key or similar_text(key, last_key) > 0.8):
            continue
        out.append(ln)
        last_key = key
    return out


def clean_raw_transcription(text: str) -> str:
    """Clean a raw transcription chunk"""
    if not text:
        return ""

    lines = []
    for line in text.split("\n"):
        # Skip lines that match our skip patterns
        if should_skip_content(line):
            continue

        # Clean lines with timestamps specifically
        timestamp_match = re.match(r"(\[[\d\.]+s - [\d\.]+s\]:)(.*)", line)
        if timestamp_match:
            timestamp = timestamp_match.group(1)
            content = timestamp_match.group(2).strip()
            if content and not should_skip_content(content):
                lines.append(f"{timestamp} {clean_text_content(content)}")
        else:
            # Other non-timestamp lines that shouldn't be skipped
            if line.strip():
                lines.append(line)

    return "\n".join(lines)


def post_process_transcript(text: str) -> str:
    """Final post-processing of the cleaned transcript"""
    if not text:
        return ""

    # Process line by line
    lines = []
    for line in text.split("\n"):
        if should_skip_content(line):
            continue

        lines.append(line)

    # Deduplicate lines using shared helper
    deduped_lines = dedup_lines(lines)
    prev_line = None

        # Skip if this line is identical or very similar to the previous
        if prev_line and line.strip() == prev_line.strip():
            continue

        # Compare content after timestamps to catch near-duplicates
        if prev_line and ":" in prev_line and ":" in line:
            prev_content = prev_line[prev_line.find(":") + 1 :].strip().lower()
            current_content = line[line.find(":") + 1 :].strip().lower()

            # Skip if content is nearly identical (allowing for minor differences)
            if (
                prev_content
                and current_content
                and similar_text(prev_content, current_content) > 0.8
            ):
                continue

        deduped_lines.append(line)
        prev_line = line

    # Combine lines and ensure reasonable spacing
    result = "\n".join(deduped_lines)

    # Replace multiple newlines with a maximum of two consecutive newlines
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()
