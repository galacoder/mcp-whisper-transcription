import os
import argparse
import logging
from pathlib import Path
# Fix ffmpeg import for ffmpeg-python package
try:
    import ffmpeg
except ImportError:
    # If direct import fails, try to install it
    import subprocess
    import sys
    print("Installing ffmpeg-python package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ffmpeg-python"])
    import ffmpeg
from tqdm import tqdm
import json
from datetime import datetime, timedelta
import psutil
from typing import Optional, Dict, Any, List, Set
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import time
import humanize
import sys
import math
import re

# Import shared utilities
from whisper_utils import (
    TranscriptionStats,
    PerformanceReport,
    OutputFormatter,
    extract_audio,
    cleanup_temp_files,
    setup_logger,
    get_video_duration,
)

# Try various import patterns for mlx-whisper
try:
    # Try direct import first
    import mlx_whisper
    from mlx_whisper import transcribe as mlx_whisper_transcribe

    print("Successfully imported mlx_whisper directly")
except ImportError:
    try:
        # Try alternate import structure
        from mlx_whisper import whisper
        from mlx_whisper.whisper import transcribe as mlx_whisper_transcribe

        print("Successfully imported from mlx_whisper.whisper")
    except ImportError:
        try:
            # Use system path as a last resort
            sys.path.append(os.path.expanduser("~/.local/lib/python3.9/site-packages"))
            import mlx_whisper
            from mlx_whisper import transcribe as mlx_whisper_transcribe

            print("Successfully imported mlx_whisper from site-packages")
        except ImportError:
            try:
                # If direct import fails, try to install it
                import subprocess
                print("Installing mlx-whisper package...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "mlx-whisper"])
                # Try importing again after installation
                import mlx_whisper
                from mlx_whisper import transcribe as mlx_whisper_transcribe
                print("Successfully imported mlx_whisper after installing")
            except ImportError:
                raise ImportError(
                    "Could not import mlx-whisper. Please install it with 'pip install mlx-whisper'"
                )


class WhisperTranscriber:
    def __init__(
        self, model_name="mlx-community/whisper-large-v3-mlx", output_formats=None
    ):
        """Initialize with MLX optimized model"""
        self.base_dir = Path(__file__).parent
        self.raw_files_dir = self.base_dir / "raw_files"
        self.output_dir = self.base_dir / "transcripts"
        self.logs_dir = self.base_dir / "logs"

        # Create directories if they don't exist
        for dir_path in [self.raw_files_dir, self.output_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)

        # Set up logger first
        self.logger = setup_logger(self.logs_dir, "WhisperTranscriber")

        # Set output formats (default if none specified)
        self.output_formats = (
            set(output_formats.split(",")) if output_formats else {"txt", "md", "srt"}
        )
        self.logger.info(f"Using output formats: {', '.join(self.output_formats)}")

        # Create formatter with specified formats
        self.formatter = OutputFormatter(self.output_formats)

        # Set model and load it
        self.model_name = model_name
        self.logger.info(f"Starting transcription with model: {self.model_name}")
        self.load_model()

        # Create performance report
        self.performance_report = PerformanceReport()

        # Log successful initialization
        self.logger.info("WhisperTranscriber initialized successfully")

    def load_model(self):
        try:
            self.logger.info(f"Loading MLX Whisper model: {self.model_name}")
            print(f"\nLoading MLX Whisper model: {self.model_name}")
            pbar = tqdm(total=1, desc="Loading Model", position=0)

            if not Path(self.model_name).exists():
                print(f"Using model from HuggingFace: {self.model_name}")

            pbar.update(1)
            pbar.close()

            print(f"\n✨ Model ready!")
            print(f"  📚 Model: {self.model_name}")
            print(f"  🖥️  Device: MLX")

            self.logger.info("MLX Whisper model configured successfully")

        except Exception as e:
            self.logger.error(f"Failed to configure model: {str(e)}")
            raise

    def transcribe_audio(self, audio_path: Path, output_path: Path) -> Dict[str, Any]:
        try:
            self.logger.info(f"Starting transcription: {audio_path}")
            print(f"\nTranscribing: {audio_path.name}")

            main_pbar = tqdm(total=100, desc="Overall Progress", position=0, leave=True)
            text_pbar = tqdm(
                desc="Transcribing", position=1, leave=True, bar_format="{desc}"
            )

            try:
                main_pbar.update(10)
                main_pbar.set_description("Processing audio")
                self.logger.info(f"Processing audio file: {audio_path}")

                # Use MLX Whisper for transcription
                start_time = time.time()
                result = mlx_whisper_transcribe(
                    str(audio_path),
                    path_or_hf_repo=self.model_name,
                    language="en",  # Explicitly set language
                    task="transcribe",  # Specify task
                    temperature=0.0,  # Reduce randomness
                    no_speech_threshold=0.45,  # Adjusted silence detection (less strict)
                    condition_on_previous_text=True,  # Better context handling
                    initial_prompt="The following is a high-quality transcript. Mark silence and pauses appropriately.",  # Help set expectations
                )
                processing_time = time.time() - start_time
                self.logger.info(
                    f"Transcription completed in {processing_time:.2f} seconds"
                )

                # Convert string result to dictionary if needed
                if isinstance(result, str):
                    result = {
                        "text": result,
                        "segments": [],  # Initialize empty segments
                    }

                main_pbar.update(90)
                text_pbar.set_description(f"🎯 Transcription completed")

            except Exception as e:
                self.logger.error(f"Transcription error: {str(e)}")
                raise
            finally:
                main_pbar.close()
                text_pbar.close()

            if result:
                print("\nSaving transcription files...")
                self.logger.info(f"Saving transcription results to {output_path}")
                # Use the shared formatter to save in requested formats
                self.formatter.save_transcription(result, output_path)
                self.logger.info(f"Transcription saved successfully")
                print(f"\n✨ Transcription completed successfully!")

            return result

        except Exception as e:
            self.logger.error(f"Failed to transcribe {audio_path}: {str(e)}")
            raise

    def process_videos(self):
        video_files = self.get_video_files()
        if not video_files:
            msg = "No media files found in the raw_files directory."
            print(msg)
            self.logger.warning(msg)
            return

        processed_videos = self.get_processed_videos()
        videos_to_process = [
            video
            for video in video_files
            if self.needs_processing(video, processed_videos)
        ]

        if not videos_to_process:
            msg = "All files are up to date - no processing needed."
            print(msg)
            self.logger.info(msg)
            return

        self.logger.info(f"Found {len(videos_to_process)} files to process")
        print(f"\nFound {len(videos_to_process)} files to process")

        temp_dir = self.base_dir / "temp"
        temp_dir.mkdir(exist_ok=True)

        try:
            max_workers = min(os.cpu_count() or 1, len(videos_to_process))
            self.logger.info(f"Using {max_workers} worker threads for processing")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(
                    tqdm(
                        executor.map(
                            partial(self.process_single_file, temp_dir=temp_dir),
                            videos_to_process,
                        ),
                        total=len(videos_to_process),
                        desc="Processing videos",
                        unit="video",
                    )
                )

            # Generate and print performance report
            report = self.performance_report.generate_report()
            print(report)

            # Save report to file
            # Create reports directory if it doesn't exist
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)

            report_path = (
                reports_dir / f"performance_report_{datetime.now():%Y%m%d_%H%M%S}.txt"
            )
            with open(report_path, "w") as f:
                f.write(report)
            self.logger.info(f"Performance report saved to {report_path}")

        finally:
            # Use the shared cleanup utility
            cleanup_temp_files(temp_dir, self.logger)

    def get_processed_videos(self) -> set:
        processed = set()

        # Check for all possible output formats
        for ext in [".txt", ".md", ".json", ".srt"]:
            for file in self.output_dir.glob(f"*{ext}"):
                processed.add(file.stem)

        return processed

    def needs_processing(self, video_path: Path, processed_videos: set) -> bool:
        """Enhanced check to determine if a video needs processing"""
        video_stem = video_path.stem

        # Check if any requested format exists
        transcript_paths = []
        for ext in [".txt", ".md", ".json", ".srt"]:
            if ext[1:] in self.output_formats:
                transcript_paths.append(self.output_dir / f"{video_stem}{ext}")

        # If no transcript files exist for requested formats, video needs processing
        if not any(path.exists() for path in transcript_paths):
            self.logger.info(f"No existing transcripts found for {video_path.name}")
            return True

        # Check if video is newer than any existing transcript
        video_mtime = video_path.stat().st_mtime
        for transcript_path in transcript_paths:
            if transcript_path.exists():
                # Check if video is newer than transcript
                if video_mtime > transcript_path.stat().st_mtime:
                    self.logger.info(
                        f"Video {video_path.name} is newer than transcript {transcript_path.name}"
                    )
                    return True

                # Validate JSON content
                if transcript_path.suffix == ".json":
                    try:
                        with open(transcript_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            # Check for minimum content requirements
                            if not isinstance(data, dict):
                                self.logger.warning(
                                    f"Invalid JSON format in {transcript_path}"
                                )
                                return True
                            if not data.get("text"):
                                self.logger.warning(
                                    f"No text content in {transcript_path}"
                                )
                                return True
                            # Check for empty or suspiciously small content
                            if len(data.get("text", "").split()) < 10:
                                self.logger.warning(
                                    f"Suspiciously small transcript in {transcript_path}"
                                )
                                return True
                    except (json.JSONDecodeError, IOError) as e:
                        self.logger.warning(
                            f"Error reading transcript {transcript_path}: {e}"
                        )
                        return True

                # Validate text/markdown content
                elif transcript_path.suffix in [".txt", ".md"]:
                    try:
                        content = transcript_path.read_text(encoding="utf-8")
                        if len(content.strip().split()) < 10:
                            self.logger.warning(
                                f"Suspiciously small transcript in {transcript_path}"
                            )
                            return True
                    except IOError as e:
                        self.logger.warning(
                            f"Error reading transcript {transcript_path}: {e}"
                        )
                        return True

        # All checks passed, no processing needed
        self.logger.info(f"Skipping {video_path.name} - valid transcripts exist")
        return False

    def process_single_file(self, file_path: Path, temp_dir: Path):
        """Modified to handle chunked processing"""
        try:
            self.logger.info(f"Processing file: {file_path}")
            start_time = time.time()
            duration = get_video_duration(file_path, self.logger)

            print(f"\nProcessing: {file_path.name} ({humanize.naturaldelta(duration)})")
            self.logger.info(f"File duration: {duration:.2f} seconds")

            # Convert to audio first
            audio_path = temp_dir / f"{file_path.stem}_processed.wav"
            if not extract_audio(file_path, audio_path, self.logger):
                self.logger.error(f"Audio extraction failed for {file_path}")
                return False

            # Split into chunks if longer than 20 minutes
            chunks = self.split_audio_into_chunks(audio_path)
            if not chunks:
                self.logger.error(f"Failed to create chunks for {file_path}")
                return False

            # Process each chunk
            chunk_transcripts = []
            total_chunks = len(chunks)
            self.logger.info(f"Processing {total_chunks} chunks for {file_path.name}")

            for i, chunk in enumerate(chunks, 1):
                print(f"\nProcessing chunk {i} of {total_chunks}")
                self.logger.info(f"Processing chunk {i} of {total_chunks}: {chunk}")

                chunk_output = temp_dir / f"{file_path.stem}_chunk_{i}"

                # Process the chunk
                if self.transcribe_audio(chunk, chunk_output):
                    chunk_transcripts.append(chunk_output.with_suffix(".txt"))
                    self.logger.info(f"Chunk {i} processed successfully")
                else:
                    self.logger.warning(f"Failed to process chunk {i}")

            # Merge transcripts
            if chunk_transcripts:
                self.logger.info(f"Merging {len(chunk_transcripts)} transcript chunks")
                final_output = self.output_dir / file_path.stem
                self.merge_transcripts(chunk_transcripts, final_output)

                # Clean up chunks
                self.logger.info("Cleaning up temporary chunk files")
                for chunk in chunks:
                    if chunk != audio_path:  # Don't delete if it's the original
                        chunk.unlink(missing_ok=True)

                # Clean up chunk transcripts
                for transcript in chunk_transcripts:
                    transcript.unlink(missing_ok=True)

            # Clean up processed audio
            if audio_path.exists():
                audio_path.unlink()
                self.logger.info(f"Temporary audio file removed: {audio_path}")

            end_time = time.time()
            processing_time = end_time - start_time

            # Add stats to performance report
            words_transcribed = 0
            # Estimate word count from the merged text files
            merged_txt_path = self.output_dir / f"{file_path.stem}.txt"
            if merged_txt_path.exists():
                text_content = merged_txt_path.read_text(encoding="utf-8")
                words_transcribed = len(text_content.split())

            stats = TranscriptionStats(
                video_name=file_path.name,
                video_duration=duration,
                processing_time=processing_time,
                model_name=self.model_name,
                peak_memory=self._get_peak_memory_usage(),
                chunks_processed=total_chunks,
                words_transcribed=words_transcribed,
                processing_speed=(duration / 60) / (processing_time / 60),
                device="MLX",
                batch_size=32,
                chunk_size=30,
            )
            self.performance_report.add_stats(stats)

            self.logger.info(
                f"Completed processing {file_path.name} in {processing_time:.2f} seconds"
            )
            print(f"Completed in {humanize.naturaldelta(processing_time)}")
            return True

        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
            return False

    def _get_peak_memory_usage(self) -> float:
        """Get peak memory usage in GB"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024**3)  # Convert to GB
        except:
            return 0.0

    def get_video_files(self):
        """Get all valid video and audio files"""
        media_extensions = {
            # Video extensions
            ".mp4",
            ".mov",
            ".avi",
            ".mkv",
            ".webm",
            # Audio extensions
            ".mp3",
            ".wav",
            ".m4a",
            ".aac",
            ".flac",
        }
        valid_files = []

        self.logger.info(f"Scanning {self.raw_files_dir} for media files")
        for f in self.raw_files_dir.iterdir():
            if f.suffix.lower() in media_extensions:
                try:
                    probe = ffmpeg.probe(str(f))
                    has_audio = any(
                        stream["codec_type"] == "audio" for stream in probe["streams"]
                    )
                    if has_audio:
                        valid_files.append(f)
                        self.logger.info(f"Found valid media file: {f}")
                    else:
                        self.logger.warning(f"No audio stream found in: {f}")
                except ffmpeg.Error:
                    self.logger.warning(f"Could not read media file: {f}")

        self.logger.info(f"Found {len(valid_files)} valid media files")
        return valid_files

    def force_reprocess_video(self, video_path: Path) -> None:
        self.logger.info(f"Forcing reprocessing of {video_path}")
        for ext in [".txt", ".md", ".json", ".srt"]:
            transcript_path = self.output_dir / f"{video_path.stem}{ext}"
            if transcript_path.exists():
                transcript_path.unlink()
                self.logger.info(f"Removed existing transcript: {transcript_path}")

    def split_audio_into_chunks(
        self, audio_path: Path, chunk_duration: int = 1200
    ) -> list[Path]:
        """Split audio into 20-minute chunks (1200 seconds)"""
        try:
            duration = get_video_duration(audio_path, self.logger)
            chunks = []

            # If file is shorter than chunk_duration, return original
            if duration <= chunk_duration:
                self.logger.info(
                    f"Audio file is shorter than chunk duration, using as single chunk"
                )
                return [audio_path]

            chunk_dir = audio_path.parent / "chunks"
            chunk_dir.mkdir(exist_ok=True)

            total_chunks = math.ceil(duration / chunk_duration)
            self.logger.info(f"Splitting audio into {total_chunks} chunks")
            print(f"\nSplitting audio into {total_chunks} chunks...")

            for i in range(0, total_chunks):
                start_time = i * chunk_duration
                chunk_path = chunk_dir / f"{audio_path.stem}_chunk_{i + 1}.wav"
                self.logger.info(
                    f"Creating chunk {i + 1}/{total_chunks} at position {start_time}s"
                )

                try:
                    stream = ffmpeg.input(
                        str(audio_path), ss=start_time, t=chunk_duration
                    )
                    stream = ffmpeg.output(
                        stream, str(chunk_path), acodec="pcm_s16le", ac=1, ar="16000"
                    )
                    ffmpeg.run(stream, capture_stderr=True, overwrite_output=True)
                    chunks.append(chunk_path)
                    self.logger.info(f"Created chunk: {chunk_path}")
                except ffmpeg.Error as e:
                    self.logger.error(f"Error creating chunk {i + 1}: {str(e)}")
                    continue

            self.logger.info(f"Successfully created {len(chunks)} chunks")
            return chunks
        except Exception as e:
            self.logger.error(f"Error splitting audio: {str(e)}")
            return []

    def merge_transcripts(self, transcript_paths: list[Path], output_path: Path):
        """Merge and clean multiple transcript files into one, with all formats"""
        try:
            self.logger.info(
                f"Merging {len(transcript_paths)} transcript files to {output_path}"
            )

            # 1. First merge the raw text transcripts
            merged_text = []
            total_words = 0
            segments = []
            current_time_offset = 0.0

            for i, path in enumerate(transcript_paths, 1):
                if path.exists():
                    # Read text content
                    text = path.read_text(encoding="utf-8")

                    # Clean unwanted content from individual chunks before merging
                    text = self._clean_raw_transcription(text)

                    merged_text.append(f"[Chunk {i}]\n{text}\n")
                    self.logger.info(f"Adding chunk {i} to merged transcript")

                    # Read corresponding JSON if exists
                    json_path = path.with_suffix(".json")
                    if json_path.exists():
                        try:
                            with open(json_path, "r", encoding="utf-8") as f:
                                chunk_data = json.load(f)

                                # Clean segments text
                                if (
                                    isinstance(chunk_data, dict)
                                    and "segments" in chunk_data
                                ):
                                    for segment in chunk_data["segments"]:
                                        if "text" in segment:
                                            segment["text"] = self._clean_text_content(
                                                segment["text"]
                                            )

                                # Adjust timestamps for segments
                                if (
                                    isinstance(chunk_data, dict)
                                    and "segments" in chunk_data
                                ):
                                    for segment in chunk_data["segments"]:
                                        # Skip empty or irrelevant segments
                                        if not segment.get(
                                            "text"
                                        ) or self._should_skip_content(
                                            segment.get("text", "")
                                        ):
                                            continue

                                        segment["start"] += current_time_offset
                                        segment["end"] += current_time_offset
                                        segments.append(segment)

                                    # Update time offset for next chunk
                                    if chunk_data["segments"]:
                                        current_time_offset = chunk_data["segments"][
                                            -1
                                        ]["end"]

                                # Count words
                                if (
                                    isinstance(chunk_data, dict)
                                    and "text" in chunk_data
                                ):
                                    cleaned_text = self._clean_text_content(
                                        chunk_data["text"]
                                    )
                                    total_words += len(cleaned_text.split())
                                    self.logger.debug(
                                        f"Chunk {i} word count: {len(cleaned_text.split())}"
                                    )
                        except json.JSONDecodeError:
                            self.logger.warning(f"Could not parse JSON for chunk {i}")

            # Merge all chunks
            merged = "\n".join(merged_text)

            # Run full cleaning on the merged transcript
            cleaned = self.clean_transcript(merged)

            # Additional post-processing to ensure clean output
            cleaned = self._post_process_transcript(cleaned)

            # Create merged result dictionary
            # Create a proper chunks list that can be iterated
            chunk_objects = []
            for i, segment in enumerate(segments):
                # Skip segments with problematic content
                if self._should_skip_content(segment.get("text", "")):
                    continue

                chunk_objects.append(
                    {
                        "text": self._clean_text_content(segment.get("text", "")),
                        "timestamp": [segment.get("start", 0), segment.get("end", 0)],
                    }
                )

            # Extract text content from original txt files to create better chunk objects
            chunks_text = []
            for path in transcript_paths:
                if path.exists():
                    content = path.read_text(encoding="utf-8")
                    for line in content.split("\n"):
                        timestamp_match = re.match(
                            r"\[([\d\.]+)s - ([\d\.]+)s\]:", line
                        )
                        if timestamp_match:
                            start_time = float(timestamp_match.group(1))
                            end_time = float(timestamp_match.group(2))
                            text_content = line[line.find(":") + 1 :].strip()

                            # Skip problematic content
                            if self._should_skip_content(text_content):
                                continue

                            if text_content:  # Only add non-empty content
                                chunks_text.append(
                                    {
                                        "text": self._clean_text_content(text_content),
                                        "timestamp": [start_time, end_time],
                                    }
                                )

            # If we extracted text from timestamps, use it
            if chunks_text:
                chunk_objects = chunks_text

            # Fallback to creating chunks from segments
            if not chunk_objects:
                for i, segment in enumerate(segments):
                    # Skip problematic segments
                    if self._should_skip_content(segment.get("text", "")):
                        continue

                    chunk_objects.append(
                        {
                            "text": self._clean_text_content(segment.get("text", "")),
                            "timestamp": [
                                segment.get("start", 0),
                                segment.get("end", 0),
                            ],
                        }
                    )

            # Add the text from the original transcript if chunks are empty
            if cleaned and not any(chunk.get("text") for chunk in chunk_objects):
                # Preserve the main text even without timestamps
                chunk_objects.append(
                    {
                        "text": cleaned,
                        "timestamp": [0, 0],
                    }
                )

            merged_result = {
                "text": cleaned,
                "segments": segments,
                "words": total_words,
                "chunks": chunk_objects,
                "timestamp": datetime.now().isoformat(),
            }

            # Use formatter to save in the requested formats
            self.formatter.save_transcription(merged_result, output_path)
            self.logger.info(f"Successfully merged and saved transcripts")

            # Save original version for comparison
            original_path = output_path.with_stem(f"{output_path.stem}_original")
            original_path.write_text(merged, encoding="utf-8")
            self.logger.info(
                f"Original unprocessed transcript saved to: {original_path}"
            )

        except Exception as e:
            self.logger.error(f"Error merging transcripts: {str(e)}")
            raise

    def _should_skip_content(self, text: str) -> bool:
        """Check if content should be skipped based on various criteria"""
        # Skip empty content
        if not text or not text.strip():
            return True

        # Skip lines that likely contain noise or errors from the model
        skip_patterns = [
            r"downloading",
            r"^\s*\d{3,}\s*$",  # Just numbers (line numbers)
            r"^\s*$",  # Empty lines
            r"^\s*\[.*?EMPTY.*?\]\s*$",  # Empty markers
            r"^line\s+\d+\s*$",  # Line markers
            r"^file\s+\d+\s*$",  # File markers
        ]

        for pattern in skip_patterns:
            if re.search(pattern, text.lower()):
                return True

        return False

    def _handle_silence(self, text: str) -> str:
        """Handle silence markers in a consistent way"""
        # Replace sequences of dots (silence markers) with a proper silence notation
        if re.match(r"^\s*\.\.+\s*$", text):
            return "[silence]"

        # Also replace ellipsis at the end of sentences which might be unfinished thoughts
        text = re.sub(r"\s*\.\.+\s*$", " [unfinished thought]", text)

        # Replace ellipsis in the middle of text with a proper pause notation
        text = re.sub(r"\s*\.\.+\s*", " [pause] ", text)

        return text

    def _clean_text_content(self, text: str) -> str:
        """Apply cleaning to a text segment"""
        if not text:
            return ""

        # Handle silence and pauses
        text = self._handle_silence(text)

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

    def _clean_raw_transcription(self, text: str) -> str:
        """Clean a raw transcription chunk"""
        if not text:
            return ""

        lines = []
        for line in text.split("\n"):
            # Check if line is just dots (silence)
            if re.match(r"^\s*\.\.+\s*$", line):
                lines.append(self._handle_silence(line))
                continue

            # Skip lines that match our skip patterns
            if self._should_skip_content(line):
                continue

            # Clean lines with timestamps specifically
            timestamp_match = re.match(r"(\[[\d\.]+s - [\d\.]+s\]:)(.*)", line)
            if timestamp_match:
                timestamp = timestamp_match.group(1)
                content = timestamp_match.group(2).strip()
                if content:
                    # Handle silence in timestamped content
                    content = self._handle_silence(content)
                    if not self._should_skip_content(content):
                        lines.append(f"{timestamp} {self._clean_text_content(content)}")
            else:
                # Other non-timestamp lines that shouldn't be skipped
                if line.strip():
                    lines.append(line)

        return "\n".join(lines)

    def _post_process_transcript(self, text: str) -> str:
        """Final post-processing of the cleaned transcript"""
        if not text:
            return ""

        # Process line by line
        lines = []
        for line in text.split("\n"):
            if self._should_skip_content(line):
                continue

            lines.append(line)

        # Remove any section of 3+ consecutive empty lines
        result = []
        empty_count = 0
        for line in lines:
            if not line.strip():
                empty_count += 1
                if empty_count <= 2:  # Allow up to 2 consecutive empty lines
                    result.append(line)
            else:
                empty_count = 0
                result.append(line)

        return "\n".join(result)

    def clean_transcript(self, text: str) -> str:
        """Clean and format transcript text more thoroughly"""
        self.logger.info("Cleaning transcript text")

        # Process the text line by line for more control
        lines = []
        for line in text.split("\n"):
            # Check if line is just dots (silence)
            if re.match(r"^\s*\.\.+\s*$", line):
                lines.append(self._handle_silence(line))
                continue

            # Skip lines with known issues
            if self._should_skip_content(line):
                continue

            # Remove any lines related to the transcription completion timestamp
            if re.match(r"Transcription completed at:.*?", line):
                continue

            # Skip chunk markers but add a newline for separation
            if re.match(r"\[Chunk \d+\]", line):
                lines.append("")  # Add empty line for spacing between chunks
                continue

            # Process timestamped lines specially
            timestamp_match = re.match(r"(\[[\d\.]+s - [\d\.]+s\]:)(.*)", line)
            if timestamp_match:
                timestamp = timestamp_match.group(1)
                content = timestamp_match.group(2).strip()
                if content:
                    # Handle silence in timestamped content
                    content = self._handle_silence(content)
                    if not self._should_skip_content(content):
                        lines.append(f"{timestamp} {self._clean_text_content(content)}")
            else:
                # Other non-timestamp lines
                if line.strip():
                    # Apply silence handling before general cleaning
                    line = self._handle_silence(line)
                    lines.append(self._clean_text_content(line))

        # Remove duplicate or near-duplicate lines, even if blank lines separate them
        deduped_lines = []
        last_content_key = None  # content of last non-empty line (cleaned, lowercase)
        for line in lines:
            stripped = line.strip()

            # Always allow a single blank line but do not reset duplicate tracking
            if not stripped:
                deduped_lines.append(line)
                continue

            # Build a comparison key that ignores timestamp prefixes like "[0.0s - 1.0s]:"
            if stripped.startswith("[") and "]:" in stripped:
                key = stripped[stripped.find("]: ") + 3 :].lower()
            else:
                key = stripped.lower()

            # Skip if this line is identical or highly similar to the last non-empty line
            if last_content_key and (
                key == last_content_key or similar_text(key, last_content_key) > 0.8
            ):
                continue

            deduped_lines.append(line)
            last_content_key = key

        # Combine lines and ensure reasonable spacing
        result = "\n".join(deduped_lines)
        result = re.sub(r"\n{3,}", "\n\n", result)

        return result.strip()


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


def list_available_models():
    """Display available MLX Whisper models and their details"""
    models = {
        "mlx-community/whisper-large-v3-turbo": {
            "size": "2.5GB",
            "description": "MLX optimized turbo model (4 decoder layers)",
            "speed": "Ultra fast on Apple Silicon",
        },
    }

    print("\nAvailable MLX Whisper Models:")
    print("-" * 60)
    for model_id, info in models.items():
        print(f"\n{model_id}")
        print(f"  Size: {info['size']}")
        print(f"  Description: {info['description']}")
        print(f"  Speed: {info['speed']}")
    print("\nUsing mlx-community/whisper-large-v3-turbo for optimal speed")


def main():
    parser = argparse.ArgumentParser(description="Transcribe videos using MLX Whisper")
    parser.add_argument(
        "--model",
        default="mlx-community/whisper-large-v3-turbo",
        choices=["mlx-community/whisper-large-v3-turbo"],
        help="MLX Whisper model to use",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force reprocessing of all videos"
    )
    parser.add_argument(
        "--force-video",
        help="Force reprocessing of a specific video (filename without extension)",
    )
    parser.add_argument(
        "--output-formats",
        default="txt,md,srt",
        help="Comma-separated list of output formats (txt,md,srt,json)",
    )

    args = parser.parse_args()

    try:
        # Create logger to capture startup info
        logger = logging.getLogger("WhisperTranscriber")
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler())
        logger.info(f"Starting MLX Whisper Transcriber")

        transcriber = WhisperTranscriber(
            model_name=args.model, output_formats=args.output_formats
        )

        if args.force:
            logger.info("Forcing reprocessing of all transcripts")
            for ext in [".txt", ".md", ".json", ".srt"]:
                for f in transcriber.output_dir.glob(f"*{ext}"):
                    f.unlink()
                    logger.info(f"Removed: {f}")
            logger.info("Forced removal of all existing transcripts")

        elif args.force_video:
            video_path = transcriber.raw_files_dir / f"{args.force_video}"
            transcriber.force_reprocess_video(video_path)

        transcriber.process_videos()
        logger.info("Transcription process completed successfully")

    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
