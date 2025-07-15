#!/usr/bin/env python3
"""
Whisper Transcription MCP Server
FastMCP-based server for audio/video transcription using MLX-optimized Whisper models
"""

from fastmcp import FastMCP
import os
import sys
import time
import json
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastMCP with metadata
mcp = FastMCP(
    name="Whisper Transcription MCP",
    instructions="""
    This MCP server provides audio/video transcription using MLX-optimized Whisper models.
    Optimized for Apple Silicon devices with ultra-fast performance.
    
    Key Features:
    - Automatic smart routing: Files >5min processed asynchronously
    - Voice Activity Detection (VAD) for faster processing
    - Hallucination prevention filters
    - Job queue for long-running transcriptions
    
    Available tools:
    - transcribe_file: Transcribe a single file (sync/async)
    - batch_transcribe: Process multiple files
    - list_models: Show available Whisper models
    - check_job_status: Check async job status
    - get_job_result: Get completed job results
    - list_jobs: List all transcription jobs
    - cancel_job: Cancel a running job
    
    Supports multiple output formats: txt, md, srt, json
    """,
)

# Track server start time for performance metrics
server_start_time = time.time()

# Import and Initialize Components
sys.path.append(str(Path(__file__).parent.parent))

from transcribe_mlx import WhisperTranscriber
from whisper_utils import (
    PerformanceReport,
    setup_logger,
    extract_audio,
    get_video_duration,
)
sys.path.append(str(Path(__file__).parent.parent))
from whisper_config import WhisperConfig
from hallucination_filter import post_process_transcription
from job_manager import JobManager, JobStatus, SmartRouter

# Global instances
logger = setup_logger(Path("logs"), "WhisperMCP")
transcriber = None  # Lazy initialization
performance_report = PerformanceReport()
job_manager = None  # Lazy initialization

# Configuration Management
# Configuration from environment
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "mlx-community/whisper-large-v3-mlx")
DEFAULT_FORMATS = os.getenv("OUTPUT_FORMATS", "txt,md,srt")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
TEMP_DIR = Path(os.getenv("TEMP_DIR", "./temp"))

# Ensure directories exist
TEMP_DIR.mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)


# Lazy Transcriber Initialization
def get_transcriber(model_name: str = None, use_vad: bool = False) -> WhisperTranscriber:
    """Get or create WhisperTranscriber instance with lazy loading"""
    global transcriber
    model = model_name or DEFAULT_MODEL

    # Check if we need to reinitialize due to VAD setting change
    reinit_needed = (transcriber is None or 
                    transcriber.model_name != model or
                    getattr(transcriber, 'use_vad', False) != use_vad)
    
    if reinit_needed:
        logger.info(f"Initializing transcriber with model: {model}, VAD: {use_vad}")
        transcriber = WhisperTranscriber(
            model_name=model, 
            output_formats=DEFAULT_FORMATS,
            use_vad=use_vad
        )
    return transcriber


# Lazy Job Manager Initialization
async def get_job_manager() -> JobManager:
    """Get or create JobManager instance with lazy loading"""
    global job_manager
    
    if job_manager is None:
        logger.info("Initializing job manager")
        job_manager = JobManager(
            jobs_dir=Path("jobs"),
            max_workers=MAX_WORKERS
        )
        
        # Set transcribe callback
        def transcribe_sync(file_path: str, options: dict) -> dict:
            """Synchronous wrapper for transcription"""
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    _transcribe_file_internal(
                        file_path=file_path,
                        **options
                    )
                )
                return result
            finally:
                loop.close()
        
        job_manager.transcribe_callback = transcribe_sync
        
        # Start the job manager
        await job_manager.start()
    
    return job_manager


# Error Handling Setup
class TranscriptionError(Exception):
    """Custom exception for transcription errors"""

    pass


# Shutdown Handler
# Note: FastMCP handles cleanup automatically
# We'll clean up temp files manually when needed


# Internal transcription function (not decorated)
async def _transcribe_file_internal(
    file_path: str,
    model: str = None,
    output_formats: str = None,
    language: str = "en",
    task: str = "transcribe",
    output_dir: str = None,
    temperature: float = 0.0,
    no_speech_threshold: float = 0.45,
    initial_prompt: str = None,
    use_vad: bool = False,
) -> dict:
    """
    Transcribe a single audio/video file using MLX Whisper.

    Args:
        file_path: Path to audio/video file (required)
        model: Whisper model to use (default: from environment)
        output_formats: Comma-separated formats (txt,md,srt,json)
        language: Language code (default: en)
        task: Task type - 'transcribe' or 'translate'
        output_dir: Directory for output files (default: same as input)
        temperature: Sampling temperature (0.0 = deterministic)
        no_speech_threshold: Silence detection threshold
        initial_prompt: Optional prompt to guide transcription style

    Returns:
        dict with:
        - text: Full transcription text
        - segments: List of timestamped segments
        - output_files: Paths to generated files
        - duration: Audio duration in seconds
        - processing_time: Time taken to transcribe
        - model_used: Name of the model used
    """
    try:
        # Validate file exists
        input_path = Path(file_path)
        if not input_path.exists():
            raise TranscriptionError(f"File not found: {file_path}")

        # Get transcriber instance
        transcriber = get_transcriber(model, use_vad)

        # Set output directory
        output_path = Path(output_dir) if output_dir else input_path.parent
        # Ensure output directory exists
        output_path.mkdir(parents=True, exist_ok=True)
        output_base = output_path / input_path.stem

        # Override output formats if specified
        original_formats = None
        if output_formats:
            # Save original formats
            if hasattr(transcriber, "formatter") and transcriber.formatter:
                original_formats = transcriber.formatter.output_formats.copy()
                transcriber.formatter.output_formats = set(output_formats.split(","))

        # Extract audio if needed (video file)
        if input_path.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv"]:
            audio_path = TEMP_DIR / f"{input_path.stem}_audio.wav"
            success = extract_audio(input_path, audio_path, logger)
            if not success:
                raise TranscriptionError(f"Failed to extract audio from video: {file_path}")
        else:
            audio_path = input_path

        # Perform transcription
        # Note: transcribe_audio currently uses fixed parameters internally
        # TODO: Extend WhisperTranscriber to accept these parameters
        if language != "en" or task != "transcribe" or temperature != 0.0:
            logger.warning(
                f"Custom transcription parameters not yet supported. "
                f"Using defaults: language=en, task=transcribe, temperature=0.0"
            )

        start_time = time.time()
        result = transcriber.transcribe_audio(audio_path, output_base)
        processing_time = time.time() - start_time

        # Get list of output files
        output_files = []
        # Use the formats we requested, not the transcriber's default
        formats_to_check = (
            output_formats.split(",") if output_formats else transcriber.formatter.output_formats
        )
        for fmt in formats_to_check:
            output_file_path = f"{output_base}.{fmt}"
            if Path(output_file_path).exists():
                output_files.append(str(output_file_path))

        # Restore original formats if they were changed
        if original_formats is not None:
            transcriber.formatter.output_formats = original_formats

        # Clean up temp audio if extracted
        if audio_path != input_path:
            audio_path.unlink()

        # Prepare result
        transcription_result = {
            "text": result.get("text", ""),
            "segments": result.get("segments", []),
            "output_files": output_files,
            "duration": get_video_duration(input_path),
            "processing_time": processing_time,
            "model_used": transcriber.model_name,
        }

        # Record transcription in history
        record_transcription(file_path, transcription_result)

        # Update performance report
        performance_report.add_file(
            transcription_result["duration"], transcription_result["processing_time"]
        )

        return transcription_result

    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}", exc_info=True)
        raise TranscriptionError(f"Transcription failed: {str(e)}")


# MCP Tools Implementation
@mcp.tool
async def transcribe_file(
    file_path: str,
    model: str = None,
    output_formats: str = None,
    language: str = "en",
    task: str = "transcribe",
    output_dir: str = None,
    temperature: float = 0.0,
    no_speech_threshold: float = 0.45,
    initial_prompt: str = None,
    use_vad: bool = False,
    force_async: bool = False,
) -> dict:
    """
    Transcribe a single audio/video file using MLX Whisper.

    Args:
        file_path: Path to audio/video file (required)
        model: Whisper model to use (default: from environment)
        output_formats: Comma-separated formats (txt,md,srt,json)
        language: Language code (default: en)
        task: Task type - 'transcribe' or 'translate'
        output_dir: Directory for output files (default: same as input)
        temperature: Sampling temperature (0.0 = deterministic)
        no_speech_threshold: Silence detection threshold
        initial_prompt: Optional prompt to guide transcription style
        use_vad: Enable Voice Activity Detection to remove silence
        force_async: Force async processing (for testing)
    Returns:
        dict with:
        - text: Full transcription text (if sync)
        - job_id: Job ID for async processing (if async)
        - message: Status message
        - segments: List of timestamped segments (if sync)
        - output_files: Paths to generated files (if sync)
        - duration: Audio duration in seconds
        - processing_time: Time taken to transcribe (if sync)
        - model_used: Name of the model used
    """
    try:
        # Get file info for routing decision
        input_path = Path(file_path)
        if not input_path.exists():
            return {"error": f"File not found: {file_path}"}
        
        # Get duration and file size
        duration = get_video_duration(input_path)
        file_size = input_path.stat().st_size
        
        # Smart routing decision
        should_async = force_async or SmartRouter.should_use_async(
            file_path, duration, file_size
        )
        
        if should_async:
            # Use async job queue
            logger.info(f"Using async processing for {file_path} (duration: {duration}s)")
            
            # Get job manager
            jm = await get_job_manager()
            
            # Prepare options
            options = {
                "model": model,
                "output_formats": output_formats,
                "language": language,
                "task": task,
                "output_dir": output_dir,
                "temperature": temperature,
                "no_speech_threshold": no_speech_threshold,
                "initial_prompt": initial_prompt,
                "use_vad": use_vad,
            }
            
            # Create job
            job_id = jm.create_job(file_path, options, duration)
            
            # Estimate processing time
            model_name = model or DEFAULT_MODEL
            estimated_time = SmartRouter.estimate_processing_time(duration, model_name)
            
            return {
                "job_id": job_id,
                "message": "Transcription job created. Use 'check_job_status' to monitor progress.",
                "estimated_time": estimated_time,
                "estimated_time_formatted": f"{int(estimated_time // 60)}m {int(estimated_time % 60)}s",
                "duration": duration,
                "file": file_path,
                "async": True
            }
        else:
            # Use synchronous processing
            logger.info(f"Using sync processing for {file_path} (duration: {duration}s)")
            
            result = await _transcribe_file_internal(
                file_path=file_path,
                model=model,
                output_formats=output_formats,
                language=language,
                task=task,
                output_dir=output_dir,
                temperature=temperature,
                no_speech_threshold=no_speech_threshold,
                initial_prompt=initial_prompt,
                use_vad=use_vad,
            )
            
            result["async"] = False
            return result
            
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}", exc_info=True)
        return {"error": f"Transcription failed: {str(e)}"}


@mcp.tool
async def batch_transcribe(
    directory: str,
    pattern: str = "*",
    recursive: bool = False,
    max_workers: int = None,
    output_formats: str = None,
    skip_existing: bool = True,
    output_dir: str = None,
) -> dict:
    """
    Batch transcribe multiple files matching pattern.

    Args:
        directory: Directory containing media files
        pattern: Glob pattern for files (e.g., "*.mp4", "audio_*.m4a")
        recursive: Search subdirectories recursively
        max_workers: Number of parallel workers (default: from env)
        output_formats: Output formats for all files
        skip_existing: Skip files that already have transcripts
        output_dir: Optional separate output directory

    Returns:
        dict with:
        - total_files: Number of files found
        - processed: Number of files processed
        - skipped: Number of files skipped
        - failed: Number of files that failed
        - results: Array of individual file results
        - performance_report: Overall performance statistics
    """
    try:
        # Find matching files
        dir_path = Path(directory)
        if not dir_path.exists():
            raise TranscriptionError(f"Directory not found: {directory}")

        if recursive:
            files = list(dir_path.rglob(pattern))
        else:
            files = list(dir_path.glob(pattern))

        # Filter to supported formats
        supported_extensions = {".mp3", ".wav", ".m4a", ".mp4", ".mov", ".avi", ".mkv"}
        media_files = [f for f in files if f.suffix.lower() in supported_extensions]

        if not media_files:
            return {
                "total_files": 0,
                "processed": 0,
                "skipped": 0,
                "failed": 0,
                "results": [],
                "performance_report": "No media files found",
            }

        # Check which files need processing
        files_to_process = []
        skipped_files = []

        for file in media_files:
            output_base = Path(output_dir) / file.stem if output_dir else file.parent / file.stem
            # Check if any output format already exists
            needs_processing = True
            if skip_existing:
                for fmt in (output_formats or DEFAULT_FORMATS).split(","):
                    if Path(f"{output_base}.{fmt}").exists():
                        needs_processing = False
                        break

            if needs_processing:
                files_to_process.append(file)
            else:
                skipped_files.append(file)

        # Process files in parallel
        results = []
        failed = []
        workers = max_workers or MAX_WORKERS

        # Create progress tracking
        logger.info(f"Processing {len(files_to_process)} files with {workers} workers")

        # Process files using asyncio gather for true async
        import asyncio

        semaphore = asyncio.Semaphore(workers)

        async def process_with_semaphore(file_path):
            async with semaphore:
                try:
                    return await _transcribe_file_internal(
                        file_path=str(file_path),
                        output_formats=output_formats,
                        output_dir=output_dir,
                    )
                except Exception as e:
                    return {"error": str(e)}

        # Process all tasks
        task_results = await asyncio.gather(
            *[process_with_semaphore(file) for file in files_to_process]
        )

        # Collect results
        for file, result in zip(files_to_process, task_results):
            if "error" in result:
                failed.append({"file": str(file), "error": result["error"]})
            else:
                results.append({"file": str(file), **result})

        # Generate performance report
        total_duration = sum(r.get("duration", 0) for r in results)
        total_processing = sum(r.get("processing_time", 0) for r in results)

        performance_summary = {
            "total_audio_duration": total_duration,
            "total_processing_time": total_processing,
            "average_speed": total_duration / total_processing if total_processing > 0 else 0,
            "files_per_minute": len(results) / (total_processing / 60)
            if total_processing > 0
            else 0,
        }

        return {
            "total_files": len(media_files),
            "processed": len(results),
            "skipped": len(skipped_files),
            "failed": len(failed),
            "results": results,
            "failed_files": failed,
            "performance_report": performance_summary,
        }

    except Exception as e:
        logger.error(f"Batch transcription failed: {str(e)}", exc_info=True)
        raise TranscriptionError(f"Batch transcription failed: {str(e)}")


# Internal functions (not decorated)
def _list_models_internal() -> dict:
    """Internal function to list all available MLX Whisper models with details."""
    models = [
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

    # Get current model
    current = None
    if transcriber:
        current = transcriber.model_name
    else:
        current = DEFAULT_MODEL

    return {
        "models": models,
        "current_model": current,
        "cache_dir": str(Path.home() / ".cache" / "huggingface"),
    }

# Support Tools
@mcp.tool
def list_models() -> dict:
    """List all available MLX Whisper models with details."""
    return _list_models_internal()


@mcp.tool
def get_model_info(model_id: str) -> dict:
    """Get detailed information about a specific Whisper model.

    Args:
        model_id: The model identifier (e.g., 'mlx-community/whisper-tiny-mlx')

    Returns:
        dict with model details including size, speed, accuracy, and requirements
    """
    model_info = {
        "mlx-community/whisper-tiny-mlx": {
            "size": "39M",
            "speed": "~10x realtime",
            "accuracy": "Good",
            "parameters": "39M",
            "memory_required": "~150MB",
            "recommended_for": "Quick transcriptions, draft quality",
        },
        "mlx-community/whisper-base-mlx": {
            "size": "74M",
            "speed": "~7x realtime",
            "accuracy": "Better",
            "parameters": "74M",
            "memory_required": "~250MB",
            "recommended_for": "Good balance of speed and accuracy",
        },
        "mlx-community/whisper-small-mlx": {
            "size": "244M",
            "speed": "~5x realtime",
            "accuracy": "Very Good",
            "parameters": "244M",
            "memory_required": "~600MB",
            "recommended_for": "High quality transcriptions",
        },
        "mlx-community/whisper-medium-mlx": {
            "size": "769M",
            "speed": "~3x realtime",
            "accuracy": "Excellent",
            "parameters": "769M",
            "memory_required": "~1.5GB",
            "recommended_for": "Professional transcriptions",
        },
        "mlx-community/whisper-large-v3-mlx": {
            "size": "1550M",
            "speed": "~2x realtime",
            "accuracy": "Best",
            "parameters": "1550M",
            "memory_required": "~3GB",
            "recommended_for": "Maximum accuracy, challenging audio",
        },
        "mlx-community/whisper-large-v3-turbo": {
            "size": "809M",
            "speed": "~4x realtime",
            "accuracy": "Excellent",
            "parameters": "809M",
            "memory_required": "~1.6GB",
            "recommended_for": "Fast high-quality transcriptions",
        },
    }

    if model_id not in model_info:
        return {
            "error": f"Model '{model_id}' not found",
            "available_models": list(model_info.keys()),
        }

    info = model_info[model_id].copy()
    info["id"] = model_id
    info["is_current"] = transcriber and transcriber.model_name == model_id

    # Check if model is cached
    cache_path = Path.home() / ".cache" / "huggingface" / "hub"
    model_cache = cache_path / f"models--{model_id.replace('/', '--')}"
    info["is_cached"] = model_cache.exists()

    return info


@mcp.tool
def clear_cache(model_id: str = None) -> dict:
    """Clear downloaded model cache.

    Args:
        model_id: Specific model to clear, or None for all models

    Returns:
        dict with cleared models and freed space
    """
    cache_path = Path.home() / ".cache" / "huggingface" / "hub"
    cleared_models = []
    total_freed = 0

    if not cache_path.exists():
        return {"message": "No cache directory found", "cleared_models": [], "freed_space": "0MB"}

    try:
        if model_id:
            # Clear specific model
            model_cache = cache_path / f"models--{model_id.replace('/', '--')}"
            if model_cache.exists():
                # Calculate size
                size = sum(f.stat().st_size for f in model_cache.rglob("*") if f.is_file())
                total_freed += size

                # Remove directory
                shutil.rmtree(model_cache)
                cleared_models.append(model_id)
            else:
                return {
                    "error": f"Model '{model_id}' not found in cache",
                    "cache_dir": str(cache_path),
                }
        else:
            # Clear all MLX Whisper models
            for model_dir in cache_path.glob("models--mlx-community--whisper-*"):
                # Calculate size
                size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
                total_freed += size

                # Extract model name
                model_name = model_dir.name.replace("models--", "").replace("--", "/")
                cleared_models.append(model_name)

                # Remove directory
                shutil.rmtree(model_dir)

        # Format size
        if total_freed > 1024 * 1024 * 1024:  # GB
            freed_str = f"{total_freed / (1024 * 1024 * 1024):.2f}GB"
        else:  # MB
            freed_str = f"{total_freed / (1024 * 1024):.2f}MB"

        return {
            "message": f"Successfully cleared {len(cleared_models)} model(s)",
            "cleared_models": cleared_models,
            "freed_space": freed_str,
        }

    except Exception as e:
        return {"error": f"Failed to clear cache: {str(e)}", "cache_dir": str(cache_path)}


@mcp.tool
def estimate_processing_time(file_path: str, model: str = None) -> dict:
    """Estimate processing time for a file.

    Args:
        file_path: Path to the audio/video file
        model: Whisper model to use (default: current model)

    Returns:
        dict with duration, estimated_time, and model_speed
    """
    try:
        # Validate file exists
        input_path = Path(file_path)
        if not input_path.exists():
            return {"error": f"File not found: {file_path}"}

        # Get file duration
        duration = get_video_duration(input_path)

        # Model speed estimates (conservative)
        model_speeds = {
            "mlx-community/whisper-tiny-mlx": 10.0,
            "mlx-community/whisper-base-mlx": 7.0,
            "mlx-community/whisper-small-mlx": 5.0,
            "mlx-community/whisper-medium-mlx": 3.0,
            "mlx-community/whisper-large-v3-mlx": 2.0,
            "mlx-community/whisper-large-v3-turbo": 4.0,
        }

        # Get model to use
        model_name = model or (transcriber.model_name if transcriber else DEFAULT_MODEL)

        # Get speed estimate
        speed = model_speeds.get(model_name, 2.0)  # Default to 2x if unknown

        # Calculate estimated time
        estimated_time = duration / speed

        # Add overhead for file loading, model initialization if needed
        overhead = 2.0  # seconds
        if not transcriber or transcriber.model_name != model_name:
            overhead += 5.0  # Model loading time

        total_time = estimated_time + overhead

        return {
            "file": str(input_path.name),
            "duration": duration,
            "duration_formatted": f"{int(duration // 60)}:{int(duration % 60):02d}",
            "model": model_name,
            "model_speed": f"{speed}x realtime",
            "estimated_time": total_time,
            "estimated_time_formatted": f"{int(total_time // 60)}:{int(total_time % 60):02d}",
            "includes_overhead": True,
            "overhead_seconds": overhead,
        }

    except Exception as e:
        return {"error": f"Failed to estimate time: {str(e)}"}


@mcp.tool
def validate_media_file(file_path: str) -> dict:
    """Validate if a file can be transcribed.

    Args:
        file_path: Path to the media file to validate

    Returns:
        dict with is_valid, format, duration, and any issues
    """
    issues = []

    try:
        # Check file exists
        input_path = Path(file_path)
        if not input_path.exists():
            return {
                "is_valid": False,
                "file": str(file_path),
                "error": "File not found",
                "issues": ["File does not exist"],
            }

        # Check file size
        file_size = input_path.stat().st_size
        if file_size == 0:
            issues.append("File is empty")

        # Check format
        suffix = input_path.suffix.lower()
        supported_audio = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
        supported_video = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        supported_all = supported_audio | supported_video

        if suffix not in supported_all:
            issues.append(f"Unsupported format: {suffix}")

        # Get file info using ffmpeg
        try:
            import subprocess

            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration,bit_rate:stream=codec_type,codec_name,channels,sample_rate",
                    "-of",
                    "json",
                    str(input_path),
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                import json

                probe_data = json.loads(result.stdout)

                # Get duration
                duration = float(probe_data.get("format", {}).get("duration", 0))

                # Check streams
                has_audio = False
                audio_info = {}

                for stream in probe_data.get("streams", []):
                    if stream.get("codec_type") == "audio":
                        has_audio = True
                        audio_info = {
                            "codec": stream.get("codec_name", "unknown"),
                            "channels": stream.get("channels", 0),
                            "sample_rate": stream.get("sample_rate", "0"),
                        }
                        break

                if not has_audio:
                    issues.append("No audio stream found")

                # Check duration limits
                if duration == 0:
                    issues.append("Unable to determine duration")
                elif duration > 3600:  # 1 hour
                    issues.append(
                        f"Very long file ({duration/3600:.1f} hours) - processing may take a while"
                    )

                return {
                    "is_valid": len(issues) == 0,
                    "file": str(input_path.name),
                    "format": suffix,
                    "file_size": f"{file_size / (1024*1024):.2f}MB",
                    "duration": duration,
                    "duration_formatted": f"{int(duration // 60)}:{int(duration % 60):02d}",
                    "audio_info": audio_info,
                    "issues": issues,
                }
            else:
                issues.append("Failed to probe file with ffmpeg")

        except Exception as e:
            issues.append(f"FFmpeg error: {str(e)}")

        return {
            "is_valid": False,
            "file": str(input_path.name),
            "format": suffix,
            "file_size": f"{file_size / (1024*1024):.2f}MB",
            "issues": issues,
        }

    except Exception as e:
        return {
            "is_valid": False,
            "file": str(file_path),
            "error": str(e),
            "issues": [f"Validation error: {str(e)}"],
        }


def _get_supported_formats_internal() -> dict:
    """Internal function to get lists of supported input and output formats."""
    return {
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


@mcp.tool
def get_supported_formats() -> dict:
    """Get lists of supported input and output formats."""
    return _get_supported_formats_internal()


@mcp.tool
async def check_job_status(job_id: str) -> dict:
    """Check the status of a transcription job.
    
    Args:
        job_id: The job ID returned by transcribe_file
        
    Returns:
        dict with job status, progress, and results if completed
    """
    jm = await get_job_manager()
    job = jm.get_job(job_id)
    
    if not job:
        return {"error": f"Job {job_id} not found"}
    
    response = {
        "job_id": job_id,
        "status": job.status.value,
        "created_at": job.created_at.isoformat(),
        "file": job.file_path,
        "progress": job.progress
    }
    
    if job.started_at:
        response["started_at"] = job.started_at.isoformat()
        
    if job.status == JobStatus.COMPLETED:
        response["completed_at"] = job.completed_at.isoformat()
        response["result"] = job.result
        # Calculate processing time
        if job.started_at and job.completed_at:
            processing_time = (job.completed_at - job.started_at).total_seconds()
            response["processing_time"] = processing_time
            
    elif job.status == JobStatus.FAILED:
        response["completed_at"] = job.completed_at.isoformat()
        response["error"] = job.error
        
    elif job.status == JobStatus.CANCELLED:
        response["completed_at"] = job.completed_at.isoformat()
        response["message"] = "Job was cancelled"
    
    return response


@mcp.tool
async def get_job_result(job_id: str) -> dict:
    """Get the transcription result for a completed job.
    
    Args:
        job_id: The job ID
        
    Returns:
        dict with transcription result or error
    """
    jm = await get_job_manager()
    job = jm.get_job(job_id)
    
    if not job:
        return {"error": f"Job {job_id} not found"}
    
    if job.status == JobStatus.COMPLETED:
        return job.result
    elif job.status == JobStatus.FAILED:
        return {"error": f"Job failed: {job.error}"}
    elif job.status == JobStatus.CANCELLED:
        return {"error": "Job was cancelled"}
    else:
        return {
            "error": f"Job not completed yet (status: {job.status.value})",
            "status": job.status.value,
            "progress": job.progress
        }


@mcp.tool
async def list_jobs(
    status: str = None,
    limit: int = 10
) -> dict:
    """List transcription jobs.
    
    Args:
        status: Filter by status (pending, running, completed, failed, cancelled)
        limit: Maximum number of jobs to return (default: 10)
        
    Returns:
        dict with list of jobs and statistics
    """
    jm = await get_job_manager()
    
    # Parse status filter
    status_filter = None
    if status:
        try:
            status_filter = JobStatus(status.lower())
        except ValueError:
            return {"error": f"Invalid status: {status}. Valid options: pending, running, completed, failed, cancelled"}
    
    # Get jobs
    jobs = jm.list_jobs(status_filter, limit)
    
    # Format jobs for response
    job_list = []
    for job in jobs:
        job_info = {
            "job_id": job.id,
            "status": job.status.value,
            "file": Path(job.file_path).name,
            "created_at": job.created_at.isoformat(),
            "progress": job.progress
        }
        
        if job.estimated_duration:
            job_info["estimated_duration"] = job.estimated_duration
            
        if job.status == JobStatus.COMPLETED and job.completed_at:
            job_info["completed_at"] = job.completed_at.isoformat()
        elif job.status == JobStatus.FAILED:
            job_info["error"] = job.error
            
        job_list.append(job_info)
    
    # Get statistics
    stats = jm.get_stats()
    
    return {
        "jobs": job_list,
        "total_returned": len(job_list),
        "statistics": stats
    }


@mcp.tool
async def cancel_job(job_id: str) -> dict:
    """Cancel a pending or running transcription job.
    
    Args:
        job_id: The job ID to cancel
        
    Returns:
        dict with cancellation result
    """
    jm = await get_job_manager()
    
    success = jm.cancel_job(job_id)
    
    if success:
        return {
            "job_id": job_id,
            "message": "Job cancelled successfully",
            "status": "cancelled"
        }
    else:
        job = jm.get_job(job_id)
        if not job:
            return {"error": f"Job {job_id} not found"}
        else:
            return {
                "error": f"Cannot cancel job in status: {job.status.value}",
                "status": job.status.value
            }


@mcp.tool
async def clean_old_jobs(days: int = 7) -> dict:
    """Remove completed/failed jobs older than specified days.
    
    Args:
        days: Number of days to keep (default: 7)
        
    Returns:
        dict with cleanup results
    """
    jm = await get_job_manager()
    
    # Get stats before cleanup
    stats_before = jm.get_stats()
    
    # Clean old jobs
    jm.clean_old_jobs(days)
    
    # Get stats after cleanup
    stats_after = jm.get_stats()
    
    cleaned = stats_before["total_jobs"] - stats_after["total_jobs"]
    
    return {
        "message": f"Cleaned {cleaned} old jobs",
        "jobs_removed": cleaned,
        "cutoff_days": days,
        "remaining_jobs": stats_after["total_jobs"]
    }


# History Management
def record_transcription(file_path: str, result: dict) -> None:
    """Record transcription in history.

    Args:
        file_path: Path to the transcribed file
        result: Transcription result dict with model_used, duration, processing_time, output_files
    """
    history_file = Path("logs") / "transcription_history.json"

    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "file_path": file_path,
        "model": result.get("model_used", "unknown"),
        "duration": result.get("duration", 0),
        "processing_time": result.get("processing_time", 0),
        "output_files": result.get("output_files", []),
    }

    # Load existing history
    if history_file.exists():
        with open(history_file) as f:
            history = json.load(f)
    else:
        history = []
        # Ensure logs directory exists
        history_file.parent.mkdir(exist_ok=True)

    # Append and save
    history.append(entry)
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)


# MCP Resource Endpoints
@mcp.resource("transcription://history")
async def get_transcription_history() -> dict:
    """Get list of recent transcriptions."""
    history_file = Path("logs") / "transcription_history.json"

    if history_file.exists():
        with open(history_file) as f:
            history = json.load(f)
    else:
        history = []

    # Return last 50 transcriptions
    return {"transcriptions": history[-50:], "total_count": len(history)}


@mcp.resource("transcription://history/{transcription_id}")
async def get_transcription_details(transcription_id: str) -> dict:
    """Get detailed information about a specific transcription."""
    history_file = Path("logs") / "transcription_history.json"

    if not history_file.exists():
        return {"error": f"Transcription {transcription_id} not found"}

    with open(history_file) as f:
        history = json.load(f)

    # Find the transcription by ID
    for entry in history:
        if entry.get("id") == transcription_id:
            return entry

    return {"error": f"Transcription {transcription_id} not found"}


@mcp.resource("transcription://models")
def get_models_resource() -> dict:
    """Resource endpoint for available models."""
    return _list_models_internal()


@mcp.resource("transcription://config")
def get_config_resource() -> dict:
    """Get current server configuration."""
    return {
        "default_model": DEFAULT_MODEL,
        "output_formats": DEFAULT_FORMATS,
        "max_workers": MAX_WORKERS,
        "temp_dir": str(TEMP_DIR),
        "version": "1.0.0",
    }


@mcp.resource("transcription://formats")
def get_formats_resource() -> dict:
    """Resource for supported formats."""
    return _get_supported_formats_internal()


@mcp.resource("transcription://performance")
async def get_performance_stats() -> dict:
    """Get server performance statistics."""
    # Calculate uptime
    uptime = time.time() - server_start_time

    # Get performance data from the report
    return {
        "total_transcriptions": getattr(performance_report, "total_files", 0),
        "total_audio_hours": getattr(performance_report, "total_duration", 0) / 3600,
        "average_speed": getattr(performance_report, "average_speed", 0),
        "uptime": uptime,
    }


# Main Entry Point
if __name__ == "__main__":
    # Check dependencies
    try:
        import mlx  # noqa: F401
        import ffmpeg  # noqa: F401
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        print(f"Error: {e}")
        print("Please install all dependencies: poetry install")
        sys.exit(1)

    # Run the server
    logger.info("Starting Whisper Transcription MCP Server")
    mcp.run()
