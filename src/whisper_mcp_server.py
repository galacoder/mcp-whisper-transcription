#!/usr/bin/env python3
"""
Whisper Transcription MCP Server
FastMCP-based server for audio/video transcription using MLX-optimized Whisper models
"""

from fastmcp import FastMCP
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastMCP with metadata
mcp = FastMCP(
    name="Whisper Transcription MCP",
    instructions="""
    This MCP server provides audio/video transcription using MLX-optimized Whisper models.
    Optimized for Apple Silicon devices with ultra-fast performance.
    
    Available tools:
    - transcribe_file: Transcribe a single file
    - batch_transcribe: Process multiple files
    - list_models: Show available Whisper models
    
    Supports multiple output formats: txt, md, srt, json
    """
)

# Import and Initialize Components
sys.path.append(str(Path(__file__).parent.parent))

from transcribe_mlx import WhisperTranscriber
from whisper_utils import (
    TranscriptionStats,
    PerformanceReport,
    OutputFormatter,
    setup_logger,
    cleanup_temp_files,
)

# Global instances
logger = setup_logger(Path("logs"), "WhisperMCP")
transcriber = None  # Lazy initialization
performance_report = PerformanceReport()

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
def get_transcriber(model_name: str = None) -> WhisperTranscriber:
    """Get or create WhisperTranscriber instance with lazy loading"""
    global transcriber
    model = model_name or DEFAULT_MODEL
    
    if transcriber is None or transcriber.model_name != model:
        logger.info(f"Initializing transcriber with model: {model}")
        transcriber = WhisperTranscriber(
            model_name=model,
            output_formats=DEFAULT_FORMATS
        )
    return transcriber


# Error Handling Setup
class TranscriptionError(Exception):
    """Custom exception for transcription errors"""
    pass


# Shutdown Handler
# Note: FastMCP handles cleanup automatically
# We'll clean up temp files manually when needed


# Main Entry Point
if __name__ == "__main__":
    # Check dependencies
    try:
        import mlx
        import ffmpeg
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        print(f"Error: {e}")
        print("Please install all dependencies: poetry install")
        sys.exit(1)
    
    # Run the server
    logger.info("Starting Whisper Transcription MCP Server")
    mcp.run()