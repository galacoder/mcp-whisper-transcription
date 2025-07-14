#!/usr/bin/env python3
"""
Whisper Transcription MCP Server
FastMCP-based server for audio/video transcription using MLX-optimized Whisper models
"""

from fastmcp import FastMCP
import os
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

# TODO: Import and initialize components in next task
# TODO: Add configuration management
# TODO: Implement tools and resources

if __name__ == "__main__":
    # TODO: Add dependency checks
    # TODO: Start the server
    print("Whisper MCP Server - Implementation pending")
    pass