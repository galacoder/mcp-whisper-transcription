#!/bin/bash

# MCP Whisper Transcription Server Wrapper
# This script ensures the correct Python with dependencies is used
# Following the same pattern as working MCP servers (memos)

# Set the project directory
PROJECT_DIR="/Users/sangle/Dev/action/projects/mcp-servers/tools/mcp-whisper-transcription"

# Change to the project directory
cd "$PROJECT_DIR" || {
    echo "Error: Could not change to project directory: $PROJECT_DIR" >&2
    exit 1
}

# Set environment variables
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR"

# Use the full path to pyenv Python which has the dependencies
# This avoids Poetry virtual environment issues in Claude Desktop
exec /Users/sangle/.pyenv/shims/python3 -m src.whisper_mcp_server "$@"