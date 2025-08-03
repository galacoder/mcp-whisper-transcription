#!/bin/bash
# Add MCP Whisper Transcription to Claude Code with user scope

echo "Adding MCP Whisper Transcription to Claude Code..."

# Add with user scope for global availability
claude mcp add-json whisper-transcription --scope user '{
  "command": "poetry",
  "args": ["run", "python", "-m", "src.whisper_mcp_server"],
  "cwd": "/Users/sangle/Dev/action/projects/mcp-servers/tools/mcp-whisper-transcription",
  "env": {
    "PYTHONUNBUFFERED": "1"
  }
}'

echo "✅ Added to Claude Code with user scope!"
echo "Run '/mcp' in Claude Code to verify connection"
