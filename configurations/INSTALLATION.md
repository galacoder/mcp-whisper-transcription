# MCP Whisper Transcription Installation Guide

## 🚀 Claude Code Installation

Run the following command in your terminal:
```bash
cd /Users/sangle/Dev/action/projects/mcp-servers/tools/mcp-whisper-transcription
./configurations/claude_code_setup.sh
```

Then verify in Claude Code by running: `/mcp`

## 🖥️ Claude Desktop Installation

### Option 1: Fresh Installation
If you don't have any existing MCP servers configured:

1. Create the configuration directory (if it doesn't exist):
   ```bash
   mkdir -p "/Users/sangle/Library/Application Support/Claude"
   ```

2. Copy the configuration:
   ```bash
   cp /Users/sangle/Dev/action/projects/mcp-servers/tools/mcp-whisper-transcription/configurations/claude_desktop_config.json "/Users/sangle/Library/Application Support/Claude/claude_desktop_config.json"
   ```

3. Restart Claude Desktop

### Option 2: Add to Existing Configuration
If you already have MCP servers configured:

1. Open your existing configuration:
   ```bash
   open "/Users/sangle/Library/Application Support/Claude/claude_desktop_config.json"
   ```

2. Add this to your existing "mcpServers" section:
   ```json
   "whisper-transcription": {
     "command": "poetry",
     "args": [
       "run",
       "python",
       "-m",
       "src.whisper_mcp_server"
     ],
     "cwd": "/Users/sangle/Dev/action/projects/mcp-servers/tools/mcp-whisper-transcription",
     "env": {
       "PYTHONUNBUFFERED": "1",
       "PYTHONPATH": "/Users/sangle/Dev/action/projects/mcp-servers/tools/mcp-whisper-transcription"
     }
   }
   ```

3. Restart Claude Desktop

## ✅ Verification

### Claude Code
- Run `/mcp` and look for "whisper-transcription: connected"

### Claude Desktop
- Look for the 🔌 icon in the bottom left of the input box
- Click it to see connected MCP servers
- "whisper-transcription" should appear in the list

## 🎯 Usage

Once connected, you can:
- Drag and drop audio files for transcription
- Use commands like "transcribe this audio file"
- Long files (>5 min) automatically use async processing
- Monitor job progress with "check transcription status"

## 🔧 Troubleshooting

1. **Server not connecting**: 
   - Ensure Poetry is installed: `curl -sSL https://install.python-poetry.org | python3 -`
   - Run `poetry install` in the project directory

2. **Python not found**:
   - Make sure Python 3.11+ is installed
   - Check that poetry is in your PATH

3. **Permission issues**:
   - Ensure the project directory is accessible
   - Check file permissions on the configuration
