#!/bin/bash
# Setup MCP Whisper Transcription for both Claude Code and Claude Desktop

echo "🎯 Setting up MCP Whisper Transcription Integration"
echo "=================================================="

# Get the absolute path of the current directory
PROJECT_DIR=$(pwd)

# Create configuration directory if needed
CONFIG_DIR="configurations"
mkdir -p "$CONFIG_DIR"

# 1. Create Claude Code configuration command
echo ""
echo "📝 Creating Claude Code configuration..."
cat > "$CONFIG_DIR/claude_code_setup.sh" << EOF
#!/bin/bash
# Add MCP Whisper Transcription to Claude Code with user scope

echo "Adding MCP Whisper Transcription to Claude Code..."

# Add with user scope for global availability
claude mcp add-json whisper-transcription --scope user '{
  "command": "poetry",
  "args": ["run", "python", "-m", "src.whisper_mcp_server"],
  "cwd": "$PROJECT_DIR",
  "env": {
    "PYTHONUNBUFFERED": "1"
  }
}'

echo "✅ Added to Claude Code with user scope!"
echo "Run '/mcp' in Claude Code to verify connection"
EOF

chmod +x "$CONFIG_DIR/claude_code_setup.sh"

# 2. Create Claude Desktop configuration
echo ""
echo "📝 Creating Claude Desktop configuration..."

# Detect OS for correct config path
if [[ "$OSTYPE" == "darwin"* ]]; then
    CLAUDE_CONFIG_PATH="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
    CONFIG_DIR_PATH="$HOME/Library/Application Support/Claude"
else
    CLAUDE_CONFIG_PATH="$HOME/.config/Claude/claude_desktop_config.json"
    CONFIG_DIR_PATH="$HOME/.config/Claude"
fi

# Create the Claude Desktop configuration snippet
cat > "$CONFIG_DIR/claude_desktop_config.json" << EOF
{
  "mcpServers": {
    "whisper-transcription": {
      "command": "poetry",
      "args": [
        "run",
        "python",
        "-m",
        "src.whisper_mcp_server"
      ],
      "cwd": "$PROJECT_DIR",
      "env": {
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": "$PROJECT_DIR"
      }
    }
  }
}
EOF

# 3. Create manual installation instructions
cat > "$CONFIG_DIR/INSTALLATION.md" << EOF
# MCP Whisper Transcription Installation Guide

## 🚀 Claude Code Installation

Run the following command in your terminal:
\`\`\`bash
cd $PROJECT_DIR
./configurations/claude_code_setup.sh
\`\`\`

Then verify in Claude Code by running: \`/mcp\`

## 🖥️ Claude Desktop Installation

### Option 1: Fresh Installation
If you don't have any existing MCP servers configured:

1. Create the configuration directory (if it doesn't exist):
   \`\`\`bash
   mkdir -p "$CONFIG_DIR_PATH"
   \`\`\`

2. Copy the configuration:
   \`\`\`bash
   cp $PROJECT_DIR/configurations/claude_desktop_config.json "$CLAUDE_CONFIG_PATH"
   \`\`\`

3. Restart Claude Desktop

### Option 2: Add to Existing Configuration
If you already have MCP servers configured:

1. Open your existing configuration:
   \`\`\`bash
   open "$CLAUDE_CONFIG_PATH"
   \`\`\`

2. Add this to your existing "mcpServers" section:
   \`\`\`json
   "whisper-transcription": {
     "command": "poetry",
     "args": [
       "run",
       "python",
       "-m",
       "src.whisper_mcp_server"
     ],
     "cwd": "$PROJECT_DIR",
     "env": {
       "PYTHONUNBUFFERED": "1",
       "PYTHONPATH": "$PROJECT_DIR"
     }
   }
   \`\`\`

3. Restart Claude Desktop

## ✅ Verification

### Claude Code
- Run \`/mcp\` and look for "whisper-transcription: connected"

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
   - Ensure Poetry is installed: \`curl -sSL https://install.python-poetry.org | python3 -\`
   - Run \`poetry install\` in the project directory

2. **Python not found**:
   - Make sure Python 3.11+ is installed
   - Check that poetry is in your PATH

3. **Permission issues**:
   - Ensure the project directory is accessible
   - Check file permissions on the configuration
EOF

# 4. Create a merge helper script for existing configs
cat > "$CONFIG_DIR/merge_config.py" << 'EOF'
#!/usr/bin/env python3
"""Helper script to merge whisper-transcription into existing Claude Desktop config"""

import json
import sys
import os
from pathlib import Path

def get_config_path():
    """Get the Claude Desktop config path based on OS"""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    else:
        return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"

def main():
    config_path = get_config_path()
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load existing config or create new
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✓ Found existing config at: {config_path}")
    else:
        config = {}
        print(f"✓ Creating new config at: {config_path}")
    
    # Ensure mcpServers exists
    if "mcpServers" not in config:
        config["mcpServers"] = {}
    
    # Add whisper-transcription
    config["mcpServers"]["whisper-transcription"] = {
        "command": "poetry",
        "args": ["run", "python", "-m", "src.whisper_mcp_server"],
        "cwd": project_dir,
        "env": {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": project_dir
        }
    }
    
    # Backup existing config
    if config_path.exists():
        backup_path = config_path.with_suffix('.json.backup')
        config_path.rename(backup_path)
        print(f"✓ Backed up existing config to: {backup_path}")
    
    # Write merged config
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Successfully added whisper-transcription to Claude Desktop config!")
    print(f"   Please restart Claude Desktop to activate the MCP server.")

if __name__ == "__main__":
    main()
EOF

chmod +x "$CONFIG_DIR/merge_config.py"

# 5. Display summary
echo ""
echo "✅ Setup files created successfully!"
echo ""
echo "📁 Created files in $CONFIG_DIR/:"
echo "   - claude_code_setup.sh     : Run this to add to Claude Code"
echo "   - claude_desktop_config.json: Claude Desktop configuration"
echo "   - merge_config.py          : Helper to merge with existing config"
echo "   - INSTALLATION.md          : Detailed installation instructions"
echo ""
echo "🚀 Quick Start:"
echo ""
echo "For Claude Code:"
echo "  ./configurations/claude_code_setup.sh"
echo ""
echo "For Claude Desktop (merge with existing):"
echo "  python3 ./configurations/merge_config.py"
echo ""
echo "Or manually copy the configuration from:"
echo "  ./configurations/claude_desktop_config.json"