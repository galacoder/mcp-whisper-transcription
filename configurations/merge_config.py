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
