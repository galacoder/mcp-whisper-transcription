# Troubleshooting Guide - MCP Whisper Transcription Server

This guide provides solutions to common issues you might encounter when setting up or using the MCP Whisper Transcription Server.

## 📋 Table of Contents

- [Installation Issues](#-installation-issues)
- [Configuration Problems](#-configuration-problems)
- [Runtime Errors](#-runtime-errors)
- [Performance Issues](#-performance-issues)
- [Model-Related Problems](#-model-related-problems)
- [File Format Issues](#-file-format-issues)
- [Memory and Resource Issues](#-memory-and-resource-issues)
- [Claude Desktop Integration](#-claude-desktop-integration)
- [Network and Firewall Issues](#-network-and-firewall-issues)
- [Advanced Debugging](#-advanced-debugging)

## 🔧 Installation Issues

### Poetry Installation Problems

#### Issue: `poetry: command not found`

**Symptoms**: 
```bash
poetry --version
# bash: poetry: command not found
```

**Solutions**:

1. **Check if Poetry is installed**:
   ```bash
   ls -la ~/.local/bin/poetry
   ```

2. **Reinstall Poetry**:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Add Poetry to PATH**:
   ```bash
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
   source ~/.zshrc
   ```

4. **Alternative installation**:
   ```bash
   # Via Homebrew
   brew install poetry
   ```

#### Issue: Poetry uses wrong Python version

**Symptoms**:
```bash
poetry run python --version
# Python 3.8.x (but you need 3.10+)
```

**Solutions**:

1. **Set correct Python version**:
   ```bash
   poetry env use python3.10
   # or
   poetry env use python3.11
   ```

2. **Verify Python installation**:
   ```bash
   python3.10 --version
   # If not found, install:
   brew install python@3.10
   ```

3. **Recreate virtual environment**:
   ```bash
   poetry env remove --all
   poetry install
   ```

### FFmpeg Installation Issues

#### Issue: `ffmpeg: command not found`

**Solutions**:

1. **Install via Homebrew**:
   ```bash
   brew install ffmpeg
   ```

2. **Verify installation**:
   ```bash
   ffmpeg -version
   which ffmpeg
   ```

3. **Fix PATH issues**:
   ```bash
   echo $PATH | grep -o homebrew
   # If empty, add Homebrew to PATH:
   echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc
   source ~/.zshrc
   ```

#### Issue: FFmpeg permission errors

**Symptoms**:
```bash
ffmpeg: Operation not permitted
```

**Solutions**:

1. **Remove quarantine attribute**:
   ```bash
   sudo xattr -r -d com.apple.quarantine /opt/homebrew/bin/ffmpeg
   ```

2. **Reinstall FFmpeg**:
   ```bash
   brew uninstall ffmpeg
   brew install ffmpeg
   ```

### MLX Installation Problems

#### Issue: MLX Whisper installation fails

**Symptoms**:
```bash
ERROR: Failed building wheel for mlx-whisper
```

**Solutions**:

1. **Update Python and pip**:
   ```bash
   python3 -m pip install --upgrade pip
   poetry run pip install --upgrade pip
   ```

2. **Install Xcode Command Line Tools**:
   ```bash
   xcode-select --install
   ```

3. **Force reinstall MLX packages**:
   ```bash
   poetry run pip install --force-reinstall mlx-whisper
   ```

4. **Check Apple Silicon compatibility**:
   ```bash
   uname -m
   # Should show: arm64
   ```

## ⚙️ Configuration Problems

### Environment Variables

#### Issue: Configuration not loading

**Symptoms**: Server uses default settings despite `.env` file

**Solutions**:

1. **Check `.env` file location**:
   ```bash
   ls -la .env
   # Should be in project root
   ```

2. **Verify `.env` syntax**:
   ```bash
   cat .env
   # Check for syntax errors, no spaces around =
   ```

3. **Test environment loading**:
   ```bash
   poetry run python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('DEFAULT_MODEL'))"
   ```

4. **Create proper `.env` file**:
   ```bash
   cat > .env << EOF
   DEFAULT_MODEL=mlx-community/whisper-large-v3-turbo
   OUTPUT_FORMATS=txt,md,srt
   MAX_WORKERS=4
   TEMP_DIR=./temp
   EOF
   ```

### File Permissions

#### Issue: Permission denied errors

**Symptoms**:
```bash
PermissionError: [Errno 13] Permission denied: './temp'
```

**Solutions**:

1. **Create necessary directories**:
   ```bash
   mkdir -p temp logs
   chmod 755 temp logs
   ```

2. **Fix file permissions**:
   ```bash
   chmod +x src/whisper_mcp_server.py
   chmod -R 755 .
   ```

3. **Check directory ownership**:
   ```bash
   ls -la .
   # Ensure you own the directory
   ```

## 🏃 Runtime Errors

### MCP Server Startup Issues

#### Issue: Server fails to start

**Symptoms**:
```bash
poetry run python src/whisper_mcp_server.py
# No output or immediate exit
```

**Solutions**:

1. **Check Python syntax**:
   ```bash
   poetry run python -m py_compile src/whisper_mcp_server.py
   ```

2. **Run with verbose output**:
   ```bash
   poetry run python src/whisper_mcp_server.py --verbose
   ```

3. **Check dependencies**:
   ```bash
   poetry check
   poetry install --no-dev
   ```

4. **Test FastMCP import**:
   ```bash
   poetry run python -c "import fastmcp; print('FastMCP OK')"
   ```

#### Issue: Import errors

**Symptoms**:
```bash
ModuleNotFoundError: No module named 'mlx_whisper'
```

**Solutions**:

1. **Verify virtual environment**:
   ```bash
   poetry env info
   poetry shell
   ```

2. **Reinstall dependencies**:
   ```bash
   poetry install --no-cache
   ```

3. **Check package installation**:
   ```bash
   poetry show mlx-whisper
   ```

### Tool Execution Errors

#### Issue: Tool calls fail with JSON errors

**Symptoms**:
```bash
JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**Solutions**:

1. **Check tool function signatures**:
   ```bash
   poetry run python -c "from src.whisper_mcp_server import app; print(app.list_tools())"
   ```

2. **Test tool individually**:
   ```bash
   poetry run python examples/test_tools.py
   ```

3. **Validate JSON inputs**:
   ```python
   import json
   params = {"file_path": "test.mp3"}
   print(json.dumps(params))
   ```

## 🚀 Performance Issues

### Slow Transcription

#### Issue: Transcription takes too long

**Symptoms**: Processing time much longer than expected

**Solutions**:

1. **Choose appropriate model**:
   ```bash
   # For speed, use smaller models
   export DEFAULT_MODEL=mlx-community/whisper-base-mlx
   ```

2. **Check system resources**:
   ```bash
   # Monitor during transcription
   top -pid $(pgrep -f whisper_mcp_server)
   ```

3. **Optimize settings**:
   ```bash
   # Reduce concurrent workers
   export MAX_WORKERS=2
   ```

4. **Profile the operation**:
   ```python
   import time
   start = time.time()
   # Your transcription code
   print(f"Time taken: {time.time() - start}")
   ```

### Memory Usage Issues

#### Issue: High memory consumption

**Solutions**:

1. **Use smaller models**:
   - `tiny`: ~150MB
   - `base`: ~250MB  
   - `small`: ~600MB
   - `medium`: ~1.5GB
   - `large-v3-turbo`: ~1.6GB

2. **Reduce concurrent processing**:
   ```bash
   export MAX_WORKERS=1
   ```

3. **Monitor memory usage**:
   ```bash
   # During transcription
   memory_pressure
   ```

4. **Clear model cache**:
   ```bash
   poetry run python -c "
   from src.whisper_mcp_server import app
   print(app.call_tool('clear_cache', {}))
   "
   ```

## 🤖 Model-Related Problems

### Model Download Issues

#### Issue: Model download fails

**Symptoms**:
```bash
ConnectionError: Failed to download model
```

**Solutions**:

1. **Check internet connection**:
   ```bash
   ping huggingface.co
   ```

2. **Clear corrupted cache**:
   ```bash
   rm -rf ~/.cache/huggingface/transformers/models--*whisper*
   ```

3. **Download manually**:
   ```python
   from huggingface_hub import snapshot_download
   snapshot_download("mlx-community/whisper-base-mlx")
   ```

4. **Check disk space**:
   ```bash
   df -h ~/.cache
   ```

#### Issue: Model not found

**Symptoms**:
```bash
Error: Model 'whisper-invalid' not found
```

**Solutions**:

1. **List available models**:
   ```bash
   poetry run python -c "
   from src.whisper_mcp_server import app
   print(app.call_tool('list_models', {}))
   "
   ```

2. **Use correct model ID**:
   ```bash
   # Correct format:
   mlx-community/whisper-large-v3-turbo
   # NOT: whisper-large-v3-turbo
   ```

3. **Test model availability**:
   ```python
   from huggingface_hub import list_models
   models = list_models(search="mlx-community/whisper")
   for model in models:
       print(model.modelId)
   ```

### Model Loading Errors

#### Issue: Model fails to load

**Symptoms**:
```bash
RuntimeError: Failed to load model weights
```

**Solutions**:

1. **Check model compatibility**:
   ```bash
   # Ensure using MLX-optimized models
   DEFAULT_MODEL=mlx-community/whisper-base-mlx
   ```

2. **Clear and re-download**:
   ```bash
   rm -rf ~/.cache/huggingface/transformers/models--mlx-community*
   ```

3. **Check Apple Silicon**:
   ```bash
   system_profiler SPSoftwareDataType | grep "Chip"
   # Should show Apple Silicon
   ```

## 📁 File Format Issues

### Unsupported File Formats

#### Issue: File format not supported

**Symptoms**:
```bash
Error: Unsupported format: .xyz
```

**Solutions**:

1. **Check supported formats**:
   ```bash
   poetry run python -c "
   from src.whisper_mcp_server import app
   print(app.call_tool('get_supported_formats', {}))
   "
   ```

2. **Convert to supported format**:
   ```bash
   # Convert to MP3
   ffmpeg -i input.xyz output.mp3
   
   # Convert to WAV
   ffmpeg -i input.xyz output.wav
   ```

3. **Validate file**:
   ```bash
   ffprobe input.mp3
   ```

### Corrupted Files

#### Issue: Audio/video file corruption

**Symptoms**:
```bash
Error: Invalid data found when processing input
```

**Solutions**:

1. **Test with FFmpeg**:
   ```bash
   ffmpeg -v error -i suspicious_file.mp3 -f null -
   ```

2. **Try file repair**:
   ```bash
   # Attempt to fix
   ffmpeg -i broken.mp3 -c copy fixed.mp3
   ```

3. **Check file integrity**:
   ```bash
   file suspicious_file.mp3
   ls -la suspicious_file.mp3
   ```

## 💾 Memory and Resource Issues

### Out of Memory Errors

#### Issue: Process killed due to memory

**Symptoms**:
```bash
Killed: 9
# Or: MemoryError
```

**Solutions**:

1. **Check available memory**:
   ```bash
   vm_stat | head -5
   ```

2. **Close unnecessary applications**:
   ```bash
   # Check memory-intensive processes
   top -o mem
   ```

3. **Use smaller model**:
   ```bash
   export DEFAULT_MODEL=mlx-community/whisper-tiny-mlx
   ```

4. **Process files individually**:
   ```bash
   export MAX_WORKERS=1
   ```

5. **Split large files**:
   ```bash
   # Split 3-hour file into 30-minute chunks
   ffmpeg -i large_file.mp3 -f segment -segment_time 1800 chunk_%03d.mp3
   ```

### Disk Space Issues

#### Issue: No space left on device

**Solutions**:

1. **Clear model cache**:
   ```bash
   du -sh ~/.cache/huggingface
   rm -rf ~/.cache/huggingface/transformers/models--*
   ```

2. **Clean temporary files**:
   ```bash
   rm -rf ./temp/*
   rm -rf /tmp/whisper_*
   ```

3. **Check disk usage**:
   ```bash
   df -h
   du -sh ~/.cache
   ```

## 🖥️ Claude Desktop Integration

### Configuration Issues

#### Issue: MCP server not appearing in Claude Desktop

**Symptoms**: No transcription tools available in Claude

**Solutions**:

1. **Check config file location**:
   ```bash
   # macOS
   ls -la ~/Library/Application\ Support/Claude/claude_desktop_config.json
   
   # Windows
   dir "%APPDATA%\Claude\claude_desktop_config.json"
   ```

2. **Validate JSON syntax**:
   ```bash
   python3 -m json.tool ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```

3. **Check absolute paths**:
   ```json
   {
     "mcpServers": {
       "whisper-transcription": {
         "command": "poetry",
         "args": ["run", "python", "src/whisper_mcp_server.py"],
         "cwd": "/full/absolute/path/to/mcp-whisper-transcription"
       }
     }
   }
   ```

4. **Test server manually**:
   ```bash
   cd /path/to/mcp-whisper-transcription
   poetry run python src/whisper_mcp_server.py
   ```

5. **Restart Claude Desktop** after configuration changes

#### Issue: Server starts but tools don't work

**Solutions**:

1. **Check server logs**:
   ```bash
   tail -f logs/whisper_mcp.log
   ```

2. **Test MCP protocol**:
   ```bash
   poetry run python examples/test_mcp_client.py
   ```

3. **Verify tool registration**:
   ```python
   from src.whisper_mcp_server import app
   print([tool.name for tool in app.list_tools()])
   ```

### Permission Issues

#### Issue: Claude Desktop can't access files

**Symptoms**: "Permission denied" when accessing user files

**Solutions**:

1. **Grant Full Disk Access to Claude**:
   - System Preferences → Security & Privacy → Privacy
   - Select "Full Disk Access"
   - Add Claude Desktop app

2. **Use accessible file paths**:
   ```python
   # Instead of protected directories, use:
   file_path = "~/Documents/audio.mp3"
   # Or copy files to project directory
   ```

## 🌐 Network and Firewall Issues

### Model Download Problems

#### Issue: Downloads fail due to network restrictions

**Solutions**:

1. **Configure proxy** (if needed):
   ```bash
   export HTTP_PROXY=http://proxy.company.com:8080
   export HTTPS_PROXY=http://proxy.company.com:8080
   ```

2. **Use alternative model source**:
   ```python
   # Manual download
   wget https://huggingface.co/mlx-community/whisper-base-mlx/resolve/main/pytorch_model.bin
   ```

3. **Check firewall settings**:
   ```bash
   # Test connection to Hugging Face
   curl -I https://huggingface.co
   ```

## 🔍 Advanced Debugging

### Verbose Logging

Enable detailed logging for troubleshooting:

```bash
# Set log level
export LOG_LEVEL=DEBUG

# Or modify code temporarily
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Testing Components

#### Test MLX Whisper directly:

```python
import mlx_whisper

# Test model loading
model = mlx_whisper.load_model("mlx-community/whisper-base-mlx")
print("Model loaded successfully")

# Test transcription
result = mlx_whisper.transcribe("test_audio.wav", model=model)
print(result["text"])
```

#### Test FastMCP server:

```python
from fastmcp import FastMCP
from src.whisper_mcp_server import app

# Test server creation
print("Server tools:", [tool.name for tool in app.list_tools()])
print("Server resources:", [res.uri for res in app.list_resources()])
```

#### Test file processing:

```bash
# Test FFmpeg processing
ffmpeg -i test_video.mp4 -vn -ar 16000 -ac 1 -f wav test_audio.wav

# Test Python file handling
poetry run python -c "
import os
print('File exists:', os.path.exists('test_audio.wav'))
print('File size:', os.path.getsize('test_audio.wav'))
"
```

### Performance Profiling

```python
import cProfile
import pstats

# Profile transcription function
cProfile.run('transcribe_file("test.mp3")', 'transcription_profile.stats')

# Analyze results
stats = pstats.Stats('transcription_profile.stats')
stats.sort_stats('cumulative').print_stats(20)
```

### System Information Collection

For bug reports, collect this information:

```bash
#!/bin/bash
echo "=== System Information ==="
uname -a
sw_vers

echo "=== Python Information ==="
python3 --version
poetry --version

echo "=== MLX Information ==="
poetry run python -c "import mlx; print(mlx.__version__)"

echo "=== FFmpeg Information ==="
ffmpeg -version | head -5

echo "=== Memory Information ==="
vm_stat | head -5

echo "=== Disk Space ==="
df -h

echo "=== Environment ==="
env | grep -E "(DEFAULT_MODEL|MAX_WORKERS|TEMP_DIR)"
```

## 📞 Getting Additional Help

If you're still experiencing issues:

1. **Check the FAQ** in the main README.md
2. **Search existing issues** on GitHub
3. **Create a detailed bug report** with:
   - System information (use script above)
   - Exact error messages
   - Steps to reproduce
   - Log files (if available)

4. **Include test case**:
   ```bash
   # Minimal reproduction case
   poetry run python -c "
   from src.whisper_mcp_server import app
   try:
       result = app.call_tool('transcribe_file', {'file_path': 'test.mp3'})
       print('SUCCESS:', result)
   except Exception as e:
       print('ERROR:', str(e))
       import traceback
       traceback.print_exc()
   "
   ```

Remember: Most issues are related to environment setup, file permissions, or model compatibility. Following this troubleshooting guide systematically will resolve the majority of problems.