# Setup Guide - MCP Whisper Transcription Server

This guide provides detailed installation and configuration instructions for the MCP Whisper Transcription Server.

## 📋 System Requirements

### Hardware Requirements

- **Apple Silicon Mac** (M1, M2, M3, or later)
  - Intel Macs are not supported due to MLX optimization
- **Memory**: 4GB RAM minimum, 8GB+ recommended
- **Storage**: 2GB+ free space for model cache
- **Network**: Internet connection for initial model downloads

### Software Requirements

- **macOS**: 12.0 (Monterey) or later
- **Python**: 3.10 or later
- **Command Line Tools**: Xcode Command Line Tools
- **Package Manager**: Homebrew (recommended)

## 🔧 Pre-Installation Setup

### 1. Install Xcode Command Line Tools

```bash
xcode-select --install
```

### 2. Install Homebrew (if not already installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 3. Update Homebrew

```bash
brew update && brew upgrade
```

### 4. Install Python 3.10+

Check your Python version:
```bash
python3 --version
```

If you need to install or upgrade Python:
```bash
brew install python@3.10
```

### 5. Install FFmpeg

FFmpeg is required for video file processing:
```bash
brew install ffmpeg
```

Verify installation:
```bash
ffmpeg -version
```

## 📦 Installation

### Step 1: Clone the Repository

```bash
# Create a directory for your MCP servers (optional but recommended)
mkdir -p ~/mcp-servers
cd ~/mcp-servers

# Clone the repository
git clone https://github.com/galacoder/mcp-whisper-transcription.git
cd mcp-whisper-transcription
```

### Step 2: Install Poetry

Poetry is used for dependency management:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Add Poetry to your PATH:
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

Verify installation:
```bash
poetry --version
```

### Step 3: Install Dependencies

```bash
# Install project dependencies
poetry install

# Verify installation
poetry run python src/whisper_mcp_server.py --help
```

## ⚙️ Configuration

### Environment Configuration

Create a `.env` file in the project root:

```bash
cp .env.example .env  # If .env.example exists
# OR create manually:
touch .env
```

Edit `.env` with your preferred settings:

```bash
# Model Configuration
DEFAULT_MODEL=mlx-community/whisper-large-v3-turbo
OUTPUT_FORMATS=txt,md,srt,json

# Performance Settings
MAX_WORKERS=4
TEMP_DIR=./temp

# Logging
LOG_LEVEL=INFO

# Optional: Future cloud features
# OPENAI_API_KEY=your_key_here
```

### Model Selection

Choose the appropriate model based on your needs:

| Model | Size | Memory | Speed | Best For |
|-------|------|--------|-------|----------|
| `mlx-community/whisper-tiny-mlx` | 39M | ~150MB | ~10x | Quick testing, drafts |
| `mlx-community/whisper-base-mlx` | 74M | ~250MB | ~7x | General use |
| `mlx-community/whisper-small-mlx` | 244M | ~600MB | ~5x | High quality |
| `mlx-community/whisper-medium-mlx` | 769M | ~1.5GB | ~3x | Professional work |
| `mlx-community/whisper-large-v3-mlx` | 1550M | ~3GB | ~2x | Maximum accuracy |
| `mlx-community/whisper-large-v3-turbo` | 809M | ~1.6GB | ~4x | **Recommended** |

## 🔗 Claude Desktop Integration

### Configuration File Location

Find your Claude Desktop configuration file:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

### Add MCP Server Configuration

Edit the configuration file and add the whisper-transcription server:

```json
{
  "mcpServers": {
    "whisper-transcription": {
      "command": "poetry",
      "args": ["run", "python", "src/whisper_mcp_server.py"],
      "cwd": "/absolute/path/to/mcp-whisper-transcription",
      "env": {
        "DEFAULT_MODEL": "mlx-community/whisper-large-v3-turbo",
        "MAX_WORKERS": "4"
      }
    }
  }
}
```

**Important**: Replace `/absolute/path/to/mcp-whisper-transcription` with the actual path to your installation.

### Example Full Configuration

```json
{
  "mcpServers": {
    "whisper-transcription": {
      "command": "poetry",
      "args": ["run", "python", "src/whisper_mcp_server.py"],
      "cwd": "/Users/yourusername/mcp-servers/mcp-whisper-transcription",
      "env": {
        "DEFAULT_MODEL": "mlx-community/whisper-large-v3-turbo",
        "OUTPUT_FORMATS": "txt,md,srt",
        "MAX_WORKERS": "4",
        "TEMP_DIR": "./temp"
      }
    },
    "other-mcp-server": {
      "command": "other-server-command",
      "args": ["--option", "value"]
    }
  }
}
```

## 🧪 Testing the Installation

### 1. Test MCP Server Directly

```bash
# Test server startup
poetry run python src/whisper_mcp_server.py

# Should show server information and available tools
```

### 2. Test with Sample Audio

```bash
# Create a test audio file
poetry run python examples/generate_test_audio.py

# Test transcription
poetry run python examples/usage_example.py
```

### 3. Run Test Suite

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src --cov-report=html

# Test specific functionality
poetry run pytest tests/test_mcp_tools.py -v
```

## 🐛 Troubleshooting Installation

### Common Issues

#### 1. Poetry Installation Issues

```bash
# If poetry command not found
which poetry
# Should show: /Users/yourusername/.local/bin/poetry

# If not found, check PATH
echo $PATH | grep -o "[^:]*\.local/bin[^:]*"

# Reinstall poetry if needed
curl -sSL https://install.python-poetry.org | python3 - --uninstall
curl -sSL https://install.python-poetry.org | python3 -
```

#### 2. Python Version Issues

```bash
# Check Python version
python3 --version

# If version < 3.10, install newer version
brew install python@3.11
brew link python@3.11

# Update poetry to use correct Python
poetry env use python3.11
poetry install
```

#### 3. FFmpeg Issues

```bash
# Verify FFmpeg installation
ffmpeg -version

# If not found, install
brew install ffmpeg

# If permission issues
sudo xattr -r -d com.apple.quarantine /opt/homebrew/bin/ffmpeg
```

#### 4. MLX Dependencies

```bash
# If MLX installation fails
poetry run pip install --upgrade mlx-whisper

# For Apple Silicon compatibility issues
poetry run pip install --force-reinstall mlx-whisper
```

#### 5. Memory Issues

If you encounter memory issues:

1. **Use smaller models**:
   ```bash
   export DEFAULT_MODEL=mlx-community/whisper-base-mlx
   ```

2. **Reduce concurrent workers**:
   ```bash
   export MAX_WORKERS=2
   ```

3. **Close other applications** to free up memory

#### 6. Permission Issues

```bash
# Fix file permissions
chmod +x src/whisper_mcp_server.py

# Fix directory permissions
chmod -R 755 .

# Create necessary directories
mkdir -p temp logs
```

## 🔄 Updating

### Update the Project

```bash
# Pull latest changes
git pull origin main

# Update dependencies
poetry lock --no-update
poetry install

# Run tests to verify update
poetry run pytest
```

### Update Models

Models are automatically cached. To force re-download:

```bash
# Clear model cache
poetry run python src/whisper_mcp_server.py --clear-cache

# Or manually remove cache
rm -rf ~/.cache/huggingface/transformers/models--*whisper*
```

## 🔒 Security Considerations

### File Permissions

Ensure appropriate file permissions:

```bash
# Secure configuration files
chmod 600 .env

# Secure log files
chmod 644 logs/*.log

# Secure script files
chmod 755 src/*.py
```

### Network Security

- The MCP server runs locally only
- No external network access required after model download
- Models are cached locally for privacy

### Data Privacy

- All transcription happens locally on your device
- No data is sent to external services
- Temporary files are cleaned up automatically

## 📞 Getting Help

If you encounter issues:

1. **Check the logs**: Look in the `logs/` directory for error messages
2. **Run tests**: `poetry run pytest -v` to identify specific issues
3. **Check dependencies**: `poetry check` to verify dependency integrity
4. **Consult troubleshooting**: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed solutions
5. **Open an issue**: Report bugs on the project's GitHub issues page

## 🎯 Next Steps

After successful installation:

1. **Read the API documentation**: [API.md](API.md)
2. **Explore examples**: [EXAMPLES.md](EXAMPLES.md)
3. **Learn about models**: [MODELS.md](MODELS.md)
4. **Configure for your needs**: Adjust `.env` settings
5. **Start transcribing**: Begin with small test files

Congratulations! Your MCP Whisper Transcription Server is now ready to use.