# MCP Whisper Transcription Server

An MCP (Model Context Protocol) server for audio/video transcription using MLX-optimized Whisper models. Optimized for Apple Silicon devices with ultra-fast performance.

## ✨ Features

- 🚀 **MLX-Optimized**: Leverages Apple Silicon for blazing-fast transcription (up to 10x faster)
- 🎯 **Multiple Formats**: Supports txt, md, srt, and json output formats
- 🎬 **Video Support**: Automatically extracts audio from video files (MP4, MOV, AVI, MKV)
- 📦 **Batch Processing**: Process multiple files in parallel with configurable workers
- 🔧 **MCP Integration**: Full MCP protocol support with tools and resources
- 📊 **Performance Tracking**: Built-in performance monitoring and reporting
- 🎛️ **Flexible Models**: Choose from 6 different Whisper models (tiny to large-v3-turbo)
- 🛠️ **Error Handling**: Robust error handling and validation
- 📈 **Concurrent Processing**: Thread-safe concurrent transcription support
- 🔇 **Voice Activity Detection**: Optional VAD to remove silence and speed up processing
- 🧹 **Hallucination Prevention**: Advanced filtering to remove common transcription artifacts

## 🏆 Performance

- **Speed**: Up to 10x realtime transcription on Apple Silicon
- **Memory**: Optimized memory usage (< 500MB for most files)
- **Concurrent**: Handle multiple transcriptions simultaneously
- **Scalable**: Batch process hundreds of files efficiently

## 🚀 Quick Start

### Prerequisites

- **Apple Silicon Mac** (M1, M2, M3, or later)
- **Python 3.10+**
- **FFmpeg** (for video support)

### Installation

1. **Install FFmpeg** (if not already installed):
   ```bash
   brew install ffmpeg
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/galacoder/mcp-whisper-transcription.git
   cd mcp-whisper-transcription
   ```

3. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

4. **Install dependencies**:
   ```bash
   poetry install
   ```

5. **Test the installation**:
   ```bash
   poetry run python src/whisper_mcp_server.py --help
   ```

## 📋 Configuration

### Environment Variables

Create a `.env` file to customize settings:

```bash
# Model Configuration
DEFAULT_MODEL=mlx-community/whisper-large-v3-turbo
OUTPUT_FORMATS=txt,md,srt,json

# Performance Settings
MAX_WORKERS=4
TEMP_DIR=./temp

# Optional: API Keys for future cloud features
# OPENAI_API_KEY=your_key_here
```

### Available Models

| Model | Size | Speed | Memory | Best For |
|-------|------|-------|--------|----------|
| `whisper-tiny-mlx` | 39M | ~10x | ~150MB | Quick drafts |
| `whisper-base-mlx` | 74M | ~7x | ~250MB | Balanced performance |
| `whisper-small-mlx` | 244M | ~5x | ~600MB | High quality |
| `whisper-medium-mlx` | 769M | ~3x | ~1.5GB | Professional use |
| `whisper-large-v3-mlx` | 1550M | ~2x | ~3GB | Maximum accuracy |
| `whisper-large-v3-turbo` | 809M | ~4x | ~1.6GB | **Recommended** |

## 🔧 Usage

### Claude Desktop Integration

Add to your Claude Desktop configuration file:

```json
{
  "mcpServers": {
    "whisper-transcription": {
      "command": "poetry",
      "args": ["run", "python", "src/whisper_mcp_server.py"],
      "cwd": "/absolute/path/to/mcp-whisper-transcription"
    }
  }
}
```

**📍 Configuration File Locations:**
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

### Standalone Usage

```bash
# Run the MCP server directly
poetry run python src/whisper_mcp_server.py

# Or use the development server
poetry run python -m src.whisper_mcp_server
```

## 🛠️ Available Tools & Resources

### MCP Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `transcribe_file` | Transcribe a single audio/video file | `file_path`, `model`, `output_formats` |
| `batch_transcribe` | Process multiple files in a directory | `directory`, `pattern`, `max_workers` |
| `list_models` | Show available Whisper models | None |
| `get_model_info` | Get details about a specific model | `model_id` |
| `clear_cache` | Clear model cache | `model_id` (optional) |
| `estimate_processing_time` | Estimate transcription time | `file_path`, `model` |
| `validate_media_file` | Check file compatibility | `file_path` |
| `get_supported_formats` | List supported input/output formats | None |

### MCP Resources

| Resource | Description | Data Provided |
|----------|-------------|---------------|
| `transcription://history` | Recent transcriptions | List of all transcriptions |
| `transcription://history/{id}` | Specific transcription details | Full transcription metadata |
| `transcription://models` | Available models | Model specifications and status |
| `transcription://config` | Current configuration | Server settings and environment |
| `transcription://formats` | Supported formats | Input/output format details |
| `transcription://performance` | Performance statistics | Speed, memory, and uptime metrics |

### Quick Examples

```python
# Single file transcription
result = await client.call_tool("transcribe_file", {
    "file_path": "interview.mp4",
    "output_formats": "txt,srt",
    "model": "mlx-community/whisper-large-v3-turbo"
})

# Transcription with Voice Activity Detection
result = await client.call_tool("transcribe_file", {
    "file_path": "long_interview.mp4",
    "output_formats": "txt,srt",
    "use_vad": True  # Remove silence for faster processing
})

# Batch processing
result = await client.call_tool("batch_transcribe", {
    "directory": "./podcasts",
    "pattern": "*.mp3",
    "max_workers": 4
})

# Check supported formats
formats = await client.call_tool("get_supported_formats", {})
```

## 🧪 Development

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src --cov-report=html

# Run specific test file
poetry run pytest tests/test_mcp_tools.py -v
```

### Code Quality

```bash
# Format code
poetry run black .
poetry run isort .

# Type checking (optional)
poetry run mypy src/

# Lint code
poetry run flake8 src/
```

### Project Structure

```
mcp-whisper-transcription/
├── src/
│   └── whisper_mcp_server.py    # Main MCP server
├── tests/                       # Comprehensive test suite
├── examples/                    # Usage examples and test files
├── transcribe_mlx.py           # MLX Whisper integration
├── whisper_utils.py            # Utility functions
└── pyproject.toml              # Project configuration
```

## 📊 Performance Benchmarks

### Test Results (Apple M3 Max)

| Model | Audio Duration | Processing Time | Speed | Memory |
|-------|----------------|-----------------|-------|--------|
| tiny | 10 minutes | 1.2 minutes | 8.3x | 150MB |
| base | 10 minutes | 1.8 minutes | 5.6x | 250MB |
| small | 10 minutes | 2.5 minutes | 4.0x | 600MB |
| medium | 10 minutes | 4.2 minutes | 2.4x | 1.5GB |
| large-v3 | 10 minutes | 5.8 minutes | 1.7x | 3GB |
| large-v3-turbo | 10 minutes | 3.1 minutes | 3.2x | 1.6GB |

## 🔧 Troubleshooting

### Common Issues

1. **FFmpeg not found**
   ```bash
   brew install ffmpeg
   ```

2. **Model download slow**
   - Models are cached in `~/.cache/huggingface/`
   - First download can be slow but subsequent runs are fast

3. **Memory issues**
   - Use smaller models (tiny/base) for large files
   - Reduce `MAX_WORKERS` for concurrent processing

4. **Permission errors**
   - Ensure proper file permissions
   - Check output directory write access

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed solutions.

## 📋 Requirements

- **Python 3.10+**
- **Apple Silicon Mac** (M1, M2, M3, or later)
- **FFmpeg** (for video file support)
- **4GB+ RAM** (8GB+ recommended for large models)
- **2GB+ free disk space** (for model cache)

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 🙏 Acknowledgments

- Built with [FastMCP](https://github.com/jlowin/fastmcp) - Modern MCP server framework
- Powered by [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) - Apple Silicon optimization
- Original [Whisper](https://github.com/openai/whisper) by OpenAI - Revolutionary speech recognition
- Thanks to the MLX team at Apple for the incredible performance optimizations