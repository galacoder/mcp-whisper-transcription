# MCP Whisper Transcription Server

An MCP (Model Context Protocol) server for audio/video transcription using MLX-optimized Whisper models. Optimized for Apple Silicon devices with ultra-fast performance.

## Features

- 🚀 **MLX-Optimized**: Leverages Apple Silicon for blazing-fast transcription
- 🎯 **Multiple Formats**: Supports txt, md, srt, and json output formats
- 🎬 **Video Support**: Automatically extracts audio from video files
- 📦 **Batch Processing**: Process multiple files in parallel
- 🔧 **MCP Integration**: Full MCP protocol support with tools and resources

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/galacoder/mcp-whisper-transcription.git
   cd mcp-whisper-transcription
   ```

2. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Install dependencies**:
   ```bash
   poetry install
   ```

4. **Copy environment configuration**:
   ```bash
   cp .env.example .env
   ```

## Configuration

Edit `.env` to configure:
- `DEFAULT_MODEL`: Choose from tiny, base, small, medium, large-v3, or large-v3-turbo
- `OUTPUT_FORMATS`: Comma-separated list of output formats
- `MAX_WORKERS`: Number of parallel workers for batch processing
- `TEMP_DIR`: Directory for temporary files

## Usage

### As MCP Server

Add to your Claude Code configuration:

```json
{
  "mcpServers": {
    "whisper-transcription": {
      "command": "poetry",
      "args": ["run", "python", "-m", "src.whisper_mcp_server"],
      "cwd": "/path/to/mcp-whisper-transcription"
    }
  }
}
```

### Available MCP Tools

- **transcribe_file**: Transcribe a single audio/video file
- **batch_transcribe**: Process multiple files in a directory
- **list_models**: Show available Whisper models
- **get_model_info**: Get details about a specific model
- **clear_cache**: Clear model cache
- **estimate_processing_time**: Estimate transcription time
- **validate_media_file**: Check file compatibility
- **get_supported_formats**: List supported input/output formats

### Available MCP Resources

- `transcription://history` - Recent transcriptions
- `transcription://history/{id}` - Specific transcription details
- `transcription://models` - Available models
- `transcription://config` - Current configuration
- `transcription://formats` - Supported formats
- `transcription://performance` - Performance statistics

## Development

### Running Tests
```bash
poetry run pytest
```

### Code Formatting
```bash
poetry run black .
poetry run isort .
```

### Type Checking
```bash
poetry run mypy src/
```

## Requirements

- Python 3.9+
- Apple Silicon Mac (for MLX optimization)
- ffmpeg (for video file support)

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Built with [FastMCP](https://github.com/jlowin/fastmcp)
- Powered by [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
- Original Whisper by [OpenAI](https://github.com/openai/whisper)