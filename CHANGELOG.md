# Changelog

All notable changes to the MCP Whisper Transcription Server will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation suite (README, SETUP, API, MODELS, TROUBLESHOOTING, EXAMPLES, CONTRIBUTING)
- Complete test coverage with 87% coverage rate
- Performance monitoring and reporting features
- Support for 6 different MLX-optimized Whisper models
- Concurrent processing with configurable worker threads
- Multiple output formats (txt, md, srt, json)
- File validation and processing time estimation tools

### Changed
- Upgraded to FastMCP 2.0 framework for better MCP protocol compliance
- Improved error handling and validation across all components
- Enhanced performance with MLX optimization for Apple Silicon

### Security
- All processing happens locally with no external data transmission
- Secure file handling with proper validation and sanitization

## [0.1.0] - 2024-07-14

### Added
- Initial release of MCP Whisper Transcription Server
- FastMCP-based MCP server implementation
- MLX-optimized Whisper model integration
- Support for audio and video file transcription
- Batch processing capabilities
- MCP tools for transcription operations:
  - `transcribe_file` - Single file transcription
  - `batch_transcribe` - Directory batch processing
  - `list_models` - Available model listing
  - `get_model_info` - Model information retrieval
  - `clear_cache` - Model cache management
  - `estimate_processing_time` - Processing time estimation
  - `validate_media_file` - File validation
  - `get_supported_formats` - Format compatibility checking
- MCP resources for monitoring and status:
  - `transcription://history` - Transcription history
  - `transcription://history/{id}` - Specific transcription details
  - `transcription://models` - Available models
  - `transcription://config` - Server configuration
  - `transcription://formats` - Supported formats
  - `transcription://performance` - Performance statistics
- Support for multiple input formats:
  - Audio: MP3, WAV, M4A, FLAC, OGG
  - Video: MP4, MOV, AVI, MKV, WEBM
- Support for multiple output formats:
  - TXT: Plain text with timestamps
  - MD: Markdown formatted text
  - SRT: SubRip subtitle format
  - JSON: Full transcription data with segments
- Performance optimizations for Apple Silicon (M1/M2/M3)
- Comprehensive error handling and logging
- Environment configuration via .env files
- Poetry-based dependency management
- Development tooling (Black, isort, pytest, flake8)

### Technical Details
- Python 3.10+ requirement
- FastMCP framework integration
- MLX Whisper library for Apple Silicon optimization
- FFmpeg integration for video file processing
- Async/await support throughout
- Thread-safe concurrent processing
- Comprehensive test suite with pytest
- Type hints and proper error handling
- Modular architecture with utility functions

### Supported Models
- `mlx-community/whisper-tiny-mlx` (39M parameters)
- `mlx-community/whisper-base-mlx` (74M parameters)
- `mlx-community/whisper-small-mlx` (244M parameters)
- `mlx-community/whisper-medium-mlx` (769M parameters)
- `mlx-community/whisper-large-v3-mlx` (1550M parameters)
- `mlx-community/whisper-large-v3-turbo` (809M parameters) - Recommended

### Performance Benchmarks
- Up to 10x realtime transcription speed on Apple Silicon
- Memory usage optimized for each model size
- Concurrent processing support with configurable workers
- Efficient batch processing for multiple files

---

## Version History

### [0.1.0] - 2024-07-14
- Initial release with full MCP integration
- MLX-optimized Whisper transcription
- Comprehensive tooling and documentation

---

## Planned Future Releases

### [0.2.0] - Planned
- Enhanced model management with automatic updates
- Real-time transcription capabilities
- Improved error recovery and retry mechanisms
- Additional output format support (VTT, DOCX)
- Performance optimizations and memory usage improvements

### [0.3.0] - Planned
- Speaker diarization support
- Cloud model integration options
- Advanced audio preprocessing
- Plugin architecture for custom models
- REST API interface in addition to MCP

### [1.0.0] - Planned
- Production-ready stability
- Full platform compatibility (Windows, Linux)
- Enterprise features and configuration
- Advanced analytics and reporting
- Professional support options

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for details on:
- How to submit bug reports and feature requests
- Development setup and guidelines
- Code style and testing requirements
- Pull request process

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [FastMCP](https://github.com/jlowin/fastmcp) - Modern MCP server framework
- [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) - Apple Silicon optimization
- [OpenAI Whisper](https://github.com/openai/whisper) - Original speech recognition model
- Apple MLX team for incredible performance optimizations on Apple Silicon
- The MCP community for protocol development and adoption