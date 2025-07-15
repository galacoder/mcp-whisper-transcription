# API Reference - MCP Whisper Transcription Server

This document provides comprehensive API documentation for all MCP tools and resources provided by the Whisper Transcription Server.

## 📋 Table of Contents

- [Tools](#-tools)
  - [transcribe_file](#transcribe_file)
  - [batch_transcribe](#batch_transcribe)
  - [list_models](#list_models)
  - [get_model_info](#get_model_info)
  - [clear_cache](#clear_cache)
  - [estimate_processing_time](#estimate_processing_time)
  - [validate_media_file](#validate_media_file)
  - [get_supported_formats](#get_supported_formats)
- [Resources](#-resources)
  - [transcription://history](#transcriptionhistory)
  - [transcription://history/{id}](#transcriptionhistoryid)
  - [transcription://models](#transcriptionmodels)
  - [transcription://config](#transcriptionconfig)
  - [transcription://formats](#transcriptionformats)
  - [transcription://performance](#transcriptionperformance)
- [Error Handling](#-error-handling)
- [Rate Limits](#-rate-limits)

## 🛠️ Tools

### transcribe_file

Transcribe a single audio or video file using MLX-optimized Whisper models.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file_path` | string | ✅ | - | Path to the audio/video file |
| `model` | string | ❌ | env:DEFAULT_MODEL | Whisper model to use |
| `output_formats` | string | ❌ | env:OUTPUT_FORMATS | Comma-separated output formats |
| `language` | string | ❌ | "en" | Language code (ISO 639-1) |
| `task` | string | ❌ | "transcribe" | Task type: "transcribe" or "translate" |
| `output_dir` | string | ❌ | same as input | Directory for output files |
| `temperature` | float | ❌ | 0.0 | Sampling temperature (0.0-1.0) |
| `no_speech_threshold` | float | ❌ | 0.45 | Silence detection threshold |
| `initial_prompt` | string | ❌ | null | Optional prompt to guide transcription |

#### Example Request

```python
result = await client.call_tool("transcribe_file", {
    "file_path": "/Users/john/Documents/interview.mp4",
    "model": "mlx-community/whisper-large-v3-turbo",
    "output_formats": "txt,srt,json",
    "language": "en",
    "output_dir": "/Users/john/Documents/transcripts"
})
```

#### Response

```json
{
  "text": "Hello, this is a test transcription...",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Hello, this is a test transcription"
    }
  ],
  "output_files": [
    "/Users/john/Documents/transcripts/interview.txt",
    "/Users/john/Documents/transcripts/interview.srt",
    "/Users/john/Documents/transcripts/interview.json"
  ],
  "duration": 125.4,
  "processing_time": 31.2,
  "model_used": "mlx-community/whisper-large-v3-turbo"
}
```

#### Supported Input Formats

**Audio**: `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`
**Video**: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`

#### Supported Output Formats

- **txt**: Timestamped plain text
- **md**: Clean markdown format
- **srt**: SubRip subtitle format
- **json**: Full transcription data with segments

---

### batch_transcribe

Process multiple files in a directory with parallel execution.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `directory` | string | ✅ | - | Directory containing media files |
| `pattern` | string | ❌ | "*" | Glob pattern for file matching |
| `recursive` | boolean | ❌ | false | Search subdirectories recursively |
| `max_workers` | integer | ❌ | env:MAX_WORKERS | Number of parallel workers |
| `output_formats` | string | ❌ | env:OUTPUT_FORMATS | Output formats for all files |
| `skip_existing` | boolean | ❌ | true | Skip files with existing transcripts |
| `output_dir` | string | ❌ | null | Optional separate output directory |

#### Example Request

```python
result = await client.call_tool("batch_transcribe", {
    "directory": "/Users/john/Podcasts",
    "pattern": "*.mp3",
    "max_workers": 4,
    "output_formats": "txt,md",
    "skip_existing": true
})
```

#### Response

```json
{
  "total_files": 15,
  "processed": 12,
  "skipped": 3,
  "failed": 0,
  "results": [
    {
      "file": "/Users/john/Podcasts/episode1.mp3",
      "text": "Welcome to our podcast...",
      "duration": 3600.0,
      "processing_time": 450.2,
      "output_files": ["episode1.txt", "episode1.md"]
    }
  ],
  "failed_files": [],
  "performance_report": {
    "total_audio_duration": 43200.0,
    "total_processing_time": 5400.0,
    "average_speed": 8.0,
    "files_per_minute": 1.33
  }
}
```

---

### list_models

List all available MLX Whisper models with specifications.

#### Parameters

None.

#### Example Request

```python
result = await client.call_tool("list_models", {})
```

#### Response

```json
{
  "models": [
    {
      "id": "mlx-community/whisper-tiny-mlx",
      "size": "39M",
      "speed": "~10x",
      "accuracy": "Good"
    },
    {
      "id": "mlx-community/whisper-large-v3-turbo",
      "size": "809M",
      "speed": "~4x",
      "accuracy": "Excellent"
    }
  ],
  "current_model": "mlx-community/whisper-large-v3-turbo",
  "cache_dir": "/Users/john/.cache/huggingface"
}
```

---

### get_model_info

Get detailed information about a specific Whisper model.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_id` | string | ✅ | - | Model identifier |

#### Example Request

```python
result = await client.call_tool("get_model_info", {
    "model_id": "mlx-community/whisper-large-v3-turbo"
})
```

#### Response

```json
{
  "id": "mlx-community/whisper-large-v3-turbo",
  "size": "809M",
  "speed": "~4x realtime",
  "accuracy": "Excellent",
  "parameters": "809M",
  "memory_required": "~1.6GB",
  "recommended_for": "Fast high-quality transcriptions",
  "is_current": true,
  "is_cached": true
}
```

#### Error Response

```json
{
  "error": "Model 'invalid-model' not found",
  "available_models": [
    "mlx-community/whisper-tiny-mlx",
    "mlx-community/whisper-base-mlx"
  ]
}
```

---

### clear_cache

Clear downloaded model cache to free up disk space.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_id` | string | ❌ | null | Specific model to clear (null = all) |

#### Example Request

```python
# Clear specific model
result = await client.call_tool("clear_cache", {
    "model_id": "mlx-community/whisper-tiny-mlx"
})

# Clear all models
result = await client.call_tool("clear_cache", {})
```

#### Response

```json
{
  "message": "Successfully cleared 1 model(s)",
  "cleared_models": ["mlx-community/whisper-tiny-mlx"],
  "freed_space": "39.5MB"
}
```

---

### estimate_processing_time

Estimate processing time for a file before transcription.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file_path` | string | ✅ | - | Path to the media file |
| `model` | string | ❌ | current model | Whisper model to use |

#### Example Request

```python
result = await client.call_tool("estimate_processing_time", {
    "file_path": "/Users/john/Documents/lecture.mp4",
    "model": "mlx-community/whisper-medium-mlx"
})
```

#### Response

```json
{
  "file": "lecture.mp4",
  "duration": 3600.0,
  "duration_formatted": "60:00",
  "model": "mlx-community/whisper-medium-mlx",
  "model_speed": "3x realtime",
  "estimated_time": 1205.0,
  "estimated_time_formatted": "20:05",
  "includes_overhead": true,
  "overhead_seconds": 5.0
}
```

---

### validate_media_file

Validate if a file can be transcribed and check its properties.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file_path` | string | ✅ | - | Path to the media file |

#### Example Request

```python
result = await client.call_tool("validate_media_file", {
    "file_path": "/Users/john/Documents/audio.mp3"
})
```

#### Response (Valid File)

```json
{
  "is_valid": true,
  "file": "audio.mp3",
  "format": ".mp3",
  "file_size": "45.67MB",
  "duration": 2745.3,
  "duration_formatted": "45:45",
  "audio_info": {
    "codec": "mp3",
    "channels": 2,
    "sample_rate": "44100"
  },
  "issues": []
}
```

#### Response (Invalid File)

```json
{
  "is_valid": false,
  "file": "document.pdf",
  "format": ".pdf",
  "file_size": "2.34MB",
  "issues": [
    "Unsupported format: .pdf",
    "No audio stream found"
  ]
}
```

---

### get_supported_formats

Get lists of supported input and output formats.

#### Parameters

None.

#### Example Request

```python
result = await client.call_tool("get_supported_formats", {})
```

#### Response

```json
{
  "input_formats": {
    "audio": {
      ".mp3": "MPEG Audio Layer 3",
      ".wav": "Waveform Audio File",
      ".m4a": "MPEG-4 Audio",
      ".flac": "Free Lossless Audio Codec",
      ".ogg": "Ogg Vorbis"
    },
    "video": {
      ".mp4": "MPEG-4 Video",
      ".mov": "QuickTime Movie",
      ".avi": "Audio Video Interleave",
      ".mkv": "Matroska Video",
      ".webm": "WebM Video"
    }
  },
  "output_formats": {
    "txt": "Plain text with timestamps",
    "md": "Markdown formatted text", 
    "srt": "SubRip subtitle format",
    "json": "Full transcription data with segments"
  },
  "notes": {
    "audio_extraction": "Video files are automatically converted to audio",
    "format_detection": "File format is detected by extension",
    "quality": "All formats supported equally - quality depends on source"
  }
}
```

## 📊 Resources

### transcription://history

Get list of recent transcriptions.

#### Response

```json
{
  "transcriptions": [
    {
      "id": "uuid-12345",
      "timestamp": "2024-07-14T18:30:45",
      "file_path": "/Users/john/Documents/interview.mp4",
      "model": "mlx-community/whisper-large-v3-turbo",
      "duration": 1800.0,
      "processing_time": 450.0,
      "output_files": ["interview.txt", "interview.srt"]
    }
  ],
  "total_count": 25
}
```

### transcription://history/{id}

Get detailed information about a specific transcription.

#### Response

```json
{
  "id": "uuid-12345",
  "timestamp": "2024-07-14T18:30:45",
  "file_path": "/Users/john/Documents/interview.mp4",
  "model": "mlx-community/whisper-large-v3-turbo",
  "duration": 1800.0,
  "processing_time": 450.0,
  "output_files": [
    "/Users/john/Documents/interview.txt",
    "/Users/john/Documents/interview.srt"
  ]
}
```

#### Error Response

```json
{
  "error": "Transcription uuid-invalid not found"
}
```

### transcription://models

Get available models (same as `list_models` tool).

### transcription://config

Get current server configuration.

#### Response

```json
{
  "default_model": "mlx-community/whisper-large-v3-turbo",
  "output_formats": "txt,md,srt,json",
  "max_workers": 4,
  "temp_dir": "./temp",
  "version": "1.0.0"
}
```

### transcription://formats

Get supported formats (same as `get_supported_formats` tool).

### transcription://performance

Get server performance statistics.

#### Response

```json
{
  "total_transcriptions": 147,
  "total_audio_hours": 23.5,
  "average_speed": 4.2,
  "uptime": 86400.0
}
```

## ⚠️ Error Handling

### Error Response Format

All errors follow this format:

```json
{
  "error": "Description of the error",
  "error_code": "ERROR_CODE",
  "details": {
    "file": "problematic_file.mp3",
    "line": 123
  }
}
```

### Common Error Codes

| Error Code | Description | Resolution |
|------------|-------------|------------|
| `FILE_NOT_FOUND` | Input file doesn't exist | Check file path |
| `UNSUPPORTED_FORMAT` | File format not supported | Use supported format |
| `INSUFFICIENT_MEMORY` | Not enough memory for model | Use smaller model |
| `MODEL_NOT_FOUND` | Requested model unavailable | Check model ID |
| `TRANSCRIPTION_FAILED` | Transcription process failed | Check file integrity |
| `PERMISSION_DENIED` | No access to file/directory | Check file permissions |
| `INVALID_PARAMETERS` | Invalid parameter values | Check parameter types |

### Error Examples

#### File Not Found

```json
{
  "error": "File not found: /path/to/missing.mp3",
  "error_code": "FILE_NOT_FOUND"
}
```

#### Invalid Model

```json
{
  "error": "Model 'invalid-model' not found",
  "error_code": "MODEL_NOT_FOUND",
  "available_models": ["mlx-community/whisper-tiny-mlx"]
}
```

#### Transcription Failed

```json
{
  "error": "Transcription failed: Audio stream not found",
  "error_code": "TRANSCRIPTION_FAILED",
  "details": {
    "file": "corrupted.mp3",
    "ffmpeg_error": "Invalid data found when processing input"
  }
}
```

## 🚦 Rate Limits

### Current Limits

- **Concurrent transcriptions**: Limited by `MAX_WORKERS` setting (default: 4)
- **File size**: No hard limit (memory dependent)
- **Request rate**: No rate limiting (local server)
- **Batch size**: No limit on number of files

### Performance Guidelines

#### Optimal Settings by System

| System | Recommended MAX_WORKERS | Model Recommendation |
|--------|------------------------|----------------------|
| M1 8GB | 2-3 | base or small |
| M1 16GB+ | 4-6 | medium or large-v3-turbo |
| M2 8GB | 3-4 | small or medium |
| M2 16GB+ | 6-8 | large-v3-turbo |
| M3 8GB | 4-5 | medium or large-v3-turbo |
| M3 16GB+ | 8-10 | large-v3 or large-v3-turbo |

#### File Size Recommendations

| File Duration | Recommended Model | Expected Processing Time |
|---------------|-------------------|--------------------------|
| < 10 minutes | Any model | < 2 minutes |
| 10-60 minutes | base, small, medium | 2-20 minutes |
| 1-3 hours | small, medium | 10-60 minutes |
| > 3 hours | tiny, base (or split file) | 30+ minutes |

## 📈 Performance Optimization

### Best Practices

1. **Choose appropriate models**:
   - Use `tiny` or `base` for long files or low memory
   - Use `large-v3-turbo` for best balance of speed/quality
   - Use `large-v3` only when maximum accuracy is needed

2. **Optimize batch processing**:
   - Set `MAX_WORKERS` to match your system capabilities
   - Process smaller files first for quick feedback
   - Use `skip_existing=true` to avoid reprocessing

3. **Monitor memory usage**:
   - Close other applications during large transcriptions
   - Use smaller models if you encounter memory errors
   - Process very large files individually

4. **Storage optimization**:
   - Clear model cache periodically: `clear_cache`
   - Use appropriate output formats (txt is smallest)
   - Clean up temporary files regularly

## 🔧 Advanced Usage

### Custom Model Loading

```python
# Pre-load a specific model
await client.call_tool("get_model_info", {
    "model_id": "mlx-community/whisper-large-v3-turbo"
})

# This caches the model for faster subsequent use
```

### Batch Processing with Filtering

```python
# Process only recent files
await client.call_tool("batch_transcribe", {
    "directory": "/Users/john/Recordings",
    "pattern": "2024-*.mp3",
    "recursive": true,
    "skip_existing": true
})
```

### Performance Monitoring

```python
# Get performance stats before processing
perf_before = await client.read_resource("transcription://performance")

# Process files...

# Check performance after
perf_after = await client.read_resource("transcription://performance")
```

This completes the comprehensive API documentation. All tools and resources are documented with examples, error handling, and best practices for optimal performance.