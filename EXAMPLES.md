# Usage Examples - MCP Whisper Transcription Server

This guide provides comprehensive examples of how to use the MCP Whisper Transcription Server for various transcription tasks.

## 📋 Table of Contents

- [Basic Examples](#-basic-examples)
- [Single File Transcription](#-single-file-transcription)
- [Batch Processing](#-batch-processing)
- [Model Management](#-model-management)
- [Advanced Usage](#-advanced-usage)
- [Integration Examples](#-integration-examples)
- [Real-World Scenarios](#-real-world-scenarios)
- [Performance Optimization](#-performance-optimization)

## 🚀 Basic Examples

### Quick Start

```python
# In Claude Desktop or using FastMCP Client
async with Client(mcp) as client:
    # Simple transcription
    result = await client.call_tool("transcribe_file", {
        "file_path": "/Users/john/Documents/meeting.mp4"
    })
    print(result["text"])
```

### Check Available Tools

```python
# List all available tools
tools = await client.list_tools()
for tool in tools:
    print(f"Tool: {tool.name}")
    print(f"Description: {tool.description}")
    print("---")
```

### Check Supported Formats

```python
# Get supported input and output formats
formats = await client.call_tool("get_supported_formats", {})
print("Audio formats:", formats["input_formats"]["audio"])
print("Video formats:", formats["input_formats"]["video"])
print("Output formats:", formats["output_formats"])
```

## 📄 Single File Transcription

### Basic Audio Transcription

```python
# Transcribe an audio file with default settings
result = await client.call_tool("transcribe_file", {
    "file_path": "/Users/john/Downloads/podcast_episode.mp3"
})

print("Transcription:")
print(result["text"])
print(f"Duration: {result['duration']} seconds")
print(f"Processing time: {result['processing_time']} seconds")
print(f"Output files: {result['output_files']}")
```

### Video Transcription with Custom Settings

```python
# Transcribe video with specific model and output formats
result = await client.call_tool("transcribe_file", {
    "file_path": "/Users/john/Documents/interview.mp4",
    "model": "mlx-community/whisper-large-v3-turbo",
    "output_formats": "txt,srt,json",
    "language": "en",
    "output_dir": "/Users/john/Documents/transcripts"
})

print("Video transcribed successfully!")
print(f"Model used: {result['model_used']}")
for file in result["output_files"]:
    print(f"Created: {file}")
```

### Transcription with Custom Parameters

```python
# Advanced transcription with fine-tuned parameters
result = await client.call_tool("transcribe_file", {
    "file_path": "/Users/john/Documents/lecture.m4a",
    "model": "mlx-community/whisper-medium-mlx",
    "language": "en",
    "task": "transcribe",  # or "translate"
    "temperature": 0.0,
    "no_speech_threshold": 0.45,
    "initial_prompt": "This is a computer science lecture about machine learning."
})

print("Lecture transcription complete:")
print(result["text"][:500] + "...")
```

### Non-English Transcription

```python
# Transcribe Spanish audio
result = await client.call_tool("transcribe_file", {
    "file_path": "/Users/john/Documents/spanish_podcast.mp3",
    "model": "mlx-community/whisper-large-v3-turbo",
    "language": "es",
    "output_formats": "txt,md"
})

print("Spanish transcription:")
print(result["text"])
```

## 📦 Batch Processing

### Simple Batch Processing

```python
# Process all MP3 files in a directory
result = await client.call_tool("batch_transcribe", {
    "directory": "/Users/john/Podcasts",
    "pattern": "*.mp3",
    "max_workers": 4,
    "output_formats": "txt,srt"
})

print(f"Processed {result['processed']} out of {result['total_files']} files")
print(f"Skipped {result['skipped']} existing files")
print(f"Failed {result['failed']} files")

# Performance summary
perf = result["performance_report"]
print(f"Total audio duration: {perf['total_audio_duration']/3600:.1f} hours")
print(f"Average processing speed: {perf['average_speed']:.1f}x realtime")
```

### Recursive Directory Processing

```python
# Process all video files recursively in subdirectories
result = await client.call_tool("batch_transcribe", {
    "directory": "/Users/john/Videos",
    "pattern": "*.{mp4,mov,avi}",
    "recursive": true,
    "max_workers": 2,
    "output_formats": "md,json",
    "skip_existing": true,
    "output_dir": "/Users/john/Video_Transcripts"
})

# Show detailed results
for file_result in result["results"]:
    print(f"File: {file_result['file']}")
    print(f"Duration: {file_result['duration']/60:.1f} minutes")
    print(f"Processing time: {file_result['processing_time']/60:.1f} minutes")
    print(f"Speed: {file_result['duration']/file_result['processing_time']:.1f}x")
    print("---")
```

### Selective Processing with Patterns

```python
# Process only recent recordings (2024 files)
result = await client.call_tool("batch_transcribe", {
    "directory": "/Users/john/Recordings",
    "pattern": "2024-*.{mp3,wav,m4a}",
    "max_workers": 3,
    "model": "mlx-community/whisper-base-mlx",  # Faster for large batches
    "output_formats": "txt"
})

print("Recent recordings processed:")
for file_result in result["results"]:
    filename = file_result["file"].split("/")[-1]
    print(f"✓ {filename}")
```

## 🤖 Model Management

### List Available Models

```python
# Get all available models with specifications
models = await client.call_tool("list_models", {})

print("Available Whisper Models:")
print("=" * 50)
for model in models["models"]:
    print(f"Model: {model['id']}")
    print(f"Size: {model['size']}")
    print(f"Speed: {model['speed']}")
    print(f"Accuracy: {model['accuracy']}")
    print("---")

print(f"Current model: {models['current_model']}")
print(f"Cache directory: {models['cache_dir']}")
```

### Get Specific Model Information

```python
# Get detailed info about a specific model
model_info = await client.call_tool("get_model_info", {
    "model_id": "mlx-community/whisper-large-v3-turbo"
})

print(f"Model: {model_info['id']}")
print(f"Parameters: {model_info['parameters']}")
print(f"Memory required: {model_info['memory_required']}")
print(f"Speed: {model_info['speed']}")
print(f"Recommended for: {model_info['recommended_for']}")
print(f"Currently loaded: {model_info['is_current']}")
print(f"Cached locally: {model_info['is_cached']}")
```

### Clear Model Cache

```python
# Clear specific model from cache
result = await client.call_tool("clear_cache", {
    "model_id": "mlx-community/whisper-tiny-mlx"
})
print(f"Freed space: {result['freed_space']}")

# Clear all models from cache
result = await client.call_tool("clear_cache", {})
print(f"Cleared {len(result['cleared_models'])} models")
print(f"Total space freed: {result['freed_space']}")
```

## 🔍 Advanced Usage

### File Validation Before Processing

```python
# Validate a file before transcription
validation = await client.call_tool("validate_media_file", {
    "file_path": "/Users/john/Documents/suspicious_file.mp3"
})

if validation["is_valid"]:
    print("File is valid for transcription")
    print(f"Format: {validation['format']}")
    print(f"Duration: {validation['duration_formatted']}")
    print(f"File size: {validation['file_size']}")
    
    # Proceed with transcription
    result = await client.call_tool("transcribe_file", {
        "file_path": "/Users/john/Documents/suspicious_file.mp3"
    })
else:
    print("File cannot be transcribed:")
    for issue in validation["issues"]:
        print(f"- {issue}")
```

### Estimate Processing Time

```python
# Estimate how long transcription will take
estimate = await client.call_tool("estimate_processing_time", {
    "file_path": "/Users/john/Documents/long_meeting.mp4",
    "model": "mlx-community/whisper-medium-mlx"
})

print(f"File: {estimate['file']}")
print(f"Duration: {estimate['duration_formatted']}")
print(f"Model: {estimate['model']} ({estimate['model_speed']})")
print(f"Estimated processing time: {estimate['estimated_time_formatted']}")

# Decide whether to proceed based on estimate
if estimate["estimated_time"] > 1800:  # 30 minutes
    print("This will take a while. Consider using a faster model.")
    
    # Try with faster model
    fast_estimate = await client.call_tool("estimate_processing_time", {
        "file_path": "/Users/john/Documents/long_meeting.mp4",
        "model": "mlx-community/whisper-base-mlx"
    })
    print(f"With base model: {fast_estimate['estimated_time_formatted']}")
```

### Multi-Language Processing

```python
# Process files in different languages
languages = [
    {"file": "english_meeting.mp3", "lang": "en", "name": "English"},
    {"file": "spanish_interview.mp3", "lang": "es", "name": "Spanish"},
    {"file": "french_lecture.mp4", "lang": "fr", "name": "French"},
    {"file": "german_podcast.mp3", "lang": "de", "name": "German"}
]

for item in languages:
    print(f"Processing {item['name']} file...")
    
    result = await client.call_tool("transcribe_file", {
        "file_path": f"/Users/john/MultiLang/{item['file']}",
        "language": item["lang"],
        "model": "mlx-community/whisper-large-v3-turbo",  # Best for multilingual
        "output_formats": "txt,md"
    })
    
    print(f"✓ {item['name']} transcription complete")
    print(f"Duration: {result['duration']/60:.1f} minutes")
    print("Preview:", result["text"][:100] + "...")
    print("---")
```

## 📊 Integration Examples

### Using Resources for Monitoring

```python
# Check transcription history
history = await client.read_resource("transcription://history")
print(f"Total transcriptions: {len(history['transcriptions'])}")

for trans in history["transcriptions"][:5]:  # Show last 5
    print(f"File: {trans['file_path'].split('/')[-1]}")
    print(f"Time: {trans['timestamp']}")
    print(f"Model: {trans['model']}")
    print(f"Duration: {trans['duration']/60:.1f} min")
    print("---")

# Get performance statistics
perf = await client.read_resource("transcription://performance")
print(f"Server uptime: {perf['uptime']/3600:.1f} hours")
print(f"Total transcriptions: {perf['total_transcriptions']}")
print(f"Total audio processed: {perf['total_audio_hours']:.1f} hours")
print(f"Average speed: {perf['average_speed']:.1f}x realtime")
```

### Configuration and Status Checking

```python
# Check current server configuration
config = await client.read_resource("transcription://config")
print("Current Configuration:")
print(f"Default model: {config['default_model']}")
print(f"Output formats: {config['output_formats']}")
print(f"Max workers: {config['max_workers']}")
print(f"Temp directory: {config['temp_dir']}")
print(f"Server version: {config['version']}")

# Get available models (via resource)
models = await client.read_resource("transcription://models")
print(f"\\nAvailable models: {len(models['models'])}")
```

## 🌍 Real-World Scenarios

### Podcast Production Workflow

```python
# Complete podcast processing pipeline
podcast_dir = "/Users/john/PodcastProduction"
episode_files = ["raw_recording.wav", "intro.mp3", "outro.mp3"]

print("=== Podcast Production Workflow ===")

# 1. Validate all files first
valid_files = []
for file in episode_files:
    validation = await client.call_tool("validate_media_file", {
        "file_path": f"{podcast_dir}/{file}"
    })
    
    if validation["is_valid"]:
        valid_files.append(file)
        print(f"✓ {file} - Valid ({validation['duration_formatted']})")
    else:
        print(f"✗ {file} - Issues: {', '.join(validation['issues'])}")

# 2. Batch process valid files
if valid_files:
    result = await client.call_tool("batch_transcribe", {
        "directory": podcast_dir,
        "pattern": "*.{wav,mp3}",
        "model": "mlx-community/whisper-large-v3-turbo",
        "output_formats": "txt,srt,md",
        "max_workers": 2
    })
    
    print(f"\\nProcessed {result['processed']} files")
    print(f"Total processing time: {result['performance_report']['total_processing_time']/60:.1f} minutes")

# 3. Generate final transcript
print("\\n=== Final Transcripts ===")
for file in valid_files:
    try:
        with open(f"{podcast_dir}/{file.replace('.wav', '.md').replace('.mp3', '.md')}", 'r') as f:
            content = f.read()
            print(f"\\n{file} transcript preview:")
            print(content[:200] + "...")
    except FileNotFoundError:
        print(f"Transcript for {file} not found")
```

### Meeting Transcription and Summary

```python
# Process meeting recordings with detailed analysis
meeting_file = "/Users/john/Meetings/team_meeting_2024-07-14.mp4"

print("=== Meeting Transcription Workflow ===")

# 1. Estimate processing time
estimate = await client.call_tool("estimate_processing_time", {
    "file_path": meeting_file,
    "model": "mlx-community/whisper-large-v3-turbo"
})

print(f"Meeting duration: {estimate['duration_formatted']}")
print(f"Estimated processing: {estimate['estimated_time_formatted']}")

# 2. Transcribe with optimized settings for meetings
result = await client.call_tool("transcribe_file", {
    "file_path": meeting_file,
    "model": "mlx-community/whisper-large-v3-turbo",
    "output_formats": "txt,json,srt",
    "language": "en",
    "initial_prompt": "This is a team meeting discussion about project planning and updates."
})

print(f"\\nTranscription completed in {result['processing_time']/60:.1f} minutes")
print(f"Speed: {result['duration']/result['processing_time']:.1f}x realtime")

# 3. Extract meeting segments (using JSON output with timestamps)
import json
with open(result["output_files"][2], 'r') as f:  # JSON file
    transcript_data = json.load(f)

print("\\n=== Meeting Segments ===")
for i, segment in enumerate(transcript_data["segments"][:10]):  # First 10 segments
    start_min = int(segment["start"] // 60)
    start_sec = int(segment["start"] % 60)
    print(f"[{start_min:02d}:{start_sec:02d}] {segment['text']}")
```

### Academic Research Processing

```python
# Process academic interviews and lectures
research_dir = "/Users/john/Research/Interviews"

print("=== Academic Research Processing ===")

# 1. Process all interview files
result = await client.call_tool("batch_transcribe", {
    "directory": research_dir,
    "pattern": "interview_*.{mp3,wav,m4a}",
    "model": "mlx-community/whisper-large-v3-turbo",  # High accuracy for research
    "output_formats": "txt,json",
    "max_workers": 2,
    "skip_existing": True
})

print(f"Processed {result['processed']} interview files")

# 2. Generate research summary
interview_summaries = []
for file_result in result["results"]:
    filename = file_result["file"].split("/")[-1]
    duration_min = file_result["duration"] / 60
    word_count = len(file_result["text"].split())
    
    interview_summaries.append({
        "file": filename,
        "duration_minutes": duration_min,
        "word_count": word_count,
        "words_per_minute": word_count / duration_min,
        "preview": file_result["text"][:200]
    })

print("\\n=== Interview Analysis ===")
for summary in interview_summaries:
    print(f"File: {summary['file']}")
    print(f"Duration: {summary['duration_minutes']:.1f} minutes")
    print(f"Word count: {summary['word_count']}")
    print(f"Speech rate: {summary['words_per_minute']:.1f} words/min")
    print(f"Preview: {summary['preview']}...")
    print("---")

# 3. Calculate research corpus statistics
total_duration = sum(s["duration_minutes"] for s in interview_summaries)
total_words = sum(s["word_count"] for s in interview_summaries)
print(f"\\nCorpus Statistics:")
print(f"Total interviews: {len(interview_summaries)}")
print(f"Total duration: {total_duration/60:.1f} hours")
print(f"Total words: {total_words:,}")
print(f"Average interview length: {total_duration/len(interview_summaries):.1f} minutes")
```

## ⚡ Performance Optimization

### Memory-Efficient Large File Processing

```python
# Process very large files efficiently
large_file = "/Users/john/Recordings/conference_day1.mp4"  # 4 hours

print("=== Large File Processing ===")

# 1. Check file and estimate requirements
validation = await client.call_tool("validate_media_file", {
    "file_path": large_file
})

if validation["duration"] > 7200:  # More than 2 hours
    print("Large file detected - using memory-efficient approach")
    
    # Use smaller, faster model for long files
    model = "mlx-community/whisper-base-mlx"
    print(f"Using {model} for better memory efficiency")
else:
    model = "mlx-community/whisper-large-v3-turbo"

# 2. Process with optimized settings
result = await client.call_tool("transcribe_file", {
    "file_path": large_file,
    "model": model,
    "output_formats": "txt,srt",  # Skip JSON to save memory
    "language": "en"
})

print(f"Large file processed successfully!")
print(f"Duration: {result['duration']/3600:.1f} hours")
print(f"Processing time: {result['processing_time']/3600:.1f} hours")
print(f"Speed ratio: {result['duration']/result['processing_time']:.1f}x")
```

### Concurrent Processing with Resource Monitoring

```python
# Monitor system resources during batch processing
import asyncio

async def monitor_performance():
    """Monitor performance during processing"""
    while True:
        perf = await client.read_resource("transcription://performance")
        print(f"Active transcriptions: checking...")
        await asyncio.sleep(30)  # Check every 30 seconds

# Start monitoring in background
monitor_task = asyncio.create_task(monitor_performance())

# Process multiple directories concurrently
directories = [
    "/Users/john/Podcasts",
    "/Users/john/Meetings", 
    "/Users/john/Lectures"
]

batch_tasks = []
for directory in directories:
    task = client.call_tool("batch_transcribe", {
        "directory": directory,
        "pattern": "*.{mp3,mp4,wav}",
        "max_workers": 2,  # Conservative for parallel processing
        "model": "mlx-community/whisper-small-mlx",  # Balance speed/quality
        "output_formats": "txt",
        "skip_existing": True
    })
    batch_tasks.append(task)

# Wait for all batches to complete
results = await asyncio.gather(*batch_tasks)

# Stop monitoring
monitor_task.cancel()

# Summarize results
total_processed = sum(r["processed"] for r in results)
total_failed = sum(r["failed"] for r in results)
print(f"\\nBatch Processing Complete:")
print(f"Total files processed: {total_processed}")
print(f"Total files failed: {total_failed}")
```

### Model Switching Strategy

```python
# Dynamic model selection based on file characteristics
files_to_process = [
    "/Users/john/test_short.mp3",     # 5 minutes
    "/Users/john/medium_meeting.mp4", # 45 minutes  
    "/Users/john/long_lecture.wav"    # 2.5 hours
]

for file_path in files_to_process:
    # 1. Validate and get file info
    validation = await client.call_tool("validate_media_file", {
        "file_path": file_path
    })
    
    duration_minutes = validation["duration"] / 60
    
    # 2. Choose optimal model based on duration
    if duration_minutes < 15:
        model = "mlx-community/whisper-large-v3-turbo"  # Best quality for short files
        print(f"Short file ({duration_minutes:.1f}m) - using large-v3-turbo")
    elif duration_minutes < 60:
        model = "mlx-community/whisper-medium-mlx"  # Good balance
        print(f"Medium file ({duration_minutes:.1f}m) - using medium")
    else:
        model = "mlx-community/whisper-base-mlx"  # Faster for long files
        print(f"Long file ({duration_minutes:.1f}m) - using base for speed")
    
    # 3. Process with selected model
    result = await client.call_tool("transcribe_file", {
        "file_path": file_path,
        "model": model,
        "output_formats": "txt,md"
    })
    
    speed_ratio = result["duration"] / result["processing_time"]
    print(f"✓ Processed at {speed_ratio:.1f}x realtime")
    print("---")
```

This completes the comprehensive examples guide. These examples cover everything from basic usage to advanced real-world scenarios, helping users understand how to effectively use the MCP Whisper Transcription Server for their specific needs.