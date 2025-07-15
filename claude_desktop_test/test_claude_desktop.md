# Claude Desktop MCP Whisper Test Guide

## Quick Setup

1. Ensure the MCP server is configured in Claude Desktop:
   ```json
   {
     "mcpServers": {
       "whisper-transcription": {
         "command": "python",
         "args": ["-m", "src.whisper_mcp_server"],
         "cwd": "/path/to/mcp-whisper-transcription"
       }
     }
   }
   ```

2. Restart Claude Desktop after configuration

## Test Scenarios

### Test 1: Simple Transcription
**File**: Upload any audio file (MP3, WAV, M4A, etc.)
**Prompt**: "Transcribe this audio file"
**Expected**: 
- Files <5 min: Immediate result
- Files >5 min: Job ID with progress updates

### Test 2: Model Selection
**Prompt**: "Transcribe this using the large model for better accuracy"
**Available Models**:
- tiny (fastest)
- base (balanced)
- small
- medium
- large (most accurate)

### Test 3: Output Formats
**Prompt**: "Generate subtitles from this audio"
**Formats**:
- txt (timestamped text)
- srt (subtitle format)
- md (markdown)
- vtt (web subtitles)
- json (structured data)

### Test 4: Language Support
**Prompt**: "Transcribe this Spanish audio and translate to English"
**Features**:
- Auto-detect language
- Specify language: "Transcribe this as French"
- Translation: "Translate this to English"

### Test 5: VAD Optimization
**Prompt**: "Transcribe with voice activity detection"
**Benefits**:
- 3-5x faster processing
- Removes silence
- Better for podcasts/meetings

## Advanced Commands

### Job Management
- "List my transcription jobs"
- "Check status of job [job-id]"
- "Cancel job [job-id]"
- "Get result of job [job-id]"

### Batch Processing
- Upload multiple files at once
- "Transcribe all these files with the base model"

### Custom Settings
- "Transcribe with temperature 0.2 for more consistent output"
- "Set initial prompt: 'Technical podcast about AI'"
- "Save output to custom directory"

## Troubleshooting

### If transcription seems stuck:
1. Check job status: "List my transcription jobs"
2. Long files automatically use async processing
3. Check logs in: `logs/transcription_history.json`

### Common Issues:
- **Timeout**: Files >5 min use async by default
- **Memory**: Large model needs ~8GB RAM
- **Format**: Ensure audio is valid (not video files)

## Performance Tips

1. **Use VAD** for long recordings
2. **Choose appropriate model**:
   - tiny: 39M params, very fast
   - base: 74M params, good balance
   - large: 1550M params, best quality

3. **File preparation**:
   - Convert video to audio first
   - Split very long files (>2 hours)
   - Use standard formats (WAV, MP3, M4A)