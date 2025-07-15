# Claude Desktop Whisper Transcription Test

## Quick Test Instructions

1. **Test Short Audio (Sync Processing)**:
   - Drag  into Claude Desktop
   - Ask: "Please transcribe this audio file"
   - Expected: Immediate transcription result

2. **Test Long Audio (Async Processing)**:
   - Drag  into Claude Desktop  
   - Ask: "Please transcribe this audio file"
   - Expected: Job ID returned, then progress updates

3. **Test Multiple Files**:
   - Select and drag both files at once
   - Ask: "Please transcribe all these audio files"
   - Expected: Smart routing for each file

## What to Look For

- Short files (<5 min) → Immediate results
- Long files (>5 min) → Job ID + progress monitoring
- No timeouts even for very long files
- VAD optimization for faster processing

## Advanced Commands

After uploading a file, you can also ask:
- "Transcribe this with the large model for better accuracy"
- "Generate subtitles in SRT format"
- "Translate this to English" (for non-English audio)
- "List all my transcription jobs"
- "Cancel job [job-id]"
