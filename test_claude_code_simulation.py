#!/usr/bin/env python3
"""
Claude Code Simulation Test
Simulates how Claude Code would interact with the Whisper MCP server.

This demonstrates:
1. File upload simulation (like dragging a file into Claude Desktop)
2. Smart routing decision (sync vs async)
3. Progress monitoring for async jobs
4. Result retrieval
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
import shutil
import tempfile
from typing import Dict, Any

from fastmcp import Client
from src.whisper_mcp_server import mcp


class ClaudeCodeSimulator:
    """Simulates Claude Code's interaction with MCP server"""
    
    def __init__(self):
        self.conversation_history = []
        
    def log_interaction(self, role: str, content: str):
        """Log the conversation as it would appear in Claude Desktop"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {role}: {content}"
        self.conversation_history.append(entry)
        print(f"\n{'='*60}")
        print(entry)
        print('='*60)
        
    async def simulate_file_upload(self, file_path: Path) -> Dict[str, Any]:
        """Simulate user dragging/uploading a file to Claude Desktop"""
        self.log_interaction(
            "User", 
            f"[Uploads file: {file_path.name}]\n"
            f"Please transcribe this audio file."
        )
        
        # Claude Code would analyze the request
        self.log_interaction(
            "Claude",
            f"I'll transcribe the audio file '{file_path.name}' for you. "
            f"Let me check the file duration to determine the best approach..."
        )
        
        async with Client(mcp) as client:
            # First, get file info
            await asyncio.sleep(0.5)  # Simulate thinking
            
            # Call transcribe_file tool
            result = await client.call_tool("transcribe_file", {
                "file_path": str(file_path),
                "model": "mlx-community/whisper-base-mlx",
                "output_formats": "txt,srt,md",
                "use_vad": True
            })
            
            result_data = result.content if hasattr(result, 'content') else result
            # Handle the case where result might be wrapped in a list
            if isinstance(result_data, list) and result_data:
                result_data = result_data[0]
            return result_data
    
    async def handle_transcription_result(self, result: Dict[str, Any]):
        """Handle the transcription result based on sync/async mode"""
        if result.get('async'):
            # Async mode - got job ID
            job_id = result.get('job_id')
            duration_min = result.get('duration', 0) / 60
            
            self.log_interaction(
                "Claude",
                f"This is a long audio file ({duration_min:.1f} minutes), so I've started "
                f"an asynchronous transcription job to avoid timeout issues.\n\n"
                f"Job ID: {job_id}\n"
                f"Estimated time: {result.get('estimated_time_formatted', 'calculating...')}\n\n"
                f"I'll monitor the progress and let you know when it's complete."
            )
            
            # Monitor the job
            completed_result = await self.monitor_job_progress(job_id)
            return completed_result
            
        else:
            # Sync mode - got immediate result
            duration = result.get('duration', 0)
            processing_time = result.get('processing_time', 0)
            
            self.log_interaction(
                "Claude",
                f"✅ Transcription completed successfully!\n\n"
                f"Audio duration: {duration:.1f} seconds\n"
                f"Processing time: {processing_time:.1f} seconds\n"
                f"Output files:\n" + 
                "\n".join([f"  - {f}" for f in result.get('output_files', [])])
            )
            
            # Show text preview
            text = result.get('text', '')
            if text:
                preview = text[:500] + "..." if len(text) > 500 else text
                self.log_interaction(
                    "Claude",
                    f"Here's a preview of the transcription:\n\n{preview}"
                )
            
            return result
    
    async def monitor_job_progress(self, job_id: str) -> Dict[str, Any]:
        """Monitor async job progress as Claude would"""
        async with Client(mcp) as client:
            check_count = 0
            last_progress = 0
            
            while True:
                await asyncio.sleep(3)  # Check every 3 seconds
                check_count += 1
                
                # Check status
                status_result = await client.call_tool("check_job_status", {
                    "job_id": job_id
                })
                
                status_data = status_result.content if hasattr(status_result, 'content') else status_result
                status = status_data.get('status')
                progress = status_data.get('progress', 0)
                
                # Only log significant progress updates
                if progress - last_progress >= 20 or status in ['completed', 'failed']:
                    if status == 'running':
                        self.log_interaction(
                            "Claude",
                            f"Transcription progress: {progress:.0f}%"
                        )
                    last_progress = progress
                
                if status == 'completed':
                    # Get the full result
                    result = await client.call_tool("get_job_result", {
                        "job_id": job_id
                    })
                    
                    result_data = result.content if hasattr(result, 'content') else result
                    
                    self.log_interaction(
                        "Claude",
                        f"✅ Transcription completed!\n\n"
                        f"Processing time: {status_data.get('processing_time', 0)/60:.1f} minutes\n"
                        f"Output files:\n" + 
                        "\n".join([f"  - {f}" for f in result_data.get('output_files', [])])
                    )
                    
                    # Show text preview
                    text = result_data.get('text', '')
                    if text:
                        preview = text[:500] + "..." if len(text) > 500 else text
                        self.log_interaction(
                            "Claude",
                            f"Here's a preview of the transcription:\n\n{preview}"
                        )
                    
                    return result_data
                    
                elif status == 'failed':
                    error = status_data.get('error', 'Unknown error')
                    self.log_interaction(
                        "Claude",
                        f"❌ Transcription failed: {error}"
                    )
                    return status_data
                
                # Safety limit
                if check_count > 100:
                    self.log_interaction(
                        "Claude",
                        "The transcription is taking longer than expected. "
                        "You can check the status later using the job ID."
                    )
                    break
    
    async def simulate_multiple_file_upload(self, files: list[Path]):
        """Simulate uploading multiple files at once"""
        file_names = [f.name for f in files]
        self.log_interaction(
            "User",
            f"[Uploads {len(files)} files: {', '.join(file_names)}]\n"
            f"Please transcribe all these audio files."
        )
        
        self.log_interaction(
            "Claude",
            f"I'll transcribe all {len(files)} audio files for you. "
            f"Let me process them efficiently..."
        )
        
        async with Client(mcp) as client:
            # Start all transcriptions
            jobs = []
            for i, file_path in enumerate(files):
                result = await client.call_tool("transcribe_file", {
                    "file_path": str(file_path),
                    "model": "mlx-community/whisper-tiny-mlx",  # Faster for multiple files
                    "output_formats": "txt",
                    "use_vad": True
                })
                
                result_data = result.content if hasattr(result, 'content') else result
                
                if result_data.get('async'):
                    jobs.append({
                        'file': file_path.name,
                        'job_id': result_data.get('job_id'),
                        'duration': result_data.get('duration', 0)
                    })
                else:
                    # Sync completion
                    self.log_interaction(
                        "Claude",
                        f"✅ {file_path.name} - Completed immediately "
                        f"(duration: {result_data.get('duration', 0):.1f}s)"
                    )
            
            if jobs:
                self.log_interaction(
                    "Claude",
                    f"Started {len(jobs)} async transcription jobs:\n" +
                    "\n".join([f"  - {j['file']} ({j['duration']/60:.1f} min) - Job: {j['job_id'][:8]}..." 
                              for j in jobs])
                )
                
                # Check queue status
                list_result = await client.call_tool("list_jobs", {"limit": 10})
                list_data = list_result.content if hasattr(list_result, 'content') else list_result
                
                stats = list_data.get('statistics', {})
                self.log_interaction(
                    "Claude",
                    f"\nQueue status: {stats.get('running', 0)} running, "
                    f"{stats.get('pending', 0)} pending, "
                    f"{stats.get('workers', 0)} workers available"
                )


async def create_test_interface():
    """Create a simple test interface for Claude Desktop"""
    test_dir = Path(__file__).parent / "claude_desktop_test"
    test_dir.mkdir(exist_ok=True)
    
    # Create README for Claude Desktop
    readme_content = """# Claude Desktop Whisper Transcription Test

## Quick Test Instructions

1. **Test Short Audio (Sync Processing)**:
   - Drag `test_short.wav` into Claude Desktop
   - Ask: "Please transcribe this audio file"
   - Expected: Immediate transcription result

2. **Test Long Audio (Async Processing)**:
   - Drag `test_long.m4a` into Claude Desktop  
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
"""
    
    readme_path = test_dir / "README.md"
    readme_path.write_text(readme_content)
    
    # Copy test files
    test_files_dir = Path(__file__).parent / "test_files"
    if test_files_dir.exists():
        # Copy short test file
        short_src = test_files_dir / "gary-4min.m4a"
        if short_src.exists():
            shutil.copy(short_src, test_dir / "test_short.m4a")
        
        # Copy long test file  
        long_src = test_files_dir / "gary-37min.m4a"
        if long_src.exists():
            # Create a symlink to save space
            long_dst = test_dir / "test_long.m4a"
            if long_dst.exists():
                long_dst.unlink()
            long_dst.symlink_to(long_src.absolute())
    
    print(f"\n✅ Created Claude Desktop test directory: {test_dir}")
    print(f"   - README.md with instructions")
    print(f"   - test_short.m4a (4 min)")
    print(f"   - test_long.m4a (37 min)")
    print(f"\nYou can now open this directory and drag files into Claude Desktop!")
    
    return test_dir


async def run_simulation():
    """Run the Claude Code simulation"""
    print("\n🤖 Claude Code Simulation Test")
    print("="*60)
    
    simulator = ClaudeCodeSimulator()
    test_files = Path(__file__).parent / "test_files"
    
    # Test 1: Short file upload
    print("\n📝 Simulation 1: Short Audio File Upload")
    short_file = test_files / "gary-4min.m4a"
    if short_file.exists():
        result = await simulator.simulate_file_upload(short_file)
        await simulator.handle_transcription_result(result)
    
    # Test 2: Long file upload
    print("\n\n📝 Simulation 2: Long Audio File Upload")
    long_file = test_files / "gary-37min.m4a"
    if long_file.exists():
        result = await simulator.simulate_file_upload(long_file)
        await simulator.handle_transcription_result(result)
    
    # Test 3: Multiple files
    print("\n\n📝 Simulation 3: Multiple File Upload")
    files = [short_file, long_file]
    if all(f.exists() for f in files):
        await simulator.simulate_multiple_file_upload(files)
    
    # Create test interface
    print("\n\n📁 Creating Claude Desktop Test Interface...")
    test_dir = await create_test_interface()
    
    print("\n\n✅ Simulation Complete!")
    print("\nConversation History:")
    print("-" * 60)
    for entry in simulator.conversation_history[-5:]:  # Show last 5 entries
        print(entry)


if __name__ == "__main__":
    asyncio.run(run_simulation())