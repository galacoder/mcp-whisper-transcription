#!/usr/bin/env python3
"""
Real-world test simulating Claude Desktop timeout scenario.

This test demonstrates:
1. Short files (<5 min) work fine with sync processing
2. Long files (>5 min) automatically use async to avoid timeout
3. Simulates Claude Desktop's ~30-60 second timeout
"""

import asyncio
import json
import time
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from fastmcp import Client
from src.whisper_mcp_server import mcp


class TimeoutSimulator:
    """Simulates Claude Desktop timeout behavior"""
    
    CLAUDE_DESKTOP_TIMEOUT = 30  # Simulated timeout in seconds
    
    def __init__(self):
        self.test_files = {
            "short": Path(__file__).parent / "test_files" / "gary-4min.m4a",
            "long": Path(__file__).parent / "test_files" / "gary-37min.m4a"
        }
        
    async def test_short_file_sync(self):
        """Test that short files complete within timeout window"""
        print("\n" + "="*60)
        print("📝 Test 1: Short File (4 minutes) - Synchronous Processing")
        print("="*60)
        
        async with Client(mcp) as client:
            start_time = time.time()
            
            print(f"\n⏰ Starting transcription at {datetime.now():%H:%M:%S}")
            print(f"   File: {self.test_files['short'].name}")
            print(f"   Expected: Complete within {self.CLAUDE_DESKTOP_TIMEOUT}s timeout")
            
            try:
                # Simulate timeout with asyncio.wait_for
                result = await asyncio.wait_for(
                    client.call_tool("transcribe_file", {
                        "file_path": str(self.test_files["short"]),
                        "model": "mlx-community/whisper-base-mlx",  # Faster model
                        "output_formats": "txt",
                        "use_vad": True  # Speed up with VAD
                    }),
                    timeout=self.CLAUDE_DESKTOP_TIMEOUT
                )
                
                elapsed = time.time() - start_time
                
                # Extract result
                if hasattr(result, 'data'):
                    result_data = result.data
                else:
                    result_data = result
                
                if not result_data.get('async'):
                    print(f"\n✅ SUCCESS: Transcription completed in {elapsed:.1f}s")
                    print(f"   Mode: Synchronous")
                    print(f"   Duration: {result_data.get('duration', 0):.1f}s")
                    print(f"   Processing time: {result_data.get('processing_time', 0):.1f}s")
                    print(f"   Text preview: {result_data.get('text', '')[:100]}...")
                else:
                    print(f"\n⚠️  Unexpected: File was processed asynchronously")
                    
            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                print(f"\n❌ TIMEOUT: Request timed out after {elapsed:.1f}s")
                print("   This would fail in Claude Desktop!")
                
    async def test_long_file_timeout(self):
        """Test that long files would timeout without async"""
        print("\n" + "="*60)
        print("🚫 Test 2: Long File (37 minutes) - Simulated Sync Timeout")
        print("="*60)
        
        print(f"\n⏰ Simulating what would happen without async processing...")
        print(f"   File: {self.test_files['long'].name}")
        print(f"   Duration: ~37 minutes")
        
        # Estimate processing time
        estimated_time = 37 * 60 * 0.3  # ~30% of realtime with base model
        
        print(f"\n📊 Estimated processing time: {estimated_time/60:.1f} minutes")
        print(f"⏱️  Claude Desktop timeout: {self.CLAUDE_DESKTOP_TIMEOUT} seconds")
        
        if estimated_time > self.CLAUDE_DESKTOP_TIMEOUT:
            print(f"\n❌ Would TIMEOUT in Claude Desktop!")
            print(f"   Processing would take {estimated_time:.0f}s but timeout is {self.CLAUDE_DESKTOP_TIMEOUT}s")
            print(f"   This is why we need async processing!")
        
    async def test_long_file_async(self):
        """Test that long files work with async processing"""
        print("\n" + "="*60)
        print("✅ Test 3: Long File (37 minutes) - Async Processing")
        print("="*60)
        
        async with Client(mcp) as client:
            start_time = time.time()
            
            print(f"\n⏰ Starting transcription at {datetime.now():%H:%M:%S}")
            print(f"   File: {self.test_files['long'].name}")
            print(f"   Expected: Return immediately with job ID")
            
            try:
                # This should return immediately
                result = await asyncio.wait_for(
                    client.call_tool("transcribe_file", {
                        "file_path": str(self.test_files["long"]),
                        "model": "mlx-community/whisper-base-mlx",
                        "output_formats": "txt,srt",
                        "use_vad": True  # Speed up with VAD
                    }),
                    timeout=self.CLAUDE_DESKTOP_TIMEOUT
                )
                
                elapsed = time.time() - start_time
                
                # Extract result
                if hasattr(result, 'data'):
                    result_data = result.data
                else:
                    result_data = result
                
                if result_data.get('async'):
                    print(f"\n✅ SUCCESS: Returned immediately in {elapsed:.1f}s")
                    print(f"   Mode: Asynchronous")
                    print(f"   Job ID: {result_data.get('job_id')}")
                    print(f"   Estimated time: {result_data.get('estimated_time_formatted')}")
                    print(f"   File duration: {result_data.get('duration', 0)/60:.1f} minutes")
                    
                    return result_data.get('job_id')
                else:
                    print(f"\n❌ ERROR: File was processed synchronously (would timeout!)")
                    
            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                print(f"\n❌ TIMEOUT: Even async job creation timed out after {elapsed:.1f}s")
                print("   This should not happen!")
                
    async def monitor_async_job(self, job_id: str):
        """Monitor the progress of an async job"""
        print("\n" + "="*60)
        print("📊 Test 4: Monitor Async Job Progress")
        print("="*60)
        
        async with Client(mcp) as client:
            print(f"\n📋 Monitoring job: {job_id}")
            print("   Checking status every 10 seconds...")
            
            start_time = time.time()
            check_count = 0
            
            while True:
                check_count += 1
                
                # Check job status
                status_result = await client.call_tool("check_job_status", {
                    "job_id": job_id
                })
                
                # Extract result
                if hasattr(status_result, 'data'):
                    status_data = status_result.data
                else:
                    status_data = status_result
                
                status = status_data.get('status')
                progress = status_data.get('progress', 0)
                elapsed = time.time() - start_time
                
                print(f"\n   Check #{check_count} at {elapsed:.0f}s: Status={status}, Progress={progress:.1f}%")
                
                if status == 'completed':
                    print(f"\n✅ Job completed successfully!")
                    if status_data.get('processing_time'):
                        print(f"   Processing time: {status_data['processing_time']/60:.1f} minutes")
                    
                    # Get the full result
                    result = await client.call_tool("get_job_result", {
                        "job_id": job_id
                    })
                    
                    if hasattr(result, 'data'):
                        result_data = result.data
                    else:
                        result_data = result
                    
                    print(f"   Output files: {result_data.get('output_files', [])}")
                    print(f"   Text length: {len(result_data.get('text', ''))} characters")
                    break
                    
                elif status in ['failed', 'cancelled']:
                    print(f"\n❌ Job {status}!")
                    break
                
                # Don't check too frequently
                await asyncio.sleep(10)
                
                # Stop after 5 minutes of monitoring
                if elapsed > 300:
                    print("\n⏱️  Stopping monitor after 5 minutes")
                    break
    
    async def test_multiple_async_jobs(self):
        """Test handling multiple concurrent async jobs"""
        print("\n" + "="*60)
        print("🔄 Test 5: Multiple Concurrent Async Jobs")
        print("="*60)
        
        async with Client(mcp) as client:
            # Submit 3 jobs quickly
            print("\n📤 Submitting multiple jobs...")
            
            jobs = []
            for i in range(3):
                result = await client.call_tool("transcribe_file", {
                    "file_path": str(self.test_files["long"]),
                    "model": "mlx-community/whisper-tiny-mlx",  # Fastest model
                    "output_formats": "txt",
                    "force_async": True,
                    "output_dir": str(Path(__file__).parent / "test_files" / f"job_{i}")
                })
                
                if hasattr(result, 'data'):
                    result_data = result.data
                else:
                    result_data = result
                
                if result_data.get('async'):
                    jobs.append(result_data.get('job_id'))
                    print(f"   Job {i+1}: {result_data.get('job_id')[:8]}...")
            
            # Check queue status
            print("\n📊 Checking job queue status...")
            
            list_result = await client.call_tool("list_jobs", {
                "limit": 10
            })
            
            if hasattr(list_result, 'data'):
                list_data = list_result.data
            else:
                list_data = list_result
            
            stats = list_data.get('statistics', {})
            print(f"   Total jobs: {stats.get('total_jobs', 0)}")
            print(f"   Pending: {stats.get('pending', 0)}")
            print(f"   Running: {stats.get('running', 0)}")
            print(f"   Workers: {stats.get('workers', 0)}")
            
            return jobs
    
    async def run_all_tests(self):
        """Run all timeout simulation tests"""
        print("\n🚀 Real-World Timeout Simulation Tests")
        print("Demonstrating async processing for Claude Desktop")
        print(f"Simulated timeout: {self.CLAUDE_DESKTOP_TIMEOUT} seconds")
        
        try:
            # Test 1: Short file works fine
            await self.test_short_file_sync()
            
            # Test 2: Show what would happen with long file
            await self.test_long_file_timeout()
            
            # Test 3: Long file with async
            job_id = await self.test_long_file_async()
            
            if job_id:
                # Test 4: Monitor the job
                await self.monitor_async_job(job_id)
            
            # Test 5: Multiple jobs
            jobs = await self.test_multiple_async_jobs()
            
            print("\n" + "="*60)
            print("✅ All tests completed!")
            print("="*60)
            print("\n📌 Key Takeaways:")
            print("1. Short files (<5 min) process synchronously without issues")
            print("2. Long files (>5 min) automatically use async to avoid timeout")
            print("3. Async jobs return immediately with job ID")
            print("4. Jobs can be monitored and managed independently")
            print("5. Multiple jobs can run concurrently")
            
        except Exception as e:
            print(f"\n❌ Test failed: {str(e)}")
            import traceback
            traceback.print_exc()


async def main():
    """Main test runner"""
    simulator = TimeoutSimulator()
    
    # Check if test files exist
    for name, path in simulator.test_files.items():
        if not path.exists():
            print(f"❌ Test file not found: {path}")
            print("Please ensure test files are created first")
            return
    
    await simulator.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())