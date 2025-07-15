#!/usr/bin/env python3
"""
Test script for async job processing functionality.

This script tests:
1. Smart routing decision logic
2. Async job creation and management
3. Job status tracking
4. Result retrieval
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
from job_manager import SmartRouter


class AsyncJobTester:
    """Test async job processing functionality."""
    
    def __init__(self):
        self.test_files = {
            "short": "/Users/sangle/Dev/action/projects/@ai/whisper-transcription/raw_files/test_short.mp3",
            "medium": "/Users/sangle/Dev/action/projects/@ai/whisper-transcription/raw_files/test_numbers.mp3",
            "long": "/Users/sangle/Dev/action/projects/@ai/whisper-transcription/raw_files/gary-quick-call-to-calm-me.m4a"
        }
        
    async def test_smart_routing(self):
        """Test smart routing decisions."""
        print("\n" + "="*60)
        print("🧠 Testing Smart Routing Logic")
        print("="*60)
        
        # Test different file scenarios
        test_cases = [
            {"file": "short_audio.mp3", "duration": 120, "size": 5*1024*1024, "expected": False},
            {"file": "medium_audio.mp3", "duration": 240, "size": 20*1024*1024, "expected": False},
            {"file": "long_audio.mp3", "duration": 600, "size": 50*1024*1024, "expected": True},
            {"file": "huge_file.mp3", "duration": 180, "size": 150*1024*1024, "expected": True},
        ]
        
        print("\nRouting Decision Tests:")
        print("-" * 40)
        for case in test_cases:
            should_async = SmartRouter.should_use_async(
                case["file"], case["duration"], case["size"]
            )
            
            status = "✅" if should_async == case["expected"] else "❌"
            print(f"{status} {case['file']}: duration={case['duration']}s, size={case['size']/(1024*1024):.1f}MB")
            print(f"   Decision: {'ASYNC' if should_async else 'SYNC'} (expected: {'ASYNC' if case['expected'] else 'SYNC'})")
    
    async def test_sync_processing(self):
        """Test synchronous processing for short files."""
        print("\n" + "="*60)
        print("⚡ Testing Synchronous Processing")
        print("="*60)
        
        async with Client(mcp) as client:
            # Test short file (should be sync)
            print("\n📝 Testing short file (should use sync)...")
            
            result = await client.call_tool("transcribe_file", {
                "file_path": self.test_files["short"],
                "output_formats": "txt",
                "model": "mlx-community/whisper-base-mlx"  # Fast model for testing
            })
            
            # Handle response
            if isinstance(result, list):
                result = result[0]
            
            # Extract content from CallToolResult
            if hasattr(result, 'content'):
                result = result.content
            elif hasattr(result, 'text'):
                result = json.loads(result.text)
            
            print(f"Processing mode: {'ASYNC' if result.get('async') else 'SYNC'}")
            
            if not result.get('async'):
                print("✅ Correctly used synchronous processing")
                print(f"Duration: {result.get('duration', 0):.1f}s")
                print(f"Processing time: {result.get('processing_time', 0):.1f}s")
                print(f"Text preview: {result.get('text', '')[:100]}...")
            else:
                print("❌ Unexpectedly used async processing")
    
    async def test_async_job_creation(self):
        """Test async job creation and management."""
        print("\n" + "="*60)
        print("📋 Testing Async Job Creation")
        print("="*60)
        
        async with Client(mcp) as client:
            # Test long file (should be async)
            print("\n📝 Creating async job for long file...")
            
            result = await client.call_tool("transcribe_file", {
                "file_path": self.test_files["long"],
                "output_formats": "txt,json",
                "model": "mlx-community/whisper-base-mlx",  # Fast model for testing
                "force_async": True  # Force async for testing
            })
            
            # Handle response
            if isinstance(result, list):
                result = result[0]
            
            # Extract content from CallToolResult
            if hasattr(result, 'content'):
                result = result.content
            elif hasattr(result, 'text'):
                result = json.loads(result.text)
            
            print(f"Processing mode: {'ASYNC' if result.get('async') else 'SYNC'}")
            
            if result.get('async'):
                job_id = result.get('job_id')
                print(f"✅ Job created: {job_id}")
                print(f"Estimated time: {result.get('estimated_time_formatted')}")
                print(f"File duration: {result.get('duration'):.1f}s")
                
                return job_id
            else:
                print("❌ Unexpectedly used sync processing")
                return None
    
    async def test_job_status_tracking(self, job_id: str):
        """Test job status checking."""
        print("\n" + "="*60)
        print("📊 Testing Job Status Tracking")
        print("="*60)
        
        async with Client(mcp) as client:
            # Check status multiple times
            max_checks = 30  # Max 30 seconds
            check_interval = 1  # Check every second
            
            for i in range(max_checks):
                # Check job status
                status_result = await client.call_tool("check_job_status", {
                    "job_id": job_id
                })
                
                # Handle response
                if isinstance(status_result, list):
                    status_result = status_result[0]
                
                # Extract content from CallToolResult
                if hasattr(status_result, 'content'):
                    status_result = status_result.content
                elif hasattr(status_result, 'text'):
                    status_result = json.loads(status_result.text)
                
                status = status_result.get('status')
                progress = status_result.get('progress', 0)
                
                print(f"\rCheck {i+1}: Status={status}, Progress={progress:.1f}%", end="")
                
                if status in ['completed', 'failed', 'cancelled']:
                    print(f"\n✅ Job finished with status: {status}")
                    return status_result
                
                await asyncio.sleep(check_interval)
            
            print("\n⏱️ Job still running after 30 seconds")
            return None
    
    async def test_job_result_retrieval(self, job_id: str):
        """Test getting job results."""
        print("\n" + "="*60)
        print("📄 Testing Job Result Retrieval")
        print("="*60)
        
        async with Client(mcp) as client:
            # Get job result
            result = await client.call_tool("get_job_result", {
                "job_id": job_id
            })
            
            # Handle response
            if isinstance(result, list):
                result = result[0]
            
            # Extract content from CallToolResult
            if hasattr(result, 'content'):
                result = result.content
            elif hasattr(result, 'text'):
                result = json.loads(result.text)
            
            if not result.get('error'):
                print("✅ Successfully retrieved job result")
                print(f"Text length: {len(result.get('text', ''))} characters")
                print(f"Output files: {result.get('output_files', [])}")
                print(f"Processing time: {result.get('processing_time', 0):.1f}s")
            else:
                print(f"❌ Error retrieving result: {result.get('error')}")
    
    async def test_job_listing(self):
        """Test listing jobs."""
        print("\n" + "="*60)
        print("📋 Testing Job Listing")
        print("="*60)
        
        async with Client(mcp) as client:
            # List all jobs
            result = await client.call_tool("list_jobs", {
                "limit": 5
            })
            
            # Handle response
            if isinstance(result, list):
                result = result[0]
            
            # Extract content from CallToolResult
            if hasattr(result, 'content'):
                result = result.content
            elif hasattr(result, 'text'):
                result = json.loads(result.text)
            
            jobs = result.get('jobs', [])
            stats = result.get('statistics', {})
            
            print(f"\n📊 Job Statistics:")
            print(f"  Total jobs: {stats.get('total_jobs', 0)}")
            print(f"  Pending: {stats.get('pending', 0)}")
            print(f"  Running: {stats.get('running', 0)}")
            print(f"  Completed: {stats.get('completed', 0)}")
            print(f"  Failed: {stats.get('failed', 0)}")
            
            if jobs:
                print(f"\n📄 Recent Jobs:")
                for job in jobs[:5]:
                    print(f"  - {job['job_id'][:8]}... | {job['status']} | {job['file']}")
    
    async def test_forced_async(self):
        """Test forced async processing on a short file."""
        print("\n" + "="*60)
        print("🔧 Testing Forced Async Processing")
        print("="*60)
        
        async with Client(mcp) as client:
            # Force async on short file
            print("\n📝 Forcing async on short file...")
            
            result = await client.call_tool("transcribe_file", {
                "file_path": self.test_files["short"],
                "output_formats": "txt",
                "model": "mlx-community/whisper-tiny-mlx",  # Fastest model
                "force_async": True
            })
            
            # Handle response
            if isinstance(result, list):
                result = result[0]
            
            # Extract content from CallToolResult
            if hasattr(result, 'content'):
                result = result.content
            elif hasattr(result, 'text'):
                result = json.loads(result.text)
            
            if result.get('async'):
                job_id = result.get('job_id')
                print(f"✅ Successfully forced async: {job_id}")
                
                # Wait for completion
                print("⏳ Waiting for job completion...")
                await asyncio.sleep(5)
                
                # Check final status
                final_status = await client.call_tool("check_job_status", {
                    "job_id": job_id
                })
                
                if isinstance(final_status, list):
                    final_status = final_status[0]
                
                # Extract content from CallToolResult
                if hasattr(final_status, 'content'):
                    final_status = final_status.content
                elif hasattr(final_status, 'text'):
                    final_status = json.loads(final_status.text)
                
                print(f"Final status: {final_status.get('status')}")
                
                return job_id
            else:
                print("❌ Failed to force async processing")
                return None
    
    async def run_all_tests(self):
        """Run all async job tests."""
        print("\n🚀 Starting Async Job Processing Tests")
        print("Testing async job queue functionality")
        
        try:
            # Test 1: Smart Routing
            await self.test_smart_routing()
            
            # Test 2: Sync Processing
            await self.test_sync_processing()
            
            # Test 3: Job Listing
            await self.test_job_listing()
            
            # Test 4: Forced Async
            job_id = await self.test_forced_async()
            
            # Test 5: Create Async Job (if files exist)
            if Path(self.test_files["long"]).exists():
                job_id = await self.test_async_job_creation()
                
                if job_id:
                    # Test 6: Status Tracking
                    status = await self.test_job_status_tracking(job_id)
                    
                    # Test 7: Result Retrieval
                    if status and status.get('status') == 'completed':
                        await self.test_job_result_retrieval(job_id)
            
            print("\n" + "="*60)
            print("✅ All async job tests completed!")
            print("="*60)
            
        except Exception as e:
            print(f"\n❌ Test failed: {str(e)}")
            import traceback
            traceback.print_exc()


async def main():
    """Main test runner."""
    tester = AsyncJobTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())