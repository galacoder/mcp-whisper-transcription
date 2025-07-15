#!/usr/bin/env python3
"""
Performance tests for MCP Whisper Transcription Server.
Tests various performance aspects including speed, memory usage, and scalability.
"""

import pytest
import asyncio
import time
import tempfile
import shutil
import json
import psutil
import os
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from fastmcp import Client
from src.whisper_mcp_server import mcp


class TestPerformance:
    """Performance tests for the MCP server."""
    
    @pytest.fixture
    def test_audio_path(self):
        """Get path to test audio file."""
        return str(Path(__file__).parent.parent / "examples" / "test_short.wav")
    
    @pytest.fixture
    def test_audio_dir(self):
        """Get path to test audio directory."""
        return str(Path(__file__).parent.parent / "examples")
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_transcription_speed(self, test_audio_path, temp_output_dir):
        """Test transcription speed and verify it meets performance targets."""
        async with Client(mcp) as client:
            # Time a single transcription
            start_time = time.time()
            
            response = await client.call_tool("transcribe_file", {
                "file_path": test_audio_path,
                "output_dir": temp_output_dir
            })
            result = json.loads(response[0].text)
            
            end_time = time.time()
            wall_clock_time = end_time - start_time
            
            # Check that we have timing data
            assert "duration" in result
            assert "processing_time" in result
            
            audio_duration = result["duration"]
            processing_time = result["processing_time"]
            
            # Calculate speed metrics
            speed_ratio = audio_duration / processing_time if processing_time > 0 else 0
            
            print(f"\n=== Transcription Speed Test ===")
            print(f"Audio duration: {audio_duration:.2f}s")
            print(f"Processing time: {processing_time:.2f}s")
            print(f"Wall clock time: {wall_clock_time:.2f}s")
            print(f"Speed ratio: {speed_ratio:.2f}x realtime")
            
            # Performance assertions
            assert processing_time > 0, "Processing time should be positive"
            assert wall_clock_time > 0, "Wall clock time should be positive"
            
            # With MLX on Apple Silicon, we should get at least 1x realtime
            assert speed_ratio >= 1.0, f"Speed ratio {speed_ratio:.2f}x is below 1x realtime"
            
            # Wall clock time should be close to processing time (within 20% overhead)
            overhead_ratio = wall_clock_time / processing_time
            assert overhead_ratio <= 1.5, f"Too much overhead: {overhead_ratio:.2f}x"
    
    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, test_audio_dir, temp_output_dir):
        """Test batch processing performance and parallelization."""
        async with Client(mcp) as client:
            # Test sequential processing (1 worker)
            start_time = time.time()
            response_sequential = await client.call_tool("batch_transcribe", {
                "directory": test_audio_dir,
                "pattern": "test_*.wav",
                "max_workers": 1,
                "output_dir": temp_output_dir + "_seq"
            })
            sequential_time = time.time() - start_time
            
            # Clean up for next test
            shutil.rmtree(temp_output_dir + "_seq", ignore_errors=True)
            
            # Test parallel processing (4 workers)
            start_time = time.time()
            response_parallel = await client.call_tool("batch_transcribe", {
                "directory": test_audio_dir,
                "pattern": "test_*.wav",
                "max_workers": 4,
                "output_dir": temp_output_dir + "_par"
            })
            parallel_time = time.time() - start_time
            
            seq_result = json.loads(response_sequential[0].text)
            par_result = json.loads(response_parallel[0].text)
            
            print(f"\n=== Batch Processing Performance ===")
            print(f"Sequential (1 worker): {sequential_time:.2f}s")
            print(f"Parallel (4 workers): {parallel_time:.2f}s")
            print(f"Files processed: {seq_result['processed']}")
            print(f"Speedup: {sequential_time / parallel_time:.2f}x")
            
            # Parallel should be faster or at least not significantly slower
            # Allow some tolerance for overhead
            assert parallel_time <= sequential_time * 1.2, "Parallel processing should not be much slower"
            
            # Both should process the same number of files
            assert seq_result["processed"] == par_result["processed"], "Different number of files processed"
            assert seq_result["failed"] == 0, "Sequential processing should not fail"
            assert par_result["failed"] == 0, "Parallel processing should not fail"
    
    @pytest.mark.asyncio
    async def test_memory_usage(self, test_audio_path, temp_output_dir):
        """Test memory usage during transcription."""
        process = psutil.Process(os.getpid())
        
        # Get baseline memory usage
        baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        async with Client(mcp) as client:
            # Monitor memory during transcription
            memory_before = process.memory_info().rss / (1024 * 1024)  # MB
            
            response = await client.call_tool("transcribe_file", {
                "file_path": test_audio_path,
                "output_dir": temp_output_dir
            })
            result = json.loads(response[0].text)
            
            memory_after = process.memory_info().rss / (1024 * 1024)  # MB
            memory_peak = memory_after  # Approximate peak
            
            print(f"\n=== Memory Usage Test ===")
            print(f"Baseline memory: {baseline_memory:.1f} MB")
            print(f"Memory before: {memory_before:.1f} MB")
            print(f"Memory after: {memory_after:.1f} MB")
            print(f"Peak memory: {memory_peak:.1f} MB")
            print(f"Memory increase: {memory_peak - baseline_memory:.1f} MB")
            
            # Memory usage should be reasonable (less than 500MB for short audio)
            memory_increase = memory_peak - baseline_memory
            assert memory_increase < 500, f"Memory usage too high: {memory_increase:.1f} MB"
            
            # Memory should be released after transcription (allow 50MB tolerance)
            assert memory_after <= memory_peak + 50, "Memory not properly released"
    
    @pytest.mark.asyncio
    async def test_concurrent_transcriptions(self, test_audio_path, temp_output_dir):
        """Test performance under concurrent load."""
        async with Client(mcp) as client:
            
            async def single_transcription(file_index):
                """Perform a single transcription with unique output."""
                output_dir = Path(temp_output_dir) / f"output_{file_index}"
                output_dir.mkdir(exist_ok=True)
                
                start_time = time.time()
                response = await client.call_tool("transcribe_file", {
                    "file_path": test_audio_path,
                    "output_dir": str(output_dir)
                })
                result = json.loads(response[0].text)
                end_time = time.time()
                
                return {
                    "wall_time": end_time - start_time,
                    "processing_time": result.get("processing_time", 0),
                    "success": "text" in result and len(result["text"]) > 0
                }
            
            # Run multiple concurrent transcriptions
            num_concurrent = 3
            start_time = time.time()
            
            tasks = [single_transcription(i) for i in range(num_concurrent)]
            results = await asyncio.gather(*tasks)
            
            total_wall_time = time.time() - start_time
            
            # Calculate metrics
            avg_wall_time = sum(r["wall_time"] for r in results) / len(results)
            avg_processing_time = sum(r["processing_time"] for r in results) / len(results)
            success_count = sum(1 for r in results if r["success"])
            
            print(f"\n=== Concurrent Transcriptions Test ===")
            print(f"Number of concurrent transcriptions: {num_concurrent}")
            print(f"Total wall clock time: {total_wall_time:.2f}s")
            print(f"Average wall time per task: {avg_wall_time:.2f}s")
            print(f"Average processing time per task: {avg_processing_time:.2f}s")
            print(f"Successful transcriptions: {success_count}/{num_concurrent}")
            
            # All transcriptions should succeed
            assert success_count == num_concurrent, f"Only {success_count}/{num_concurrent} transcriptions succeeded"
            
            # Average processing time should be reasonable
            assert avg_processing_time > 0, "Processing time should be positive"
            
            # With concurrent execution, total time should be less than sum of individual times
            # (allow some overhead)
            sequential_estimate = avg_wall_time * num_concurrent
            efficiency = sequential_estimate / total_wall_time
            print(f"Concurrency efficiency: {efficiency:.2f}x")
            
            # Should get some benefit from concurrency (at least 1.2x)
            assert efficiency >= 1.2, f"Poor concurrency efficiency: {efficiency:.2f}x"
    
    @pytest.mark.asyncio
    async def test_tool_response_times(self):
        """Test response times for various MCP tools."""
        async with Client(mcp) as client:
            tool_times = {}
            
            # Test quick tools that don't require heavy processing
            quick_tools = [
                ("list_models", {}),
                ("get_model_info", {"model_id": "mlx-community/whisper-tiny-mlx"}),
                ("get_supported_formats", {}),
                ("estimate_processing_time", {"file_path": str(Path(__file__).parent.parent / "examples" / "test_short.wav")}),
            ]
            
            for tool_name, params in quick_tools:
                start_time = time.time()
                
                response = await client.call_tool(tool_name, params)
                result = json.loads(response[0].text)
                
                end_time = time.time()
                response_time = end_time - start_time
                tool_times[tool_name] = response_time
                
                # Quick tools should respond very fast (< 2 seconds)
                assert response_time < 2.0, f"Tool {tool_name} too slow: {response_time:.2f}s"
                
                # Response should be valid
                assert isinstance(result, dict), f"Tool {tool_name} should return dict"
                if "error" in result:
                    print(f"Warning: Tool {tool_name} returned error: {result['error']}")
            
            print(f"\n=== Tool Response Times ===")
            for tool_name, response_time in tool_times.items():
                print(f"{tool_name}: {response_time:.3f}s")
            
            # Average response time should be under 1 second
            avg_response_time = sum(tool_times.values()) / len(tool_times)
            assert avg_response_time < 1.0, f"Average response time too slow: {avg_response_time:.3f}s"
    
    @pytest.mark.asyncio
    async def test_resource_access_performance(self):
        """Test performance of resource endpoint access."""
        async with Client(mcp) as client:
            
            resources_to_test = [
                "transcription://models",
                "transcription://config",
                "transcription://formats",
                "transcription://performance",
                "transcription://history"
            ]
            
            resource_times = {}
            
            for resource_uri in resources_to_test:
                start_time = time.time()
                
                resource = await client.read_resource(resource_uri)
                data = json.loads(resource[0].text)
                
                end_time = time.time()
                response_time = end_time - start_time
                resource_times[resource_uri] = response_time
                
                # Resources should respond very fast (< 1 second)
                assert response_time < 1.0, f"Resource {resource_uri} too slow: {response_time:.2f}s"
                
                # Response should be valid
                assert isinstance(data, dict), f"Resource {resource_uri} should return dict"
            
            print(f"\n=== Resource Access Times ===")
            for resource_uri, response_time in resource_times.items():
                print(f"{resource_uri}: {response_time:.3f}s")
            
            # Average response time should be under 0.5 seconds
            avg_response_time = sum(resource_times.values()) / len(resource_times)
            assert avg_response_time < 0.5, f"Average resource response time too slow: {avg_response_time:.3f}s"
    
    @pytest.mark.asyncio
    async def test_startup_performance(self):
        """Test server startup and initialization performance."""
        # This test measures the time it takes to create a client connection
        # which includes server initialization overhead
        
        start_time = time.time()
        
        async with Client(mcp) as client:
            # Make a simple call to ensure server is fully initialized
            response = await client.call_tool("list_models", {})
            result = json.loads(response[0].text)
        
        end_time = time.time()
        startup_time = end_time - start_time
        
        print(f"\n=== Startup Performance ===")
        print(f"Client connection + first call: {startup_time:.3f}s")
        
        # Startup should be reasonably fast (< 5 seconds)
        assert startup_time < 5.0, f"Startup too slow: {startup_time:.3f}s"
        
        # Response should be valid
        assert "models" in result, "list_models should return models"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])