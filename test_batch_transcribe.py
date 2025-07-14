#!/usr/bin/env python3
"""Test script for the batch_transcribe MCP tool using FastMCP Client"""

import asyncio
import sys
import json
import shutil
import subprocess
from pathlib import Path
from fastmcp import Client

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from src.whisper_mcp_server import mcp


async def create_test_files(test_dir: Path, num_files: int = 3):
    """Create test audio files in a directory"""
    test_dir.mkdir(exist_ok=True)

    files = []
    for i in range(num_files):
        filename = f"test_audio_{i}.wav"
        filepath = test_dir / filename

        # Create a 1-second sine wave test audio with different frequencies
        freq = 440 + (i * 110)  # 440Hz, 550Hz, 660Hz
        subprocess.run(
            [
                "ffmpeg",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={freq}:duration=1",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(filepath),
                "-y",
            ],
            check=True,
            capture_output=True,
        )

        files.append(filepath)

    return files


async def test_basic_batch():
    """Test basic batch transcription"""
    print("\nTesting basic batch transcription...")

    test_dir = Path("test_batch_dir")

    try:
        # Create test files
        test_files = await create_test_files(test_dir)
        print(f"✓ Created {len(test_files)} test files")

        async with Client(mcp) as client:
            result = await client.call_tool(
                "batch_transcribe",
                {"directory": str(test_dir), "pattern": "*.wav", "output_formats": "txt"},
            )

            if result and len(result) > 0:
                data = json.loads(result[0].text)
            else:
                raise Exception("No result returned from tool")

            # Verify result structure
            assert "total_files" in data, "Result should contain 'total_files'"
            assert "processed" in data, "Result should contain 'processed'"
            assert "skipped" in data, "Result should contain 'skipped'"
            assert "failed" in data, "Result should contain 'failed'"
            assert "results" in data, "Result should contain 'results'"
            assert "performance_report" in data, "Result should contain 'performance_report'"

            # Debug print
            print(f"Debug - Full result: {json.dumps(data, indent=2)}")

            # Verify counts
            assert data["total_files"] == 3, f"Expected 3 files, got {data['total_files']}"
            assert data["processed"] == 3, f"Expected 3 processed, got {data['processed']}"
            assert data["skipped"] == 0, f"Expected 0 skipped, got {data['skipped']}"
            assert data["failed"] == 0, f"Expected 0 failed, got {data['failed']}"

            # Verify performance report
            perf = data["performance_report"]
            assert isinstance(perf, dict), "Performance report should be a dictionary"
            assert "total_audio_duration" in perf, "Should have total_audio_duration"
            assert "total_processing_time" in perf, "Should have total_processing_time"
            assert "average_speed" in perf, "Should have average_speed"
            assert "files_per_minute" in perf, "Should have files_per_minute"

            print("✓ Basic batch transcription completed successfully")
            print(f"  - Total files: {data['total_files']}")
            print(f"  - Processed: {data['processed']}")
            print(f"  - Processing time: {perf['total_processing_time']:.2f}s")
            print(f"  - Average speed: {perf['average_speed']:.2f}x realtime")

            return True

    except Exception as e:
        print(f"❌ Basic batch test failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Clean up
        shutil.rmtree(test_dir, ignore_errors=True)


async def test_pattern_matching():
    """Test pattern matching functionality"""
    print("\nTesting pattern matching...")

    test_dir = Path("test_pattern_dir")

    try:
        test_dir.mkdir(exist_ok=True)

        # Create files with different patterns
        files = []
        # Create audio_*.wav files
        for i in range(2):
            filepath = test_dir / f"audio_{i}.wav"
            subprocess.run(
                [
                    "ffmpeg",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:duration=0.5",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    str(filepath),
                    "-y",
                ],
                check=True,
                capture_output=True,
            )
            files.append(filepath)

        # Create video_*.wav files
        for i in range(2):
            filepath = test_dir / f"video_{i}.wav"
            subprocess.run(
                [
                    "ffmpeg",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=550:duration=0.5",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    str(filepath),
                    "-y",
                ],
                check=True,
                capture_output=True,
            )
            files.append(filepath)

        # Create other.wav file
        filepath = test_dir / "other.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=660:duration=0.5",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(filepath),
                "-y",
            ],
            check=True,
            capture_output=True,
        )
        files.append(filepath)

        async with Client(mcp) as client:
            # Test audio_* pattern
            result = await client.call_tool(
                "batch_transcribe",
                {"directory": str(test_dir), "pattern": "audio_*.wav", "output_formats": "txt"},
            )

            data = json.loads(result[0].text)
            assert data["total_files"] == 2, f"Expected 2 audio_* files, got {data['total_files']}"
            assert data["processed"] == 2, f"Expected 2 processed, got {data['processed']}"

            print("✓ Pattern matching 'audio_*.wav' worked correctly")

            # Test video_* pattern
            result = await client.call_tool(
                "batch_transcribe",
                {"directory": str(test_dir), "pattern": "video_*.wav", "output_formats": "txt"},
            )

            data = json.loads(result[0].text)
            assert data["total_files"] == 2, f"Expected 2 video_* files, got {data['total_files']}"

            print("✓ Pattern matching 'video_*.wav' worked correctly")

            return True

    except Exception as e:
        print(f"❌ Pattern matching test failed: {e}")
        return False
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


async def test_skip_existing():
    """Test skip_existing functionality"""
    print("\nTesting skip_existing functionality...")

    test_dir = Path("test_skip_dir")

    try:
        # Create test file
        test_files = await create_test_files(test_dir, num_files=1)

        async with Client(mcp) as client:
            # First transcription
            result = await client.call_tool(
                "batch_transcribe", {"directory": str(test_dir), "output_formats": "txt"}
            )

            data = json.loads(result[0].text)
            assert data["processed"] == 1, "Should process 1 file initially"

            # Second transcription with skip_existing=True (default)
            result = await client.call_tool(
                "batch_transcribe", {"directory": str(test_dir), "output_formats": "txt"}
            )

            data = json.loads(result[0].text)
            assert data["processed"] == 0, "Should process 0 files when skipping existing"
            assert data["skipped"] == 1, "Should skip 1 file"

            print("✓ Skip existing functionality working correctly")

            # Third transcription with skip_existing=False
            result = await client.call_tool(
                "batch_transcribe",
                {"directory": str(test_dir), "output_formats": "txt", "skip_existing": False},
            )

            data = json.loads(result[0].text)
            assert data["processed"] == 1, "Should process 1 file when not skipping"
            assert data["skipped"] == 0, "Should skip 0 files"

            print("✓ Force reprocessing working correctly")

            return True

    except Exception as e:
        print(f"❌ Skip existing test failed: {e}")
        return False
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


async def test_recursive_search():
    """Test recursive directory search"""
    print("\nTesting recursive directory search...")

    test_dir = Path("test_recursive_dir")

    try:
        # Create nested directory structure
        test_dir.mkdir(exist_ok=True)
        subdir1 = test_dir / "subdir1"
        subdir2 = test_dir / "subdir2"
        subdir1.mkdir(exist_ok=True)
        subdir2.mkdir(exist_ok=True)

        # Create files in different directories
        files = []
        # Root directory
        await create_test_files(test_dir, num_files=1)
        # Subdirectory 1
        await create_test_files(subdir1, num_files=2)
        # Subdirectory 2
        await create_test_files(subdir2, num_files=2)

        async with Client(mcp) as client:
            # Test non-recursive (default)
            result = await client.call_tool(
                "batch_transcribe",
                {"directory": str(test_dir), "pattern": "*.wav", "output_formats": "txt"},
            )

            data = json.loads(result[0].text)
            assert (
                data["total_files"] == 1
            ), f"Non-recursive should find 1 file, got {data['total_files']}"

            print("✓ Non-recursive search found only root files")

            # Test recursive
            result = await client.call_tool(
                "batch_transcribe",
                {
                    "directory": str(test_dir),
                    "pattern": "*.wav",
                    "recursive": True,
                    "output_formats": "txt",
                },
            )

            data = json.loads(result[0].text)
            assert (
                data["total_files"] == 5
            ), f"Recursive should find 5 files, got {data['total_files']}"

            print("✓ Recursive search found all nested files")

            return True

    except Exception as e:
        print(f"❌ Recursive search test failed: {e}")
        return False
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


async def test_custom_output_dir():
    """Test custom output directory"""
    print("\nTesting custom output directory...")

    test_dir = Path("test_batch_input")
    output_dir = Path("test_batch_output")

    try:
        # Create test files
        test_files = await create_test_files(test_dir, num_files=2)

        async with Client(mcp) as client:
            result = await client.call_tool(
                "batch_transcribe",
                {
                    "directory": str(test_dir),
                    "output_dir": str(output_dir),
                    "output_formats": "txt",
                },
            )

            data = json.loads(result[0].text)
            print(f"Debug custom output dir result: {json.dumps(data, indent=2)}")
            assert data["processed"] == 2, f"Should process 2 files, got {data['processed']}"

            # Check that output files are in the custom directory
            output_files = list(output_dir.glob("*.txt"))
            assert len(output_files) == 2, f"Expected 2 output files, got {len(output_files)}"

            print("✓ Custom output directory working correctly")
            print(f"  - Output files: {[f.name for f in output_files]}")

            return True

    except Exception as e:
        print(f"❌ Custom output directory test failed: {e}")
        return False
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)


async def test_parallel_processing():
    """Test parallel processing with different worker counts"""
    print("\nTesting parallel processing...")

    test_dir = Path("test_parallel_dir")

    try:
        # Create more files for parallel testing
        test_files = await create_test_files(test_dir, num_files=6)

        async with Client(mcp) as client:
            # Test with max_workers=2
            start_time = asyncio.get_event_loop().time()
            result = await client.call_tool(
                "batch_transcribe",
                {"directory": str(test_dir), "max_workers": 2, "output_formats": "txt"},
            )
            elapsed_2_workers = asyncio.get_event_loop().time() - start_time

            data = json.loads(result[0].text)
            assert data["processed"] == 6, "Should process all 6 files"

            print(f"✓ Processed 6 files with 2 workers in {elapsed_2_workers:.2f}s")

            # Clean up outputs for next test
            for f in test_dir.glob("*.txt"):
                f.unlink()

            # Test with max_workers=4
            start_time = asyncio.get_event_loop().time()
            result = await client.call_tool(
                "batch_transcribe",
                {"directory": str(test_dir), "max_workers": 4, "output_formats": "txt"},
            )
            elapsed_4_workers = asyncio.get_event_loop().time() - start_time

            data = json.loads(result[0].text)
            assert data["processed"] == 6, "Should process all 6 files"

            print(f"✓ Processed 6 files with 4 workers in {elapsed_4_workers:.2f}s")

            # More workers should generally be faster (though not guaranteed with small files)
            print(f"  - Speed improvement: {elapsed_2_workers / elapsed_4_workers:.2f}x")

            return True

    except Exception as e:
        print(f"❌ Parallel processing test failed: {e}")
        return False
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


async def test_error_handling():
    """Test error handling for invalid inputs"""
    print("\nTesting error handling...")

    async with Client(mcp) as client:
        # Test non-existent directory
        try:
            result = await client.call_tool(
                "batch_transcribe", {"directory": "/non/existent/directory"}
            )
            print("❌ Should have raised error for non-existent directory")
            return False
        except Exception as e:
            if "Directory not found" in str(e):
                print("✓ Correctly raised error for non-existent directory")
            else:
                print(f"❌ Unexpected error: {e}")
                return False

        # Test empty directory
        empty_dir = Path("test_empty_dir")
        empty_dir.mkdir(exist_ok=True)

        try:
            result = await client.call_tool("batch_transcribe", {"directory": str(empty_dir)})

            data = json.loads(result[0].text)
            assert data["total_files"] == 0, "Should find 0 files"
            assert (
                data["performance_report"] == "No media files found"
            ), "Should have appropriate message"

            print("✓ Correctly handled empty directory")

        finally:
            empty_dir.rmdir()

        return True


async def test_list_tools():
    """Test that batch_transcribe tool is properly registered"""
    print("\nTesting tool registration...")

    async with Client(mcp) as client:
        tools = await client.list_tools()
        tool_names = [tool.name for tool in tools]

        if "batch_transcribe" in tool_names:
            print("✓ batch_transcribe tool is properly registered")

            # Find and display the tool schema
            for tool in tools:
                if tool.name == "batch_transcribe":
                    print(f"  - Description: {tool.description}")
                    if tool.inputSchema:
                        print(f"  - Parameters: {tool.inputSchema}")
            return True
        else:
            print("❌ batch_transcribe tool not found in registered tools")
            print(f"Available tools: {tool_names}")
            return False


async def main():
    """Run all tests"""
    print("=== Testing batch_transcribe MCP Tool ===\n")

    tests = [
        test_list_tools(),
        test_basic_batch(),
        test_pattern_matching(),
        test_skip_existing(),
        test_recursive_search(),
        test_custom_output_dir(),
        test_parallel_processing(),
        test_error_handling(),
    ]

    results = await asyncio.gather(*tests)

    passed = sum(results)
    total = len(results)

    print(f"\n=== Test Results: {passed}/{total} passed ===")

    if passed == total:
        print("\n✅ All tests passed! The batch_transcribe tool is working correctly.")
    else:
        print(f"\n❌ {total - passed} tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
