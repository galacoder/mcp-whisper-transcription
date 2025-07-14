#!/usr/bin/env python3
"""Test script for the transcribe_file MCP tool using FastMCP Client"""

import asyncio
import sys
import json
import shutil
from pathlib import Path
from fastmcp import Client

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from src.whisper_mcp_server import mcp


async def test_file_validation():
    """Test that non-existent files raise proper errors"""
    print("Testing file validation...")

    async with Client(mcp) as client:
        try:
            result = await client.call_tool(
                "transcribe_file", {"file_path": "nonexistent_file.mp3"}
            )
            print("❌ Expected error for non-existent file, but got result")
            return False
        except Exception as e:
            if "File not found" in str(e):
                print("✓ Correctly raised error for non-existent file")
                return True
            else:
                print(f"❌ Unexpected error: {e}")
                return False


async def test_basic_transcription():
    """Test basic transcription with a test audio file"""
    print("\nTesting basic transcription...")

    # First, let's create a simple test audio file using ffmpeg
    test_audio = Path("test_audio.wav")

    if not test_audio.exists():
        print("Creating test audio file...")
        import subprocess

        try:
            # Create a 3-second sine wave test audio
            subprocess.run(
                [
                    "ffmpeg",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:duration=3",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    str(test_audio),
                    "-y",
                ],
                check=True,
                capture_output=True,
            )
            print("✓ Created test audio file")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create test audio: {e}")
            return False

    async with Client(mcp) as client:
        try:
            # Test transcription
            result = await client.call_tool(
                "transcribe_file", {"file_path": str(test_audio), "output_formats": "txt,md"}
            )

            # The result is a list of content objects
            # Extract the data from the first content object
            if result and len(result) > 0:
                # The content should be TextContent with text as JSON
                data = json.loads(result[0].text)
            else:
                raise Exception("No result returned from tool")

            # Verify result structure
            assert isinstance(data, dict), "Result data should be a dictionary"
            assert "text" in data, "Result should contain 'text'"
            assert "segments" in data, "Result should contain 'segments'"
            assert "output_files" in data, "Result should contain 'output_files'"
            assert "duration" in data, "Result should contain 'duration'"
            assert "processing_time" in data, "Result should contain 'processing_time'"
            assert "model_used" in data, "Result should contain 'model_used'"

            print("✓ Transcription completed successfully")
            print(f"  - Duration: {data['duration']:.2f}s")
            print(f"  - Processing time: {data['processing_time']:.2f}s")
            print(f"  - Model: {data['model_used']}")
            print(f"  - Output files: {data['output_files']}")

            # Clean up output files
            for file_path in data["output_files"]:
                Path(file_path).unlink(missing_ok=True)

            return True

        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            import traceback

            traceback.print_exc()
            return False
        finally:
            # Clean up test audio
            test_audio.unlink(missing_ok=True)


async def test_output_formats():
    """Test different output format combinations"""
    print("\nTesting output formats...")

    # Create test audio
    test_audio = Path("test_formats.wav")
    import subprocess

    subprocess.run(
        [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(test_audio),
            "-y",
        ],
        check=True,
        capture_output=True,
    )

    async with Client(mcp) as client:
        try:
            # Test with all formats
            result = await client.call_tool(
                "transcribe_file",
                {"file_path": str(test_audio), "output_formats": "txt,md,srt,json"},
            )

            if result and len(result) > 0:
                data = json.loads(result[0].text)
            else:
                raise Exception("No result returned from tool")

            # Check that all formats were created
            output_files = [Path(f) for f in data["output_files"]]
            extensions = {f.suffix for f in output_files}

            expected_extensions = {".txt", ".md", ".srt", ".json"}
            missing = expected_extensions - extensions

            if missing:
                print(f"❌ Missing output formats: {missing}")
                return False
            else:
                print("✓ All output formats created successfully")

            # Clean up
            for file_path in output_files:
                file_path.unlink(missing_ok=True)

            return True

        except Exception as e:
            print(f"❌ Output format test failed: {e}")
            return False
        finally:
            test_audio.unlink(missing_ok=True)


async def test_custom_output_dir():
    """Test custom output directory"""
    print("\nTesting custom output directory...")

    # Create test audio and output dir
    test_audio = Path("test_output_dir.wav")
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    import subprocess

    subprocess.run(
        [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(test_audio),
            "-y",
        ],
        check=True,
        capture_output=True,
    )

    async with Client(mcp) as client:
        try:
            result = await client.call_tool(
                "transcribe_file",
                {
                    "file_path": str(test_audio),
                    "output_dir": str(output_dir),
                    "output_formats": "txt",
                },
            )

            if result and len(result) > 0:
                data = json.loads(result[0].text)
            else:
                raise Exception("No result returned from tool")

            # Check output was created in custom directory
            output_files = [Path(f) for f in data["output_files"]]
            assert all(
                f.parent == output_dir for f in output_files
            ), "Output files should be in custom directory"

            print("✓ Custom output directory working correctly")

            # Clean up - use shutil to remove directory and all contents
            shutil.rmtree(output_dir, ignore_errors=True)

            return True

        except Exception as e:
            print(f"❌ Custom output directory test failed: {e}")
            import traceback

            traceback.print_exc()
            return False
        finally:
            test_audio.unlink(missing_ok=True)
            shutil.rmtree(output_dir, ignore_errors=True)


async def test_list_tools():
    """Test that transcribe_file tool is properly registered"""
    print("\nTesting tool registration...")

    async with Client(mcp) as client:
        tools = await client.list_tools()
        tool_names = [tool.name for tool in tools]

        if "transcribe_file" in tool_names:
            print("✓ transcribe_file tool is properly registered")

            # Find and display the tool schema
            for tool in tools:
                if tool.name == "transcribe_file":
                    print(f"  - Description: {tool.description}")
                    if tool.inputSchema:
                        print(f"  - Parameters: {tool.inputSchema}")
            return True
        else:
            print("❌ transcribe_file tool not found in registered tools")
            print(f"Available tools: {tool_names}")
            return False


async def main():
    """Run all tests"""
    print("=== Testing transcribe_file MCP Tool ===\n")

    tests = [
        test_list_tools(),
        test_file_validation(),
        test_basic_transcription(),
        test_output_formats(),
        test_custom_output_dir(),
    ]

    results = await asyncio.gather(*tests)

    passed = sum(results)
    total = len(results)

    print(f"\n=== Test Results: {passed}/{total} passed ===")

    if passed == total:
        print("\n✅ All tests passed! The transcribe_file tool is working correctly.")
    else:
        print(f"\n❌ {total - passed} tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
