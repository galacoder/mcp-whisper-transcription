#!/usr/bin/env python3
"""Test script for the support MCP tools using FastMCP Client"""

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


async def test_list_models():
    """Test list_models tool"""
    print("\nTesting list_models...")

    async with Client(mcp) as client:
        result = await client.call_tool("list_models", {})

        if result and len(result) > 0:
            data = json.loads(result[0].text)
        else:
            raise Exception("No result returned from tool")

        # Verify structure
        assert "models" in data, "Result should contain 'models'"
        assert "current_model" in data, "Result should contain 'current_model'"
        assert "cache_dir" in data, "Result should contain 'cache_dir'"

        # Verify models list
        models = data["models"]
        assert len(models) == 6, f"Expected 6 models, got {len(models)}"

        # Check model structure
        for model in models:
            assert "id" in model, "Model should have 'id'"
            assert "size" in model, "Model should have 'size'"
            assert "speed" in model, "Model should have 'speed'"
            assert "accuracy" in model, "Model should have 'accuracy'"

        print("✓ list_models working correctly")
        print(f"  - Found {len(models)} models")
        print(f"  - Current model: {data['current_model']}")

        return True


async def test_get_model_info():
    """Test get_model_info tool"""
    print("\nTesting get_model_info...")

    async with Client(mcp) as client:
        # Test valid model
        result = await client.call_tool(
            "get_model_info", {"model_id": "mlx-community/whisper-tiny-mlx"}
        )

        data = json.loads(result[0].text)

        # Verify structure
        assert "id" in data, "Result should contain 'id'"
        assert "size" in data, "Result should contain 'size'"
        assert "speed" in data, "Result should contain 'speed'"
        assert "accuracy" in data, "Result should contain 'accuracy'"
        assert "parameters" in data, "Result should contain 'parameters'"
        assert "memory_required" in data, "Result should contain 'memory_required'"
        assert "recommended_for" in data, "Result should contain 'recommended_for'"
        assert "is_current" in data, "Result should contain 'is_current'"
        assert "is_cached" in data, "Result should contain 'is_cached'"

        print("✓ get_model_info working for valid model")
        print(f"  - Model: {data['id']}")
        print(f"  - Speed: {data['speed']}")
        print(f"  - Cached: {data['is_cached']}")

        # Test invalid model
        result = await client.call_tool("get_model_info", {"model_id": "invalid/model"})

        data = json.loads(result[0].text)
        assert "error" in data, "Should return error for invalid model"
        assert "available_models" in data, "Should list available models"

        print("✓ get_model_info handles invalid model correctly")

        return True


async def test_clear_cache():
    """Test clear_cache tool"""
    print("\nTesting clear_cache...")

    async with Client(mcp) as client:
        # Note: This test is non-destructive - it won't actually clear real models
        # Test with specific model that doesn't exist
        result = await client.call_tool(
            "clear_cache", {"model_id": "mlx-community/whisper-nonexistent-mlx"}
        )

        data = json.loads(result[0].text)

        # Should get error for non-existent model
        if "error" in data:
            print("✓ clear_cache correctly reports non-existent model")
            print(f"  - Error: {data['error']}")
        else:
            print("✓ clear_cache executed")
            print(f"  - Message: {data.get('message', 'No message')}")
            print(f"  - Cleared models: {data.get('cleared_models', [])}")
            print(f"  - Freed space: {data.get('freed_space', '0MB')}")

        return True


async def test_estimate_processing_time():
    """Test estimate_processing_time tool"""
    print("\nTesting estimate_processing_time...")

    # Create a test audio file
    test_audio = Path("test_estimate.wav")
    subprocess.run(
        [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=10",
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

    try:
        async with Client(mcp) as client:
            # Test with existing file
            result = await client.call_tool(
                "estimate_processing_time", {"file_path": str(test_audio)}
            )

            data = json.loads(result[0].text)

            # Verify structure
            assert "file" in data, "Result should contain 'file'"
            assert "duration" in data, "Result should contain 'duration'"
            assert "duration_formatted" in data, "Result should contain 'duration_formatted'"
            assert "model" in data, "Result should contain 'model'"
            assert "model_speed" in data, "Result should contain 'model_speed'"
            assert "estimated_time" in data, "Result should contain 'estimated_time'"
            assert (
                "estimated_time_formatted" in data
            ), "Result should contain 'estimated_time_formatted'"
            assert "includes_overhead" in data, "Result should contain 'includes_overhead'"
            assert "overhead_seconds" in data, "Result should contain 'overhead_seconds'"

            print("✓ estimate_processing_time working correctly")
            print(f"  - File duration: {data['duration_formatted']}")
            print(f"  - Model speed: {data['model_speed']}")
            print(f"  - Estimated time: {data['estimated_time_formatted']}")

            # Test with non-existent file
            result = await client.call_tool(
                "estimate_processing_time", {"file_path": "nonexistent.mp3"}
            )

            data = json.loads(result[0].text)
            assert "error" in data, "Should return error for non-existent file"

            print("✓ estimate_processing_time handles missing files correctly")

            return True

    finally:
        test_audio.unlink(missing_ok=True)


async def test_validate_media_file():
    """Test validate_media_file tool"""
    print("\nTesting validate_media_file...")

    # Create test files
    valid_audio = Path("test_valid.wav")
    empty_file = Path("test_empty.wav")
    invalid_format = Path("test_invalid.txt")

    # Create valid audio
    subprocess.run(
        [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=5",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(valid_audio),
            "-y",
        ],
        check=True,
        capture_output=True,
    )

    # Create empty file
    empty_file.touch()

    # Create invalid format file
    invalid_format.write_text("This is not an audio file")

    try:
        async with Client(mcp) as client:
            # Test valid audio file
            result = await client.call_tool("validate_media_file", {"file_path": str(valid_audio)})

            data = json.loads(result[0].text)

            assert "is_valid" in data, "Result should contain 'is_valid'"
            assert "file" in data, "Result should contain 'file'"
            assert "format" in data, "Result should contain 'format'"
            assert "file_size" in data, "Result should contain 'file_size'"
            assert "duration" in data, "Result should contain 'duration'"
            assert "duration_formatted" in data, "Result should contain 'duration_formatted'"
            assert "audio_info" in data, "Result should contain 'audio_info'"
            assert "issues" in data, "Result should contain 'issues'"

            assert data["is_valid"] == True, "Valid audio should be valid"
            assert len(data["issues"]) == 0, "Valid audio should have no issues"

            print("✓ validate_media_file correctly validates good audio")
            print(f"  - Format: {data['format']}")
            print(f"  - Duration: {data['duration_formatted']}")
            print(f"  - Size: {data['file_size']}")

            # Test empty file
            result = await client.call_tool("validate_media_file", {"file_path": str(empty_file)})

            data = json.loads(result[0].text)
            assert data["is_valid"] == False, "Empty file should be invalid"
            assert "File is empty" in data["issues"], "Should detect empty file"

            print("✓ validate_media_file detects empty files")

            # Test invalid format
            result = await client.call_tool(
                "validate_media_file", {"file_path": str(invalid_format)}
            )

            data = json.loads(result[0].text)
            assert data["is_valid"] == False, "Text file should be invalid"
            assert any(
                "Unsupported format" in issue for issue in data["issues"]
            ), "Should detect unsupported format"

            print("✓ validate_media_file detects invalid formats")

            # Test non-existent file
            result = await client.call_tool("validate_media_file", {"file_path": "nonexistent.mp3"})

            data = json.loads(result[0].text)
            assert data["is_valid"] == False, "Non-existent file should be invalid"
            assert "File not found" in data.get("error", ""), "Should report file not found"

            print("✓ validate_media_file handles missing files")

            return True

    finally:
        valid_audio.unlink(missing_ok=True)
        empty_file.unlink(missing_ok=True)
        invalid_format.unlink(missing_ok=True)


async def test_get_supported_formats():
    """Test get_supported_formats tool"""
    print("\nTesting get_supported_formats...")

    async with Client(mcp) as client:
        result = await client.call_tool("get_supported_formats", {})

        data = json.loads(result[0].text)

        # Verify structure
        assert "input_formats" in data, "Result should contain 'input_formats'"
        assert "output_formats" in data, "Result should contain 'output_formats'"
        assert "notes" in data, "Result should contain 'notes'"

        # Check input formats
        input_formats = data["input_formats"]
        assert "audio" in input_formats, "Should have audio formats"
        assert "video" in input_formats, "Should have video formats"

        # Check specific formats
        assert ".mp3" in input_formats["audio"], "Should support MP3"
        assert ".wav" in input_formats["audio"], "Should support WAV"
        assert ".mp4" in input_formats["video"], "Should support MP4"
        assert ".mov" in input_formats["video"], "Should support MOV"

        # Check output formats
        output_formats = data["output_formats"]
        assert "txt" in output_formats, "Should support TXT output"
        assert "md" in output_formats, "Should support Markdown output"
        assert "srt" in output_formats, "Should support SRT output"
        assert "json" in output_formats, "Should support JSON output"

        print("✓ get_supported_formats working correctly")
        print(f"  - Audio formats: {len(input_formats['audio'])}")
        print(f"  - Video formats: {len(input_formats['video'])}")
        print(f"  - Output formats: {len(output_formats)}")

        return True


async def test_tools_registration():
    """Test that all support tools are properly registered"""
    print("\nTesting tools registration...")

    async with Client(mcp) as client:
        tools = await client.list_tools()
        tool_names = [tool.name for tool in tools]

        expected_tools = [
            "list_models",
            "get_model_info",
            "clear_cache",
            "estimate_processing_time",
            "validate_media_file",
            "get_supported_formats",
        ]

        all_registered = True
        for expected in expected_tools:
            if expected in tool_names:
                print(f"✓ {expected} tool is registered")
            else:
                print(f"❌ {expected} tool NOT found")
                all_registered = False

        return all_registered


async def main():
    """Run all tests"""
    print("=== Testing Support MCP Tools ===\n")

    tests = [
        test_tools_registration(),
        test_list_models(),
        test_get_model_info(),
        test_clear_cache(),
        test_estimate_processing_time(),
        test_validate_media_file(),
        test_get_supported_formats(),
    ]

    results = await asyncio.gather(*tests)

    passed = sum(results)
    total = len(results)

    print(f"\n=== Test Results: {passed}/{total} passed ===")

    if passed == total:
        print("\n✅ All tests passed! The support tools are working correctly.")
    else:
        print(f"\n❌ {total - passed} tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
