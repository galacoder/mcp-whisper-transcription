#!/usr/bin/env python3
"""Simple test of async functionality"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from fastmcp import Client
from src.whisper_mcp_server import mcp


async def test_simple():
    """Test basic functionality"""
    async with Client(mcp) as client:
        # Test with a short file (should be sync)
        print("Testing sync processing...")
        result = await client.call_tool("transcribe_file", {
            "file_path": str(Path(__file__).parent / "examples" / "test_short.wav"),
            "output_formats": "txt",
            "model": "mlx-community/whisper-tiny-mlx"
        })
        
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")
        
        # Test forced async
        print("\nTesting forced async...")
        result2 = await client.call_tool("transcribe_file", {
            "file_path": str(Path(__file__).parent / "examples" / "test_short.wav"),
            "output_formats": "txt",
            "model": "mlx-community/whisper-tiny-mlx",
            "force_async": True
        })
        
        print(f"Result2 type: {type(result2)}")
        print(f"Result2: {result2}")


if __name__ == "__main__":
    asyncio.run(test_simple())