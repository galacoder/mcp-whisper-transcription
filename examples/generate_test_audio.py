#!/usr/bin/env python3
"""
Generate test audio files for testing the Whisper MCP server.
Creates short audio files with speech synthesis.
"""

import os
import sys
import subprocess
from pathlib import Path

def generate_test_audio():
    """Generate test audio files using system TTS."""
    
    examples_dir = Path(__file__).parent
    
    # Test texts for different durations
    test_cases = [
        {
            "name": "test_short.wav",
            "text": "Hello, this is a short test audio file for Whisper transcription.",
            "description": "10 second test"
        },
        {
            "name": "test_medium.wav", 
            "text": "This is a medium length test audio file. It contains multiple sentences to test the Whisper transcription accuracy. The transcription should handle punctuation and capitalization correctly. This audio file is approximately thirty seconds long.",
            "description": "30 second test"
        },
        {
            "name": "test_numbers.wav",
            "text": "Testing numbers: one, two, three, four, five. The year is 2024. The temperature is 72 degrees Fahrenheit. There are 365 days in a year.",
            "description": "Number transcription test"
        }
    ]
    
    print("Generating test audio files...")
    
    for test in test_cases:
        output_path = examples_dir / test["name"]
        
        # Use macOS 'say' command to generate audio
        if sys.platform == "darwin":  # macOS
            cmd = [
                "say",
                "-o", str(output_path),
                "--data-format=LEF32@22050",  # 22kHz sample rate
                test["text"]
            ]
            
            try:
                subprocess.run(cmd, check=True)
                print(f"✓ Generated {test['name']} - {test['description']}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to generate {test['name']}: {e}")
        else:
            print(f"⚠ Platform {sys.platform} not supported. Please add test audio files manually.")
            break
    
    # Create a README for the examples
    readme_content = """# Test Audio Files

This directory contains test audio files for the Whisper MCP server.

## Files:
- `test_short.wav` - 10 second test audio
- `test_medium.wav` - 30 second test audio  
- `test_numbers.wav` - Number transcription test

## Generating Test Files:
Run `python generate_test_audio.py` to regenerate test files (macOS only).

For other platforms, you can:
1. Record your own test audio files
2. Download sample audio from free sources
3. Use online TTS services to generate test files
"""
    
    readme_path = examples_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"\n✓ Created README.md in examples directory")

if __name__ == "__main__":
    generate_test_audio()