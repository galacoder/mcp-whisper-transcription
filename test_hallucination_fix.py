#!/usr/bin/env python3
"""
Test script to validate hallucination detection and prevention improvements.

Tests the enhanced transcription with the problematic gary audio file.
"""

import asyncio
import sys
from pathlib import Path
import json
import time
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

from fastmcp import Client
from whisper_mcp_server import mcp


# Audio file that exhibited severe hallucinations
TEST_AUDIO_FILE = "/Users/sangle/Dev/action/projects/@ai/whisper-transcription/raw_files/gary-quick-call-to-calm-me.m4a"
OUTPUT_DIR = Path("output/hallucination_test")


async def test_enhanced_transcription():
    """Test the enhanced transcription with hallucination prevention."""
    print("🧪 Testing Enhanced Transcription with Hallucination Prevention")
    print("=" * 60)
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if test file exists
    if not Path(TEST_AUDIO_FILE).exists():
        print(f"❌ Test audio file not found: {TEST_AUDIO_FILE}")
        return
    
    async with Client(mcp) as client:
        print(f"\n📁 Test Audio: {Path(TEST_AUDIO_FILE).name}")
        print(f"📏 File Size: {Path(TEST_AUDIO_FILE).stat().st_size / (1024*1024):.1f} MB")
        
        # Test with whisper-large-v3-turbo (previously had severe hallucinations)
        print("\n🔄 Testing with whisper-large-v3-turbo model...")
        start_time = time.time()
        
        try:
            response = await client.call_tool("transcribe_file", {
                "file_path": TEST_AUDIO_FILE,
                "model": "mlx-community/whisper-large-v3-turbo",
                "output_formats": "txt,json",
                "output_dir": str(OUTPUT_DIR),
            })
            
            result = json.loads(response[0].text) if hasattr(response[0], 'text') else response
            elapsed = time.time() - start_time
            
            print(f"✅ Transcription completed in {elapsed:.1f}s")
            print(f"⚡ Speed: {result['duration'] / elapsed:.1f}x realtime")
            
            # Analyze the transcript for hallucinations
            analyze_transcript(result, "whisper-large-v3-turbo")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            
        # Test with whisper-base-mlx for comparison
        print("\n🔄 Testing with whisper-base-mlx model...")
        start_time = time.time()
        
        try:
            response = await client.call_tool("transcribe_file", {
                "file_path": TEST_AUDIO_FILE,
                "model": "mlx-community/whisper-base-mlx",
                "output_formats": "txt,json",
                "output_dir": str(OUTPUT_DIR / "base_model"),
            })
            
            result = json.loads(response[0].text) if hasattr(response[0], 'text') else response
            elapsed = time.time() - start_time
            
            print(f"✅ Transcription completed in {elapsed:.1f}s")
            print(f"⚡ Speed: {result['duration'] / elapsed:.1f}x realtime")
            
            # Analyze the transcript for hallucinations
            analyze_transcript(result, "whisper-base-mlx")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")


def analyze_transcript(result: dict, model_name: str):
    """Analyze transcript for hallucination patterns."""
    print(f"\n📊 Analyzing {model_name} transcript for hallucinations...")
    
    text = result.get("text", "")
    segments = result.get("segments", [])
    
    # Check for repetitive patterns
    repetitive_segments = []
    word_repetitions = []
    
    # Analyze segments
    for i, segment in enumerate(segments):
        seg_text = segment.get("text", "").strip()
        
        # Check for excessive word repetitions
        words = seg_text.lower().split()
        if len(words) > 1:
            # Count consecutive repetitions
            for j in range(len(words) - 1):
                if words[j] == words[j + 1]:
                    count = 1
                    k = j + 1
                    while k < len(words) and words[k] == words[j]:
                        count += 1
                        k += 1
                    if count >= 3:
                        word_repetitions.append({
                            "segment": i,
                            "time": segment.get("start", 0),
                            "word": words[j],
                            "count": count,
                            "text": seg_text[:50] + "..." if len(seg_text) > 50 else seg_text
                        })
                        
        # Check for segment-level repetitions
        if i > 0:
            for prev_idx in range(max(0, i-5), i):
                prev_text = segments[prev_idx].get("text", "").strip().lower()
                if seg_text.lower() == prev_text and len(seg_text) > 10:
                    repetitive_segments.append({
                        "segment": i,
                        "time": segment.get("start", 0),
                        "repeats_segment": prev_idx,
                        "text": seg_text[:50] + "..." if len(seg_text) > 50 else seg_text
                    })
    
    # Check for hallucination zones if present
    hallucination_zones = result.get("hallucination_zones", [])
    
    # Print analysis results
    print(f"📈 Total segments: {len(segments)}")
    print(f"🔍 Word repetitions found: {len(word_repetitions)}")
    print(f"🔄 Repetitive segments found: {len(repetitive_segments)}")
    print(f"⚠️  Hallucination zones detected: {len(hallucination_zones)}")
    
    # Show examples of issues found
    if word_repetitions:
        print("\n❌ Word Repetition Examples:")
        for rep in word_repetitions[:3]:  # Show first 3
            print(f"  - At {rep['time']:.1f}s: '{rep['word']}' repeated {rep['count']} times")
            print(f"    Segment: {rep['text']}")
            
    if repetitive_segments:
        print("\n❌ Repetitive Segment Examples:")
        for rep in repetitive_segments[:3]:  # Show first 3
            print(f"  - At {rep['time']:.1f}s: Repeats segment {rep['repeats_segment']}")
            print(f"    Text: {rep['text']}")
            
    if hallucination_zones:
        print("\n⚠️  Hallucination Zones:")
        for zone in hallucination_zones:
            print(f"  - {zone[0]:.1f}s to {zone[1]:.1f}s ({zone[1]-zone[0]:.1f}s duration)")
    
    # Overall quality assessment
    quality_score = 100
    quality_score -= min(50, len(word_repetitions) * 5)  # -5 per word repetition, max -50
    quality_score -= min(30, len(repetitive_segments) * 10)  # -10 per segment repetition, max -30
    quality_score -= min(20, len(hallucination_zones) * 10)  # -10 per zone, max -20
    
    print(f"\n🎯 Quality Score: {quality_score}/100")
    
    if quality_score >= 90:
        print("✅ Excellent quality - minimal hallucinations detected")
    elif quality_score >= 70:
        print("⚠️  Good quality - some minor issues detected")
    elif quality_score >= 50:
        print("⚠️  Fair quality - noticeable hallucination issues")
    else:
        print("❌ Poor quality - significant hallucination problems")
    
    # Save detailed analysis
    analysis_file = OUTPUT_DIR / f"{model_name}_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump({
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "total_segments": len(segments),
            "word_repetitions": word_repetitions,
            "repetitive_segments": repetitive_segments,
            "hallucination_zones": hallucination_zones,
            "quality_score": quality_score,
        }, f, indent=2)
    
    print(f"\n💾 Detailed analysis saved to: {analysis_file}")


if __name__ == "__main__":
    print("🚀 Whisper Hallucination Prevention Test")
    print("Testing enhanced transcription with problematic audio file")
    print()
    
    asyncio.run(test_enhanced_transcription())
    
    print("\n✅ Test completed. Check output directory for results.")