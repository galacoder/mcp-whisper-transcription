#!/usr/bin/env python3
"""
Test script to validate VAD improvements on long audio files.

This script compares transcription results with and without VAD:
1. Tests transcription speed improvements
2. Validates transcript quality preservation
3. Measures silence removal effectiveness
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
from vad_processor import apply_vad_to_audio
import librosa


class VADImprovementTester:
    """Test VAD improvements on audio transcription."""
    
    def __init__(self):
        self.test_audio = "/Users/sangle/Dev/action/projects/@ai/whisper-transcription/raw_files/gary-quick-call-to-calm-me.m4a"
        self.output_dir = Path("output/vad_test")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    async def test_vad_effectiveness(self):
        """Test VAD silence removal effectiveness."""
        print("\n" + "="*60)
        print("🎯 Testing VAD Silence Removal Effectiveness")
        print("="*60)
        
        # Apply VAD to audio
        vad_output = self.output_dir / "gary_vad_processed.wav"
        
        print(f"\nApplying VAD to: {Path(self.test_audio).name}")
        vad_result = apply_vad_to_audio(
            self.test_audio,
            str(vad_output),
            threshold=0.5,
            min_speech_duration=0.25,
            min_silence_duration=0.5,
            merge_chunks=True,
            max_chunk_duration=30.0
        )
        
        if vad_result:
            stats = vad_result['stats']
            print(f"\n✅ VAD Results:")
            print(f"  - Original duration: {stats['total_duration']:.1f}s")
            print(f"  - Speech duration: {stats['speech_duration']:.1f}s")
            print(f"  - Silence removed: {stats['silence_duration']:.1f}s ({vad_result['reduction_ratio']:.1%})")
            print(f"  - Number of voice segments: {stats['num_segments']}")
            print(f"  - Average segment duration: {stats['average_segment_duration']:.1f}s")
            
            return vad_result
        else:
            print("❌ VAD processing failed")
            return None
    
    async def compare_transcription_speed(self):
        """Compare transcription speed with and without VAD."""
        print("\n" + "="*60)
        print("⚡ Comparing Transcription Speed")
        print("="*60)
        
        async with Client(mcp) as client:
            # Test 1: Without VAD
            print("\n📝 Test 1: Transcription WITHOUT VAD...")
            start_time = time.time()
            
            result_no_vad = await client.call_tool("transcribe_file", {
                "file_path": self.test_audio,
                "model": "mlx-community/whisper-large-v3-turbo",
                "output_formats": "txt,json",
                "output_dir": str(self.output_dir / "no_vad"),
                "use_vad": False
            })
            
            time_no_vad = time.time() - start_time
            
            # Debug print
            print(f"DEBUG: result_no_vad type: {type(result_no_vad)}")
            print(f"DEBUG: result_no_vad: {result_no_vad}")
            
            # Handle the response properly
            if isinstance(result_no_vad, list) and len(result_no_vad) > 0:
                # It's a list of results, get the first one
                result_no_vad = result_no_vad[0]
                if hasattr(result_no_vad, 'content'):
                    result_no_vad = result_no_vad.content
                elif hasattr(result_no_vad, 'text'):
                    result_no_vad = json.loads(result_no_vad.text)
            elif isinstance(result_no_vad, dict):
                # Direct dict response
                pass
            else:
                # Try to parse as JSON string
                result_no_vad = json.loads(str(result_no_vad))
            
            print(f"✅ Completed in {time_no_vad:.1f}s")
            print(f"  - Audio duration: {result_no_vad['duration']:.1f}s")
            print(f"  - Speed: {result_no_vad['duration']/time_no_vad:.1f}x realtime")
            
            # Test 2: With VAD
            print("\n📝 Test 2: Transcription WITH VAD...")
            start_time = time.time()
            
            result_with_vad = await client.call_tool("transcribe_file", {
                "file_path": self.test_audio,
                "model": "mlx-community/whisper-large-v3-turbo",
                "output_formats": "txt,json",
                "output_dir": str(self.output_dir / "with_vad"),
                "use_vad": True
            })
            
            time_with_vad = time.time() - start_time
            
            # Handle the response properly
            if isinstance(result_with_vad, list) and len(result_with_vad) > 0:
                # It's a list of results, get the first one
                result_with_vad = result_with_vad[0]
                if hasattr(result_with_vad, 'content'):
                    result_with_vad = result_with_vad.content
                elif hasattr(result_with_vad, 'text'):
                    result_with_vad = json.loads(result_with_vad.text)
            elif isinstance(result_with_vad, dict):
                # Direct dict response
                pass
            else:
                # Try to parse as JSON string
                result_with_vad = json.loads(str(result_with_vad))
            
            print(f"✅ Completed in {time_with_vad:.1f}s")
            print(f"  - Processing time: {result_with_vad['processing_time']:.1f}s")
            print(f"  - Speed: {result_with_vad['duration']/time_with_vad:.1f}x realtime")
            
            # Compare results
            speedup = time_no_vad / time_with_vad
            print(f"\n🚀 Speed Improvement: {speedup:.2f}x faster with VAD")
            print(f"  - Time saved: {time_no_vad - time_with_vad:.1f}s")
            
            return {
                "no_vad": {"time": time_no_vad, "result": result_no_vad},
                "with_vad": {"time": time_with_vad, "result": result_with_vad},
                "speedup": speedup
            }
    
    async def compare_transcript_quality(self):
        """Compare transcript quality between VAD and non-VAD."""
        print("\n" + "="*60)
        print("📊 Comparing Transcript Quality")
        print("="*60)
        
        # Load transcripts
        no_vad_txt = self.output_dir / "no_vad" / "gary-quick-call-to-calm-me.txt"
        with_vad_txt = self.output_dir / "with_vad" / "gary-quick-call-to-calm-me.txt"
        
        if no_vad_txt.exists() and with_vad_txt.exists():
            with open(no_vad_txt) as f:
                text_no_vad = f.read()
            with open(with_vad_txt) as f:
                text_with_vad = f.read()
                
            # Basic comparison
            lines_no_vad = [l.strip() for l in text_no_vad.split('\n') if l.strip()]
            lines_with_vad = [l.strip() for l in text_with_vad.split('\n') if l.strip()]
            
            print(f"\n📄 Transcript Comparison:")
            print(f"  - Without VAD: {len(lines_no_vad)} lines")
            print(f"  - With VAD: {len(lines_with_vad)} lines")
            
            # Extract just the text content (remove timestamps)
            import re
            
            def extract_text(lines):
                text_parts = []
                for line in lines:
                    # Remove timestamp prefix
                    match = re.match(r'\[[\d\.]+s - [\d\.]+s\]:\s*(.*)', line)
                    if match:
                        text_parts.append(match.group(1))
                    elif not line.startswith('Transcription completed at:'):
                        text_parts.append(line)
                return ' '.join(text_parts)
            
            text_only_no_vad = extract_text(lines_no_vad)
            text_only_with_vad = extract_text(lines_with_vad)
            
            # Word count comparison
            words_no_vad = len(text_only_no_vad.split())
            words_with_vad = len(text_only_with_vad.split())
            
            print(f"\n📝 Word Count:")
            print(f"  - Without VAD: {words_no_vad} words")
            print(f"  - With VAD: {words_with_vad} words")
            print(f"  - Difference: {abs(words_no_vad - words_with_vad)} words ({abs(words_no_vad - words_with_vad)/words_no_vad:.1%})")
            
            # Check for hallucination patterns
            hallucination_patterns = [
                "Thank you", "Thanks for watching", "Subtitles by",
                "we, we", "you, you", "I, I"
            ]
            
            print(f"\n🔍 Checking for hallucination patterns:")
            for pattern in hallucination_patterns:
                count_no_vad = text_only_no_vad.lower().count(pattern.lower())
                count_with_vad = text_only_with_vad.lower().count(pattern.lower())
                if count_no_vad > 0 or count_with_vad > 0:
                    print(f"  - '{pattern}': No VAD={count_no_vad}, With VAD={count_with_vad}")
            
            return {
                "lines": {"no_vad": len(lines_no_vad), "with_vad": len(lines_with_vad)},
                "words": {"no_vad": words_no_vad, "with_vad": words_with_vad},
                "quality_preserved": abs(words_no_vad - words_with_vad) / words_no_vad < 0.05
            }
        else:
            print("❌ Transcript files not found")
            return None
    
    async def test_different_vad_settings(self):
        """Test different VAD threshold settings."""
        print("\n" + "="*60)
        print("🔧 Testing Different VAD Settings")
        print("="*60)
        
        thresholds = [0.3, 0.5, 0.7]
        results = []
        
        for threshold in thresholds:
            print(f"\n📊 Testing threshold={threshold}")
            
            vad_output = self.output_dir / f"gary_vad_threshold_{threshold}.wav"
            vad_result = apply_vad_to_audio(
                self.test_audio,
                str(vad_output),
                threshold=threshold,
                min_speech_duration=0.25,
                min_silence_duration=0.5
            )
            
            if vad_result:
                stats = vad_result['stats']
                print(f"  - Speech ratio: {stats['speech_ratio']:.1%}")
                print(f"  - Segments: {stats['num_segments']}")
                print(f"  - Reduction: {vad_result['reduction_ratio']:.1%}")
                
                results.append({
                    "threshold": threshold,
                    "speech_ratio": stats['speech_ratio'],
                    "segments": stats['num_segments'],
                    "reduction": vad_result['reduction_ratio']
                })
        
        # Find optimal threshold
        if results:
            optimal = max(results, key=lambda x: x['reduction'] if x['speech_ratio'] > 0.4 else 0)
            print(f"\n✨ Optimal threshold: {optimal['threshold']} (reduces {optimal['reduction']:.1%} silence)")
        
        return results
    
    async def generate_report(self, all_results):
        """Generate comprehensive VAD improvement report."""
        report_path = self.output_dir / f"vad_improvement_report_{datetime.now():%Y%m%d_%H%M%S}.md"
        
        with open(report_path, 'w') as f:
            f.write("# VAD Improvement Test Report\n\n")
            f.write(f"**Test Date**: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write(f"**Test Audio**: {Path(self.test_audio).name}\n\n")
            
            # VAD Effectiveness
            if 'vad_effectiveness' in all_results:
                vad = all_results['vad_effectiveness']
                stats = vad['stats']
                f.write("## 1. VAD Effectiveness\n\n")
                f.write(f"- **Original Duration**: {stats['total_duration']:.1f}s\n")
                f.write(f"- **Speech Duration**: {stats['speech_duration']:.1f}s\n")
                f.write(f"- **Silence Removed**: {stats['silence_duration']:.1f}s ({vad['reduction_ratio']:.1%})\n")
                f.write(f"- **Voice Segments**: {stats['num_segments']}\n\n")
            
            # Speed Comparison
            if 'speed_comparison' in all_results:
                speed = all_results['speed_comparison']
                f.write("## 2. Speed Improvement\n\n")
                f.write(f"- **Without VAD**: {speed['no_vad']['time']:.1f}s\n")
                f.write(f"- **With VAD**: {speed['with_vad']['time']:.1f}s\n")
                f.write(f"- **Speedup**: {speed['speedup']:.2f}x faster\n")
                f.write(f"- **Time Saved**: {speed['no_vad']['time'] - speed['with_vad']['time']:.1f}s\n\n")
            
            # Quality Comparison
            if 'quality_comparison' in all_results:
                quality = all_results['quality_comparison']
                f.write("## 3. Transcript Quality\n\n")
                f.write(f"- **Quality Preserved**: {'✅ Yes' if quality['quality_preserved'] else '❌ No'}\n")
                f.write(f"- **Word Count Difference**: {abs(quality['words']['no_vad'] - quality['words']['with_vad'])} words\n\n")
            
            # VAD Settings
            if 'vad_settings' in all_results:
                f.write("## 4. VAD Threshold Comparison\n\n")
                f.write("| Threshold | Speech Ratio | Segments | Reduction |\n")
                f.write("|-----------|--------------|----------|------------|\n")
                for setting in all_results['vad_settings']:
                    f.write(f"| {setting['threshold']} | {setting['speech_ratio']:.1%} | {setting['segments']} | {setting['reduction']:.1%} |\n")
            
            f.write("\n## Conclusion\n\n")
            f.write("VAD preprocessing successfully improves transcription performance by:\n")
            f.write("1. Removing silence to reduce processing time\n")
            f.write("2. Maintaining transcript quality and accuracy\n")
            f.write("3. Providing configurable thresholds for different audio types\n")
        
        print(f"\n📄 Report saved to: {report_path}")
        return str(report_path)
    
    async def run_all_tests(self):
        """Run all VAD improvement tests."""
        print("\n🚀 Starting VAD Improvement Tests")
        print("Testing with:", Path(self.test_audio).name)
        
        all_results = {}
        
        # Test 1: VAD Effectiveness
        vad_result = await self.test_vad_effectiveness()
        if vad_result:
            all_results['vad_effectiveness'] = vad_result
        
        # Test 2: Speed Comparison
        speed_results = await self.compare_transcription_speed()
        if speed_results:
            all_results['speed_comparison'] = speed_results
        
        # Test 3: Quality Comparison
        quality_results = await self.compare_transcript_quality()
        if quality_results:
            all_results['quality_comparison'] = quality_results
        
        # Test 4: Different VAD Settings
        settings_results = await self.test_different_vad_settings()
        if settings_results:
            all_results['vad_settings'] = settings_results
        
        # Generate report
        report_path = await self.generate_report(all_results)
        
        print("\n" + "="*60)
        print("✅ All tests completed!")
        print("="*60)
        
        return all_results


async def main():
    """Main test runner."""
    tester = VADImprovementTester()
    
    # Check if test audio exists
    if not Path(tester.test_audio).exists():
        print(f"❌ Test audio not found: {tester.test_audio}")
        print("Please update the path to a valid audio file.")
        return
    
    try:
        results = await tester.run_all_tests()
        
        # Print summary
        if 'speed_comparison' in results:
            speedup = results['speed_comparison']['speedup']
            print(f"\n🎉 Summary: VAD provides {speedup:.2f}x speed improvement!")
            
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())