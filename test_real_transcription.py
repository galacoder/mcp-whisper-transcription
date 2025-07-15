#!/usr/bin/env python3
"""
Comprehensive FastMCP Client Test for Whisper Transcription Server
Testing with real audio file using latest FastMCP patterns
"""

import asyncio
import time
import json
from pathlib import Path
from typing import Dict, Any

from fastmcp import Client

# Test Configuration
AUDIO_FILE_PATH = "/Users/sangle/Dev/action/projects/@ai/whisper-transcription/raw_files/gary-quick-call-to-calm-me.m4a"
OUTPUT_DIR = Path("./output")
SERVER_SCRIPT = "src/whisper_mcp_server.py"

class TranscriptionTester:
    """Comprehensive test suite for MCP Whisper Transcription Server."""
    
    def __init__(self):
        self.results = []
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(exist_ok=True)
        
    def log_result(self, test_name: str, result: Dict[str, Any], duration: float = None):
        """Log test result with timing."""
        entry = {
            "test": test_name,
            "result": result,
            "timestamp": time.time(),
            "duration": duration
        }
        self.results.append(entry)
        
        print(f"\n{'='*60}")
        print(f"TEST: {test_name}")
        if duration:
            print(f"Duration: {duration:.2f}s")
        print(f"{'='*60}")
        
        if isinstance(result, dict):
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                # Pretty print key results
                if "text" in result:
                    text_preview = result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"]
                    print(f"📝 Transcription Preview: {text_preview}")
                
                if "duration" in result and "processing_time" in result:
                    speed_ratio = result["duration"] / result["processing_time"] if result["processing_time"] > 0 else 0
                    print(f"⏱️  Audio Duration: {result['duration']:.1f}s")
                    print(f"🚀 Processing Time: {result['processing_time']:.1f}s")
                    print(f"⚡ Speed Ratio: {speed_ratio:.1f}x realtime")
                
                if "model_used" in result:
                    print(f"🤖 Model Used: {result['model_used']}")
                
                if "output_files" in result:
                    print(f"📁 Output Files: {len(result['output_files'])} created")
                    for file in result["output_files"]:
                        print(f"   - {file}")
        else:
            print(f"Result: {result}")

    async def test_in_memory_approach(self):
        """Test using in-memory FastMCP server (recommended for development)."""
        print("\n🧪 Testing In-Memory FastMCP Client Approach")
        
        try:
            # Import the server module to get the mcp instance
            import sys
            sys.path.append(str(Path("src").absolute()))
            from whisper_mcp_server import mcp
            
            start_time = time.time()
            
            # Test with in-memory transport (fastest, recommended for testing)
            async with Client(mcp) as client:
                # Test server connectivity
                await client.ping()
                
                # Basic file validation
                validation_result = await client.call_tool("validate_media_file", {
                    "file_path": AUDIO_FILE_PATH
                })
                self.log_result("In-Memory: File Validation", validation_result.data)
                
                if validation_result.data.get("is_valid"):
                    # Estimate processing time
                    estimate_result = await client.call_tool("estimate_processing_time", {
                        "file_path": AUDIO_FILE_PATH,
                        "model": "mlx-community/whisper-large-v3-turbo"
                    })
                    self.log_result("In-Memory: Time Estimation", estimate_result.data)
                    
                    # Perform transcription
                    transcription_result = await client.call_tool("transcribe_file", {
                        "file_path": AUDIO_FILE_PATH,
                        "model": "mlx-community/whisper-large-v3-turbo",
                        "output_formats": "txt,md,srt,json",
                        "output_dir": str(self.output_dir)
                    })
                    
                    total_time = time.time() - start_time
                    self.log_result("In-Memory: Full Transcription", transcription_result.data, total_time)
                    
                    return transcription_result.data
                else:
                    print("❌ File validation failed - skipping transcription")
                    return validation_result.data
                    
        except Exception as e:
            error_result = {"error": f"In-memory test failed: {str(e)}"}
            self.log_result("In-Memory: Error", error_result)
            return error_result

    async def test_subprocess_approach(self):
        """Test using subprocess FastMCP client (real-world scenario)."""
        print("\n🧪 Testing Subprocess FastMCP Client Approach")
        
        try:
            start_time = time.time()
            
            # Test with subprocess transport (realistic deployment scenario)
            async with Client(SERVER_SCRIPT) as client:
                # Test server connectivity
                await client.ping()
                
                # Test with a different model for comparison
                transcription_result = await client.call_tool("transcribe_file", {
                    "file_path": AUDIO_FILE_PATH,
                    "model": "mlx-community/whisper-base-mlx",  # Faster model
                    "output_formats": "txt,md",
                    "output_dir": str(self.output_dir / "subprocess_test")
                })
                
                total_time = time.time() - start_time
                self.log_result("Subprocess: Base Model Transcription", transcription_result.data, total_time)
                
                return transcription_result.data
                
        except Exception as e:
            error_result = {"error": f"Subprocess test failed: {str(e)}"}
            self.log_result("Subprocess: Error", error_result)
            return error_result

    async def test_mcp_resources(self):
        """Test MCP resource endpoints for monitoring and configuration."""
        print("\n🧪 Testing MCP Resource Endpoints")
        
        try:
            # Use subprocess approach for resources testing
            async with Client(SERVER_SCRIPT) as client:
                # Test configuration resource
                config_result = await client.read_resource("transcription://config")
                self.log_result("Resources: Configuration", json.loads(config_result[0].text))
                
                # Test models resource  
                models_result = await client.read_resource("transcription://models")
                models_data = json.loads(models_result[0].text)
                self.log_result("Resources: Available Models", {
                    "current_model": models_data.get("current_model"),
                    "total_models": len(models_data.get("models", [])),
                    "model_names": [m["id"] for m in models_data.get("models", [])[:3]]  # First 3
                })
                
                # Test formats resource
                formats_result = await client.read_resource("transcription://formats")
                formats_data = json.loads(formats_result[0].text)
                self.log_result("Resources: Supported Formats", {
                    "audio_formats": len(formats_data.get("input_formats", {}).get("audio", {})),
                    "video_formats": len(formats_data.get("input_formats", {}).get("video", {})),
                    "output_formats": list(formats_data.get("output_formats", {}).keys())
                })
                
                # Test performance resource
                perf_result = await client.read_resource("transcription://performance")
                perf_data = json.loads(perf_result[0].text)
                self.log_result("Resources: Performance Stats", perf_data)
                
                # Test transcription history
                history_result = await client.read_resource("transcription://history")
                history_data = json.loads(history_result[0].text)
                self.log_result("Resources: Transcription History", {
                    "total_transcriptions": history_data.get("total_count", 0),
                    "recent_transcriptions": len(history_data.get("transcriptions", []))
                })
                
        except Exception as e:
            error_result = {"error": f"Resources test failed: {str(e)}"}
            self.log_result("Resources: Error", error_result)

    async def test_model_comparison(self):
        """Test different models for performance comparison."""
        print("\n🧪 Testing Model Performance Comparison")
        
        models_to_test = [
            "mlx-community/whisper-tiny-mlx",
            "mlx-community/whisper-base-mlx", 
            "mlx-community/whisper-large-v3-turbo"
        ]
        
        try:
            async with Client(SERVER_SCRIPT) as client:
                comparison_results = []
                
                for model in models_to_test:
                    print(f"\n🔄 Testing model: {model}")
                    start_time = time.time()
                    
                    try:
                        # Get model info first
                        model_info = await client.call_tool("get_model_info", {"model_id": model})
                        
                        # Estimate processing time
                        estimate = await client.call_tool("estimate_processing_time", {
                            "file_path": AUDIO_FILE_PATH,
                            "model": model
                        })
                        
                        # Create model-specific output directory
                        model_dir = self.output_dir / f"model_comparison_{model.split('/')[-1]}"
                        model_dir.mkdir(exist_ok=True)
                        
                        # Perform transcription
                        result = await client.call_tool("transcribe_file", {
                            "file_path": AUDIO_FILE_PATH,
                            "model": model,
                            "output_formats": "txt",
                            "output_dir": str(model_dir)
                        })
                        
                        total_time = time.time() - start_time
                        
                        comparison_entry = {
                            "model": model,
                            "estimated_time": estimate.data.get("estimated_time", 0),
                            "actual_time": result.data.get("processing_time", 0),
                            "speed_ratio": result.data.get("duration", 0) / result.data.get("processing_time", 1),
                            "accuracy": model_info.data.get("accuracy", "Unknown"),
                            "total_test_time": total_time
                        }
                        
                        comparison_results.append(comparison_entry)
                        self.log_result(f"Model Comparison: {model}", comparison_entry, total_time)
                        
                    except Exception as e:
                        error_entry = {"model": model, "error": str(e)}
                        comparison_results.append(error_entry)
                        self.log_result(f"Model Comparison Error: {model}", error_entry)
                
                # Summary comparison
                self.log_result("Model Comparison Summary", {
                    "models_tested": len(comparison_results),
                    "successful_tests": len([r for r in comparison_results if "error" not in r]),
                    "results": comparison_results
                })
                
        except Exception as e:
            error_result = {"error": f"Model comparison failed: {str(e)}"}
            self.log_result("Model Comparison: Error", error_result)

    async def test_output_formats(self):
        """Test different output formats."""
        print("\n🧪 Testing Output Format Generation")
        
        try:
            async with Client(SERVER_SCRIPT) as client:
                # Test all output formats
                formats_dir = self.output_dir / "format_test"
                formats_dir.mkdir(exist_ok=True)
                
                result = await client.call_tool("transcribe_file", {
                    "file_path": AUDIO_FILE_PATH,
                    "model": "mlx-community/whisper-base-mlx",  # Fast model for format testing
                    "output_formats": "txt,md,srt,json",
                    "output_dir": str(formats_dir)
                })
                
                # Analyze generated files
                format_analysis = {}
                for output_file in result.data.get("output_files", []):
                    file_path = Path(output_file)
                    if file_path.exists():
                        file_size = file_path.stat().st_size
                        format_analysis[file_path.suffix] = {
                            "file": str(file_path),
                            "size_bytes": file_size,
                            "size_kb": file_size / 1024,
                            "exists": True
                        }
                        
                        # Read a sample of each format
                        if file_path.suffix == ".txt":
                            with open(file_path, 'r') as f:
                                sample = f.read()[:150]
                                format_analysis[file_path.suffix]["sample"] = sample + "..."
                        elif file_path.suffix == ".json":
                            with open(file_path, 'r') as f:
                                try:
                                    json_data = json.load(f)
                                    format_analysis[file_path.suffix]["segments_count"] = len(json_data.get("segments", []))
                                except json.JSONDecodeError:
                                    format_analysis[file_path.suffix]["error"] = "Invalid JSON"
                
                self.log_result("Output Formats Test", {
                    "transcription_result": {
                        "duration": result.data.get("duration"),
                        "processing_time": result.data.get("processing_time"),
                        "files_created": len(result.data.get("output_files", []))
                    },
                    "format_analysis": format_analysis
                })
                
        except Exception as e:
            error_result = {"error": f"Output formats test failed: {str(e)}"}
            self.log_result("Output Formats: Error", error_result)

    async def run_all_tests(self):
        """Run comprehensive test suite."""
        print("🚀 Starting Comprehensive FastMCP Whisper Transcription Tests")
        print(f"📁 Audio File: {AUDIO_FILE_PATH}")
        print(f"📁 Output Directory: {self.output_dir}")
        
        start_time = time.time()
        
        # Run all test categories
        await self.test_in_memory_approach()
        await self.test_subprocess_approach()
        await self.test_mcp_resources()
        await self.test_output_formats()
        await self.test_model_comparison()
        
        total_time = time.time() - start_time
        
        # Generate comprehensive summary
        print(f"\n{'='*80}")
        print("🎯 COMPREHENSIVE TEST SUMMARY")
        print(f"{'='*80}")
        print(f"📊 Total Tests Run: {len(self.results)}")
        print(f"⏱️  Total Test Duration: {total_time:.2f}s")
        print(f"📁 Output Directory: {self.output_dir}")
        
        # Count successful vs failed tests
        successful_tests = len([r for r in self.results if "error" not in r.get("result", {})])
        failed_tests = len(self.results) - successful_tests
        
        print(f"✅ Successful Tests: {successful_tests}")
        print(f"❌ Failed Tests: {failed_tests}")
        
        # Save detailed results
        results_file = self.output_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "summary": {
                    "total_tests": len(self.results),
                    "successful_tests": successful_tests,
                    "failed_tests": failed_tests,
                    "total_duration": total_time,
                    "audio_file": AUDIO_FILE_PATH,
                    "output_dir": str(self.output_dir)
                },
                "test_results": self.results
            }, f, indent=2, default=str)
        
        print(f"📄 Detailed results saved to: {results_file}")
        
        # List all generated files
        output_files = list(self.output_dir.rglob("*"))
        output_files = [f for f in output_files if f.is_file()]
        
        if output_files:
            print(f"\n📁 Generated Files ({len(output_files)}):")
            for file in sorted(output_files):
                rel_path = file.relative_to(self.output_dir)
                file_size = file.stat().st_size
                print(f"   - {rel_path} ({file_size / 1024:.1f}KB)")
        
        print(f"\n🎉 Testing Complete! Check {self.output_dir} for all generated files.")

async def main():
    """Main test execution."""
    # Ensure we're in the right directory
    if not Path("src/whisper_mcp_server.py").exists():
        print("❌ Error: Please run this script from the project root directory")
        print("   Expected to find: src/whisper_mcp_server.py")
        return
    
    # Check if audio file exists
    if not Path(AUDIO_FILE_PATH).exists():
        print(f"❌ Error: Audio file not found: {AUDIO_FILE_PATH}")
        return
    
    # Run comprehensive tests
    tester = TranscriptionTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())