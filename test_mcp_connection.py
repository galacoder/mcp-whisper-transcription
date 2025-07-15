#!/usr/bin/env python3
"""
Test script to verify MCP server can start without errors
"""
import sys
import json
import asyncio
from pathlib import Path

# Add the project directory to Python path
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

async def test_mcp_server():
    """Test that MCP server can be imported and initialized"""
    try:
        # Import the MCP server module
        from src.whisper_mcp_server import mcp
        
        print("✅ MCP server module imported successfully")
        
        # Test that we can access the server instance
        if hasattr(mcp, 'list_tools'):
            tools = await mcp.list_tools()
            print(f"✅ Server has {len(tools.tools)} tools available")
            
            # List the tools
            for tool in tools.tools:
                print(f"   - {tool.name}: {tool.description}")
        
        if hasattr(mcp, 'list_resources'):
            resources = await mcp.list_resources()
            print(f"✅ Server has {len(resources.resources)} resources available")
            
            # List the resources
            for resource in resources.resources:
                print(f"   - {resource.uri}: {resource.description}")
        
        print("✅ MCP server test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error testing MCP server: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_mcp_server())
    sys.exit(0 if success else 1)