import os
from mcp import ClientSession
from mcp.client.sse import sse_client



MCP_URL = f"http://127.0.0.1:8000/sse"

async def list_mcp_tools():
    """Fetches the list of tools from the running MCP server."""
    try:
        async with sse_client(url=MCP_URL) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                response = await session.list_tools()
                return response.tools
    except Exception as e:
        print(f"Error: Could not connect to MCP server on {MCP_URL}. {e}")
        return []
      
async def call_mcp_tool(tool_name: str, arguments: dict):
    """Invokes an MCP tool and returns the text result."""
    try:
        async with sse_client(url=MCP_URL) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                
                # Extract the first text content block
                if result.content:
                    return result.content[0].text
                return "No information found."
    except Exception as e:
        return f"Client error calling tool '{tool_name}': {str(e)}"
    
