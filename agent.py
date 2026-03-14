
from mcp_client import list_mcp_tools, call_mcp_tool
from llm_hf import call_llm
#from llm_ollama import call_llm
from prompt import get_custom_prompt


class ChatAgent:
    
    def needs_tool_execution(self, llm_response) -> bool:
        # Check if the LLM response contains tool_calls
        return hasattr(llm_response, 'tool_calls') and bool(llm_response.tool_calls)
    
    async def run(self, query: str) -> str:
        # 1. Fetch the tools from the MCP Server
        # If your function returns 'response.tools', this list is correct.
        # If it returns the whole response, change to 'await list_mcp_tools().tools'
        tool_list = await list_mcp_tools() 
        

        # 2. Format tools for the LLM
        formatted_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema if hasattr(tool, 'inputSchema') else {},
                 },
            }
            for tool in tool_list 
        ]

        # 3. First LLM Call: Check for Intent
        messages = get_custom_prompt(query)
        print(query)
        print(formatted_tools)
        llm_response = await call_llm(messages, tools=formatted_tools)
        print(llm_response)
        # 4. Step-by-Step Tool Handling
        if self.needs_tool_execution(llm_response):
            tool_call = llm_response.tool_calls[0]
            tool_name = tool_call['name']
            tool_args = tool_call['args']

            # Execute the tool call
            tool_result = await call_mcp_tool(tool_name, tool_args)

            print(tool_result)

            # 5. Second LLM Call: Summarize the results
            # We call get_custom_prompt again, this time passing the tool result
            final_messages = get_custom_prompt(query, tool_result)
            print(final_messages)
            final_response = await call_llm(final_messages)
            print(final_response)
            
            return final_response.content if hasattr(final_response, 'content') else str(final_response)

        # If no tool was needed, just return the first response
        return llm_response.content


async def feed_knowledgebase(text: str):
    """
    Orchestrates the feeding of new information to the MCP server.
    """
    # Define the tool name as registered on your MCP server
    TOOL_NAME = "append_to_knowledge_base"
    
    # Define the arguments matching your server's method signature
    arguments = {"text": text}
    
    print(f"🤖 Agent: Forwarding new knowledge to MCP...")
    
    # Invoke your existing call_mcp_tool helper
    result = await call_mcp_tool(TOOL_NAME, arguments)
    
    return result

# Note: Ensure your call_mcp_tool(tool_name, arguments) 
# helper is either in this file or imported here.
