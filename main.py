from typing import List, Dict, Any, Union
from gradio.components.chatbot import ChatMessage
import json
import os
import asyncio
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from groq import Groq
from dotenv import load_dotenv

# A fair bit of this code is from this git repo: https://github.com/xmassmx/mcp
# It is heavily edited for my use
# I also fixed some error in the original code

load_dotenv()
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

print(MODEL)
loop = asyncio.new_event_loop()


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.client = None
        self.api_key = os.getenv("GROQ_API_KEY")

    def set_api_key(self, api_key: str):
        """Set the Groq API key and initialize the client"""
        self.api_key = api_key
        self.client = Groq(api_key=api_key)
        return "API key set successfully"

    def connect(self, server_script_path: str):
        return loop.run_until_complete(self.connect_to_server(server_script_path))

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command, args=[server_script_path], env=None
        )
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        await self.session.initialize()
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    # Main Message Processor
    async def _process_query(
        self, message: str, history: List[Union[Dict[str, Any], ChatMessage]]
    ):
        result_messages = history.copy()
        result_messages.append({"role": "user", "content": message})
        response = await self.session.list_tools()
        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in response.tools
        ]
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=result_messages,
            stream=False,
            tools=available_tools,
            tool_choice="auto",
            max_completion_tokens=5000,
        )
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        if not tool_calls:
            print(response)
            if response_message.content:
                print("\n\n" + response.choices[0].message.content)
        if response_message.content:
            result_messages.append(
                {"role": "assistant", "content": response_message.content}
            )
        if tool_calls:
            print("\n\n")
            print("TOOL CALLED")
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                print("FUNCTION ARGUMENTS: \n\n")
                print(function_args)
                print("\n\n")
                result = await self.session.call_tool(function_name, function_args)
                result_content = result.content
                if isinstance(result_content, list):
                    result_content = "\n".join(
                        str(item.text) for item in result_content
                    )
                result_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": result_content,
                    }
                )
                print("INFO FROM TOOL CALLED: \n\n\n\n")
                print(function_name)
                print(result_content)
                print("\n\n\n")
                print("Result Messages: ")
                print(result_messages)
            return await self._process_query(
                "Use the results from the tools to answer the question", result_messages
            )

        result_messages.append(
            {"role": "assistant", "content": response_message.content}
        )
        return result_messages

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        # System Prompt, Change if needed
        history = [
            {
                "role": "system",
                "content": """You are an research assistant that does in-depth research helps users by researching topics using available tools. 
                When a user asks a question that requires information:
                1. Use the web_search tool with a clear, specific query string as the 'query' parameter
                2. Review the search results
                3. If you need more details from a specific URL, use the scrape tool with the 'url' parameter
                4. Synthesize the information to answer the user's question
                CRITICAL: When calling tools, you MUST provide ALL required parameters. Check the tool schema to see which parameters are required.""",
            }
        ]

        # loop of Messages
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == "quit":
                    break
                history = await self._process_query(query, history)
                print("\n\n History:")
                print(history)
            except Exception as e:
                print(e)

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    client = MCPClient()
    try:
        await client.connect_to_server("server.py")
        client.set_api_key(os.getenv("GROQ_API_KEY"))
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
