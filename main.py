import asyncio
import os
import json
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from groq import Groq

load_dotenv()

groqLLM =  Groq(api_key = os.getenv("GROQ_API_KEY"))


def mcp_tools_to_openai_format(tools):
    """Convert MCP tools to OpenAI function calling format."""
    tool_descriptions = {
        "web_search": "Search the web for information about a topic. Returns search results with URLs, titles, and snippets. Use this FIRST to discover relevant URLs and get an overview. After getting search results, consider using the scrape tool on the most relevant URLs for detailed information.",
        "scrape": "Scrape detailed content from a specific URL. Returns the full markdown content of the webpage. Use this AFTER web_search to get comprehensive, detailed information from URLs found in search results. Extract URLs from web_search results and scrape the most relevant ones (typically 1-3 URLs) for in-depth content."
    }
    
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool_descriptions.get(tool.name, tool.description or ""),
                "parameters": getattr(tool, 'inputSchema', None) or getattr(tool, 'input_schema', None) or {}
            }
        }
        for tool in tools
    ]


def extract_tool_result(tool_result):
    """Extract string content from MCP tool result (handles various formats)."""
    if hasattr(tool_result, 'content'):

        parts = []
        for item in tool_result.content:
            if hasattr(item, 'text'):
                parts.append(item.text)
            elif isinstance(item, dict):
                parts.append(json.dumps(item))
            else:
                parts.append(str(item))
        return "\n".join(parts) if parts else str(tool_result)
    elif isinstance(tool_result, (dict, list)):
        return json.dumps(tool_result, indent=2)
    else:
        return str(tool_result)


def truncate_content(content: str, max_chars: int = 5000) -> str:
    """Truncate content to prevent token limit errors."""
    if len(content) <= max_chars:
        return content
    
    truncated = content[:max_chars]
    return f"{truncated}\n\n[Content truncated - showing first {max_chars} characters of {len(content)} total]"


def limit_message_history(messages: list, max_messages: int = 15) -> list:
    """Limit message history to prevent token overflow. Keeps system message and recent messages."""
    if len(messages) <= max_messages:
        return messages
    
    system_msg = None
    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        system_msg = messages[0]
    
    if system_msg:
        recent_messages = messages[-(max_messages-1):]
        return [system_msg] + recent_messages
    else:
        return messages[-max_messages:]


async def run_agent(query: str):
    """Run an agentic loop using MCP tools and Groq LLM."""
    params = StdioServerParameters(
        command="py",
        args=["server.py"],
        env=os.environ.copy(),
    )

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            tools = tools_result.tools
            print(f"Available MCP tools: {[t.name for t in tools]}")

            tools_for_llm = mcp_tools_to_openai_format(tools)

            messages = [
                {
                    "role": "system",
                    "content": """You are a helpful research assistant with access to two powerful tools:

1. **web_search**: Use this to find relevant URLs and get an overview of information on a topic. This returns search results with URLs, titles, and snippets.

2. **scrape**: Use this to get detailed, comprehensive content from specific URLs. This returns the full markdown content of webpages.

**Recommended workflow for research queries:**
- Start with web_search to discover relevant URLs and get an overview
- Review the search results and identify the most relevant URLs (typically 1-3 URLs)
- Use scrape on those URLs to get detailed, comprehensive information
- Synthesize information from both search results and scraped content

**When to use both tools:**
- For research queries, news topics, or when you need comprehensive information, use BOTH tools: search first, then scrape the most relevant URLs
- For simple factual questions, web_search alone may be sufficient
- If you already have a specific URL and need detailed content, you can use scrape directly

Use your judgment to determine when both tools would be helpful, but generally prefer using both for thorough research."""
                },
                {"role": "user", "content": query}
            ]

            for iteration in range(1, 11): 
                print(f"\n--- Iteration {iteration} ---")
                
                messages = limit_message_history(messages, max_messages=15)

                response = groqLLM.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=messages,
                    tools=tools_for_llm,
                    tool_choice="auto",
                    temperature=0.7,
                    max_tokens=2048
                )

                msg = response.choices[0].message
                messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                        }
                        for tc in (msg.tool_calls or [])
                    ]
                })
                if not msg.tool_calls:
                    print(f"\n✅ Final answer: {msg.content}")
                    return msg.content

                for tool_call in msg.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    print(f"🔧 Calling MCP tool: {tool_name}({tool_args})")
                    
                    result = await session.call_tool(tool_name, tool_args)
                    result_str = extract_tool_result(result)
                    
                    max_chars = 5000 if tool_name == "scrape" else 8000
                    result_str = truncate_content(result_str, max_chars=max_chars)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": result_str
                    })
                    
                    print(f"📊 Result: {result_str[:150]}...")
                
                messages = limit_message_history(messages, max_messages=15)

            print("\n⚠️ Reached max iterations")
            return messages[-1].get("content", "No response")


if __name__ == "__main__":
    q = input("Enter your research query: ")
    asyncio.run(run_agent(q))
