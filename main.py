import asyncio
import os
import json
import re
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from groq import Groq

load_dotenv()


class LLMClient:
    """Pluggable LLM client interface."""
    
    def __init__(self, client, model: str):
        self.client = client
        self.model = model
    
    def generate(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 2048) -> str:
        """Generate a response from the LLM."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content


def build_tool_list(tools) -> Dict[str, Dict]:
    """Build a dictionary of available MCP tools.
    
    Args:
        tools: List of MCP tool objects with name, description, inputSchema
        
    Returns:
        Dictionary mapping tool names to their schemas
    """
    tool_dict = {}
    for tool in tools:
        tool_dict[tool.name] = {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": tool.inputSchema
        }
    return tool_dict


def build_system_prompt(tool_dict: Dict[str, Dict]) -> str:
    """Build the system prompt with ReAct instructions.
    
    Args:
        tool_dict: Dictionary of available tools
        
    Returns:
        System prompt string
    """
    tool_descriptions = "\n".join([
        f"- {name}: {info['description']}"
        for name, info in tool_dict.items()
    ])
    
    return f"""You are a research assistant using the ReAct (Reasoning and Acting) framework.

Available tools:
{tool_descriptions}

You must respond using this EXACT format:

PLAN:
[Describe your step-by-step plan to answer the query. List each step clearly.]

ACTION: {{"tool": "tool_name", "arguments": {{"arg1": "value1", "arg2": "value2"}}}}
[Use this ONLY when you need to call a tool. The JSON must be valid and the tool name must exist.]

REFLECTION:
[After receiving tool results, analyze what you learned. Determine if you need more information or can provide the final answer.]

FINAL_ANSWER:
[Provide your comprehensive answer to the user's query. Use this ONLY when you have enough information.]

Rules:
1. Always start with PLAN: section
2. Use ACTION: only when calling a tool - the JSON must be valid
3. After tool results, provide REFLECTION: section
4. Use FINAL_ANSWER: only when ready to answer
5. You can have multiple PLAN → ACTION → REFLECTION cycles before FINAL_ANSWER
6. Choose tools based on reasoning, not fixed sequences
7. If you need more information, create a new PLAN and ACTION"""


def extract_tool_result(tool_result) -> str:
    """Extract string content from MCP tool result.
    
    Args:
        tool_result: MCP tool result object
        
    Returns:
        String representation of the result
    """
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


def parse_action(response: str) -> Optional[Dict]:
    """Parse ACTION section from LLM response.
    
    Args:
        response: LLM response text
        
    Returns:
        Dictionary with 'tool' and 'arguments' keys, or None if no valid action
    """
    action_match = re.search(r'ACTION:\s*(\{.*?\})', response, re.DOTALL)
    if not action_match:
        return None
    
    try:
        action_json = action_match.group(1)
        action = json.loads(action_json)
        
        if not isinstance(action, dict):
            return None
        if "tool" not in action or "arguments" not in action:
            return None
        
        return action
    except json.JSONDecodeError:
        return None


def parse_sections(response: str) -> Dict[str, str]:
    """Parse all sections from LLM response.
    
    Args:
        response: LLM response text
        
    Returns:
        Dictionary with section names as keys and content as values
    """
    sections = {}
    
    plan_match = re.search(r'PLAN:\s*(.*?)(?=ACTION:|REFLECTION:|FINAL_ANSWER:|$)', response, re.DOTALL)
    if plan_match:
        sections['plan'] = plan_match.group(1).strip()
    
    action = parse_action(response)
    if action:
        sections['action'] = action
    
    reflection_match = re.search(r'REFLECTION:\s*(.*?)(?=PLAN:|ACTION:|FINAL_ANSWER:|$)', response, re.DOTALL)
    if reflection_match:
        sections['reflection'] = reflection_match.group(1).strip()
    
    final_match = re.search(r'FINAL_ANSWER:\s*(.*?)$', response, re.DOTALL)
    if final_match:
        sections['final_answer'] = final_match.group(1).strip()
    
    return sections


async def summarize_content(llm: LLMClient, content: str, max_length: int = 2000) -> str:
    """Summarize large content using the LLM.
    
    Args:
        llm: LLM client
        content: Content to summarize
        max_length: Maximum length of summary
        
    Returns:
        Summarized content
    """
    if len(content) <= max_length:
        return content
    
    summary_prompt = f"""Summarize the following content concisely, preserving key information and facts:

{content[:10000]}

Provide a concise summary:"""
    
    messages = [
        {"role": "user", "content": summary_prompt}
    ]
    
    try:
        summary = llm.generate(messages, temperature=0.3, max_tokens=max_length)
        return summary
    except Exception as e:
        print(f"Warning: Summarization failed: {e}")
        return content[:max_length] + "\n\n[Content truncated]"


async def run_step(
    llm: LLMClient,
    session: ClientSession,
    tool_dict: Dict[str, Dict],
    messages: List[Dict],
    iteration: int
) -> Tuple[Optional[str], bool]:
    """Run one ReAct step.
    
    Args:
        llm: LLM client
        session: MCP client session
        tool_dict: Dictionary of available tools
        messages: Current conversation messages
        iteration: Current iteration number
        
    Returns:
        Tuple of (final_answer or None, should_continue)
    """
    print(f"\n--- Iteration {iteration} ---")
    
    response = llm.generate(messages, temperature=0.7, max_tokens=2048)
    print(f"LLM Response:\n{response}\n")
    
    sections = parse_sections(response)
    
    messages.append({
        "role": "assistant",
        "content": response
    })
    
    if 'final_answer' in sections:
        print(f"\n✅ Final Answer: {sections['final_answer']}")
        return sections['final_answer'], False
    
    if 'action' in sections:
        action = sections['action']
        tool_name = action.get('tool')
        tool_args = action.get('arguments', {})
        
        if tool_name not in tool_dict:
            print(f"❌ Error: Tool '{tool_name}' not found. Available tools: {list(tool_dict.keys())}")
            # Repair step
            messages.append({
                "role": "user",
                "content": f"Your last action was invalid. The tool '{tool_name}' does not exist. Available tools are: {list(tool_dict.keys())}. Fix your plan and try again."
            })
            return None, True
        
        print(f"🔧 Executing tool: {tool_name} with args: {tool_args}")
        try:
            result = await session.call_tool(tool_name, tool_args)
            result_str = extract_tool_result(result)
            
            if len(result_str) > 3000:
                print("📝 Summarizing large tool result...")
                result_str = await summarize_content(llm, result_str, max_length=2000)
            
            messages.append({
                "role": "user",
                "content": f"Tool result from {tool_name}:\n{result_str}"
            })
            
            print(f"📊 Tool result (first 200 chars): {result_str[:200]}...")
            
            if 'reflection' not in sections:
                messages.append({
                    "role": "user",
                    "content": "Provide a REFLECTION: section analyzing the tool result. Do you need more information or can you provide a FINAL_ANSWER?"
                })
            
        except Exception as e:
            print(f"❌ Error executing tool: {e}")
            messages.append({
                "role": "user",
                "content": f"Tool execution failed: {str(e)}. Fix your plan and try again."
            })
            return None, True
    
    return None, True


async def run_agent(query: str, llm: Optional[LLMClient] = None):
    """Run the ReAct agent loop.
    
    Args:
        query: User query
        llm: Optional LLM client (defaults to Groq)
        
    Returns:
        Final answer string
    """
    if llm is None:
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        llm = LLMClient(groq_client, "llama-3.1-8b-instant")
    
    # Setup MCP connection
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
            tool_dict = build_tool_list(tools)
            
            print(f"Available MCP tools: {list(tool_dict.keys())}")
            
            system_prompt = build_system_prompt(tool_dict)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            max_iterations = 15
            for iteration in range(1, max_iterations + 1):
                final_answer, should_continue = await run_step(
                    llm, session, tool_dict, messages, iteration
                )
                
                if not should_continue:
                    return final_answer
            
            print("\n⚠️ Reached max iterations")
            last_response = messages[-1].get("content", "")
            final_match = re.search(r'FINAL_ANSWER:\s*(.*?)$', last_response, re.DOTALL)
            if final_match:
                return final_match.group(1).strip()
            return "Maximum iterations reached. Unable to provide a complete answer."


if __name__ == "__main__":
    q = input("Enter your research query: ")
    asyncio.run(run_agent(q))
