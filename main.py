import asyncio
import os
import json
from typing import Dict, List, Optional, AsyncIterator, Callable, Any, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from groq import Groq
from jsonschema import validate, ValidationError, Draft7Validator

load_dotenv()


# JSON Schema for ReAct response format
REACT_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["plan"],
    "properties": {
        "plan": {
            "type": "string",
            "description": "Step-by-step plan to answer the query"
        },
        "action": {
            "type": "object",
            "properties": {
                "tool": {
                    "type": "string",
                    "description": "Name of the tool to call"
                },
                "arguments": {
                    "type": "object",
                    "description": "Arguments for the tool call"
                }
            },
            "required": ["tool", "arguments"],
            "additionalProperties": False
        },
        "reflection": {
            "type": "string",
            "description": "Analysis of tool results and next steps"
        },
        "final_answer": {
            "type": "string",
            "description": "Final answer to the user's query"
        }
    },
    "additionalProperties": False
}


@dataclass
class StreamEvent:
    """Event for streaming."""
    type: str  # 'llm_output', 'tool_start', 'tool_result', 'tool_error', 'final_answer'
    data: Any
    iteration: Optional[int] = None


class LLMClient:
    """Pluggable LLM client interface with streaming support."""
    
    def __init__(self, client, model: str):
        self.client = client
        self.model = model
    
    def generate(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: Optional[Dict] = None
    ) -> str:
        """Generate a response from the LLM."""
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if response_format:
            params["response_format"] = response_format
        
        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content
    
    async def generate_stream(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: Optional[Dict] = None
    ) -> AsyncIterator[str]:
        """Generate a streaming response from the LLM."""
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        if response_format:
            params["response_format"] = response_format
        
        stream = self.client.chat.completions.create(**params)
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class ToolRegistry:
    """Registry for MCP tools with auto-generated schemas."""
    
    def __init__(self, tools):
        self.tools = {}
        self.schemas = {}
        self._build_registry(tools)
    
    def _build_registry(self, tools):
        """Build tool registry from MCP tools."""
        for tool in tools:
            self.tools[tool.name] = {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema
            }
            # Store validator for each tool
            if tool.inputSchema:
                try:
                    self.schemas[tool.name] = Draft7Validator(tool.inputSchema)
                except Exception as e:
                    print(f"Warning: Could not create validator for {tool.name}: {e}")
    
    def get_tool(self, name: str) -> Optional[Dict]:
        """Get tool definition by name."""
        return self.tools.get(name)
    
    def validate_arguments(self, tool_name: str, arguments: Dict) -> Tuple[bool, Optional[str]]:
        """Validate tool arguments against schema."""
        if tool_name not in self.tools:
            return False, f"Tool '{tool_name}' not found"
        
        validator = self.schemas.get(tool_name)
        if not validator:
            return True, None  # No schema to validate against
        
        try:
            validator.validate(arguments)
            return True, None
        except ValidationError as e:
            return False, f"Validation error: {e.message}"
    
    def generate_tool_documentation(self) -> str:
        """Auto-generate tool documentation for LLM."""
        docs = []
        for name, tool in self.tools.items():
            docs.append(f"Tool: {name}")
            docs.append(f"  Description: {tool['description']}")
            if tool.get('inputSchema'):
                props = tool['inputSchema'].get('properties', {})
                if props:
                    docs.append("  Arguments:")
                    for arg_name, arg_schema in props.items():
                        arg_type = arg_schema.get('type', 'any')
                        arg_desc = arg_schema.get('description', 'No description')
                        required = arg_name in tool['inputSchema'].get('required', [])
                        req_marker = " (required)" if required else " (optional)"
                        docs.append(f"    - {arg_name} ({arg_type}){req_marker}: {arg_desc}")
            docs.append("")
        return "\n".join(docs)


class JSONRepair:
    """Handles JSON repair attempts."""
    
    @staticmethod
    async def repair_json(
        llm: LLMClient,
        invalid_json: str,
        schema_description: str,
        max_attempts: int = 3
    ) -> Optional[Dict]:
        """Attempt to repair invalid JSON using LLM."""
        for attempt in range(max_attempts):
            repair_prompt = f"""The following JSON is invalid. Fix it to be valid JSON that matches this schema:

Schema: {schema_description}

Invalid JSON:
{invalid_json}

Return ONLY valid JSON, no explanation, no markdown, no code blocks."""
            
            try:
                response = llm.generate(
                    [{"role": "user", "content": repair_prompt}],
                    temperature=0.1,
                    max_tokens=2048
                )
                # Try to extract JSON
                cleaned = response.strip()
                if cleaned.startswith("```"):
                    # Remove code blocks
                    cleaned = cleaned.split("```")[1]
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:]
                    cleaned = cleaned.strip()
                
                parsed = json.loads(cleaned)
                return parsed
            except Exception as e:
                if attempt == max_attempts - 1:
                    return None
                continue
        return None


class AgentController:
    """Main agent controller with ReAct loop."""
    
    def __init__(
        self,
        session: ClientSession,
        tool_registry: ToolRegistry,
        llm: LLMClient,
        stream_callback: Optional[Callable[[StreamEvent], None]] = None
    ):
        self.session = session
        self.tool_registry = tool_registry
        self.llm = llm
        self.stream_callback = stream_callback
        self.messages: List[Dict] = []
        self.iteration = 0
        self._build_system_prompt()
    
    def _build_system_prompt(self):
        """Auto-generate system prompt from tools."""
        tool_docs = self.tool_registry.generate_tool_documentation()
        
        prompt = f"""You are an AI assistant using the ReAct (Reasoning and Acting) framework.

Available tools:
{tool_docs}

You MUST respond with ONLY valid JSON in this exact format:
{{
    "plan": "Your step-by-step plan",
    "action": {{"tool": "tool_name", "arguments": {{"arg": "value"}}}},  // Optional, only when calling a tool
    "reflection": "Your analysis after tool results",  // Optional
    "final_answer": "Your final answer"  // Optional, only when ready to answer
}}

CRITICAL RULES:
1. ALWAYS include "plan" - describe your reasoning and steps in detail
2. For research queries, information lookups, or any question requiring current/external data, you MUST use tools
3. Include "action" when you need to call a tool - DO NOT skip tools if the query requires information you don't have
4. Include "reflection" after receiving tool results - analyze what you learned
5. Include "final_answer" ONLY when you have enough information from tools or are certain of the answer
6. The JSON must be valid and match the schema exactly
7. No text, no markdown, no code blocks - ONLY JSON
8. Tool names must exactly match available tools
9. Tool arguments must match the tool's schema

IMPORTANT: If the user asks about current events, recent information, or anything requiring external data, you MUST use the available tools. Do not provide answers without using tools when they are needed."""
        
        self.system_prompt = prompt
    
    def _emit_stream_event(self, event_type: str, data: Any):
        """Emit streaming event if callback is set."""
        if self.stream_callback:
            event = StreamEvent(
                type=event_type,
                data=data,
                iteration=self.iteration
            )
            self.stream_callback(event)
    
    async def _validate_and_repair_response(self, response: str) -> Optional[Dict]:
        """Validate JSON response and attempt repair if needed."""
        # Try to extract JSON from response
        cleaned = response.strip()
        
        # Remove markdown code blocks if present
        if cleaned.startswith("```"):
            parts = cleaned.split("```")
            if len(parts) >= 3:
                cleaned = parts[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
                cleaned = cleaned.strip()
        
        # Try to parse JSON
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # Attempt repair
            self._emit_stream_event("llm_output", {"status": "repairing_invalid_json"})
            schema_desc = json.dumps(REACT_RESPONSE_SCHEMA, indent=2)
            parsed = await JSONRepair.repair_json(self.llm, cleaned, schema_desc)
            if not parsed:
                return None
        
        # Validate against schema
        try:
            validate(instance=parsed, schema=REACT_RESPONSE_SCHEMA)
        except ValidationError as e:
            # Attempt repair
            self._emit_stream_event("llm_output", {"status": "repairing_schema_violation", "error": str(e)})
            schema_desc = json.dumps(REACT_RESPONSE_SCHEMA, indent=2)
            parsed = await JSONRepair.repair_json(self.llm, json.dumps(parsed), schema_desc)
            if not parsed:
                return None
            try:
                validate(instance=parsed, schema=REACT_RESPONSE_SCHEMA)
            except ValidationError:
                return None
        
        return parsed
    
    async def _execute_tool(self, tool_name: str, arguments: Dict) -> Tuple[bool, str]:
        """Execute a tool and return result."""
        # Validate tool exists
        if tool_name not in self.tool_registry.tools:
            return False, f"Tool '{tool_name}' not found. Available: {list(self.tool_registry.tools.keys())}"
        
        # Validate arguments
        valid, error = self.tool_registry.validate_arguments(tool_name, arguments)
        if not valid:
            return False, error
        
        # Execute tool
        self._emit_stream_event("tool_start", {"tool": tool_name, "arguments": arguments})
        try:
            result = await self.session.call_tool(tool_name, arguments)
            result_str = self._extract_tool_result(result)
            
            # Summarize if too large
            if len(result_str) > 3000:
                self._emit_stream_event("tool_result", {"tool": tool_name, "status": "summarizing"})
                result_str = await self._summarize_content(result_str)
            
            self._emit_stream_event("tool_result", {"tool": tool_name, "result_preview": result_str[:200]})
            return True, result_str
        except Exception as e:
            error_msg = str(e)
            self._emit_stream_event("tool_error", {"tool": tool_name, "error": error_msg})
            return False, error_msg
    
    def _extract_tool_result(self, tool_result) -> str:
        """Extract string from MCP tool result."""
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
    
    async def _summarize_content(self, content: str, max_length: int = 2000) -> str:
        """Summarize large content."""
        if len(content) <= max_length:
            return content
        
        summary_prompt = f"""Summarize this content concisely, preserving key facts:

{content[:10000]}

Return ONLY a concise summary:"""
        
        try:
            summary = self.llm.generate(
                [{"role": "user", "content": summary_prompt}],
                temperature=0.3,
                max_tokens=max_length
            )
            return summary
        except Exception as e:
            return content[:max_length] + "\n\n[Content truncated]"
    
    async def step(self, query: Optional[str] = None, use_streaming: bool = False) -> Tuple[Optional[str], bool]:
        """Execute one ReAct step.
        
        Returns:
            Tuple of (final_answer or None, should_continue)
        """
        self.iteration += 1
        
        # Initialize messages if first step
        if not self.messages:
            self.messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            if query:
                self.messages.append({"role": "user", "content": query})
        
        # Generate response
        print(f"🤔 Generating response (iteration {self.iteration})...")
        self._emit_stream_event("llm_output", {"status": "generating"})
        
        if use_streaming:
            full_response = ""
            async for chunk in self.llm.generate_stream(
                self.messages,
                temperature=0.7,
                max_tokens=2048,
                response_format={"type": "json_object"}
            ):
                full_response += chunk
                self._emit_stream_event("llm_output", {"chunk": chunk})
            response = full_response
        else:
            response = self.llm.generate(
                self.messages,
                temperature=0.7,
                max_tokens=2048,
                response_format={"type": "json_object"}
            )
            print(f"📥 Raw JSON Response:\n{response}\n")
        
        # Validate and parse response
        parsed = await self._validate_and_repair_response(response)
        if not parsed:
            print("❌ Invalid JSON response received")
            error_msg = "Invalid JSON response. Please respond with valid JSON only."
            self.messages.append({"role": "user", "content": error_msg})
            return None, True
        
        # Print parsed response sections
        print("\n📋 Parsed Response:")
        if "plan" in parsed and parsed.get("plan"):
            print(f"  PLAN: {parsed['plan']}")
        if "reflection" in parsed and parsed.get("reflection"):
            print(f"  REFLECTION: {parsed['reflection']}")
        if "action" in parsed and parsed.get("action"):
            action = parsed["action"]
            print(f"  ACTION: {action.get('tool')} with args: {action.get('arguments')}")
        if "final_answer" in parsed and parsed.get("final_answer"):
            print(f"  FINAL_ANSWER: {parsed['final_answer']}")
        
        # Add response to messages
        self.messages.append({
            "role": "assistant",
            "content": json.dumps(parsed, indent=2)
        })
        
        # Check for final answer
        if "final_answer" in parsed and parsed["final_answer"]:
            self._emit_stream_event("final_answer", {"answer": parsed["final_answer"]})
            return parsed["final_answer"], False
        
        # Execute action if present
        if "action" in parsed and parsed["action"]:
            action = parsed["action"]
            tool_name = action.get("tool")
            tool_args = action.get("arguments", {})
            
            success, result = await self._execute_tool(tool_name, tool_args)
            
            if not success:
                # Tool execution failed
                print(f"❌ Tool execution failed: {result}")
                self.messages.append({
                    "role": "user",
                    "content": f"Tool execution failed: {result}. Fix your plan and try again."
                })
                return None, True
            
            # Print tool result preview
            result_preview = result[:300] if len(result) > 300 else result
            print(f"📊 Tool Result Preview: {result_preview}...")
            
            # Add tool result to messages
            self.messages.append({
                "role": "user",
                "content": f"Tool '{tool_name}' result:\n{result}"
            })
            
            # Prompt for reflection if not present
            if "reflection" not in parsed or not parsed.get("reflection"):
                print("💭 No reflection provided, prompting for one...")
                self.messages.append({
                    "role": "user",
                    "content": "Provide a reflection analyzing the tool result. Do you need more information or can you provide a final answer?"
                })
            else:
                print(f"💭 Reflection provided: {parsed['reflection']}")
        
        return None, True


async def create_agent(
    server_command: str = "py",
    server_args: List[str] = None,
    llm: Optional[LLMClient] = None,
    stream_callback: Optional[Callable[[StreamEvent], None]] = None
):
    """Factory function to create an agent with MCP tools.
    
    Args:
        server_command: Command to run MCP server (default: "py")
        server_args: Arguments for server command (default: ["server.py"])
        llm: Optional LLM client (defaults to Groq)
        stream_callback: Optional callback for streaming events
        
    Returns:
        Async context manager that yields AgentController
    """
    if server_args is None:
        server_args = ["server.py"]
    
    if llm is None:
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        llm = LLMClient(groq_client, "llama-3.1-8b-instant")
    
    params = StdioServerParameters(
        command=server_command,
        args=server_args,
        env=os.environ.copy(),
    )
    
    class AgentContext:
        """Context manager for agent lifecycle."""
        
        async def __aenter__(self):
            self.stdio_ctx = stdio_client(params)
            self.read_stream, self.write_stream = await self.stdio_ctx.__aenter__()
            
            self.session = ClientSession(self.read_stream, self.write_stream)
            await self.session.__aenter__()
            await self.session.initialize()
            
            tools_result = await self.session.list_tools()
            tool_registry = ToolRegistry(tools_result.tools)
            
            print(f"✅ Agent initialized with tools: {list(tool_registry.tools.keys())}")
            
            self.agent = AgentController(self.session, tool_registry, llm, stream_callback)
            return self.agent
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            await self.session.__aexit__(exc_type, exc_val, exc_tb)
            await self.stdio_ctx.__aexit__(exc_type, exc_val, exc_tb)
    
    return AgentContext()


async def run_agent(query: str, max_iterations: int = 15, use_streaming: bool = False):
    """Run agent with a query.
    
    Args:
        query: User query
        max_iterations: Maximum number of iterations
        use_streaming: Whether to use streaming for LLM output
        
    Returns:
        Final answer string
    """
    def stream_handler(event: StreamEvent):
        """Handle streaming events."""
        if event.type == "llm_output" and "chunk" in event.data:
            print(event.data["chunk"], end="", flush=True)
        elif event.type == "tool_start":
            print(f"\n🔧 Executing: {event.data['tool']}")
        elif event.type == "tool_result":
            print(f"✅ Tool completed")
        elif event.type == "tool_error":
            print(f"❌ Tool error: {event.data['error']}")
        elif event.type == "final_answer":
            print(f"\n\n✅ Final Answer: {event.data['answer']}")
    
    async with (await create_agent(stream_callback=stream_handler)) as agent:
        for iteration in range(1, max_iterations + 1):
            print(f"\n--- Iteration {iteration} ---")
            final_answer, should_continue = await agent.step(query if iteration == 1 else None, use_streaming)
            
            if not should_continue:
                return final_answer
        
        print("\n⚠️ Reached max iterations")
        return "Maximum iterations reached."


if __name__ == "__main__":
    q = input("Enter your research query: ")
    asyncio.run(run_agent(q, use_streaming=False))
