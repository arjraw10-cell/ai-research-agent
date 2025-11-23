AI-powered research tool using Tavily Search and Groq LLMs.
This is an AI agent using the MCP protocol for agentic logic
Given a question, the agent:
1. Searches the web using Tavily
2. Summarizes findings using Groq
3. Returns a concise, structured report

Intial Setup:
Create a new file ".env" with no name, only the file extension.
Add your api keys as environment variable
Use the .example if you need help

How to Run
```powershell
py -m pip install -r requirements.txt
py main.py