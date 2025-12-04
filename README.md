AI-powered research tool using Tavily Search, Firecrawl Scraping and Groq LLMs.
This is an AI agent using the MCP protocol for agentic logic
Given a question, the agent:

1. Searches the web using Tavily
2. Summarizes findings using Groq
3. Returns a concise, structured report

For me, I find this very useful as a sort of framework to build
MCP Agents off of. I use this a lot in hackathons so I can quickly
build MCP Agents.

Intial Setup:
Create a new file ".env" with no name, only the file extension.
Add your api keys as environment variable
Use the .example if you need help

How to Run

```powershell
py -m pip install -r requirements.txt
py main.py
```
