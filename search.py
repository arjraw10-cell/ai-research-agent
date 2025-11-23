from tavily import TavilyClient
import os
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP


load_dotenv();


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("Bruh put in ur tavily api key as an environment variable. In terminal on windows, just put it in the .env")


tavily = TavilyClient(api_key = TAVILY_API_KEY)

def web_search(query: str):
    """
    Returns structured internet search results, including URLs.
    """
    result = tavily.search(
        query=query,
        max_results=5
    )
    return result["results"]