from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv();


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

print('TAVILY_API_KEY' in os.environ)

if not TAVILY_API_KEY:
    raise ValueError("Bruh put in ur tavily api key as an environment variable. In terminal on windows, just put it in the .env")


tavily = TavilyClient(api_key = TAVILY_API_KEY)

def web_search(query, max_results=5):
    """
    Returns structured internet search results.
    """
    result = tavily.search(
        query=query,
        max_results=max_results
    )

    return result["results"]
print(web_search("What is the biggest contributing factor to climate change?"))