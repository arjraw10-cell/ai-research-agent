from firecrawl import FirecrawlApp
import os
import mcp
from dotenv import load_dotenv

load_dotenv();

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

if not FIRECRAWL_API_KEY:
    raise ValueError("Set FIRECRAWL_API_KEY environment variable by creating a new .env file with no name, then follow instructions on example")

fc = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

def scrape(url: str):
    """Scrapes the content off of a url"""
    result = fc.scrape(url)
    try:
        return result.markdown
    except KeyError:
        return None
