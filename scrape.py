from firecrawl import FirecrawlApp
import os
from dotenv import load_dotenv

load_dotenv();

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

if not FIRECRAWL_API_KEY:
    raise ValueError("Set FIRECRAWL_API_KEY env var")

fc = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

def scrape_url(url: str):
    result = fc.scrape_url(url)
    try:
        return result["content"]["text"]
    except KeyError:
        return None
