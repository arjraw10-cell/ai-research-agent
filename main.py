from llm import run_llm
from search import web_search
from scrape import scrape_url

def research_agent(query):
    search_results = web_search(query)
    top_url = search_results[0]["url"]
    content = scrape_url(top_url)
    summary = run_llm(f"Summarize this:\n\n{content}")
    return {
        "query": query,
        "url": top_url,
        "summary": summary
    }

if __name__ == "__main__":
    resp = research_agent("What is quantum computing?")
    print(resp)
