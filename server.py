from mcp.server.fastmcp import FastMCP, server
from search import web_search
from scrape import scrape
Server = FastMCP("research-agent")
Server.tool()(web_search)
Server.tool()(scrape)

if __name__ == "__main__":
    Server.run(transport='stdio')
