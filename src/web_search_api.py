from typing import List

from ddgs import DDGS

from src.logger import logger


class WebSearchAPI:
    def __init__(self):
        """Initialize DuckDuckGo search client (no API key required)."""
        self.ddgs = DDGS()

    def search(self, query: str, num_results: int = 3) -> List[dict]:
        """Search the web using DuckDuckGo and return up to num_results."""
        try:
            results = list(self.ddgs.text(query, max_results=num_results))
            logger.info(f"Web search for '{query}' returned {len(results)} results")
            return results
        except Exception as exc:
            logger.exception(f"Web search failed for '{query}': '{exc}'")
            return []
