import json
import uuid
import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain_core.tools import tool

@tool
def search_web(query: str, max_results: int = 5) -> str:
    """Search web for research leads and return compact JSON list of results"""
    results = []
    with DDGS as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(
                {
                    "title": r.get("title"),
                    "href": r.get("href"),
                    "body": r.get("body")
                }
            )
    return json.dumps(results, ensure_ascii=False)

@tool
def fetch_url(url: str, max_chars: int = 4000) -> str:
    """Fetch a webpage and return readable text content."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    r = httpx.get(url, headers=headers, timeout=20.0, follow_redirects=True)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    
    text = " ".join(soup.get_text(separator=" ").split())
    return text[:max_chars]


@tool
def make_note(title: str, content: str) -> str:
    """Create a note payload to be converted into a SandboxCard"""
    return json.dumps(
        {
            "card_id": str(uuid.uuid4()),
            "title": title,
            "content": content,
        }, 
        ensure_ascii=False
    )