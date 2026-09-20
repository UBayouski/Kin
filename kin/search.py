"""Live web search through Tavily, so Kin can answer with current facts instead of guessing."""

from urllib.parse import urlparse

import httpx

from kin import config

API_URL = "https://api.tavily.com/search"
_SNIPPET_CHARS = 220


def available() -> bool:
    return bool(config.tavily_api_key())


def search(query: str, topic: str = "general", max_results: int = 4) -> str:
    """Search the web and return a compact text digest for the model: a short answer, then sourced snippets."""
    body = {
        "query": query,
        "topic": topic if topic in ("general", "news") else "general",
        "max_results": max_results,
        "include_answer": "basic",
    }
    if body["topic"] == "news":
        body["time_range"] = "week"
    response = httpx.post(
        API_URL, json=body, headers={"Authorization": f"Bearer {config.tavily_api_key()}"}, timeout=20
    )
    response.raise_for_status()
    return _digest(response.json())


def _digest(data: dict) -> str:
    lines = []
    if data.get("answer"):
        lines.append(f"Summary: {data['answer']}")
    for result in data.get("results", []):
        source = urlparse(result.get("url", "")).netloc.removeprefix("www.")
        snippet = " ".join(result.get("content", "").split())[:_SNIPPET_CHARS]
        lines.append(f"- [{source}] {result.get('title', '')}: {snippet}")
    return "\n".join(lines) or "no results"
