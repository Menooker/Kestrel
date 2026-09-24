"""Search engine URL builders, adapted from NightHeron's tools/web.py."""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlencode

SearchEngine = Literal["google", "bing"]
GOOGLE_SEARCH_URL = "https://www.google.com/search"
BING_SEARCH_URL = "https://www.bing.com/search"


def google_search_url(keyword: str) -> str:
    return search_url(keyword, engine="google")


def bing_search_url(keyword: str) -> str:
    return search_url(keyword, engine="bing")


def search_url(keyword: str, *, engine: SearchEngine = "google") -> str:
    query = keyword.strip()
    if not query:
        raise ValueError("search keyword must be non-empty")
    if engine == "google":
        base_url = GOOGLE_SEARCH_URL
    elif engine == "bing":
        base_url = BING_SEARCH_URL
    else:
        raise ValueError(f"unsupported search engine: {engine}")
    return f"{base_url}?{urlencode({'q': query})}"
