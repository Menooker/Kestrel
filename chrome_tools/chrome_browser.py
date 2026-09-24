"""Minimal Chrome DevTools MCP accessor with a synchronous facade.

Adapted from NightHeron (benchmark/reference only; NightHeron is never
imported): the useful pieces are a stdio MCP session worker, a page
open/snapshot/close fetcher and the search/open_url tools. The heavy
asynchronous machinery (Context worker loops, pools, rate limiters) is
deliberately not copied. Instead all async runs on one hidden background
event loop and callers only see the blocking ``BrowserTool.search()`` and
``BrowserTool.open_urls()`` methods.

Requires the ``mcp`` Python package and a Chrome instance exposing a DevTools
port (default http://127.0.0.1:19222).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import threading
from typing import Any, cast
from urllib.parse import urlparse

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, TextContent

from chrome_tools.search_engine import SearchEngine, search_url
from chrome_tools.snapshot_cleaner import clean_snapshot

logger = logging.getLogger("chrome_tools.chrome_browser")

CHROME_DEVTOOLS_MCP_VERSION = "1.4.0"

LINK_URL_RE = re.compile(r'url="(?P<url>[^"]*)"')
SELECTED_PAGE_RE = re.compile(r"^\s*(?P<page_id>\d+):\s+.*\s+\[selected\]\s*$", re.MULTILINE)


def mcp_text(result: CallToolResult) -> str:
    """Join all text content blocks of an MCP tool result."""
    content = cast(list[TextContent], result.content)
    return "\n".join(item.text for item in content)


def parse_selected_page_id(pages_text: str) -> int | None:
    """Extract the page id of the freshly opened Chrome tab."""
    match = SELECTED_PAGE_RE.search(pages_text)
    if not match:
        return None
    return int(match.group("page_id"))


def source_from_url(url: str) -> str:
    host = urlparse(url).netloc
    return host or "unknown-source"


class _MCPSession:
    """Own one chrome-devtools-mcp stdio process and its client session.

    The stdio client and client session are entered inside a single long-lived
    owner task and only exited from that same task (anyio cancel scopes must be
    exited from the task that entered them).
    """

    def __init__(self, browser_url: str, *, suppress_server_stderr: bool = True) -> None:
        self.browser_url = browser_url
        self.suppress_server_stderr = suppress_server_stderr
        self._owner_task: asyncio.Task[None] | None = None
        self._ready: asyncio.Event | None = None
        self._stop: asyncio.Event | None = None
        self._open_error: BaseException | None = None
        self.session: ClientSession | None = None

    def _server_params(self) -> StdioServerParameters:
        args = [
            "-y",
            f"chrome-devtools-mcp@{CHROME_DEVTOOLS_MCP_VERSION}",
            f"--browserUrl={self.browser_url}",
            "--no-usage-statistics",
            "--no-performance-crux",
        ]
        return StdioServerParameters(
            command="npx",
            args=args,
            env={
                "CHROME_DEVTOOLS_MCP_NO_UPDATE_CHECKS": "1",
                "npm_config_audit": "false",
                "npm_config_fund": "false",
                "npm_config_prefer_offline": "true",
            },
        )

    async def open(self) -> None:
        if self._owner_task is not None:
            return
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._open_error = None
        self._owner_task = asyncio.create_task(self._run(), name="chrome-mcp-session")
        try:
            await self._ready.wait()
        except BaseException:
            task, self._owner_task = self._owner_task, None
            if task is not None:
                task.cancel()
                try:
                    await task
                except BaseException:
                    pass
            raise
        if self._open_error is not None:
            error = self._open_error
            task, self._owner_task = self._owner_task, None
            if task is not None:
                try:
                    await task
                except BaseException:
                    pass
            raise RuntimeError("chrome-devtools-mcp session failed to start") from error

    async def _run(self) -> None:
        ready = self._ready
        stop = self._stop
        if ready is None or stop is None:
            raise RuntimeError("chrome MCP session lifecycle was not initialized")
        errlog = sys.stderr
        try:
            if self.suppress_server_stderr:
                errlog = open(os.devnull, "w", encoding="utf-8")
            async with stdio_client(self._server_params(), errlog=errlog) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.session = session
                    ready.set()
                    await stop.wait()
        except BaseException as exc:
            if not ready.is_set():
                self._open_error = exc
                ready.set()
            raise
        finally:
            self.session = None
            if errlog is not sys.stderr:
                errlog.close()

    async def close(self) -> None:
        task, self._owner_task = self._owner_task, None
        if task is None:
            return
        if self._stop is not None:
            self._stop.set()
        try:
            await task
        finally:
            self._ready = None
            self._stop = None
            self._open_error = None


class _ChromeFetcher:
    """Async page fetcher: open a tab, take a cleaned snapshot, close the tab."""

    def __init__(
        self,
        browser_url: str,
        *,
        max_wait_ms: int = 60000,
        max_content_length: int | None = None,
        close_opened_pages: bool = True,
    ) -> None:
        self.max_wait_ms = int(max_wait_ms)
        self.max_content_length = max_content_length
        self.close_opened_pages = close_opened_pages
        self._session = _MCPSession(browser_url)

    async def start(self) -> None:
        await self._session.open()

    async def stop(self) -> None:
        await self._session.close()

    async def fetch_snapshot(self, url: str) -> str:
        if self._session.session is None:
            raise RuntimeError("chrome fetcher is not open")
        session = self._session.session
        new_page = await session.call_tool("new_page", {"url": "about:blank", "timeout": self.max_wait_ms})
        page_id = parse_selected_page_id(mcp_text(new_page))
        try:
            await session.call_tool("navigate_page", {"url": url, "timeout": self.max_wait_ms})
            snapshot = await session.call_tool("take_snapshot", {"verbose": False})
            return clean_snapshot(mcp_text(snapshot), max_length=self.max_content_length)
        finally:
            if self.close_opened_pages and page_id is not None:
                try:
                    await session.call_tool("close_page", {"pageId": page_id})
                except Exception as exc:
                    logger.warning("failed to close chrome page pageId=%s: %s", page_id, exc)


class BrowserTool:
    """Synchronous facade over the async Chrome MCP fetcher.

    A single background event loop (daemon thread) owns the fetcher, so the
    rest of the program only calls the blocking methods ``search()`` and
    ``open_urls()``. The chrome-devtools-mcp session is started lazily on the
    first call and reused until ``close()``.
    """

    def __init__(
        self,
        browser_url: str = "http://127.0.0.1:19222",
        *,
        timeout_ms: int = 60000,
        max_content_length: int | None = 15000,
        search_engine: SearchEngine = "google",
        close_opened_pages: bool = True,
    ) -> None:
        self.browser_url = browser_url
        self.timeout_ms = timeout_ms
        self.max_content_length = max_content_length
        self.search_engine = search_engine
        self.close_opened_pages = close_opened_pages
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._loop_lock = threading.Lock()
        self._ready = threading.Event()
        self._fetcher: _ChromeFetcher | None = None

    def search(self, query: str) -> str:
        """Search the web and return the cleaned snapshot text.

        Google sometimes serves a "sorry"/captcha block page to automated
        browsers; when that is detected the query is retried on Bing so the
        model still gets real results.
        """
        snapshot = self._run(self._fetch_snapshot_async(search_url(query, engine=self.search_engine)))
        if self.search_engine == "google" and "/sorry/" in snapshot:
            logger.info("google blocked the search, falling back to bing: %s", query)
            snapshot = self._run(self._fetch_snapshot_async(search_url(query, engine="bing")))
        return snapshot

    def open_urls(self, urls: list[str]) -> dict[str, str]:
        """Open URLs one by one and return {(url): cleaned snapshot text}."""
        return self._run(self._open_urls_async(urls))

    def _fetch_snapshot(self, url: str) -> str:
        return self._run(self._fetch_snapshot_async(url))

    def close(self) -> None:
        fetcher, self._fetcher = self._fetcher, None
        if fetcher is not None and self._loop is not None:
            try:
                self._run(fetcher.stop())
            except BaseException as exc:
                logger.warning("chrome fetcher shutdown failed: %s", exc)
        with self._loop_lock:
            loop, thread = self._loop, self._loop_thread
            self._loop = None
            self._loop_thread = None
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=5)

    # --- internals ---

    async def _fetch_snapshot_async(self, url: str) -> str:
        fetcher = await self._get_fetcher()
        return await fetcher.fetch_snapshot(url)

    async def _open_urls_async(self, urls: list[str]) -> dict[str, str]:
        fetcher = await self._get_fetcher()
        pages: dict[str, str] = {}
        for url in urls:
            try:
                pages[url] = await fetcher.fetch_snapshot(url)
            except Exception as exc:
                logger.warning("fetch failed url=%s: %s", url, exc)
                pages[url] = f"[error] {type(exc).__name__}: {exc}"
        return pages

    async def _get_fetcher(self) -> _ChromeFetcher:
        if self._fetcher is None:
            fetcher = _ChromeFetcher(
                self.browser_url,
                max_wait_ms=self.timeout_ms,
                max_content_length=self.max_content_length,
                close_opened_pages=self.close_opened_pages,
            )
            await fetcher.start()
            self._fetcher = fetcher
        return self._fetcher

    def _run(self, coro: Any) -> Any:
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._loop_lock:
            if self._loop is not None and self._loop.is_running():
                return self._loop
            if self._loop_thread is not None and self._loop_thread.is_alive():
                return self._loop
            self._ready.clear()
            self._loop = None
            self._loop_thread = threading.Thread(
                target=self._run_loop,
                name="chrome-browser-loop",
                daemon=True,
            )
            self._loop_thread.start()
        self._ready.wait()
        with self._loop_lock:
            if self._loop is None:
                raise RuntimeError("chrome browser event loop failed to start")
            return self._loop

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        with self._loop_lock:
            self._loop = loop
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            with self._loop_lock:
                if self._loop is loop:
                    self._loop = None