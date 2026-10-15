"""Bounded, same-site website crawl helpers for LogHawk RAG ingestion."""

from __future__ import annotations

from collections import deque
import hashlib
import ipaddress
import socket
import time
import zipfile
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

import loghawk.config as config
from loghawk.utils.lancedbutils3 import SUPPORTED_SUFFIXES, load_document_bytes


_USER_AGENT = "LogHawkRAGBot/1.0"
_REDIRECT_CODES = {301, 302, 303, 307, 308}
_CONTENT_TYPE_SUFFIXES = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/rtf": ".rtf",
    "text/rtf": ".rtf",
    "text/html": ".html",
}


def _normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Only http:// and https:// URLs can be crawled.")
    if not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Enter a valid URL without embedded credentials.")
    if parsed.port not in {None, 80, 443}:
        raise ValueError("Website crawling only supports standard HTTP/HTTPS ports.")
    scheme = parsed.scheme.lower()
    hostname = parsed.hostname.lower()
    authority = f"[{hostname}]" if ":" in hostname else hostname
    port = parsed.port
    if port is not None and port != (443 if scheme == "https" else 80):
        authority = f"{authority}:{port}"
    normalized = parsed._replace(
        scheme=scheme,
        netloc=authority,
        fragment="",
    )
    return urldefrag(urlunparse(normalized)).url


def _validate_public_url(url: str, root_host: str) -> None:
    parsed = urlparse(_normalize_url(url))
    if parsed.hostname.lower() != root_host:
        raise ValueError("The crawler is restricted to the starting website host.")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(parsed.hostname, port)
        }
    except OSError as exc:
        raise ValueError(f"Could not resolve website host {parsed.hostname!r}.") from exc
    if not addresses or any(
        not ipaddress.ip_address(address.split("%", 1)[0]).is_global
        for address in addresses
    ):
        raise ValueError(
            "The crawler cannot access local, private, or reserved network addresses."
        )


def _safe_get(
    url: str,
    root_host: str,
    max_bytes: int,
) -> tuple[str, requests.Response, bytes]:
    current = url
    for _ in range(6):
        _validate_public_url(current, root_host)
        with requests.get(
            current,
            headers={"User-Agent": _USER_AGENT},
            timeout=(10, 30),
            allow_redirects=False,
            stream=True,
        ) as response:
            if response.status_code in _REDIRECT_CODES:
                location = response.headers.get("Location")
                if not location:
                    return current, response, b""
                current = urljoin(current, location)
                continue
            if response.status_code >= 400:
                return current, response, b""
            declared_size = response.headers.get("Content-Length")
            if declared_size and int(declared_size) > max_bytes:
                raise ValueError(
                    f"Crawl resource exceeds the {max_bytes // (1024 * 1024)} MB per-file limit."
                )
            body = bytearray()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                body.extend(chunk)
                if len(body) > max_bytes:
                    raise ValueError(
                        f"Crawl resource exceeds the {max_bytes // (1024 * 1024)} MB per-file limit."
                    )
            return current, response, bytes(body)
    raise ValueError("Website returned too many redirects.")


def _robots_rules(start_url: str, root_host: str) -> RobotFileParser:
    rules = RobotFileParser()
    robots_url = urljoin(start_url, "/robots.txt")
    try:
        _final_url, response, body = _safe_get(robots_url, root_host, 1024 * 1024)
    except (requests.RequestException, ValueError) as exc:
        raise ValueError("Could not safely retrieve robots.txt; crawl stopped.") from exc
    if response.status_code in {404, 410}:
        rules.parse([])
    elif response.status_code == 200:
        rules.parse(body.decode("utf-8", errors="replace").splitlines())
    else:
        raise ValueError(f"Could not retrieve robots.txt (HTTP {response.status_code}).")
    return rules


def _virtual_filename(url: str, content_type: str) -> str:
    path = urlparse(url).path
    filename = path.rsplit("/", 1)[-1]
    media_type = content_type.split(";", 1)[0].strip().lower()
    extension = _CONTENT_TYPE_SUFFIXES.get(media_type, "")
    if extension and not filename.lower().endswith(extension):
        filename = f"{filename or 'page'}{extension}"
    if not filename or not any(filename.lower().endswith(ext) for ext in SUPPORTED_SUFFIXES):
        filename = "page.html" if "html" in media_type else filename
    return filename


def _attach_content_hash(documents: list[Document], content: bytes) -> None:
    digest = hashlib.sha256(content).hexdigest()
    for document in documents:
        document.metadata["content_sha256"] = digest


def crawl_site(start_url: str) -> tuple[list[Document], int, list[str]]:
    """Fetch same-host pages and linked supported documents within configured limits."""
    start_url = _normalize_url(start_url)
    root_host = urlparse(start_url).hostname.lower()
    _validate_public_url(start_url, root_host)
    robots = _robots_rules(start_url, root_host)

    max_file_bytes = config.LH_RAG_CRAWL_MAX_FILE_MB * 1024 * 1024
    max_total_bytes = config.LH_RAG_CRAWL_MAX_TOTAL_MB * 1024 * 1024
    queue = deque([(start_url, 0)])
    queued = {start_url}
    max_queued_urls = config.LH_RAG_CRAWL_MAX_PAGES * 20
    visited: set[str] = set()
    documents: list[Document] = []
    skipped: list[str] = []
    total_bytes = 0
    html_pages = 0
    processed_items = 0
    last_request = 0.0

    while queue and html_pages < config.LH_RAG_CRAWL_MAX_PAGES:
        url, depth = queue.popleft()
        if url in visited:
            continue
        visited.add(url)
        if not robots.can_fetch(_USER_AGENT, url):
            skipped.append(f"Robots policy disallowed {url}")
            continue
        wait = config.LH_RAG_CRAWL_DELAY_SECONDS - (time.monotonic() - last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            final_url, response, content = _safe_get(url, root_host, max_file_bytes)
        except (requests.RequestException, ValueError) as exc:
            skipped.append(f"{url}: {exc}")
            continue
        last_request = time.monotonic()
        if response.status_code >= 400:
            skipped.append(f"{url}: HTTP {response.status_code}")
            continue
        final_url = _normalize_url(final_url)
        total_bytes += len(content)
        if total_bytes > max_total_bytes:
            skipped.append("Crawl stopped at the configured total download limit.")
            break

        content_type = response.headers.get("Content-Type", "").lower()
        filename = _virtual_filename(final_url, content_type)
        is_html = "text/html" in content_type or filename.lower().endswith((".html", ".htm"))
        if is_html:
            html_pages += 1
            parsed = load_document_bytes("page.html", content, source=final_url)
            _attach_content_hash(parsed, content)
            documents.extend(parsed)
            processed_items += 1
            if depth < config.LH_RAG_CRAWL_MAX_DEPTH:
                soup = BeautifulSoup(content, "html.parser")
                for anchor in soup.find_all("a", href=True):
                    target = urldefrag(urljoin(final_url, str(anchor["href"])))[0]
                    try:
                        target = _normalize_url(target)
                    except ValueError:
                        continue
                    target_parts = urlparse(target)
                    if target_parts.hostname.lower() != root_host or target in queued:
                        continue
                    # Enqueue normal page URLs and supported document links.
                    if len(queued) < max_queued_urls:
                        queued.add(target)
                        queue.append((target, depth + 1))
        elif any(filename.lower().endswith(ext) for ext in SUPPORTED_SUFFIXES):
            try:
                parsed = load_document_bytes(filename, content, source=final_url)
                _attach_content_hash(parsed, content)
                documents.extend(parsed)
                processed_items += 1
            except (ValueError, zipfile.BadZipFile) as exc:
                skipped.append(f"{final_url}: {exc}")
        else:
            skipped.append(f"Unsupported linked resource: {final_url}")

    return documents, processed_items, skipped
