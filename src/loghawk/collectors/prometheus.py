"""Read-only Prometheus instant query connector."""

from __future__ import annotations

from urllib.parse import urlparse

import requests

from loghawk.collectors import config


def query(promql: str) -> dict:
    base_url = config.PROMETHEUS_URL
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("Set LH_TOOLS_PROMETHEUS_URL to a reachable HTTP(S) Prometheus URL in .env_tools.")
    headers = {}
    if config.PROMETHEUS_BEARER_TOKEN:
        headers["Authorization"] = f"Bearer {config.PROMETHEUS_BEARER_TOKEN}"
    response = requests.post(
        f"{base_url}/api/v1/query",
        data={"query": promql, "limit": 100},
        headers=headers,
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "success":
        raise RuntimeError("Prometheus returned an unsuccessful query response.")
    data = payload.get("data", {})
    results = data.get("result", []) if isinstance(data, dict) else []
    return {
        "summary": f"Prometheus returned {len(results)} result series.",
        "data": {"resultType": data.get("resultType"), "result": results[:100]},
    }
