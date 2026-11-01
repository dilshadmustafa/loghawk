"""Read-only Prometheus instant query connector."""

from __future__ import annotations

import json
import os
import sys

from urllib.parse import urlparse

import requests


def query(promql: str) -> dict:
    base_url = os.environ.get("LH_TOOLS_PROMETHEUS_URL", "").strip().rstrip("/")
    bearer_token = os.environ.get("LH_TOOLS_PROMETHEUS_BEARER_TOKEN", "").strip()
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("Set LH_TOOLS_PROMETHEUS_URL to a reachable HTTP(S) Prometheus URL in .env_tools.")
    headers = {}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    response = requests.post(
        f"{base_url}/api/v1/query",
        data={"query": promql, "limit": 100},
        headers=headers,
        timeout=20,
    )
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    if not response.ok or payload.get("status") != "success":
        error_type = str(payload.get("errorType", "unknown"))[:100]
        detail = str(payload.get("error", "No structured Prometheus error details were returned."))[:1000]
        # Do not include headers, credentials or arbitrary HTML response bodies.
        # The AI execution service redacts this error before model/audit exposure.
        raise RuntimeError(f"Prometheus HTTP {response.status_code} — {error_type}: {detail}")
    data = payload.get("data", {})
    results = data.get("result", []) if isinstance(data, dict) else []
    return {
        "summary": f"Prometheus returned {len(results)} result series.",
        "data": {"resultType": data.get("resultType"), "result": results[:100]},
    }


def collect(arguments: dict) -> dict:
    promql = arguments.get("query")
    if not isinstance(promql, str) or not promql.strip():
        raise ValueError('Provide a query, for example {"query":"up"}')
    return query(promql)


def main() -> int:
    try:
        arguments = json.loads(
            os.environ.get("LH_RUNNER_ARGUMENTS_JSON", "{}")
        )
        if not isinstance(arguments, dict):
            raise ValueError("Collector arguments must be a JSON object.")
        print(json.dumps(collect(arguments), ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
