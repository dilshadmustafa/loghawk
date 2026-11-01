"""Read-only Jira Cloud JQL search connector."""

from __future__ import annotations

import json
import os
import sys

from urllib.parse import urlparse

import requests


def search_issues(jql: str, limit: int) -> dict:
    base_url = os.environ.get("LH_TOOLS_JIRA_BASE_URL", "").strip().rstrip("/")
    email = os.environ.get("LH_TOOLS_JIRA_EMAIL", "").strip()
    api_token = os.environ.get("LH_TOOLS_JIRA_API_TOKEN", "").strip()
    parsed = urlparse(base_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise RuntimeError("Set LH_TOOLS_JIRA_BASE_URL to your HTTPS Jira Cloud URL in .env_tools.")
    if not email or not api_token:
        raise RuntimeError("Set LH_TOOLS_JIRA_EMAIL and LH_TOOLS_JIRA_API_TOKEN in .env_tools.")
    response = requests.post(
        f"{base_url}/rest/api/3/search/jql",
        json={"jql": jql, "maxResults": limit, "fields": ["summary", "status", "assignee"]},
        auth=(email, api_token),
        headers={"Accept": "application/json"},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    issues = [
        {
            "key": issue.get("key"),
            "summary": issue.get("fields", {}).get("summary"),
            "status": (issue.get("fields", {}).get("status") or {}).get("name"),
            "assignee": (issue.get("fields", {}).get("assignee") or {}).get("displayName"),
        }
        for issue in payload.get("issues", [])[:limit]
    ]
    return {"summary": f"Jira returned {len(issues)} issue(s).", "data": issues}


def collect(arguments: dict) -> dict:
    jql = arguments.get("jql")
    if not isinstance(jql, str) or not jql.strip():
        raise ValueError("Provide a non-empty jql argument.")
    return search_issues(jql, arguments.get("limit", 50))


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
