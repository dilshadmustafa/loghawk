"""Read-only Jira Cloud JQL search connector."""

from __future__ import annotations

from urllib.parse import urlparse

import requests

from loghawk.collectors import config


def search_issues(jql: str, limit: int) -> dict:
    base_url = config.JIRA_BASE_URL
    parsed = urlparse(base_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise RuntimeError("Set LH_TOOLS_JIRA_BASE_URL to your HTTPS Jira Cloud URL in .env_tools.")
    if not config.JIRA_EMAIL or not config.JIRA_API_TOKEN:
        raise RuntimeError("Set LH_TOOLS_JIRA_EMAIL and LH_TOOLS_JIRA_API_TOKEN in .env_tools.")
    response = requests.post(
        f"{base_url}/rest/api/3/search/jql",
        json={"jql": jql, "maxResults": limit, "fields": ["summary", "status", "assignee"]},
        auth=(config.JIRA_EMAIL, config.JIRA_API_TOKEN),
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
