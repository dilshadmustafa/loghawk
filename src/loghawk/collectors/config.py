"""Load optional connector credentials from the separate .env_tools file."""

from __future__ import annotations

import os

from dotenv import dotenv_values

from loghawk.config import PROJECT_ROOT


_TOOLS_ENV = dotenv_values(PROJECT_ROOT / ".env_tools")


def setting(name: str, default: str = "") -> str:
    """Prefer process environment, then .env_tools, then the supplied default."""
    value = os.environ.get(name)
    if value is None:
        value = _TOOLS_ENV.get(name)
    return str(value if value is not None else default).strip()


PROMETHEUS_URL = setting("LH_TOOLS_PROMETHEUS_URL").rstrip("/")
PROMETHEUS_BEARER_TOKEN = setting("LH_TOOLS_PROMETHEUS_BEARER_TOKEN")
KUBERNETES_KUBECONFIG = setting("LH_TOOLS_KUBERNETES_KUBECONFIG")
JIRA_BASE_URL = setting("LH_TOOLS_JIRA_BASE_URL").rstrip("/")
JIRA_EMAIL = setting("LH_TOOLS_JIRA_EMAIL")
JIRA_API_TOKEN = setting("LH_TOOLS_JIRA_API_TOKEN")
