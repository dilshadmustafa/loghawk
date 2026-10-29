"""Dependency-light definitions, validation, and execution for built-in collectors."""

from __future__ import annotations

from pathlib import Path

from loghawk.collectors.contracts import validate_arguments as validate_schema_arguments


DEFINITIONS = {
    "prometheus-query": {
        "version": "1",
        "display_name": "Prometheus query",
        "description": "Query Prometheus metrics, including Kubernetes metrics, using PromQL. Examples: up, node_cpu_seconds_total, kube_pod_info. Do not pass shell commands such as kubectl.",
        "arguments_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string", "minLength": 1, "maxLength": 2000,
                    "description": "A PromQL expression, not a shell command. Example: rate(node_cpu_seconds_total[5m]).",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    "kubernetes-pods": {
        "version": "1",
        "display_name": "Kubernetes pod status",
        "description": "List pod names and status in one namespace; no cluster changes.",
        "arguments_schema": {
            "type": "object",
            "properties": {
                "namespace": {"type": "string", "default": "default", "minLength": 1, "maxLength": 253},
                "limit": {"type": "integer", "default": 100, "minimum": 1, "maximum": 200},
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    "jira-issue-search": {
        "version": "1",
        "display_name": "Jira issue search",
        "description": "Search Jira issues using a read-only JQL query.",
        "arguments_schema": {
            "type": "object",
            "properties": {
                "jql": {"type": "string", "minLength": 1, "maxLength": 2000},
                "limit": {"type": "integer", "default": 50, "minimum": 1, "maximum": 100},
            },
            "required": ["jql"],
            "additionalProperties": False,
        },
    },
}



def validate_arguments(name: str, arguments: dict) -> dict:
    definition = DEFINITIONS.get(name)
    if definition is None:
        raise KeyError(name)
    if not isinstance(arguments, dict):
        raise ValueError("Collector arguments must be a JSON object.")
    schema = definition["arguments_schema"]
    # Preserve the legacy direct-call behavior for explicitly null optional fields.
    omitted = {
        key for key, value in arguments.items()
        if value is None and key in schema["properties"] and key not in schema["required"]
    }
    normalized = validate_schema_arguments(
        schema, {key: value for key, value in arguments.items() if key not in omitted}
    )
    for key in omitted:
        normalized.pop(key, None)
    # Validate original lengths before trimming, as the built-ins always have.
    for key, value in normalized.items():
        if schema["properties"][key]["type"] == "string":
            if not value.strip():
                raise ValueError(f"'{key}' must be a non-empty string.")
            normalized[key] = value.strip()
    return normalized


SOURCE_FILES = {
    "prometheus-query": "prometheus.py",
    "kubernetes-pods": "kubernetes.py",
    "jira-issue-search": "jira.py",
}


def source_code(name: str) -> str:
    filename = SOURCE_FILES.get(name)
    if filename is None:
        raise KeyError(name)
    return Path(__file__).with_name(filename).read_text(encoding="utf-8")


def run_builtin(name: str, arguments: dict) -> dict:
    from loghawk.lhrunner import run_builtin_collector

    return run_builtin_collector(name, arguments)
