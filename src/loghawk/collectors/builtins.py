"""Dependency-light definitions, validation, and execution for built-in collectors."""

from __future__ import annotations


DEFINITIONS = {
    "prometheus-query": {
        "version": "1",
        "display_name": "Prometheus query",
        "description": "Query Prometheus metrics, including Kubernetes metrics, using PromQL. Examples: up, node_cpu_seconds_total, kube_pod_info. Do not pass shell commands such as kubectl.",
        "arguments_schema": {
            "query": {
                "type": "string", "required": True, "max_length": 2000,
                "description": "A PromQL expression, not a shell command. Example: rate(node_cpu_seconds_total[5m]).",
            },
        },
    },
    "kubernetes-pods": {
        "version": "1",
        "display_name": "Kubernetes pod status",
        "description": "List pod names and status in one namespace; no cluster changes.",
        "arguments_schema": {
            "namespace": {"type": "string", "required": False, "default": "default", "max_length": 253},
            "limit": {"type": "integer", "required": False, "default": 100, "min": 1, "max": 200},
        },
    },
    "jira-issue-search": {
        "version": "1",
        "display_name": "Jira issue search",
        "description": "Search Jira issues using a read-only JQL query.",
        "arguments_schema": {
            "jql": {"type": "string", "required": True, "max_length": 2000},
            "limit": {"type": "integer", "required": False, "default": 50, "min": 1, "max": 100},
        },
    },
}


def validate_arguments(name: str, arguments: dict) -> dict:
    definition = DEFINITIONS.get(name)
    if definition is None:
        raise KeyError(name)
    schema = definition["arguments_schema"]
    unknown = set(arguments) - set(schema)
    if unknown:
        raise ValueError(f"Unsupported argument(s): {', '.join(sorted(unknown))}")
    normalized = {}
    for key, spec in schema.items():
        value = arguments.get(key, spec.get("default"))
        if value is None:
            if spec.get("required"):
                raise ValueError(f"'{key}' is required.")
            continue
        if spec["type"] == "string":
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"'{key}' must be a non-empty string.")
            if len(value) > spec["max_length"]:
                raise ValueError(f"'{key}' is longer than {spec['max_length']} characters.")
            normalized[key] = value.strip()
        elif spec["type"] == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"'{key}' must be an integer.")
            if not spec["min"] <= value <= spec["max"]:
                raise ValueError(f"'{key}' must be between {spec['min']} and {spec['max']}.")
            normalized[key] = value
    return normalized


def run_builtin(name: str, arguments: dict) -> dict:
    if name == "prometheus-query":
        from loghawk.collectors.prometheus import query
        return query(arguments["query"])
    if name == "kubernetes-pods":
        from loghawk.collectors.kubernetes import list_pods
        return list_pods(arguments["namespace"], arguments["limit"])
    if name == "jira-issue-search":
        from loghawk.collectors.jira import search_issues
        return search_issues(arguments["jql"], arguments["limit"])
    raise KeyError(name)
