"""Read-only Kubernetes pod status connector."""

from __future__ import annotations

import json
import os
import sys


def list_pods(namespace: str, limit: int) -> dict:
    try:
        from kubernetes import client, config as kube_config
        from kubernetes.config.config_exception import ConfigException
    except ImportError as exc:
        raise RuntimeError(
            "Kubernetes collector dependency is missing; install requirements_tools.txt."
        ) from exc

    kubeconfig_path = os.environ.get("LH_TOOLS_KUBERNETES_KUBECONFIG", "").strip()
    if kubeconfig_path:
        kube_config.load_kube_config(config_file=kubeconfig_path)
    else:
        try:
            kube_config.load_incluster_config()
        except ConfigException:
            kube_config.load_kube_config()

    response = client.CoreV1Api().list_namespaced_pod(
        namespace=namespace,
        limit=limit,
        _request_timeout=20,
    )
    pods = []
    for pod in response.items[:limit]:
        ready_conditions = [
            condition.status == "True"
            for condition in (pod.status.conditions or [])
            if condition.type == "Ready"
        ]
        pods.append({
            "name": pod.metadata.name,
            "namespace": pod.metadata.namespace,
            "phase": pod.status.phase,
            "node": pod.spec.node_name,
            "ready": bool(ready_conditions) and all(ready_conditions),
            "restart_count": sum(
                item.restart_count for item in (pod.status.container_statuses or [])
            ),
        })
    return {"summary": f"Found {len(pods)} pod(s) in namespace {namespace}.", "data": pods}


def collect(arguments: dict) -> dict:
    return list_pods(
        arguments.get("namespace", "default"),
        arguments.get("limit", 100),
    )


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
