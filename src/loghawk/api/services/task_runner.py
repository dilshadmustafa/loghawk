"""Server-side execution and audit recording for policy-approved Task versions."""

from __future__ import annotations

from typing import Any

from loghawk import lhrunner
from loghawk.api.services import task_store2


def run_version(
    task: dict, artifact: dict, userid: str, arguments: dict[str, Any] | None = None, *, timeout: float = 300
) -> dict:
    if artifact.get("arguments_schema") is not None:
        from loghawk.collectors.contracts import validate_arguments
        arguments = validate_arguments(artifact["arguments_schema"], arguments or {})
    run_id, started_at = task_store2.start_task_run(task["id"], artifact["id"], userid)
    try:
        result = lhrunner.run_artifact(
            task["artifact_type"], artifact["content"], arguments=arguments, timeout=timeout
        )
    except Exception as exc:
        result = {
            "status": "FAILED", "exit_code": None, "stdout": "",
            "stderr": str(exc), "error_type": type(exc).__name__,
        }
    task_store2.finish_task_run(run_id, result)
    result = lhrunner.normalize_output(result)
    return {
        "run_id": run_id,
        "task_id": task["id"],
        "task_name": task["name"],
        "artifact_id": artifact["id"],
        "version": artifact["version"],
        "started_at": started_at,
        **result,
    }
