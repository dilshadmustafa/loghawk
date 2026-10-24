"""Authenticated endpoints for built-in read-only collectors."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from loghawk.api.routers.auth import require_console_user
from loghawk.collectors import registry as collector_registry
from loghawk.api.services import task_store2, task_runner


logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(require_console_user)])


class CollectorRunRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)


class CollectorEnabledRequest(BaseModel):
    enabled: bool


@router.get("")
def list_collectors() -> list[dict]:
    builtins = collector_registry.list_collectors()
    custom = []
    for task in task_store2.list_tasks():
        if task.get("task_kind") != "collector":
            continue
        artifact = task.get("latest_artifact") or {}
        custom.append({
            "name": task["name"], "display_name": task["name"],
            "version": str(artifact.get("version", 0)),
            "description": f"Custom collector Task in category {task.get('category', 'Uncategorized')}.",
            "arguments_schema": {"arguments": {"type": "object", "required": False}},
            "approved": bool(artifact.get("approved")),
            "enabled": bool(artifact.get("enabled")),
            "execution_policy": task.get("execution_policy", "manual"),
            "task_kind": "collector", "userid": task.get("userid"),
        })
    return [*builtins, *custom]


@router.get("/{name}")
def get_collector(name: str) -> dict:
    collector = collector_registry.get_collector(name)
    if collector is None:
        task = next(
            (item for item in task_store2.list_tasks() if item["name"].casefold() == name.casefold()),
            None,
        )
        if task is None or task.get("task_kind") != "collector":
            raise HTTPException(status_code=404, detail="Collector not found.")
        collector = task
        artifact = task.get("latest_artifact") or {}
        return {
            "name": task["name"], "task_kind": "collector",
            "artifact_type": task["artifact_type"], "category": task.get("category"),
            "userid": task.get("userid"), "version": artifact.get("version"),
            "execution_policy": task.get("execution_policy", "manual"),
            "approved": bool(artifact.get("approved")),
            "enabled": bool(artifact.get("enabled")),
            "description": f"Custom collector Task in category {task.get('category', 'Uncategorized')}.",
        }
    return collector


@router.put("/{name}")
def set_collector_enabled(
    name: str,
    body: CollectorEnabledRequest,
    user: dict = Depends(require_console_user),
) -> dict:
    if collector_registry.get_collector(name) is not None:
        if not collector_registry.set_enabled(name, body.enabled):
            raise HTTPException(status_code=404, detail="Collector not found.")
        return {"name": name, "enabled": body.enabled}
    task = next(
        (item for item in task_store2.list_tasks() if item["name"].casefold() == name.casefold()),
        None,
    )
    if task is None or task.get("task_kind") != "collector":
        raise HTTPException(status_code=404, detail="Collector not found.")
    if user["role"] not in {"Administrator", "Power User"} and task.get("userid") != user["userid"]:
        raise HTTPException(status_code=403, detail="You can modify only your own collector Tasks.")
    artifact = task.get("latest_artifact")
    if artifact is None:
        raise HTTPException(status_code=404, detail="Collector Task has no saved version.")
    updated = task_store2.set_artifact_policy(task["id"], artifact["id"], enabled=body.enabled)
    return {"name": name, "enabled": body.enabled, "artifact": updated}


@router.post("/{name}/runs")
def run_collector(name: str, body: CollectorRunRequest, user: dict = Depends(require_console_user)) -> dict:
    try:
        builtin = collector_registry.get_collector(name)
        if builtin is not None:
            return collector_registry.run_collector(name, body.arguments, user["userid"])
        task = next(
            (item for item in task_store2.list_tasks() if item["name"].casefold() == name.casefold()),
            None,
        )
        if task is None or task.get("task_kind") != "collector":
            raise KeyError(name)
        if task.get("execution_policy", "manual") == "disabled":
            raise PermissionError("This collector Task is disabled by its execution policy.")
        artifact = task.get("latest_artifact")
        if not artifact:
            raise ValueError("This collector Task has no saved version.")
        if not artifact.get("approved") or not artifact.get("enabled"):
            raise PermissionError("API execution requires this collector version to be approved and enabled.")
        return task_runner.run_version(task, artifact, user["userid"], body.arguments)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Collector not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Collector execution failed: %s", name)
        raise HTTPException(
            status_code=502,
            detail="Collector execution failed. Check the API server log for details.",
        ) from exc
