"""Web UI management of registered built-in Tasks."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, StrictBool, Field

from loghawk.api.routers.auth import require_authenticated_user
from loghawk.api.schemas import TaskExecutionPolicyWrite
from loghawk.builtin_tasks import builtin_task_registry as registry
from loghawk.builtin_tasks.loader import get_definition, source_code, KINDS
from typing import Any
import logging

router = APIRouter(dependencies=[Depends(require_authenticated_user)])


class ApprovedWrite(BaseModel):
    approved: StrictBool


@router.put("/{name}/approved")
def set_approved(name: str, body: ApprovedWrite,
                 user: dict = Depends(require_authenticated_user)) -> dict:
    if user["role"] not in {"Administrator", "Power User"}:
        raise HTTPException(status_code=403,
                            detail="Only Administrators and Power Users can change built-in Task approval.")
    if not registry.set_approved(name, body.approved):
        raise HTTPException(status_code=404, detail="Built-in Task not found.")
    return {"name": name, "approved": body.approved}


class EnabledWrite(BaseModel):
    enabled: StrictBool


@router.get("")
def list_builtins(task_kind: str | None = None) -> list[dict]:
    if task_kind is not None and task_kind not in KINDS:
        raise HTTPException(status_code=422, detail="Unknown Task kind")
    return registry.list_tasks(task_kind=task_kind)


@router.put("/{name}/enabled")
def set_enabled(name: str, body: EnabledWrite,
                user: dict = Depends(require_authenticated_user)) -> dict:
    if user["role"] not in {"Administrator", "Power User", "User"}:
        raise HTTPException(status_code=403, detail="This role cannot change Task settings.")
    if not registry.set_enabled(name, body.enabled):
        raise HTTPException(status_code=404, detail="Built-in Task not found.")
    return {"name": name, "enabled": body.enabled}


@router.put("/{name}/execution-policy")
def set_execution_policy(name: str, body: TaskExecutionPolicyWrite,
                         user: dict = Depends(require_authenticated_user)) -> dict:
    if user["role"] not in {"Administrator", "Power User"}:
        raise HTTPException(status_code=403,
                            detail="Only Administrators and Power Users can set built-in Task policies.")
    if not registry.set_execution_policy(name, body.execution_policy):
        raise HTTPException(status_code=404, detail="Built-in Task not found.")
    return {"name": name, "execution_policy": body.execution_policy}


@router.get("/{name}/source")
def get_source(name: str) -> dict:
    try:
        definition = get_definition(name)
        source = source_code(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Built-in Task not found.") from exc
    except (OSError, UnicodeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Installed Task source is unavailable.") from exc
    return {
        "name": name, "display_name": definition["display_name"],
        "version": definition["version"], "task_kind": definition["task_kind"],
        "artifact_type": definition["artifact_type"], "filename": definition["source_file"],
        "source": source, "arguments_schema": definition["arguments_schema"],
    }


@router.get("/{name}")
def get_task(name: str) -> dict:
    task = registry.get_task(name)
    if task is None:
        raise HTTPException(status_code=404, detail="Built-in Task not found.")
    return task


class RunWrite(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)
    expected_version: str | None = None


@router.post("/{name}/runs")
def run_task(name: str, body: RunWrite, user: dict = Depends(require_authenticated_user)) -> dict:
    # Origin is derived from authenticated session data, not request arguments.
    software = {"webui": "web", "console": "console", "cli": "cli"}.get(user.get("client"))
    if software is None:
        raise HTTPException(status_code=403, detail="Unsupported execution client")
    try:
        return registry.run_task(name, body.arguments, user["userid"],
                                 expected_version=body.expected_version,
                                 origin="Human", origin_software=software)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Built-in Task not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logging.getLogger(__name__).exception("Built-in Task execution failed: %s", name)
        raise HTTPException(status_code=502, detail="Task execution failed. Check the API server log.") from exc
