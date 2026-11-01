"""Web UI management of registered built-in collectors."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, StrictBool

from loghawk.api.routers.auth import require_webui_user
from loghawk.api.schemas import TaskExecutionPolicyWrite
from loghawk.builtin_tasks import builtin_task_registry as registry
from loghawk.api.routers.builtin_tasks import get_source as get_task_source

router = APIRouter(dependencies=[Depends(require_webui_user)])


def _require_collector(name: str) -> None:
    task = registry.get_task(name)
    if task is None or task["task_kind"] != "collector":
        raise HTTPException(status_code=404, detail="Built-in collector not found.")


class ApprovedWrite(BaseModel):
    approved: StrictBool


@router.put("/{name}/approved")
def set_approved(name: str, body: ApprovedWrite,
                 user: dict = Depends(require_webui_user)) -> dict:
    if user["role"] not in {"Administrator", "Power User"}:
        raise HTTPException(status_code=403,
                            detail="Only Administrators and Power Users can change built-in collector approval.")
    _require_collector(name)
    if not registry.set_approved(name, body.approved):
        raise HTTPException(status_code=404, detail="Built-in collector not found.")
    return {"name": name, "approved": body.approved}


class EnabledWrite(BaseModel):
    enabled: StrictBool


@router.get("")
def list_builtins() -> list[dict]:
    return registry.list_tasks(task_kind="collector")


@router.put("/{name}/enabled")
def set_enabled(name: str, body: EnabledWrite,
                user: dict = Depends(require_webui_user)) -> dict:
    if user["role"] not in {"Administrator", "Power User", "User"}:
        raise HTTPException(status_code=403, detail="This role cannot change collector settings.")
    _require_collector(name)
    if not registry.set_enabled(name, body.enabled):
        raise HTTPException(status_code=404, detail="Built-in collector not found.")
    return {"name": name, "enabled": body.enabled}


@router.put("/{name}/execution-policy")
def set_execution_policy(name: str, body: TaskExecutionPolicyWrite,
                         user: dict = Depends(require_webui_user)) -> dict:
    if user["role"] not in {"Administrator", "Power User"}:
        raise HTTPException(status_code=403,
                            detail="Only Administrators and Power Users can set built-in collector policies.")
    _require_collector(name)
    if not registry.set_execution_policy(name, body.execution_policy):
        raise HTTPException(status_code=404, detail="Built-in collector not found.")
    return {"name": name, "execution_policy": body.execution_policy}


@router.get("/{name}/source")
def get_source(name: str) -> dict:
    _require_collector(name)
    return get_task_source(name)
