"""Web UI management of registered built-in collectors."""
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, StrictBool

from loghawk.api.routers.auth import require_webui_user
from loghawk.api.schemas import TaskExecutionPolicyWrite
from loghawk.collectors import builtins, registry

router = APIRouter(dependencies=[Depends(require_webui_user)])


class ApprovedWrite(BaseModel):
    approved: StrictBool


@router.put("/{name}/approved")
def set_approved(name: str, body: ApprovedWrite,
                 user: dict = Depends(require_webui_user)) -> dict:
    if user["role"] not in {"Administrator", "Power User"}:
        raise HTTPException(status_code=403,
                            detail="Only Administrators and Power Users can change built-in collector approval.")
    if not registry.set_approved(name, body.approved):
        raise HTTPException(status_code=404, detail="Built-in collector not found.")
    return {"name": name, "approved": body.approved}


class EnabledWrite(BaseModel):
    enabled: StrictBool


@router.get("")
def list_builtins() -> list[dict]:
    return registry.list_collectors()


@router.put("/{name}/enabled")
def set_enabled(name: str, body: EnabledWrite,
                user: dict = Depends(require_webui_user)) -> dict:
    if user["role"] not in {"Administrator", "Power User", "User"}:
        raise HTTPException(status_code=403, detail="This role cannot change collector settings.")
    if not registry.set_enabled(name, body.enabled):
        raise HTTPException(status_code=404, detail="Built-in collector not found.")
    return {"name": name, "enabled": body.enabled}


@router.put("/{name}/execution-policy")
def set_execution_policy(name: str, body: TaskExecutionPolicyWrite,
                         user: dict = Depends(require_webui_user)) -> dict:
    if user["role"] not in {"Administrator", "Power User"}:
        raise HTTPException(status_code=403,
                            detail="Only Administrators and Power Users can set built-in collector policies.")
    if not registry.set_execution_policy(name, body.execution_policy):
        raise HTTPException(status_code=404, detail="Built-in collector not found.")
    return {"name": name, "execution_policy": body.execution_policy}


_SOURCE_FILES = {
    "prometheus-query": "prometheus.py",
    "kubernetes-pods": "kubernetes.py",
    "jira-issue-search": "jira.py",
}


@router.get("/{name}/source")
def get_source(name: str) -> dict:
    """Read installed connector source and its matching argument definition."""
    filename = _SOURCE_FILES.get(name)
    definition = builtins.DEFINITIONS.get(name)
    if filename is None or definition is None:
        raise HTTPException(status_code=404, detail="Built-in collector not found.")
    # Only known filenames are resolved; no client-provided paths or config values.
    source_path = Path(builtins.__file__).resolve().parent / filename
    try:
        source = source_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise HTTPException(status_code=503, detail="Installed collector source is unavailable.") from exc
    return {
        "name": name,
        "display_name": definition["display_name"],
        "version": str(definition["version"]),
        "filename": filename,
        "source": source,
        "arguments_schema": definition["arguments_schema"],
    }
