from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from loghawk.api.schemas import (
    TaskArtifactImport,
    TaskArtifactEnabled,
    TaskArtifactWrite,
    TaskDescriptionWrite,
    TaskExecutionPolicyWrite,
    TaskMessageWrite,
    TaskRunRequest,
    TaskWrite,
)
from loghawk.api.routers.auth import require_authenticated_user
from loghawk.api.services import n8n_service
from loghawk.api.services import task_service2 as task_service
from loghawk.api.services import task_store2 as task_store
from loghawk.api.services import task_runner
from loghawk.database.database_layer import DatabaseIntegrityError


router = APIRouter()


def _require_owner_or_manager(task: dict, user: dict) -> None:
    if user["role"] not in {"Administrator", "Power User"} and task.get("userid") != user["userid"]:
        raise HTTPException(status_code=403, detail="You can modify only your own Tasks.")


@router.get("")
def list_tasks(q: str = "") -> list[dict]:
    return task_store.list_tasks(query=q)


@router.post("", status_code=201)
def create_task(body: TaskWrite, user: dict = Depends(require_authenticated_user)) -> dict:
    if body.execution_policy == "auto-approved" and user["role"] not in {
        "Administrator", "Power User"
    }:
        raise HTTPException(
            status_code=403,
            detail="Administrator or Power User role required to select Auto-approved.",
        )
    try:
        return task_store.create_task(body.model_dump())
    except DatabaseIntegrityError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"A Task named {body.name!r} already exists.",
        ) from exc


@router.get("/{task_id}")
def read_task(task_id: str) -> dict:
    task = task_store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.put("/{task_id}/execution-policy")
def set_task_execution_policy(
    task_id: str, body: TaskExecutionPolicyWrite,
    user: dict = Depends(require_authenticated_user),
) -> dict:
    task = task_store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    _require_owner_or_manager(task, user)
    if body.execution_policy == "auto-approved" and user["role"] not in {
        "Administrator", "Power User"
    }:
        raise HTTPException(
            status_code=403,
            detail="Administrator or Power User role required to select Auto-approved.",
        )
    updated = task_store.set_task_execution_policy(task_id, body.execution_policy)
    return updated or {}


@router.put("/{task_id}/description")
def set_task_description(
    task_id: str, body: TaskDescriptionWrite,
    user: dict = Depends(require_authenticated_user),
) -> dict:
    task = task_store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    _require_owner_or_manager(task, user)
    updated = task_store.set_task_description(task_id, body.description)
    return updated or {}


@router.get("/{task_id}/artifacts/{artifact_id}")
def read_task_artifact(task_id: str, artifact_id: str) -> dict:
    artifact = task_store.get_artifact_version(task_id, artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Task artifact version not found")
    return artifact


@router.delete("/{task_id}", status_code=204)
def delete_task(task_id: str, user: dict = Depends(require_authenticated_user)) -> None:
    task = task_store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    _require_owner_or_manager(task, user)
    if not task_store.delete_task(task_id):
        raise HTTPException(status_code=404, detail="Task not found")


@router.post("/{task_id}/artifacts")
def save_task_artifact(
    task_id: str, body: TaskArtifactWrite,
    user: dict = Depends(require_authenticated_user),
) -> dict:
    task = task_store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    _require_owner_or_manager(task, user)
    try:
        result = task_service.save_artifact_version(task_id, body.content)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return result


@router.delete("/{task_id}/artifacts/{artifact_id}")
def delete_task_artifact(
    task_id: str, artifact_id: str,
    user: dict = Depends(require_authenticated_user),
) -> dict:
    task = task_store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    _require_owner_or_manager(task, user)
    if not task_store.delete_artifact_version(task_id, artifact_id):
        raise HTTPException(
            status_code=404,
            detail="Task or saved artifact version not found",
        )
    task = task_store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.post("/{task_id}/import-to-n8n")
def import_task_to_n8n(task_id: str, body: TaskArtifactImport) -> dict[str, str]:
    task = task_store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["artifact_type"] != "n8n":
        raise HTTPException(status_code=422, detail="Only n8n tasks can be imported")

    artifact = task_store.get_artifact_version(task_id, body.artifact_id)
    if artifact is None:
        raise HTTPException(
            status_code=404,
            detail="Saved Task version not found. Save the artifact before importing it.",
        )
    try:
        return n8n_service.import_workflow(task["name"], artifact["content"])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except n8n_service.N8nApiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/{task_id}/artifacts/{artifact_id}/approve")
def approve_task_artifact(
    task_id: str, artifact_id: str,
    user: dict = Depends(require_authenticated_user),
) -> dict:
    if user["role"] not in {"Administrator", "Power User"}:
        raise HTTPException(status_code=403, detail="Administrator or Power User role required to approve.")
    task = task_store.get_task(task_id)
    artifact = task_store.get_artifact_version(task_id, artifact_id)
    if task is None or artifact is None:
        raise HTTPException(status_code=404, detail="Task artifact version not found")
    updated = task_store.set_artifact_policy(
        task_id, artifact_id, approved=True, approved_by=user["userid"]
    )
    return updated or {}


@router.put("/{task_id}/artifacts/{artifact_id}/enabled")
def set_task_artifact_enabled(
    task_id: str, artifact_id: str, body: TaskArtifactEnabled,
    user: dict = Depends(require_authenticated_user),
) -> dict:
    task = task_store.get_task(task_id)
    artifact = task_store.get_artifact_version(task_id, artifact_id)
    if task is None or artifact is None:
        raise HTTPException(status_code=404, detail="Task artifact version not found")
    _require_owner_or_manager(task, user)
    updated = task_store.set_artifact_policy(task_id, artifact_id, enabled=body.enabled)
    return updated or {}


@router.post("/{task_id}/runs")
def run_task(
    task_id: str, body: TaskRunRequest,
    user: dict = Depends(require_authenticated_user),
) -> dict:
    task = task_store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    artifact = (
        task_store.get_artifact_version(task_id, body.artifact_id)
        if body.artifact_id else task.get("latest_artifact")
    )
    if artifact is None:
        raise HTTPException(status_code=404, detail="Saved Task version not found")
    if task["artifact_type"] == "n8n":
        raise HTTPException(status_code=422, detail="n8n workflows are imported into the customer-managed n8n instance; LogHawk does not execute them.")
    if task.get("execution_policy", "manual") == "disabled":
        raise HTTPException(status_code=409, detail="This Task is disabled by its execution policy.")
    if not artifact.get("approved") or not artifact.get("enabled"):
        raise HTTPException(status_code=409, detail="API execution requires this version to be approved and enabled.")
    return task_runner.run_version(task, artifact, user["userid"], body.arguments)


@router.post("/{task_id}/messages")
async def add_task_message(
    task_id: str, body: TaskMessageWrite,
    user: dict = Depends(require_authenticated_user),
) -> dict:
    task = task_store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    _require_owner_or_manager(task, user)
    try:
        result = await run_in_threadpool(
            task_service.respond_to_message, task_id, body.content
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Task assistant request failed: {exc}",
        ) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return result
