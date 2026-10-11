import sqlite3

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from loghawk.api.schemas import (
    TaskArtifactImport,
    TaskArtifactWrite,
    TaskMessageWrite,
    TaskWrite,
)
from loghawk.api.services import n8n_service, task_service, task_store


router = APIRouter()


@router.get("")
def list_tasks() -> list[dict]:
    return task_store.list_tasks()


@router.post("", status_code=201)
def create_task(body: TaskWrite) -> dict:
    try:
        return task_store.create_task(body.model_dump())
    except sqlite3.IntegrityError as exc:
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


@router.delete("/{task_id}", status_code=204)
def delete_task(task_id: str) -> None:
    if not task_store.delete_task(task_id):
        raise HTTPException(status_code=404, detail="Task not found")


@router.post("/{task_id}/artifacts")
def save_task_artifact(task_id: str, body: TaskArtifactWrite) -> dict:
    try:
        result = task_service.save_artifact_version(task_id, body.content)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return result


@router.delete("/{task_id}/artifacts/{artifact_id}")
def delete_task_artifact(task_id: str, artifact_id: str) -> dict:
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


@router.post("/{task_id}/messages")
async def add_task_message(task_id: str, body: TaskMessageWrite) -> dict:
    if task_store.get_task(task_id) is None:
        raise HTTPException(status_code=404, detail="Task not found")
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
