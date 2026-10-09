import sqlite3
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from loghawk.api.schemas import PipelineWrite
from loghawk.api.services import config_store
from loghawk.api.services.temporal_service import (
    get_run_status,
    start_configured_run,
)


router = APIRouter()


def _normalized_status(status: str) -> str:
    status = status.upper()
    if status == "COMPLETED":
        return "COMPLETED"
    if status == "RUNNING":
        return "RUNNING"
    return "FAILED"


async def _refresh_active_run(run: dict | None) -> None:
    if not run or run["status"] != "RUNNING":
        return
    try:
        temporal = await get_run_status(run["workflow_id"])
    except Exception:
        # If Temporal is temporarily unreachable, preserve the last known state.
        return
    config_store.update_run(
        run["workflow_id"],
        status=_normalized_status(temporal["status"]),
        temporal_run_id=temporal.get("run_id"),
    )


async def _refresh_pipelines() -> list[dict]:
    pipelines = config_store.list_pipelines()
    for pipeline in pipelines:
        await _refresh_active_run(pipeline.get("latest_run"))
    return config_store.list_pipelines()


@router.get("")
async def list_pipelines() -> list[dict]:
    return await _refresh_pipelines()


@router.post("", status_code=201)
def create_pipeline(body: PipelineWrite) -> dict:
    if config_store.get_config_set(body.config_set_id) is None:
        raise HTTPException(status_code=404, detail="Config Set not found")
    try:
        return config_store.create_pipeline(body.model_dump())
    except sqlite3.IntegrityError as exc:
        raise HTTPException(
            status_code=409,
            detail="Pipeline name already exists or Config Set is invalid.",
        ) from exc


@router.get("/{pipeline_id}")
async def read_pipeline(pipeline_id: str) -> dict:
    pipeline = config_store.get_pipeline(pipeline_id)
    if pipeline is None:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    await _refresh_active_run(pipeline.get("latest_run"))
    return config_store.get_pipeline(pipeline_id) or pipeline


@router.put("/{pipeline_id}")
def update_pipeline(pipeline_id: str, body: PipelineWrite) -> dict:
    if config_store.get_config_set(body.config_set_id) is None:
        raise HTTPException(status_code=404, detail="Config Set not found")
    try:
        pipeline = config_store.update_pipeline(
            pipeline_id, body.model_dump()
        )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(
            status_code=409,
            detail="Pipeline name already exists or Config Set is invalid.",
        ) from exc
    if pipeline is None:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return pipeline


@router.post("/{pipeline_id}/runs", status_code=202)
async def start_pipeline_run(pipeline_id: str) -> dict:
    pipeline = config_store.get_pipeline(pipeline_id)
    if pipeline is None:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    config_set = config_store.get_config_set(pipeline["config_set_id"])
    if config_set is None:
        raise HTTPException(status_code=404, detail="Config Set not found")

    if config_set["external_data_use"]:
        if pipeline["run_mode"] in {"train", "train-detect"} and not config_set["train_sources"]:
            raise HTTPException(status_code=400, detail="Config Set has no Train sources")
        if pipeline["run_mode"] in {"detect", "train-detect"} and not config_set["detect_sources"]:
            raise HTTPException(status_code=400, detail="Config Set has no Detect sources")
    else:
        from loghawk.api.services.storage_service import find_batch

        batch = find_batch(config_set["source_bucket"], config_set["source_batch"])
        if batch is None:
            raise HTTPException(status_code=404, detail="Selected input batch was not found")
        if pipeline["run_mode"] in {"train", "train-detect"} and not batch.train_available:
            raise HTTPException(status_code=400, detail="Selected batch has no train/ prefix")
        if pipeline["run_mode"] in {"detect", "train-detect"} and not batch.raw_available:
            raise HTTPException(status_code=400, detail="Selected batch has no raw/ prefix")

    active_run = config_store.get_active_run(pipeline_id)
    if active_run:
        try:
            temporal_status = await get_run_status(active_run["workflow_id"])
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="Cannot verify the existing Temporal run; refusing to start a duplicate.",
            ) from exc
        status = _normalized_status(temporal_status["status"])
        config_store.update_run(
            active_run["workflow_id"],
            status=status,
            temporal_run_id=temporal_status.get("run_id"),
        )
        if status == "RUNNING":
            raise HTTPException(
                status_code=409,
                detail=f"Pipeline is already running as {active_run['workflow_id']}.",
            )

    workflow_id = f"loghawk-ui-{uuid4().hex}"
    try:
        config_store.reserve_run(pipeline_id, workflow_id)
    except config_store.ActivePipelineRunError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Pipeline already has an active run") from exc

    try:
        started = await start_configured_run(
            config_set,
            pipeline["run_mode"],
            workflow_id=workflow_id,
        )
    except Exception as exc:
        config_store.update_run(
            workflow_id,
            status="FAILED",
            error=str(exc),
        )
        raise HTTPException(
            status_code=502,
            detail=f"Could not start the Temporal workflow: {exc}",
        ) from exc

    run = config_store.update_run(
        workflow_id,
        status="RUNNING",
        temporal_run_id=started.get("run_id"),
    )
    return run or {"workflow_id": workflow_id, "status": "RUNNING"}


@router.get("/runs/{workflow_id}")
async def read_pipeline_run(workflow_id: str) -> dict:
    run = config_store.get_run(workflow_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    if run["status"] == "RUNNING":
        await _refresh_active_run(run)
        run = config_store.get_run(workflow_id) or run
    run["temporal_ui_url"] = (
        "http://localhost:8233/namespaces/default/workflows"
        f"/{workflow_id}/{run.get('temporal_run_id') or ''}"
    )
    return run
