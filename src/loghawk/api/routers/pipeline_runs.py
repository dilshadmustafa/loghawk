from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

import loghawk.config as config
from loghawk.api.schemas import (
    PipelineRunRequest,
    PipelineRunStarted,
    PipelineRunStatus,
)
from loghawk.api.services.storage_service import find_batch
from loghawk.api.services.temporal_service import get_run_status, start_run


router = APIRouter()


@router.post("", response_model=PipelineRunStarted, status_code=202)
async def create_pipeline_run(
    body: PipelineRunRequest,
) -> PipelineRunStarted:
    try:
        batch = await run_in_threadpool(find_batch, body.bucket, body.batch)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not inspect the selected S3 batch: {exc}",
        ) from exc

    if batch is None and not config.LH_EXTERNAL_DATA_USE:
        raise HTTPException(
            status_code=404,
            detail=f"Batch {body.batch!r} was not found in bucket {body.bucket!r}",
        )
    train_available = (
        bool(config.LH_EXTERNAL_DATA_TRAIN)
        if config.LH_EXTERNAL_DATA_USE
        else bool(batch and batch.train_available)
    )
    raw_available = (
        bool(config.LH_EXTERNAL_DATA_RAW)
        if config.LH_EXTERNAL_DATA_USE
        else bool(batch and batch.raw_available)
    )
    if body.mode in {"train", "train-detect"} and not train_available:
        raise HTTPException(
            status_code=400,
            detail=f"Batch {body.batch!r} has no train/ input prefix",
        )
    if body.mode in {"detect", "train-detect"} and not raw_available:
        raise HTTPException(
            status_code=400,
            detail=f"Batch {body.batch!r} has no raw/ input prefix",
        )

    try:
        workflow_id = await start_run(body)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not start the Temporal workflow: {exc}",
        ) from exc

    return PipelineRunStarted(
        workflow_id=workflow_id,
        mode=body.mode,
        status="RUNNING",
    )


@router.get("/{workflow_id}", response_model=PipelineRunStatus)
async def read_pipeline_run(
    workflow_id: str,
) -> PipelineRunStatus:
    try:
        return await get_run_status(workflow_id)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not read Temporal workflow status: {exc}",
        ) from exc
