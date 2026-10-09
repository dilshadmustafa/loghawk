import asyncio
from uuid import uuid4

from temporalio.client import Client

import loghawk.config as config
from loghawk.api.schemas import PipelineRunRequest
from loghawk.workflows.temporal.workflows import LogHawkTrainDetectPipeline


TASK_QUEUE = "loghawk-pipeline"
_client: Client | None = None
_client_lock = asyncio.Lock()


async def _get_client() -> Client:
    global _client
    async with _client_lock:
        if _client is None:
            _client = await Client.connect(config.LH_TEMPORAL_ADDRESS)
        return _client


def _external_sources(urls: list[str], regions: list[str]) -> list[dict[str, str]]:
    return [
        {"url": url, "region": region}
        for url, region in zip(urls, regions)
    ]


async def start_run(request: PipelineRunRequest) -> str:
    client = await _get_client()
    train_phase = request.mode in {"train", "train-detect"}
    detect_phase = request.mode in {"detect", "train-detect"}
    batch_root = f"s3://{request.bucket}/{request.batch}"
    workflow_id = f"loghawk-ui-{uuid4().hex}"

    handle = await client.start_workflow(
        LogHawkTrainDetectPipeline.run,
        args=[
            batch_root,
            train_phase,
            detect_phase,
            config.LH_EXTERNAL_DATA_USE,
            _external_sources(
                config.LH_EXTERNAL_DATA_TRAIN,
                config.LH_EXTERNAL_S3BUCKET_REGIONS_TRAIN,
            ),
            _external_sources(
                config.LH_EXTERNAL_DATA_RAW,
                config.LH_EXTERNAL_S3BUCKET_REGIONS_RAW,
            ),
        ],
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )
    return handle.id


async def get_run_status(workflow_id: str):
    client = await _get_client()
    handle = client.get_workflow_handle(workflow_id)
    description = await handle.describe()
    status = getattr(description.status, "name", str(description.status))
    return {
        "workflow_id": description.id,
        "run_id": description.run_id,
        "status": status,
    }
