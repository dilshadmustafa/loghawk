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


async def start_configured_run(
    config_set: dict,
    mode: str,
    *,
    workflow_id: str,
) -> dict[str, str | None]:
    """Start a workflow from a saved UI Config Set."""
    client = await _get_client()
    train_phase = mode in {"train", "train-detect"}
    detect_phase = mode in {"detect", "train-detect"}
    if config_set["external_data_use"]:
        output_root = (
            f"s3://{config_set['output_bucket']}/"
            f"{config_set['output_batch']}"
        )
        train_sources = [
            {
                "url": f"s3://{item['bucket']}/"
                f"{item['folder'].strip('/') + '/' if item['folder'].strip('/') else ''}",
                "region": item["region"],
            }
            for item in config_set["train_sources"]
        ]
        detect_sources = [
            {
                "url": f"s3://{item['bucket']}/"
                f"{item['folder'].strip('/') + '/' if item['folder'].strip('/') else ''}",
                "region": item["region"],
            }
            for item in config_set["detect_sources"]
        ]
    else:
        output_root = (
            f"s3://{config_set['source_bucket']}/"
            f"{config_set['source_batch']}"
        )
        train_sources = []
        detect_sources = []

    handle = await client.start_workflow(
        LogHawkTrainDetectPipeline.run,
        args=[
            output_root,
            train_phase,
            detect_phase,
            config_set["external_data_use"],
            train_sources,
            detect_sources,
        ],
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )
    return {"workflow_id": handle.id, "run_id": None}


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
