import asyncio
from uuid import uuid4

from temporalio.client import Client

import loghawk.config as config

from loghawk.workflows.temporal.workflows import (
    LogHawkTrainDetectPipeline,
)


async def main():

    client = await Client.connect(
        config.LH_TEMPORAL_ADDRESS
    )

    batch_root = (
        f"s3://{config.LH_S3_BUCKET}/"
        f"{config.LH_S3_BATCH_FOLDER}"
    )
    workflow_id = (
        f"loghawk-{config.LH_S3_BATCH_FOLDER}-{uuid4().hex[:10]}"
    )

    result = await client.execute_workflow(
        LogHawkTrainDetectPipeline.run,
        args=[
            batch_root,
            config.LH_TRAIN_PHASE,
            config.LH_DETECT_PHASE,
            config.LH_EXTERNAL_DATA_USE,
            [
                {"url": url, "region": region}
                for url, region in zip(
                    config.LH_EXTERNAL_DATA_TRAIN,
                    config.LH_EXTERNAL_S3BUCKET_REGIONS_TRAIN,
                )
            ],
            [
                {"url": url, "region": region}
                for url, region in zip(
                    config.LH_EXTERNAL_DATA_RAW,
                    config.LH_EXTERNAL_S3BUCKET_REGIONS_RAW,
                )
            ],
        ],
        id=workflow_id,
        task_queue="loghawk-pipeline",
    )

    print()
    print("=" * 70)
    print("LogHawk Pipeline Completed")
    print("=" * 70)
    print(f"Pipeline result: {result}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
