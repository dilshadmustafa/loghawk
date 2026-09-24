import asyncio

from temporalio.client import Client

from loghawk.workflows.temporal.workflows import (
    LogHawkPipeline,
)


async def main():

    client = await Client.connect(
        "localhost:7233"
    )

    workflow_id = "loghawk-pipeline-2026-09-23"

    result = await client.execute_workflow(
        LogHawkPipeline.run,
        args=[
            "s3a://loghawk-data/raw/year=2026/month=09/day=23/",
            "s3a://loghawk-data/features/year=2026/month=09/day=23/",
            "s3://loghawk-data/anomalies/year=2026/month=09/day=23/",
        ],
        id=workflow_id,
        task_queue="loghawk-pipeline",
    )

    print()
    print("=" * 70)
    print("LogHawk Pipeline Completed")
    print("=" * 70)
    print(f"Anomaly output: {result}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())