import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from loghawk.workflows.temporal.activities import (
    run_stage_a,
    run_stage_b,
)

from loghawk.workflows.temporal.workflows import (
    LogHawkPipeline,
)


async def main():

    # ---------------------------------------------------------
    # Connect to Temporal
    # ---------------------------------------------------------

    client = await Client.connect(
        "localhost:7233"
    )

    # ---------------------------------------------------------
    # Start Worker
    # ---------------------------------------------------------

    worker = Worker(
        client,
        task_queue="loghawk-pipeline",
        workflows=[
            LogHawkPipeline,
        ],
        activities=[
            run_stage_a,
            run_stage_b,
        ],
    )

    print("=" * 70)
    print("LogHawk Temporal Worker")
    print("=" * 70)
    print("Temporal server : localhost:7233")
    print("Task queue      : loghawk-pipeline")
    print("Workflows       : LogHawkPipeline")
    print("Activities      : run_stage_a, run_stage_b")
    print("=" * 70)

    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())