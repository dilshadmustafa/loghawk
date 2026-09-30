import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from loghawk.workflows.temporal.activities import (
    run_identity_mapping,
    run_stage_a,
    run_stage_b,
    run_stage_b_train,
    run_stage_b_detect,
    run_stage_c,
)

from loghawk.workflows.temporal.workflows import (
    LogHawkPipeline,
    LogHawkTrainDetectPipeline,
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
            LogHawkTrainDetectPipeline,
        ],
        activities=[
            run_identity_mapping,
            run_stage_a,
            run_stage_b,
            run_stage_b_train,
            run_stage_b_detect,
            run_stage_c,
        ],
    )

    print("=" * 70)
    print("LogHawk Temporal Worker")
    print("=" * 70)
    print("Temporal server : localhost:7233")
    print("Task queue      : loghawk-pipeline")
    print("Workflows       : LogHawkPipeline, LogHawkTrainDetectPipeline")
    print(
        "Activities      : run_identity_mapping, run_stage_a, "
        "run_stage_b, run_stage_b_train, run_stage_b_detect, run_stage_c"
    )
    print("=" * 70)

    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
