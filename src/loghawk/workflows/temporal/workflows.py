from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from loghawk.workflows.temporal.activities import (
        run_stage_a,
        run_stage_b,
    )


@workflow.defn
class LogHawkPipeline:
    """
    LogHawk Stage A -> Stage B pipeline.

    Stage A:
        Raw logs -> Feature dataset

    Stage B:
        Feature dataset -> Anomaly results
    """

    @workflow.run
    async def run(
        self,
        input_path: str,
        feature_output_path: str,
        anomaly_output_path: str,
    ) -> str:

        # =====================================================
        # STAGE A
        # =====================================================

        feature_path = await workflow.execute_activity(
            run_stage_a,
            args=[
                input_path,
                feature_output_path,
            ],
            start_to_close_timeout=timedelta(
                hours=2
            ),
            retry_policy=workflow.RetryPolicy(
                maximum_attempts=3,
            ),
        )

        # =====================================================
        # STAGE B
        # =====================================================

        anomaly_path = await workflow.execute_activity(
            run_stage_b,
            args=[
                feature_path,
                anomaly_output_path,
            ],
            start_to_close_timeout=timedelta(
                hours=1
            ),
            retry_policy=workflow.RetryPolicy(
                maximum_attempts=3,
            ),
        )

        # =====================================================
        # PIPELINE RESULT
        # =====================================================

        return anomaly_path