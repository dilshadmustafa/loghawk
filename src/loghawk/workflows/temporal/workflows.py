from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from loghawk.workflows.temporal.activities import (
        run_identity_mapping,
        run_stage_a,
        run_stage_b,
        run_stage_c,
    )


@workflow.defn
class LogHawkPipeline:
    """
    Identity Mapping -> Stage A -> Stage B -> Stage C pipeline.

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

        # Identity mappings for every raw file must be ready before Stage A.
        await workflow.execute_activity(
            run_identity_mapping,
            args=[input_path],
            start_to_close_timeout=timedelta(hours=2),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

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
            retry_policy=RetryPolicy(
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
            retry_policy=RetryPolicy(
                maximum_attempts=3,
            ),
        )

        incident_output_path = (
            "s3://loghawk-data/"
            "incidents/year=2026/month=09/day=23/"
            "correlated_incidents.parquet"
        )

        incident_path = await workflow.execute_activity(
            run_stage_c,
            args=[
                anomaly_path,
                incident_output_path,
            ],
            start_to_close_timeout=timedelta(
                minutes=30
            ),
            retry_policy=RetryPolicy(
                maximum_attempts=3,
            ),
        )

        # =====================================================
        # PIPELINE RESULT
        # =====================================================

        #return anomaly_path
        return incident_path
