from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from loghawk.workflows.temporal.activities import (
        run_identity_mapping,
        run_stage_a,
        run_stage_b_train,
        run_stage_b_detect,
        run_stage_c,
    )


@workflow.defn
class LogHawkTrainDetectPipeline:
    """Separate model training from detection under one batch root."""

    @workflow.run
    async def run(
        self,
        batch_root: str,
        train_phase: bool,
        detect_phase: bool,
    ) -> str:
        if not train_phase and not detect_phase:
            raise ValueError("At least one of train_phase or detect_phase must be enabled")

        root = batch_root.rstrip("/")
        train_raw = root + "/train/"
        detect_raw = root + "/raw/"
        train_features = root + "/features/train/"
        detect_features = root + "/features/raw/"
        model_root = root + "/models/"
        anomaly_root = root + "/anomalies/raw/"

        if train_phase:
            await workflow.execute_activity(
                run_identity_mapping,
                args=[train_raw],
                start_to_close_timeout=timedelta(hours=2),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            await workflow.execute_activity(
                run_stage_a,
                args=[train_raw, train_features],
                start_to_close_timeout=timedelta(hours=3),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            await workflow.execute_activity(
                run_stage_b_train,
                args=[train_features, model_root],
                start_to_close_timeout=timedelta(hours=2),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

        if detect_phase:
            await workflow.execute_activity(
                run_identity_mapping,
                args=[detect_raw],
                start_to_close_timeout=timedelta(hours=2),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            await workflow.execute_activity(
                run_stage_a,
                args=[detect_raw, detect_features],
                start_to_close_timeout=timedelta(hours=3),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            await workflow.execute_activity(
                run_stage_b_detect,
                args=[detect_features, anomaly_root, model_root],
                start_to_close_timeout=timedelta(hours=2),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            incident_path = await workflow.execute_activity(
                run_stage_c,
                args=[anomaly_root, root + "/incidents/correlated_incidents.parquet"],
                start_to_close_timeout=timedelta(minutes=30),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            return incident_path

        return model_root
