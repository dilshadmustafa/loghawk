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
        external_data_use: bool,
        external_train_sources: list[dict[str, str]],
        external_raw_sources: list[dict[str, str]],
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
        train_mapping_root = root + "/identitymapping/train/"
        raw_mapping_root = root + "/identitymapping/raw/"
        train_inputs = external_train_sources if external_data_use else [
            {"url": train_raw, "region": ""}
        ]
        raw_inputs = external_raw_sources if external_data_use else [
            {"url": detect_raw, "region": ""}
        ]

        if train_phase:
            await workflow.execute_activity(
                run_identity_mapping,
                args=[train_inputs, train_mapping_root, "train", external_data_use],
                start_to_close_timeout=timedelta(hours=2),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            await workflow.execute_activity(
                run_stage_a,
                args=[train_inputs, train_features, train_mapping_root, "train", external_data_use],
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
                args=[raw_inputs, raw_mapping_root, "raw", external_data_use],
                start_to_close_timeout=timedelta(hours=2),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            await workflow.execute_activity(
                run_stage_a,
                args=[raw_inputs, detect_features, raw_mapping_root, "raw", external_data_use],
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
