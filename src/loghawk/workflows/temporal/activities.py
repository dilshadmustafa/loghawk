from datetime import timedelta

from temporalio import activity


@activity.defn
async def run_stage_a(
    input_path: str,
    output_path: str,
) -> str:
    """
    Run Stage A PySpark feature engineering.

    Stage A:
        raw logs -> feature parquet
    """

    activity.logger.info(
        f"Starting Stage A feature engineering: "
        f"{input_path} -> {output_path}"
    )

    # Import the existing Stage A implementation.
    from loghawk.feature_engineering import (
        pyspark_s3_feature_engineering
    )

    # ---------------------------------------------------------
    # IMPORTANT:
    #
    # Adapt this call to the actual callable exposed by your
    # current Stage A module.
    # ---------------------------------------------------------

    pyspark_s3_feature_engineering.run(
        input_path=input_path,
        output_path=output_path,
    )

    activity.logger.info(
        f"Stage A completed successfully: {output_path}"
    )

    return output_path


@activity.defn
async def run_stage_b(
    input_path: str,
    output_path: str,
) -> str:
    """
    Run Stage B Isolation Forest anomaly detection.

    Stage B:
        feature parquet -> anomaly results
    """

    activity.logger.info(
        f"Starting Stage B anomaly detection: "
        f"{input_path} -> {output_path}"
    )

    # Import the existing Stage B implementation.
    from loghawk.anomaly_detection import (
        scikit_s3_isolation_forest
    )

    # ---------------------------------------------------------
    # IMPORTANT:
    #
    # Adapt this call to the actual callable exposed by your
    # current Stage B module.
    # ---------------------------------------------------------

    scikit_s3_isolation_forest.run(
        input_path=input_path,
        output_path=output_path,
    )

    activity.logger.info(
        f"Stage B completed successfully: {output_path}"
    )

    return output_path