from temporalio import activity

@activity.defn
async def run_stage_a(
    input_path: str,
    output_path: str,
) -> str:
    """
    Temporal Activity for LogHawk Stage A.

    Raw logs -> Spark feature parquet
    """

    activity.logger.info(
        f"Starting Stage A: {input_path} -> {output_path}"
    )

    from loghawk.feature_engineering import (
        pyspark_s3_feature_engineering2
    )

    activity.logger.info(
        f"Stage A module: "
        f"{pyspark_s3_feature_engineering2.__file__}"
    )

    result = pyspark_s3_feature_engineering2.run(
        input_path,
        output_path,
    )

    activity.logger.info(
        f"Stage A completed: {result}"
    )

    return result


@activity.defn
async def run_stage_b(
    input_path: str,
    output_path: str,
) -> str:
    """
    Temporal Activity for LogHawk Stage B.

    Feature parquet -> Isolation Forest anomaly results
    """

    activity.logger.info(
        f"Starting Stage B: {input_path} -> {output_path}"
    )

    from loghawk.anomaly_detection import (
        scikit_s3_isolation_forest2
    )

    activity.logger.info(
        f"Stage B module: "
        f"{scikit_s3_isolation_forest2.__file__}"
    )

    result = scikit_s3_isolation_forest2.run(
        input_path,
        output_path,
    )

    activity.logger.info(
        f"Stage B completed: {result}"
    )

    return result
