from temporalio import activity


@activity.defn
async def run_identity_mapping(raw_folder: str) -> str:
    """Generate or verify identity mappings for every raw file."""
    activity.logger.info(
        f"Starting identity mapping for raw folder: {raw_folder}"
    )

    from loghawk.identity_mapping import identity_mapping5

    activity.logger.info(
        f"Identity mapping module: {identity_mapping5.__file__}"
    )
    mapping_paths = identity_mapping5.generate_identity_mappings(
        raw_folder
    )
    activity.logger.info(
        f"Identity mapping completed for {len(mapping_paths)} raw file(s)"
    )
    return raw_folder


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
        pyspark_s3_feature_engineering8
    )

    activity.logger.info(
        f"Stage A module: "
        f"{pyspark_s3_feature_engineering8.__file__}"
    )

    result = pyspark_s3_feature_engineering8.run(
        input_path,
        output_path,
    )

    activity.logger.info(
        f"Stage A completed: {result}"
    )

    return result


@activity.defn
async def run_stage_b_train(feature_root: str, model_root: str) -> str:
    """Train and persist grouped Isolation Forest model artifacts."""
    from loghawk.anomaly_detection import scikit_s3_isolation_forest6

    activity.logger.info(
        f"Training Stage B models: {feature_root} -> {model_root}; "
        f"module={scikit_s3_isolation_forest6.__file__}"
    )
    return scikit_s3_isolation_forest6.train(feature_root, model_root)


@activity.defn
async def run_stage_b_detect(
    feature_root: str,
    anomaly_root: str,
    model_root: str,
) -> str:
    """Load trained artifacts and detect anomalies in grouped raw features."""
    from loghawk.anomaly_detection import scikit_s3_isolation_forest6

    activity.logger.info(
        f"Detecting Stage B anomalies: {feature_root} -> {anomaly_root}; "
        f"models={model_root}; module={scikit_s3_isolation_forest6.__file__}"
    )
    return scikit_s3_isolation_forest6.detect(
        feature_root, anomaly_root, model_root,
    )


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
        scikit_s3_isolation_forest4
    )

    activity.logger.info(
        f"Stage B module: "
        f"{scikit_s3_isolation_forest4.__file__}"
    )

    result = scikit_s3_isolation_forest4.run(
        input_path,
        output_path,
    )

    activity.logger.info(
        f"Stage B completed: {result}"
    )

    return result

from temporalio import activity


@activity.defn
async def run_stage_c(
    input_path: str,
    output_path: str,
) -> str:

    activity.logger.info(
        f"Starting Stage C: "
        f"{input_path} -> {output_path}"
    )

    from loghawk.correlation import (
        event_correlation,
    )

    activity.logger.info(
        f"Stage C module: "
        f"{event_correlation.__file__}"
    )

    result = event_correlation.run(
        input_path,
        output_path,
        correlation_window_minutes=(
            event_correlation.DEFAULT_CORRELATION_WINDOW_MINUTES
        ),
    )

    activity.logger.info(
        f"Stage C completed: {result}"
    )

    return result

