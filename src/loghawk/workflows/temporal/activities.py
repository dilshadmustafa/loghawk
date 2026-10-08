from temporalio import activity


@activity.defn
async def run_identity_mapping(
    sources: list[dict[str, str]],
    mapping_root: str,
    phase: str,
    external: bool,
) -> str:
    """Generate or verify identity mappings for every raw file."""
    activity.logger.info(
        f"Starting identity mapping for {phase} sources: {sources}"
    )

    from loghawk.identity_mapping import identity_mapping7

    activity.logger.info(
        f"Identity mapping module: {identity_mapping7.__file__}"
    )
    mapping_paths = identity_mapping7.generate_identity_mappings(
        sources, mapping_root, phase, external
    )
    activity.logger.info(
        f"Identity mapping completed for {len(mapping_paths)} raw file(s)"
    )
    return mapping_root


@activity.defn
async def run_stage_a(
    sources: list[dict[str, str]],
    output_path: str,
    mapping_root: str,
    phase: str,
    external: bool,
) -> str:
    """
    Temporal Activity for LogHawk Stage A.

    Raw logs -> Spark feature parquet
    """

    activity.logger.info(
        f"Starting Stage A ({phase}): {sources} -> {output_path}"
    )

    from loghawk.feature_engineering import (
        pyspark_s3_feature_engineering9
    )

    activity.logger.info(
        f"Stage A module: "
        f"{pyspark_s3_feature_engineering9.__file__}"
    )

    result = pyspark_s3_feature_engineering9.run(
        sources,
        output_path,
        mapping_root,
        phase,
        external,
    )

    activity.logger.info(
        f"Stage A completed: {result}"
    )

    return result


@activity.defn
async def run_stage_b_train(feature_root: str, model_root: str) -> str:
    """Train and persist grouped anomaly detector model artifacts."""
    from loghawk.anomaly_detection import s3_anomaly_detector4

    activity.logger.info(
        f"Training Stage B models: {feature_root} -> {model_root}; "
        f"module={s3_anomaly_detector4.__file__}"
    )
    return s3_anomaly_detector4.train(feature_root, model_root)


@activity.defn
async def run_stage_b_detect(
    feature_root: str,
    anomaly_root: str,
    model_root: str,
) -> str:
    """Load trained artifacts and detect anomalies in grouped raw features."""
    from loghawk.anomaly_detection import s3_anomaly_detector4

    activity.logger.info(
        f"Detecting Stage B anomalies: {feature_root} -> {anomaly_root}; "
        f"models={model_root}; module={s3_anomaly_detector4.__file__}"
    )
    return s3_anomaly_detector4.detect(
        feature_root, anomaly_root, model_root,
    )


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

