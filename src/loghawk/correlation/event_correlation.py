"""
LogHawk Stage C - Event Correlation

Stage B anomaly events
        ↓
Temporal correlation
        ↓
Service correlation
        ↓
Incident grouping
        ↓
Correlated incidents
"""

from __future__ import annotations

import argparse
import re
from typing import List

import fsspec
import pandas as pd

import loghawk.config as config


# ============================================================
# Configuration
# ============================================================

DEFAULT_CORRELATION_WINDOW_MINUTES = config.LH_CORRELATION_WINDOW_MINUTES


# ============================================================
# S3 helpers
# ============================================================

def normalize_s3_uri(path: str) -> str:
    """
    Convert s3a:// to s3:// for fsspec/pandas.
    """

    if path.startswith("s3a://"):
        return "s3://" + path[len("s3a://"):]

    return path


def s3_storage_options() -> dict:
    """Build fsspec options for the configured S3-compatible store."""
    return {
        "key": config.LH_S3_ACCESS_KEY_ID,
        "secret": config.LH_S3_SECRET_ACCESS_KEY,
        "client_kwargs": {
            "endpoint_url": config.LH_S3_ENDPOINT,
            "region_name": config.LH_S3_REGION,
        },
        "config_kwargs": {
            "s3": {
                "addressing_style": "path",
            },
        },
    }


def read_parquet(path: str) -> pd.DataFrame:

    path = normalize_s3_uri(path)

    print(f"Reading anomaly results from: {path}")

    storage_options = (
        s3_storage_options()
        if path.startswith("s3://")
        else {}
    )

    df = pd.read_parquet(
        path,
        storage_options=storage_options,
    )

    print(f"Rows loaded: {len(df)}")

    return df


def write_parquet(
    df: pd.DataFrame,
    path: str,
) -> None:

    path = normalize_s3_uri(path)

    print(f"Writing correlated incidents to: {path}")

    storage_options = (
        s3_storage_options()
        if path.startswith("s3://")
        else {}
    )

    df.to_parquet(
        path,
        index=False,
        storage_options=storage_options,
    )

    print("Correlation results saved.")


# ============================================================
# Data preparation
# ============================================================

def prepare_anomalies(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    print()
    print("Input columns:")
    print(list(df.columns))

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    if "timestamp" not in df.columns:

        raise ValueError(
            "Stage B output must contain a 'timestamp' column."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["timestamp"]
    )

    # --------------------------------------------------------
    # Service
    # --------------------------------------------------------

    service_values = pd.Series(
        pd.NA,
        index=df.index,
        dtype="object",
    )
    service_source_columns = (
        "service",
        "application",
        "app_name",
        "application_id",
        "entity_id",
    )

    for column in service_source_columns:
        if column not in df.columns:
            continue

        candidate = (
            df[column]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )
        service_values = service_values.fillna(candidate)

    if service_values.isna().all():
        print(
            "WARNING: no service or entity identity column has values. "
            "Using 'unknown-service'."
        )

    df["service"] = (
        service_values
        .fillna("unknown-service")
        .astype(str)
    )

    # --------------------------------------------------------
    # is_anomaly
    # --------------------------------------------------------

    if "is_anomaly" not in df.columns:

        raise ValueError(
            "Stage B output must contain 'is_anomaly'."
        )

    # Handle bool / integer / string representations
    if df["is_anomaly"].dtype != bool:

        df["is_anomaly"] = (
            df["is_anomaly"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )

    # Only anomalous events enter correlation
    df = df[
        df["is_anomaly"]
    ].copy()

    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------

    if "severity" not in df.columns:

        df["severity"] = "WARNING"

    df["severity"] = (
        df["severity"]
        .fillna("WARNING")
        .astype(str)
        .str.upper()
    )

    # --------------------------------------------------------
    # Anomaly score
    # --------------------------------------------------------

    if "anomaly_score" not in df.columns:

        df["anomaly_score"] = 0.0

    df["anomaly_score"] = pd.to_numeric(
        df["anomaly_score"],
        errors="coerce",
    ).fillna(0.0)

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return df


# ============================================================
# Severity ranking
# ============================================================

SEVERITY_WEIGHT = {
    "NORMAL": 0,
    "INFO": 1,
    "WARNING": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}


def severity_weight(
    severity: str,
) -> int:

    return SEVERITY_WEIGHT.get(
        severity.upper(),
        2,
    )


# ============================================================
# Temporal correlation
# ============================================================

def create_temporal_groups(
    df: pd.DataFrame,
    correlation_window_minutes: int = DEFAULT_CORRELATION_WINDOW_MINUTES,
) -> pd.DataFrame:

    df = df.copy()

    if df.empty:

        df["correlation_group"] = pd.Series(
            dtype="int64"
        )

        return df

    window = pd.Timedelta(
        minutes=correlation_window_minutes
    )

    groups = []

    current_group = 0

    previous_timestamp = None

    for timestamp in df["timestamp"]:

        if previous_timestamp is None:

            current_group = 0

        elif timestamp - previous_timestamp > window:

            current_group += 1

        groups.append(current_group)

        previous_timestamp = timestamp

    df["correlation_group"] = groups

    return df


# ============================================================
# Incident correlation
# ============================================================

def correlate_events(
    df: pd.DataFrame,
    correlation_window_minutes: int = DEFAULT_CORRELATION_WINDOW_MINUTES,
) -> pd.DataFrame:

    df = prepare_anomalies(df)

    if df.empty:

        print("No anomalies available for correlation.")

        return pd.DataFrame()

    df = create_temporal_groups(
        df,
        correlation_window_minutes,
    )

    incidents = []

    for group_id, group in df.groupby(
        "correlation_group"
    ):

        services = sorted(
            group["service"]
            .dropna()
            .unique()
            .tolist()
        )

        severities = group[
            "severity"
        ].tolist()

        max_severity = max(
            severities,
            key=severity_weight,
        )

        start_time = group[
            "timestamp"
        ].min()

        end_time = group[
            "timestamp"
        ].max()

        duration_seconds = (
            end_time - start_time
        ).total_seconds()

        anomaly_count = len(group)

        service_count = len(
            services
        )

        # ----------------------------------------------------
        # Correlation score
        # ----------------------------------------------------

        temporal_score = min(
            1.0,
            anomaly_count / 5.0,
        )

        service_score = min(
            1.0,
            service_count / 3.0,
        )

        severity_score = (
            severity_weight(max_severity)
            / 4.0
        )

        correlation_score = (
            0.40 * temporal_score
            + 0.30 * service_score
            + 0.30 * severity_score
        )

        # ----------------------------------------------------
        # Root / primary service candidate
        #
        # This is NOT root-cause determination.
        # It is simply the service with the strongest
        # anomaly signal and should be investigated first.
        # ----------------------------------------------------

        group = group.copy()

        group["severity_weight"] = (
            group["severity"]
            .map(severity_weight)
        )

        group["event_strength"] = (
            group["severity_weight"]
            * (
                group["anomaly_score"]
                .abs()
                + 1.0
            )
        )

        primary_event = group.loc[
            group["event_strength"].idxmax()
        ]

        primary_service = (
            primary_event["service"]
        )

        incident_id = (
            f"INC-{start_time.strftime('%Y%m%d%H%M%S')}"
            f"-{int(group_id):04d}"
        )

        incidents.append(
            {
                "incident_id": incident_id,

                "start_time": start_time,

                "end_time": end_time,

                "duration_seconds":
                    duration_seconds,

                "anomaly_count":
                    anomaly_count,

                "service_count":
                    service_count,

                "services":
                    ",".join(services),

                "primary_service":
                    primary_service,

                "max_severity":
                    max_severity,

                "correlation_score":
                    round(
                        correlation_score,
                        4,
                    ),
            }
        )

    incidents_df = pd.DataFrame(
        incidents
    )

    return incidents_df


# ============================================================
# Attach incident IDs back to anomaly events
# ============================================================

def attach_incident_ids(
    df: pd.DataFrame,
    correlation_window_minutes: int = DEFAULT_CORRELATION_WINDOW_MINUTES,
) -> pd.DataFrame:

    df = prepare_anomalies(df)

    if df.empty:

        return df

    df = create_temporal_groups(
        df,
        correlation_window_minutes,
    )

    incident_map = {}

    for group_id, group in df.groupby(
        "correlation_group"
    ):

        start_time = group[
            "timestamp"
        ].min()

        incident_id = (
            f"INC-{start_time.strftime('%Y%m%d%H%M%S')}"
            f"-{int(group_id):04d}"
        )

        incident_map[group_id] = incident_id

    df["incident_id"] = (
        df["correlation_group"]
        .map(incident_map)
    )

    return df


# ============================================================
# Main Stage C
# ============================================================

def run(
    input_path: str,
    output_path: str,
    correlation_window_minutes: int = DEFAULT_CORRELATION_WINDOW_MINUTES,
) -> str:

    print("=" * 70)
    print("LogHawk Stage C - Event Correlation")
    print("=" * 70)

    print(
        f"Input : {input_path}"
    )

    print(
        f"Output: {output_path}"
    )

    print(
        f"Correlation window: "
        f"{correlation_window_minutes} minutes"
    )

    # --------------------------------------------------------
    # Load Stage B results
    # --------------------------------------------------------

    df = read_parquet(
        input_path
    )

    # --------------------------------------------------------
    # Generate incidents
    # --------------------------------------------------------

    incidents_df = correlate_events(
        df,
        correlation_window_minutes,
    )

    # --------------------------------------------------------
    # Generate event -> incident mapping
    # --------------------------------------------------------

    events_df = attach_incident_ids(
        df,
        correlation_window_minutes,
    )

    # --------------------------------------------------------
    # Save incident results
    # --------------------------------------------------------

    if not incidents_df.empty:

        write_parquet(
            incidents_df,
            output_path,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CORRELATION SUMMARY")
    print("=" * 70)

    print(
        f"Anomaly events : {len(events_df)}"
    )

    print(
        f"Correlated incidents : "
        f"{len(incidents_df)}"
    )

    if not incidents_df.empty:

        print()
        print(
            incidents_df[
                [
                    "incident_id",
                    "start_time",
                    "end_time",
                    "primary_service",
                    "services",
                    "max_severity",
                    "anomaly_count",
                    "correlation_score",
                ]
            ].to_string(
                index=False
            )
        )

    print("=" * 70)

    return output_path


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_CORRELATION_WINDOW_MINUTES,
    )

    args = parser.parse_args()

    run(
        input_path=args.input,
        output_path=args.output,
        correlation_window_minutes=args.window,
    )
