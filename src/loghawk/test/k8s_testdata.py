import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig

# Let this script import LogHawk's config when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import loghawk.config as config


TRAIN_BUCKET = "k8slogstrain"
RAW_BUCKET = "k8slogs"
START_TIME = datetime(2026, 10, 8, 10, 0, tzinfo=timezone.utc)

SERVICES = [
    (
        "payment-service",
        "payment-api",
        "payment-api-7b9d7c8d5f-2kplm",
    ),
    (
        "order-service",
        "order-api",
        "order-api-6c8f9d7c4b-v4q2n",
    ),
    (
        "inventory-service",
        "inventory-api",
        "inventory-api-5d7c6b8f9c-8zjrm",
    ),
]


def make_log(
    service: str,
    container: str,
    pod: str,
    minute: int,
    second: int,
    level: str,
    message: str,
    status_code: int | None = None,
    exception: str | None = None,
) -> dict:
    timestamp = START_TIME + timedelta(minutes=minute, seconds=second)
    return {
        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
        "service": service,
        "container": container,
        "pod": pod,
        "namespace": "production",
        "level": level,
        "message": message,
        "status_code": status_code,
        "exception": exception,
    }


def build_service_logs(
    service: str,
    container: str,
    pod: str,
    raw_phase: bool,
) -> list[dict]:
    rows = []

    # Normal baseline activity for both datasets.
    for minute in range(5):
        for second in (0, 15, 30, 45):
            rows.append(make_log(
                service, container, pod, minute, second,
                "INFO", f"{service} request processed", 200,
            ))

    # A small warning rate is part of the training baseline.
    rows.append(make_log(
        service, container, pod, 2, 35,
        "WARN", f"{service} request was delayed", 429,
    ))

    # Add a concentrated error burst only to the raw inventory logs.
    if raw_phase and service == "inventory-service":
        for second in range(0, 60, 5):
            rows.append(make_log(
                service, container, pod, 4, second,
                "ERROR",
                "Inventory database connection failed",
                503,
                "DatabaseConnectionError",
            ))

    return rows


def to_jsonl_bytes(rows: list[dict]) -> bytes:
    return (
        "\n".join(json.dumps(row) for row in rows) + "\n"
    ).encode("utf-8")


def main() -> None:
    s3 = boto3.client(
        "s3",
        endpoint_url=config.LH_S3_ENDPOINT,
        aws_access_key_id=config.LH_S3_ACCESS_KEY_ID,
        aws_secret_access_key=config.LH_S3_SECRET_ACCESS_KEY,
        region_name=config.LH_S3_REGION,
        config=BotoConfig(s3={"addressing_style": "path"}),
    )

    print(f"Uploading Kubernetes training logs to s3://{TRAIN_BUCKET}/")
    print(f"Uploading Kubernetes raw logs to s3://{RAW_BUCKET}/")

    for index, (service, container, pod) in enumerate(SERVICES, start=1):
        filename = f"k8slogs-{index}.log"
        train_rows = build_service_logs(
            service, container, pod, raw_phase=False
        )
        raw_rows = build_service_logs(
            service, container, pod, raw_phase=True
        )

        for bucket, rows in (
            (TRAIN_BUCKET, train_rows),
            (RAW_BUCKET, raw_rows),
        ):
            s3.put_object(
                Bucket=bucket,
                Key=filename,
                Body=to_jsonl_bytes(rows),
                ContentType="application/x-ndjson",
            )
            print(
                f"Uploaded s3://{bucket}/{filename} "
                f"({len(rows)} records)"
            )


if __name__ == "__main__":
    main()
