import csv
import io
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from botocore.config import Config as BotoConfig
import boto3

# Let this root-level script import the project's config module.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import loghawk.config as config


def event(timestamp, level, message, status_code, exception=None):
    return {
        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
        "service": "payment-service",
        "level": level,
        "message": message,
        "status_code": status_code,
        "exception": exception,
    }


# Training data: mostly normal activity, with occasional low-level warnings
# and errors so the model learns variation in those features.
train_start = datetime(2026, 9, 28, 10, 0, tzinfo=timezone.utc)
train_rows = []

for minute in range(60):
    minute_start = train_start + timedelta(minutes=minute)

    for index in range(6 + minute % 3):
        train_rows.append(event(
            minute_start + timedelta(seconds=index),
            "INFO",
            f"Payment completed normally, sample {index}",
            200,
        ))

    if minute % 5 == 0:
        train_rows.append(event(
            minute_start + timedelta(seconds=20),
            "WARN",
            "Payment request throttled",
            429,
        ))

    if minute in {15, 30, 45}:
        train_rows.append(event(
            minute_start + timedelta(seconds=30),
            "ERROR",
            "Transient payment gateway failure",
            500,
            "TransientGatewayError",
        ))


# Detection data: 20 HTTP 503 errors in each of three consecutive minutes.
raw_start = datetime(2026, 9, 28, 11, 5, tzinfo=timezone.utc)
raw_rows = []

for minute in range(3):
    minute_start = raw_start + timedelta(minutes=minute)

    for index in range(20):
        raw_rows.append(event(
            minute_start + timedelta(seconds=index),
            "ERROR",
            f"Database unavailable during payment {index}",
            503,
            "DatabaseUnavailable",
        ))


def to_jsonl_bytes(rows):
    return (
        "\n".join(json.dumps(row) for row in rows) + "\n"
    ).encode("utf-8")


def to_csv_bytes(rows):
    output = io.StringIO(newline="")
    fieldnames = [
        "timestamp",
        "service",
        "level",
        "message",
        "status_code",
        "exception",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


invoice_train_rows = [
    {**row, "service": "invoice-service"} for row in train_rows
]
invoice_raw_rows = [
    {**row, "service": "invoice-service"} for row in raw_rows
]


s3 = boto3.client(
    "s3",
    endpoint_url=config.LH_S3_ENDPOINT,
    aws_access_key_id=config.LH_S3_ACCESS_KEY_ID,
    aws_secret_access_key=config.LH_S3_SECRET_ACCESS_KEY,
    region_name=config.LH_S3_REGION,
    config=BotoConfig(
        s3={"addressing_style": "path"},
    ),
)

batch = config.LH_S3_BATCH_FOLDER.strip("/")
objects = [
    (
        f"{batch}/train/elasticsearch-1.log",
        train_rows,
        to_jsonl_bytes,
        "application/x-ndjson",
    ),
    (
        f"{batch}/raw/elasticsearch-1.log",
        raw_rows,
        to_jsonl_bytes,
        "application/x-ndjson",
    ),
    (
        f"{batch}/train/invoice-service-1.csv",
        invoice_train_rows,
        to_csv_bytes,
        "text/csv",
    ),
    (
        f"{batch}/raw/invoice-service-1.csv",
        invoice_raw_rows,
        to_csv_bytes,
        "text/csv",
    ),
]

for key, rows, encode_rows, content_type in objects:
    s3.put_object(
        Bucket=config.LH_S3_BUCKET,
        Key=key,
        Body=encode_rows(rows),
        ContentType=content_type,
    )
    print(
        f"Uploaded s3://{config.LH_S3_BUCKET}/{key} "
        f"({len(rows)} records)"
    )
