import boto3
from botocore.config import Config as BotoConfig

import loghawk.config as config
from loghawk.api.schemas import BatchInfo


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=config.LH_S3_ENDPOINT,
        aws_access_key_id=config.LH_S3_ACCESS_KEY_ID,
        aws_secret_access_key=config.LH_S3_SECRET_ACCESS_KEY,
        region_name=config.LH_S3_REGION,
        config=BotoConfig(s3={"addressing_style": "path"}),
    )


def list_buckets() -> list[str]:
    response = _s3_client().list_buckets()
    return sorted(
        bucket["Name"] for bucket in response.get("Buckets", [])
    )


def _prefix_has_objects(client, bucket: str, prefix: str) -> bool:
    response = client.list_objects_v2(
        Bucket=bucket,
        Prefix=prefix,
        MaxKeys=1,
    )
    return bool(response.get("Contents"))


def list_batches(bucket: str) -> list[BatchInfo]:
    client = _s3_client()
    paginator = client.get_paginator("list_objects_v2")
    batch_prefixes = set()
    for page in paginator.paginate(Bucket=bucket, Delimiter="/"):
        for item in page.get("CommonPrefixes", []):
            prefix = item.get("Prefix", "").rstrip("/")
            if prefix:
                batch_prefixes.add(prefix)

    batches = []
    for prefix in sorted(batch_prefixes):
        train_available = _prefix_has_objects(
            client, bucket, prefix + "/train/"
        )
        raw_available = _prefix_has_objects(
            client, bucket, prefix + "/raw/"
        )
        if train_available or raw_available:
            batches.append(
                BatchInfo(
                    name=prefix,
                    train_available=train_available,
                    raw_available=raw_available,
                )
            )
    return batches


def find_batch(bucket: str, batch_name: str) -> BatchInfo | None:
    return next(
        (batch for batch in list_batches(bucket) if batch.name == batch_name),
        None,
    )
