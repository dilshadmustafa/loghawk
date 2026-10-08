import argparse

import boto3

import loghawk.config as config


BUCKET = "loghawk-data"
ROOT_PREFIX = "somefolder/"
PROTECTED_PREFIXES = (
    "somefolder/raw/",
    "somefolder/train/",
)
DELETE_BATCH_SIZE = 1000
SAMPLE_KEY_LIMIT = 20


def create_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=config.LH_S3_ENDPOINT,
        region_name=config.LH_S3_REGION,
        aws_access_key_id=config.LH_S3_ACCESS_KEY_ID,
        aws_secret_access_key=config.LH_S3_SECRET_ACCESS_KEY,
    )


def is_protected(key: str) -> bool:
    return any(
        key == prefix.rstrip("/") or key.startswith(prefix)
        for prefix in PROTECTED_PREFIXES
    )


def iter_deletable_keys(client):
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=ROOT_PREFIX):
        for item in page.get("Contents", []):
            key = item["Key"]
            if not is_protected(key):
                yield key


def show_dry_run(client) -> int:
    count = 0
    sample_keys = []

    for key in iter_deletable_keys(client):
        count += 1
        if len(sample_keys) < SAMPLE_KEY_LIMIT:
            sample_keys.append(key)

    print(f"Target: s3://{BUCKET}/{ROOT_PREFIX}")
    print("Preserving:")
    for prefix in PROTECTED_PREFIXES:
        print(f"  s3://{BUCKET}/{prefix}")
    print(f"Objects that would be deleted: {count}")

    if sample_keys:
        print("Example keys:")
        for key in sample_keys:
            print(f"  s3://{BUCKET}/{key}")
        if count > len(sample_keys):
            print(f"  ... and {count - len(sample_keys)} more")

    return count


def delete_objects(client) -> int:
    deleted_count = 0
    batch = []

    for key in iter_deletable_keys(client):
        batch.append({"Key": key})
        if len(batch) == DELETE_BATCH_SIZE:
            client.delete_objects(
                Bucket=BUCKET,
                Delete={"Objects": batch, "Quiet": True},
            )
            deleted_count += len(batch)
            batch = []

    if batch:
        client.delete_objects(
            Bucket=BUCKET,
            Delete={"Objects": batch, "Quiet": True},
        )
        deleted_count += len(batch)

    return deleted_count


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Delete objects below s3://loghawk-data/somefolder/ "
            "except raw/ and train/. Defaults to deletion after confirmation."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List objects that would be deleted without deleting them.",
    )
    args = parser.parse_args()

    client = create_s3_client()
    candidate_count = show_dry_run(client)

    if args.dry_run:
        print("Dry run only. No objects were deleted.")
        return

    if candidate_count == 0:
        print("Nothing to delete.")
        return

    confirmation = input(
        'Type "delete" to continue: '
    )
    if confirmation != "delete":
        print("Confirmation did not match. Nothing was deleted.")
        return

    deleted_count = delete_objects(client)
    print(f"Deleted {deleted_count} object(s).")


if __name__ == "__main__":
    main()
