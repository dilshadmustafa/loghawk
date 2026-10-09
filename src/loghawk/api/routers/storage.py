from fastapi import APIRouter, HTTPException, Query
from botocore.exceptions import BotoCoreError, ClientError

from loghawk.api.schemas import BatchInfo
from loghawk.api.services.storage_service import list_batches, list_buckets


router = APIRouter()


@router.get("/buckets", response_model=list[str])
def get_buckets() -> list[str]:
    try:
        return list_buckets()
    except (BotoCoreError, ClientError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not list S3 buckets: {exc}",
        ) from exc


@router.get("/batches", response_model=list[BatchInfo])
def get_batches(bucket: str = Query(min_length=1)) -> list[BatchInfo]:
    if "/" in bucket or "\\" in bucket:
        raise HTTPException(status_code=400, detail="Invalid bucket name")
    try:
        return list_batches(bucket.strip())
    except ClientError as exc:
        status_code = 404 if exc.response.get("Error", {}).get("Code") in {
            "NoSuchBucket", "404", "NotFound"
        } else 502
        raise HTTPException(
            status_code=status_code,
            detail=f"Could not list batches for bucket {bucket!r}: {exc}",
        ) from exc
    except BotoCoreError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not list batches for bucket {bucket!r}: {exc}",
        ) from exc
