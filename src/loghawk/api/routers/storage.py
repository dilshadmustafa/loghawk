from fastapi import APIRouter, HTTPException, Query
from botocore.exceptions import BotoCoreError, ClientError

from loghawk.api.schemas import BatchInfo
from loghawk.api.services.storage_service import (
    list_batches,
    list_buckets,
    list_external_buckets,
    list_external_folders,
    list_external_regions,
)


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


@router.get("/external/buckets", response_model=list[str])
def get_external_buckets(region: str = Query(min_length=1)) -> list[str]:
    try:
        return list_external_buckets(region.strip())
    except (BotoCoreError, ClientError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not list external S3 buckets in {region!r}: {exc}",
        ) from exc


@router.get("/external/regions", response_model=list[str])
def get_external_regions() -> list[str]:
    return list_external_regions()


@router.get("/external/folders", response_model=list[str])
def get_external_folders(
    bucket: str = Query(min_length=1),
    region: str = Query(min_length=1),
    prefix: str = Query(default=""),
) -> list[str]:
    if "/" in bucket or "\\" in bucket:
        raise HTTPException(status_code=400, detail="Invalid bucket name")
    try:
        return list_external_folders(bucket.strip(), region.strip(), prefix)
    except ClientError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Could not list folders in external bucket {bucket!r} "
                f"for region {region!r}: {exc}"
            ),
        ) from exc
    except BotoCoreError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not list external S3 folders: {exc}",
        ) from exc
