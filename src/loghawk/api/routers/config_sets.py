import sqlite3

from fastapi import APIRouter, HTTPException

import loghawk.config as config
from loghawk.api.schemas import ConfigSetWrite
from loghawk.api.services import config_store


router = APIRouter()


@router.get("/defaults")
def get_config_set_defaults() -> dict:
    return {
        "external_data_use": config.LH_EXTERNAL_DATA_USE,
        "output_bucket": config.LH_S3_BUCKET,
        "output_batch": config.LH_S3_BATCH_FOLDER,
    }


@router.get("")
def list_config_sets() -> list[dict]:
    return config_store.list_config_sets()


@router.post("", status_code=201)
def create_config_set(body: ConfigSetWrite) -> dict:
    try:
        return config_store.save_config_set(body.model_dump())
    except sqlite3.IntegrityError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"A Config Set named {body.name!r} already exists.",
        ) from exc


@router.get("/{config_set_id}")
def read_config_set(config_set_id: str) -> dict:
    config_set = config_store.get_config_set(config_set_id)
    if config_set is None:
        raise HTTPException(status_code=404, detail="Config Set not found")
    return config_set


@router.put("/{config_set_id}")
def update_config_set(config_set_id: str, body: ConfigSetWrite) -> dict:
    try:
        result = config_store.save_config_set(
            body.model_dump(), config_set_id=config_set_id
        )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"A Config Set named {body.name!r} already exists.",
        ) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Config Set not found")
    return result
