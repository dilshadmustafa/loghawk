from typing import Literal

from pydantic import BaseModel, field_validator


RunMode = Literal["train", "detect", "train-detect"]


class BatchInfo(BaseModel):
    name: str
    train_available: bool
    raw_available: bool


class PipelineRunRequest(BaseModel):
    bucket: str
    batch: str
    mode: RunMode

    @field_validator("bucket")
    @classmethod
    def validate_bucket(cls, value: str) -> str:
        value = value.strip()
        if not value or "/" in value or "\\" in value:
            raise ValueError("bucket must be a non-empty bucket name")
        return value

    @field_validator("batch")
    @classmethod
    def validate_batch(cls, value: str) -> str:
        value = value.strip()
        if (
            not value
            or value in {".", ".."}
            or "/" in value
            or "\\" in value
        ):
            raise ValueError("batch must be one non-empty folder name")
        return value


class PipelineRunStarted(BaseModel):
    workflow_id: str
    mode: RunMode
    status: str


class PipelineRunStatus(BaseModel):
    workflow_id: str
    run_id: str | None = None
    status: str
