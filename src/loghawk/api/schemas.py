from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


RunMode = Literal["train", "detect", "train-detect"]


class ExternalS3Source(BaseModel):
    bucket: str
    folder: str = ""
    region: str

    @field_validator("bucket")
    @classmethod
    def validate_bucket(cls, value: str) -> str:
        value = value.strip()
        if not value or "/" in value or "\\" in value:
            raise ValueError("bucket must be a non-empty bucket name")
        return value

    @field_validator("folder")
    @classmethod
    def normalize_folder(cls, value: str) -> str:
        return value.strip().strip("/")

    @field_validator("region")
    @classmethod
    def validate_region(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("an AWS region is required for each external source")
        return value

    def as_workflow_source(self) -> dict[str, str]:
        suffix = f"{self.folder}/" if self.folder else ""
        return {"url": f"s3://{self.bucket}/{suffix}", "region": self.region}


class ConfigSetWrite(BaseModel):
    name: str
    external_data_use: bool
    source_bucket: str | None = None
    source_batch: str | None = None
    train_sources: list[ExternalS3Source] = Field(default_factory=list)
    detect_sources: list[ExternalS3Source] = Field(default_factory=list)
    output_bucket: str | None = None
    output_batch: str | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Config Set name is required")
        return value

    @field_validator("source_bucket", "output_bucket")
    @classmethod
    def validate_optional_bucket(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value or "/" in value or "\\" in value:
            raise ValueError("bucket must be a non-empty bucket name")
        return value

    @field_validator("source_batch", "output_batch")
    @classmethod
    def validate_optional_batch(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip().strip("/")
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError("batch must be one non-empty folder name")
        return value

    @model_validator(mode="after")
    def validate_storage_mode(self):
        if self.external_data_use:
            if not self.output_bucket or not self.output_batch:
                raise ValueError(
                    "external Config Sets require a RustFS output bucket and batch"
                )
        else:
            if not self.source_bucket or not self.source_batch:
                raise ValueError(
                    "internal Config Sets require an input bucket and batch"
                )
            if self.output_bucket not in {None, self.source_bucket}:
                raise ValueError("internal output bucket must match the input bucket")
            if self.output_batch not in {None, self.source_batch}:
                raise ValueError("internal output batch must match the input batch")
        return self


class PipelineWrite(BaseModel):
    name: str
    config_set_id: str
    run_mode: RunMode

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Pipeline name is required")
        return value


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
