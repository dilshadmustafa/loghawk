from fastapi import FastAPI

from loghawk.api.routers.config_sets import router as config_sets_router
from loghawk.api.routers.pipeline_runs import router as pipeline_runs_router
from loghawk.api.routers.pipelines import router as pipelines_router
from loghawk.api.routers.storage import router as storage_router


app = FastAPI(
    title="LogHawk API",
    version="0.1.0",
    description="Control API for LogHawk Temporal pipeline runs.",
)

app.include_router(storage_router, prefix="/api/v1/storage", tags=["storage"])
app.include_router(
    config_sets_router,
    prefix="/api/v1/config-sets",
    tags=["config sets"],
)
app.include_router(
    pipelines_router,
    prefix="/api/v1/pipelines",
    tags=["pipelines"],
)
app.include_router(
    pipeline_runs_router,
    prefix="/api/v1/pipeline-runs",
    tags=["pipeline runs"],
)


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
