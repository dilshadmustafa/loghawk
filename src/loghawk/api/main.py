import asyncio
import sys
import traceback

from fastapi import Depends, FastAPI

from loghawk.api.routers.config_sets2 import router as config_sets_router
from loghawk.api.routers.auth import require_authenticated_user, router as auth_router
from loghawk.api.routers.console import router as console_router
from loghawk.api.routers.collectors import router as collectors_router
from loghawk.api.routers.pipeline_runs import router as pipeline_runs_router
from loghawk.api.routers.pipelines2 import router as pipelines_router
from loghawk.api.routers.rag import router as rag_router
from loghawk.api.routers.storage import router as storage_router
from loghawk.api.routers.tasks2 import router as tasks_router
from loghawk.api.services.rag_store import fail_interrupted_jobs
from loghawk.api.services.auth_service import initialize_auth
from loghawk.api.services import auth_service
from loghawk.collectors.registry import ensure_registered as ensure_collectors_registered
from loghawk.utils.user_identity import set_userid_override, reset_userid_override


app = FastAPI(
    title="LogHawk API",
    version="0.1.0",
    description="Control API for LogHawk Temporal pipeline runs.",
)

_authenticated = [Depends(require_authenticated_user)]

app.include_router(auth_router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(console_router, prefix="/api/v1/console", tags=["console"])
app.include_router(collectors_router, prefix="/api/v1/console/collectors", tags=["collectors"])
app.include_router(
    storage_router, prefix="/api/v1/storage", tags=["storage"],
    dependencies=_authenticated,
)
app.include_router(
    config_sets_router,
    prefix="/api/v1/config-sets",
    tags=["config sets"],
    dependencies=_authenticated,
)
app.include_router(
    pipelines_router,
    prefix="/api/v1/pipelines",
    tags=["pipelines"],
    dependencies=_authenticated,
)
app.include_router(
    pipeline_runs_router,
    prefix="/api/v1/pipeline-runs",
    tags=["pipeline runs"],
    dependencies=_authenticated,
)
app.include_router(
    tasks_router,
    prefix="/api/v1/tasks",
    tags=["tasks"],
    dependencies=_authenticated,
)
app.include_router(
    rag_router, prefix="/api/v1/rag", tags=["rag"], dependencies=_authenticated,
)


@app.middleware("http")
async def set_authenticated_user_identity(request, call_next):
    token_value = None
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        principal = auth_service.user_for_token(authorization[7:].strip())
        if principal:
            token_value = set_userid_override(principal["userid"])
    try:
        return await call_next(request)
    finally:
        if token_value is not None:
            reset_userid_override(token_value)


@app.on_event("startup")
async def install_asyncio_exception_handler() -> None:
    loop = asyncio.get_running_loop()

    def report_exception(_loop, context) -> None:
        print(
            f"Asyncio event-loop exception: {context.get('message', 'unknown error')}",
            file=sys.stderr,
            flush=True,
        )
        exception = context.get("exception")
        if exception is not None:
            traceback.print_exception(
                type(exception), exception, exception.__traceback__,
                file=sys.stderr,
            )

    loop.set_exception_handler(report_exception)


@app.on_event("startup")
def recover_interrupted_rag_jobs() -> None:
    fail_interrupted_jobs()


@app.on_event("startup")
def seed_administrator() -> None:
    initialize_auth()


@app.on_event("startup")
def seed_collector_registry() -> None:
    ensure_collectors_registered()


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
