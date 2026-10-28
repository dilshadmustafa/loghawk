"""Authenticated Web UI RCA chat using the shared collector assistant."""
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, StrictBool
from loghawk.api.routers.auth import require_webui_user
from loghawk.api.services import assistant_chat_service, collector_tool_service

router = APIRouter(dependencies=[Depends(require_webui_user)])

class ChatRequest(BaseModel):
    messages: list[dict[str, str]] = Field(min_length=1)
    review_arguments: StrictBool = True

class Confirmation(BaseModel):
    confirmation_id: str = Field(min_length=1, max_length=64)
    approved: StrictBool
    arguments: dict[str, object] | None = None

@router.post("/chat")
def chat(body: ChatRequest, user: dict = Depends(require_webui_user)) -> dict:
    try:
        return assistant_chat_service.chat(
            body.messages, user["userid"], origin_software="web",
            review_arguments=body.review_arguments,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

@router.post("/chat/{request_id}/confirm")
def confirm(request_id: str, body: Confirmation, user: dict = Depends(require_webui_user)) -> dict:
    try:
        return collector_tool_service.resume(request_id, user["userid"], body.confirmation_id,
                                             body.approved, origin_software="web", arguments=body.arguments)
    except collector_tool_service.ArgumentReviewError as exc:
        raise HTTPException(status_code=422, detail=str(exc),
                            headers={"X-LogHawk-Execution": "not-started"}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/chat/{request_id}/explanation/retry")
def retry_explanation(request_id: str, user: dict = Depends(require_webui_user)) -> dict:
    from loghawk.api.services import collector_tool_service
    try:
        return collector_tool_service.retry_explanation(
            request_id, user["userid"], origin_software="web",
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/history")
def history(user: dict = Depends(require_webui_user), search: str = Query("", max_length=300),
            userid: str = Query("", max_length=200), origin: str = Query("", max_length=20),
            collector: str = Query("", max_length=200), status: str = Query("", max_length=40),
            date_from: str = Query("", max_length=40), date_to: str = Query("", max_length=40),
            limit: int = Query(25, ge=1, le=100), offset: int = Query(0, ge=0)):
    from datetime import datetime, UTC
    from loghawk.api.services import collector_tool_store
    def utc_value(value):
        if not value:
            return ""
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("Timezone required")
            return parsed.astimezone(UTC).isoformat()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Date filters must include a timezone.") from exc
    return collector_tool_store.history_list(user, search=search, userid=userid, origin=origin,
        collector=collector, status=status, date_from=utc_value(date_from), date_to=utc_value(date_to),
        limit=limit, offset=offset)


@router.get("/history/{request_id}")
def history_detail(request_id: str, user: dict = Depends(require_webui_user)):
    from loghawk.api.services import collector_tool_store
    detail = collector_tool_store.history_detail(request_id, user)
    if detail is None:
        raise HTTPException(status_code=404, detail="RCA request not found.")
    return detail
