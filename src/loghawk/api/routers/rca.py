"""Authenticated Web UI RCA chat using the shared collector assistant."""
from fastapi import APIRouter, Depends, HTTPException
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
