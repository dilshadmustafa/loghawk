"""Authenticated Hawk Console LLM and history endpoints."""

from __future__ import annotations

from uuid import UUID
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, StrictBool

from loghawk.api.routers.auth import require_console_user
from loghawk.database.database_layer import connect
from loghawk.api.services import task_store2 as task_store
from loghawk.llm import get_llm_client
import loghawk.config as config


router = APIRouter(dependencies=[Depends(require_console_user)])


class ChatRequest(BaseModel):
    messages: list[dict[str, str]] = Field(min_length=1)
    review_arguments: StrictBool = True


class CommandRequest(BaseModel):
    request: str
    messages: list[dict[str, str]] = Field(default_factory=list)
    shell_name: Literal["PowerShell", "Bash"]


class ChatHistoryItem(BaseModel):
    id: UUID
    role: str
    content: str
    created_at: str


class CommandHistoryItem(BaseModel):
    id: UUID
    terminal_id: str
    command: str
    created_at: str


class HistoryBatch(BaseModel):
    llm_messages: list[ChatHistoryItem] = Field(default_factory=list, max_length=5000)
    commands: list[CommandHistoryItem] = Field(default_factory=list, max_length=5000)


@router.post("/chat")
def chat(body: ChatRequest, user: dict = Depends(require_console_user)) -> dict:
    from loghawk.api.services import assistant_chat_service
    return assistant_chat_service.chat(
        body.messages, user["userid"], origin_software="console",
        review_arguments=body.review_arguments,
    )


@router.post("/command")
def propose_command(body: CommandRequest) -> dict:
    shell_name = body.shell_name
    schema = {
        "type": "object",
        "properties": {"explanation": {"type": "string"}, "command": {"type": ["string", "null"]}},
        "required": ["explanation", "command"],
        "additionalProperties": False,
    }
    content = get_llm_client().complete(
        [
            {"role": "system", "content": (
                f"Draft one {shell_name} command for the user's request. Do not execute it. "
                "Return a brief explanation and exact command. Set command to null if clarification is needed."
            )},
            *body.messages,
            {"role": "user", "content": body.request},
        ],
        response_schema=schema,
        model=config.LH_LLM_CODEASSIST_MODEL or config.LH_LLM_MODEL,
        temperature=0.1,
        max_tokens=config.LH_LLM_CODEASSIST_MAX_TOKENS,
    )
    import json
    result = json.loads(content)
    command = result.get("command")
    if command is not None and (not isinstance(command, str) or not command.strip()):
        raise ValueError("The model returned an invalid command.")
    result["command"] = command.strip() if command else None
    return result


@router.get("/history")
def get_history(user: dict = Depends(require_console_user)) -> dict:
    with connect("server") as db:
        chats = db.execute(
            "SELECT id, role, content FROM console_llm_history WHERE userid = ? ORDER BY created_at, id",
            (user["userid"],),
        ).fetchall()
        commands = db.execute(
            "SELECT id, terminal_id, command FROM console_command_history WHERE userid = ? ORDER BY created_at, id",
            (user["userid"],),
        ).fetchall()
    return {"llm_messages": [dict(row) for row in chats], "commands": [dict(row) for row in commands]}


@router.post("/history")
def save_history(body: HistoryBatch, user: dict = Depends(require_console_user)) -> dict[str, int]:
    saved_chats = 0
    saved_commands = 0
    with connect("server") as db:
        for item in body.llm_messages:
            if item.role not in {"user", "assistant"}:
                continue
            result = db.execute(
                """INSERT INTO console_llm_history (id, userid, role, content, created_at)
                   VALUES (?, ?, ?, ?, ?) ON CONFLICT (id) DO NOTHING""",
                (str(item.id), user["userid"], item.role, item.content, item.created_at),
            )
            saved_chats += max(result.rowcount, 0)
        for item in body.commands:
            if item.terminal_id not in {"command-one", "command-two"}:
                continue
            result = db.execute(
                """INSERT INTO console_command_history (id, userid, terminal_id, command, created_at)
                   VALUES (?, ?, ?, ?, ?) ON CONFLICT (id) DO NOTHING""",
                (str(item.id), user["userid"], item.terminal_id, item.command, item.created_at),
            )
            saved_commands += max(result.rowcount, 0)
            db.execute(
                """DELETE FROM console_command_history
                   WHERE userid = ? AND terminal_id = ? AND id NOT IN (
                       SELECT id FROM console_command_history
                       WHERE userid = ? AND terminal_id = ?
                       ORDER BY created_at DESC, id DESC LIMIT 500
                   )""",
                (user["userid"], item.terminal_id, user["userid"], item.terminal_id),
            )
    return {"saved_llm_messages": saved_chats, "saved_commands": saved_commands}


class ToolConfirmation(BaseModel):
    confirmation_id: str = Field(min_length=1, max_length=64)
    approved: StrictBool
    arguments: dict[str, object] | None = None


@router.post("/chat/{request_id}/confirm")
def confirm_collector(request_id: str, body: ToolConfirmation,
                      user: dict = Depends(require_console_user)) -> dict:
    from loghawk.api.services import collector_tool_service
    try:
        return collector_tool_service.resume(request_id, user["userid"], body.confirmation_id, body.approved, origin_software="console", arguments=body.arguments)
    except collector_tool_service.ArgumentReviewError as exc:
        raise HTTPException(status_code=422, detail=str(exc),
                            headers={"X-LogHawk-Execution": "not-started"}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/chat/{request_id}/explanation/retry")
def retry_explanation(request_id: str, user: dict = Depends(require_console_user)) -> dict:
    from loghawk.api.services import collector_tool_service
    try:
        return collector_tool_service.retry_explanation(
            request_id, user["userid"], origin_software="console",
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
