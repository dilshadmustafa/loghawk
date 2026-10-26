"""Shared authenticated assistant chat for Web UI and Hawk Console."""
from fastapi import HTTPException
from loghawk.api.services import task_store2 as task_store
from loghawk.llm import get_llm_client
import loghawk.config as config

def chat(chat_messages: list[dict], userid: str, *, origin_software: str,
         review_arguments: bool = True) -> dict:
    if config.LH_LLM_TOOL_CALLING_ENABLED:
        from loghawk.api.services import collector_tool_service
        try:
            return collector_tool_service.start(
                chat_messages, userid, origin_software=origin_software,
                review_arguments=review_arguments,
            )
        except collector_tool_service.ContextBudgetError as exc:
            raise HTTPException(status_code=422, detail=str(exc),
                                headers={"X-LogHawk-Execution": "not-started"}) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    candidates = [config.LH_LLM_MODEL, *config.LH_LLM_FALLBACK_MODELS]
    last_error = None
    for model in dict.fromkeys(item for item in candidates if item):
        try:
            messages = list(chat_messages)
            latest_user_message = next(
                (item.get("content", "") for item in reversed(messages)
                 if item.get("role") == "user"),
                "",
            )
            matching_tasks = task_store.relevant_tasks(latest_user_message)
            if matching_tasks:
                details = []
                for task in matching_tasks:
                    artifact = task.get("latest_artifact") or {}
                    approval = "none" if artifact.get("version") is None else (
                        "approved" if artifact.get("approved") else "not approved"
                    )
                    enabled = "enabled" if artifact.get("enabled") else "disabled"
                    details.append(
                        f"- name={task['name']}; description={task.get('description', '')}; "
                        f"category={task.get('category', '')}; kind={task.get('task_kind', 'general')}; "
                        f"type={task.get('artifact_type', '')}; policy={task.get('execution_policy', 'manual')}; "
                        f"latest_version={artifact.get('version', 'none')}; "
                        f"approval_status={approval}; enabled_status={enabled}"
                    )
                messages.insert(0, {
                    "role": "system",
                    "content": (
                        "Potentially relevant saved LogHawk Tasks are listed below. Use them to answer "
                        "questions about existing tasks or to recommend a suitable task. Do not claim "
                        "to run, approve, enable, or modify a task. State its execution policy and "
                        "latest-version approval/enabled status when relevant.\n" + "\n".join(details)
                    ),
                })
            response = get_llm_client().complete(
                messages, model=model, temperature=0.3
            )
            return {"response": response}
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Configured LLM request failed: {last_error}")

