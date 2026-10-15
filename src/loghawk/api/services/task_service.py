"""LLM-assisted task drafting and artifact validation."""

import ast
import json

import loghawk.config as config
from loghawk.api.services import task_store
from loghawk.llm.client import get_llm_client


def _response_schema(task: dict) -> dict:
    artifact_type = (
        ["object", "null"]
        if task["artifact_type"] == "n8n"
        else ["string", "null"]
    )
    return {
        "type": "object",
        "properties": {
            "reply": {"type": "string"},
            "artifact": {"type": artifact_type},
        },
        "required": ["reply", "artifact"],
        "additionalProperties": False,
    }


def _system_prompt(task: dict) -> str:
    kind = task["artifact_type"]
    if kind == "n8n":
        artifact_instructions = (
            "When providing an artifact, return a complete n8n workflow as a JSON "
            "object, not as a quoted or escaped JSON string. It must contain a nodes "
            "array and connections object. "
            "Do not claim the workflow has been imported or activated."
        )
        artifact_representation = "JSON object"
    elif kind == "python":
        artifact_instructions = (
            "Put the complete Python source only in the artifact field. The reply must "
            "be a short explanation, never source code. Include all required imports "
            "and implementation; do not return fragments or placeholders."
        )
        artifact_representation = "string"
    else:
        shell = {
            "bash": "Bash",
            "powershell": "PowerShell",
            "windows-cmd": "Windows CMD",
        }[kind]
        artifact_instructions = (
            f"When providing an artifact, return a complete {shell} script as plain text. "
            "Do not mix shell syntaxes."
        )
        artifact_representation = "string"
    return (
        "You help the user draft automation artifacts through conversation. "
        "Never execute code, access external systems, or claim an action was run. "
        "Do not include real credentials or secrets; use clearly named placeholders. "
        "Return a JSON object matching the requested response schema with a user-facing "
        f"reply and either a complete artifact {artifact_representation} or null. "
        "Return null until enough "
        "requirements are known to produce a useful draft. "
        f"Task name: {task['name']}. {artifact_instructions}"
    )


def _validate_artifact(task: dict, artifact: str | dict | None) -> str | None:
    if artifact is None:
        return None
    if task["artifact_type"] == "n8n":
        if isinstance(artifact, str):
            if not artifact.strip():
                raise ValueError("The model returned an empty artifact")
            try:
                workflow = json.loads(artifact)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Generated n8n workflow is not valid JSON: {exc}"
                ) from exc
            serialized_artifact = artifact
        elif isinstance(artifact, dict):
            workflow = artifact
            serialized_artifact = json.dumps(
                workflow, ensure_ascii=False, indent=2
            )
        else:
            raise ValueError("Generated n8n workflow must be a JSON object")
        if (
            not isinstance(workflow, dict)
            or not isinstance(workflow.get("nodes"), list)
            or not isinstance(workflow.get("connections"), dict)
        ):
            raise ValueError(
                "Generated n8n JSON must contain a nodes array and connections object"
            )
        return serialized_artifact
    elif task["artifact_type"] == "python":
        if not isinstance(artifact, str) or not artifact.strip():
            raise ValueError("The model returned an empty or invalid artifact")
        try:
            tree = ast.parse(artifact)
        except SyntaxError as exc:
            raise ValueError(f"Generated Python has a syntax error: {exc}") from exc
        if (
            len(tree.body) == 1
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, (ast.Name, ast.Attribute))
        ):
            raise ValueError(
                "Generated Python is only a bare expression, not a complete program."
            )
    elif not isinstance(artifact, str) or not artifact.strip():
        raise ValueError("The model returned an empty or invalid artifact")
    return artifact


def draft_response(task: dict, history: list[dict[str, str]]) -> tuple[str, str | None]:
    content, response = get_llm_client().complete_with_response(
        [{"role": "system", "content": _system_prompt(task)}, *history],
        response_schema=_response_schema(task),
        model=config.LH_LLM_CODEASSIST_MODEL,
        temperature=0.2,
        max_tokens=config.LH_LLM_CODEASSIST_MAX_TOKENS,
    )
    finish_reason = getattr(response.choices[0], "finish_reason", None)
    if finish_reason == "length":
        raise ValueError(
            "The code assistant reached its output-token limit. "
            "Increase LH_LLM_CODEASSIST_MAX_TOKENS and retry."
        )
    try:
        payload = json.loads(content)
        reply = payload["reply"]
        artifact = payload.get("artifact")
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError("LLM response did not match the task response schema") from exc
    expected_artifact_type = dict if task["artifact_type"] == "n8n" else str
    if not isinstance(reply, str) or (
        artifact is not None and not isinstance(artifact, expected_artifact_type)
    ):
        raise RuntimeError("LLM response has invalid reply or artifact types")
    return reply, _validate_artifact(task, artifact)


def respond_to_message(task_id: str, content: str) -> dict | None:
    task = task_store.get_task(task_id)
    if task is None:
        return None
    task_store.add_message(task_id, "user", content)
    history = task_store.get_messages(task_id)
    reply, artifact = draft_response(task, history)
    task_store.add_message(task_id, "assistant", reply)
    result = task_store.get_task(task_id)
    if result is not None:
        result["draft_artifact"] = artifact
    return result


def save_artifact_version(task_id: str, content: str) -> dict | None:
    task = task_store.get_task(task_id)
    if task is None:
        return None
    validated = _validate_artifact(task, content)
    if validated is None:
        raise ValueError("Artifact cannot be empty")
    task_store.save_artifact(task_id, validated)
    return task_store.get_task(task_id)
