"""Import saved task workflows into the configured n8n instance."""

import json
from urllib.parse import quote

import requests

import loghawk.config as config


class N8nApiError(RuntimeError):
    """Raised when the n8n API cannot create a workflow."""


def import_workflow(task_name: str, artifact_content: str) -> dict[str, str]:
    if not config.LH_N8N_API_KEY:
        raise N8nApiError("Set LH_N8N_API_KEY in .env before importing workflows.")

    try:
        workflow = json.loads(artifact_content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Workflow is not valid JSON: {exc}") from exc

    if (
        not isinstance(workflow, dict)
        or not isinstance(workflow.get("nodes"), list)
        or not isinstance(workflow.get("connections"), dict)
    ):
        raise ValueError(
            "Workflow must contain a nodes array and connections object."
        )

    # Only send workflow fields needed for creation. In particular, do not pass
    # an active flag from the artifact: imported workflows remain inactive.
    payload = {
        "name": workflow.get("name") or task_name,
        "nodes": workflow["nodes"],
        "connections": workflow["connections"],
        "settings": workflow.get("settings", {}),
    }
    try:
        response = requests.post(
            f"{config.LH_N8N_BASE_URL}/api/v1/workflows",
            headers={"X-N8N-API-KEY": config.LH_N8N_API_KEY},
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        created = response.json()
    except requests.RequestException as exc:
        detail = getattr(exc.response, "text", "") if exc.response is not None else ""
        raise N8nApiError(
            f"Could not import workflow into n8n: {detail or exc}"
        ) from exc
    except ValueError as exc:
        raise N8nApiError("n8n returned an invalid response after workflow creation.") from exc

    if not isinstance(created, dict) or not created.get("id"):
        raise N8nApiError("n8n created no workflow ID in its response.")

    workflow_id = str(created["id"])
    return {
        "workflow_id": workflow_id,
        "name": str(created.get("name") or payload["name"]),
        "url": f"{config.LH_N8N_BASE_URL}/workflow/{quote(workflow_id, safe='')}",
    }
