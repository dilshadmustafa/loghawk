"""Task lookup and artifact-version operations for Hawk CLI."""

from __future__ import annotations

from loghawk.api.services import task_service2 as task_service
from loghawk.api.services import task_store2 as task_store


_DATABASE_ROLE = "console"


def list_tasks(query: str = "") -> list[dict]:
    """Return saved Tasks from Hawk Console's local database."""
    return task_store.list_tasks(_DATABASE_ROLE, query)


def relevant_tasks(query: str, limit: int = 5) -> list[dict]:
    return task_store.relevant_tasks(query, _DATABASE_ROLE, limit)


def find_task_by_name(name: str) -> dict | None:
    """Find a Task by its case-insensitive name and return its full record."""
    normalized = name.strip().casefold()
    for task in task_store.list_tasks(_DATABASE_ROLE):
        if task["name"].casefold() == normalized:
            return task_store.get_task(task["id"], _DATABASE_ROLE)
    return None


def get_task(task_id: str) -> dict | None:
    return task_store.get_task(task_id, _DATABASE_ROLE)


def create_task(
    name: str, description: str, category: str, artifact_type: str, task_kind: str = "general",
    execution_policy: str = "manual",
) -> dict:
    """Create a Task in Hawk Console's local database."""
    return task_store.create_task(
        {
            "name": name.strip(),
            "description": description.strip(),
            "category": category.strip(),
            "task_kind": task_kind,
            "execution_policy": execution_policy,
            "artifact_type": artifact_type,
        },
        _DATABASE_ROLE,
    )


def set_description(task_id: str, description: str) -> dict | None:
    return task_store.set_task_description(task_id, description, _DATABASE_ROLE)


def set_execution_policy(task_id: str, execution_policy: str) -> dict | None:
    return task_store.set_task_execution_policy(task_id, execution_policy, _DATABASE_ROLE)


def respond_to_message(task_id: str, content: str) -> dict | None:
    """Draft a Task response and artifact with the shared code-assist service."""
    return task_service.respond_to_message(task_id, content.strip(), _DATABASE_ROLE)


def get_artifact(task_id: str, artifact_id: str) -> dict | None:
    return task_store.get_artifact_version(task_id, artifact_id, _DATABASE_ROLE)


def save_artifact(task_id: str, content: str) -> dict | None:
    return task_service.save_artifact_version(task_id, content, _DATABASE_ROLE)


def delete_artifact(task_id: str, artifact_id: str) -> bool:
    return task_store.delete_artifact_version(task_id, artifact_id, _DATABASE_ROLE)


def delete_task(task_id: str) -> bool:
    """Delete a Task and its related data from Hawk Console's local database."""
    return task_store.delete_task(task_id, _DATABASE_ROLE)
