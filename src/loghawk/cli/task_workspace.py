"""Task lookup and artifact-version operations for Hawk CLI."""

from __future__ import annotations

from loghawk.api.services import task_service, task_store


def list_tasks() -> list[dict]:
    """Return saved Tasks from the shared Web UI SQLite store."""
    return task_store.list_tasks()


def find_task_by_name(name: str) -> dict | None:
    """Find a Task by its case-insensitive name and return its full record."""
    normalized = name.strip().casefold()
    for task in task_store.list_tasks():
        if task["name"].casefold() == normalized:
            return task_store.get_task(task["id"])
    return None


def get_task(task_id: str) -> dict | None:
    return task_store.get_task(task_id)


def create_task(
    name: str, category: str, artifact_type: str, shell: str | None = None
) -> dict:
    """Create a Task using the same SQLite store as the Web UI."""
    return task_store.create_task(
        {
            "name": name.strip(),
            "category": category.strip(),
            "artifact_type": artifact_type,
            "shell": shell,
        }
    )


def respond_to_message(task_id: str, content: str) -> dict | None:
    """Draft a Task response and artifact with the shared code-assist service."""
    return task_service.respond_to_message(task_id, content.strip())


def get_artifact(task_id: str, artifact_id: str) -> dict | None:
    return task_store.get_artifact_version(task_id, artifact_id)


def save_artifact(task_id: str, content: str) -> dict | None:
    return task_service.save_artifact_version(task_id, content)


def delete_artifact(task_id: str, artifact_id: str) -> bool:
    return task_store.delete_artifact_version(task_id, artifact_id)
