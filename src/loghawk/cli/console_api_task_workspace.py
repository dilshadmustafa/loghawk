"""Task workspace operations backed by authenticated LogHawk API calls."""

from __future__ import annotations

from loghawk.cli.console_api_client import ConsoleApiClient


class ConsoleApiTaskWorkspace:
    def __init__(self, client: ConsoleApiClient):
        self.client = client

    def list_tasks(self) -> list[dict]:
        return self.client.get_json("/api/v1/tasks")

    def find_task_by_name(self, name: str) -> dict | None:
        normalized = name.strip().casefold()
        for item in self.list_tasks():
            if item["name"].casefold() == normalized:
                return self.get_task(item["id"])
        return None

    def get_task(self, task_id: str) -> dict | None:
        try:
            return self.client.get_json(f"/api/v1/tasks/{task_id}")
        except RuntimeError as exc:
            if "returned 404" in str(exc):
                return None
            raise

    def create_task(self, name: str, category: str, artifact_type: str) -> dict:
        return self.client.post_json(
            "/api/v1/tasks",
            {"name": name, "category": category, "artifact_type": artifact_type},
        )

    def respond_to_message(self, task_id: str, content: str) -> dict | None:
        return self.client.post_json(
            f"/api/v1/tasks/{task_id}/messages", {"content": content}, timeout=300
        )

    def get_artifact(self, task_id: str, artifact_id: str) -> dict | None:
        return self.client.get_json(f"/api/v1/tasks/{task_id}/artifacts/{artifact_id}")

    def save_artifact(self, task_id: str, content: str) -> dict | None:
        return self.client.post_json(
            f"/api/v1/tasks/{task_id}/artifacts", {"content": content}
        )

    def delete_artifact(self, task_id: str, artifact_id: str) -> bool:
        self.client.delete(f"/api/v1/tasks/{task_id}/artifacts/{artifact_id}")
        return True

    def delete_task(self, task_id: str) -> bool:
        return self.client.delete(f"/api/v1/tasks/{task_id}")
