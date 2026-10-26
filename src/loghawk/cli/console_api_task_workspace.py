"""Task workspace operations backed by authenticated LogHawk API calls."""

from __future__ import annotations

from urllib.parse import urlencode

from loghawk.cli.console_api_client import ConsoleApiClient


class ConsoleApiTaskWorkspace:
    def __init__(self, client: ConsoleApiClient):
        self.client = client

    def list_tasks(self, query: str = "") -> list[dict]:
        suffix = f"?{urlencode({'q': query})}" if query else ""
        return self.client.get_json(f"/api/v1/tasks{suffix}")

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

    def create_task(
        self, name: str, description: str, category: str, artifact_type: str,
        task_kind: str = "general", execution_policy: str = "manual",
    ) -> dict:
        return self.client.post_json(
            "/api/v1/tasks",
            {"name": name, "description": description, "category": category, "task_kind": task_kind,
             "execution_policy": execution_policy, "artifact_type": artifact_type},
        )


    def set_description(self, task_id: str, description: str) -> dict:
        return self.client.put_json(
            f"/api/v1/tasks/{task_id}/description", {"description": description}
        )

    def set_execution_policy(self, task_id: str, execution_policy: str) -> dict:
        return self.client.put_json(
            f"/api/v1/tasks/{task_id}/execution-policy",
            {"execution_policy": execution_policy},
        )

    def respond_to_message(self, task_id: str, content: str) -> dict | None:
        return self.client.post_json(
            f"/api/v1/tasks/{task_id}/messages", {"content": content}, timeout=300
        )

    def get_artifact(self, task_id: str, artifact_id: str) -> dict | None:
        return self.client.get_json(f"/api/v1/tasks/{task_id}/artifacts/{artifact_id}")

    def save_artifact(self, task_id: str, content: str, *, arguments_schema: dict | None = None) -> dict | None:
        return self.client.post_json(
            f"/api/v1/tasks/{task_id}/artifacts", {"content": content, "arguments_schema": arguments_schema}
        )

    def approve_artifact(self, task_id: str, artifact_id: str) -> dict:
        return self.client.post_json(
            f"/api/v1/tasks/{task_id}/artifacts/{artifact_id}/approve", {}
        )

    def set_artifact_enabled(self, task_id: str, artifact_id: str, enabled: bool) -> dict:
        return self.client.put_json(
            f"/api/v1/tasks/{task_id}/artifacts/{artifact_id}/enabled", {"enabled": enabled}
        )

    def delete_artifact(self, task_id: str, artifact_id: str) -> bool:
        self.client.delete(f"/api/v1/tasks/{task_id}/artifacts/{artifact_id}")
        return True

    def delete_task(self, task_id: str) -> bool:
        return self.client.delete(f"/api/v1/tasks/{task_id}")
