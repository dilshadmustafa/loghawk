"""Authenticated HTTP client used by signed-in Hawk Console sessions."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import requests

import loghawk.config as config


class ConsoleApiClient:
    def __init__(self, token: str):
        self.base_url = config.LH_API_SERVER_URL.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}"})

    def _request(self, method: str, path: str, **kwargs):
        response = self.session.request(
            method, f"{self.base_url}{path}", timeout=kwargs.pop("timeout", 60), **kwargs
        )
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise RuntimeError(f"LogHawk API returned {response.status_code}: {detail}")
        return response

    def get_json(self, path: str) -> Any:
        return self._request("GET", path).json()

    def post_json(self, path: str, body: dict, *, timeout: int = 60) -> Any:
        return self._request("POST", path, json=body, timeout=timeout).json()

    def put_json(self, path: str, body: dict) -> Any:
        return self._request("PUT", path, json=body).json()

    def delete(self, path: str) -> bool:
        response = self._request("DELETE", path)
        return response.status_code in {200, 202, 204}

    def upload_rag_files(self, files: list[tuple[str, str]]) -> dict:
        opened = []
        try:
            parts = []
            for file_path, source_name in files:
                stream = Path(file_path).open("rb")
                opened.append(stream)
                parts.append(("files", (source_name.replace("\\", "/"), stream)))
            return self._request(
                "POST", "/api/v1/rag/ingestions/upload", files=parts, timeout=300
            ).json()
        finally:
            for stream in opened:
                stream.close()

    def wait_for_ingestion(self, job_id: str) -> dict:
        while True:
            job = self.get_json(f"/api/v1/rag/ingestions/{job_id}")
            if job.get("status") in {"COMPLETED", "PARTIAL", "FAILED"}:
                return job
            time.sleep(2)

