"""Run a configured LogHawk Pipeline and email newly written incidents as CSV."""

from __future__ import annotations

from email.message import EmailMessage
import json
import os
import smtplib
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pyarrow.parquet as parquet
import s3fs

import loghawk.config as config


PIPELINE_NAME = os.getenv("LH_PIPELINE_NAME", "pce3").strip() or "pce3"
PIPELINE_BATCH_S3_URL = os.getenv("LH_PIPELINE_BATCH_S3_URL", "").strip()
if not PIPELINE_BATCH_S3_URL:
    raise RuntimeError("LH_PIPELINE_BATCH_S3_URL must be set by run_pipeline_email.sh")
INCIDENTS_URL = (
    f"{PIPELINE_BATCH_S3_URL.rstrip('/')}/incidents/correlated_incidents.parquet"
)
RECIPIENT = "devops-team@yourcompany.com"
POLL_INTERVAL_SECONDS = 30
WAIT_TIMEOUT_SECONDS = 24 * 60 * 60


def _api_request(
    method: str,
    path: str,
    *,
    token: str | None = None,
    payload: dict | None = None,
) -> dict | list | None:
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(
        f"{config.LH_API_SERVER_URL}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=30) as response:
            body = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LogHawk API returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach the LogHawk API: {exc.reason}") from exc

    if not body:
        return None
    return json.loads(body.decode("utf-8"))


def _login() -> str:
    userid = os.getenv("LH_PIPELINE_USERID", "").strip()
    password = os.getenv("LH_PIPELINE_PASSWORD", "")
    if not userid or not password:
        raise RuntimeError("LH_PIPELINE_USERID and LH_PIPELINE_PASSWORD are required")
    result = _api_request(
        "POST",
        "/api/v1/auth/login",
        payload={"userid": userid, "password": password, "client": "webui"},
    )
    if not isinstance(result, dict) or not result.get("access_token"):
        raise RuntimeError("The LogHawk API did not return an access token")
    return result["access_token"]


def _s3_filesystem() -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(
        key=config.LH_S3_ACCESS_KEY_ID,
        secret=config.LH_S3_SECRET_ACCESS_KEY,
        client_kwargs={
            "endpoint_url": config.LH_S3_ENDPOINT,
            "region_name": config.LH_S3_REGION,
        },
    )


def _object_key(s3_url: str) -> str:
    if not s3_url.startswith("s3://"):
        raise ValueError(f"Expected an s3:// URL, got {s3_url!r}")
    return s3_url[len("s3://"):]


def _object_signature(fs: s3fs.S3FileSystem, key: str) -> tuple | None:
    try:
        info = fs.info(key)
    except FileNotFoundError:
        return None
    modified = info.get("LastModified", info.get("last_modified", info.get("mtime")))
    return (info.get("ETag", info.get("etag")), info.get("size"), str(modified))


def _csv_attachment(fs: s3fs.S3FileSystem, key: str) -> tuple[bytes, int]:
    with fs.open(key, "rb") as source:
        table = parquet.read_table(source)
    csv_content = table.to_pandas().to_csv(index=False).encode("utf-8")
    return csv_content, table.num_rows


def _email_settings() -> dict:
    settings = {
        "host": os.getenv("LH_PIPELINE_EMAIL_SMTP_HOST", "").strip(),
        "sender": os.getenv("LH_PIPELINE_EMAIL_FROM", "").strip(),
        "port": int(os.getenv("LH_PIPELINE_EMAIL_SMTP_PORT", "587")),
        "username": os.getenv("LH_PIPELINE_EMAIL_SMTP_USERNAME", "").strip(),
        "password": os.getenv("LH_PIPELINE_EMAIL_SMTP_PASSWORD", ""),
        "starttls": os.getenv("LH_PIPELINE_EMAIL_STARTTLS", "true").strip().lower()
        in {"1", "true", "yes", "on"},
        "recipient": os.getenv("LH_PIPELINE_EMAIL_TO", RECIPIENT).strip() or RECIPIENT,
    }
    if not settings["host"] or not settings["sender"]:
        raise RuntimeError(
            "Set LH_PIPELINE_EMAIL_SMTP_HOST and LH_PIPELINE_EMAIL_FROM in .env"
        )
    if bool(settings["username"]) != bool(settings["password"]):
        raise RuntimeError(
            "Set both LH_PIPELINE_EMAIL_SMTP_USERNAME and "
            "LH_PIPELINE_EMAIL_SMTP_PASSWORD, or leave both blank"
        )
    return settings


def _send_email(
    *, workflow_id: str, output_updated: bool, csv_content: bytes | None, row_count: int
) -> None:
    settings = _email_settings()

    message = EmailMessage()
    message["Subject"] = f"LogHawk Pipeline {PIPELINE_NAME} completed"
    message["From"] = settings["sender"]
    message["To"] = settings["recipient"]
    message.set_content(
        f"Pipeline: {PIPELINE_NAME}\n"
        f"Workflow ID: {workflow_id}\n"
        f"Status: COMPLETED\n"
        f"Incident output: {INCIDENTS_URL}\n"
        + (
            f"Incident rows attached: {row_count}\n"
            if output_updated
            else "No new incident output was written by this run; no attachment is included.\n"
        )
    )
    if csv_content is not None:
        message.add_attachment(
            csv_content,
            maintype="text",
            subtype="csv",
            filename="correlated_incidents.csv",
        )

    with smtplib.SMTP(settings["host"], settings["port"], timeout=30) as server:
        if settings["starttls"]:
            server.starttls()
        if settings["username"]:
            server.login(settings["username"], settings["password"])
        server.send_message(message)


def _wait_for_completion(token: str, workflow_id: str) -> dict:
    deadline = time.monotonic() + WAIT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        result = _api_request(
            "GET", f"/api/v1/pipelines/runs/{workflow_id}", token=token
        )
        if not isinstance(result, dict):
            raise RuntimeError("The LogHawk API returned an invalid run status")
        status = str(result.get("status", "")).upper()
        print(f"Pipeline {PIPELINE_NAME} ({workflow_id}): {status}", flush=True)
        if status == "COMPLETED":
            return result
        if status == "FAILED":
            raise RuntimeError(f"Pipeline {PIPELINE_NAME} failed: {result.get('error', '')}")
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(
        f"Pipeline {PIPELINE_NAME} did not finish within {WAIT_TIMEOUT_SECONDS} seconds"
    )


def run() -> None:
    _email_settings()
    token = _login()
    try:
        pipelines = _api_request("GET", "/api/v1/pipelines", token=token)
        if not isinstance(pipelines, list):
            raise RuntimeError("The LogHawk API returned an invalid Pipeline list")
        matches = [item for item in pipelines if item.get("name", "").casefold() == PIPELINE_NAME.casefold()]
        if not matches:
            raise RuntimeError(f"Pipeline {PIPELINE_NAME!r} was not found")
        if len(matches) > 1:
            raise RuntimeError(f"More than one Pipeline is named {PIPELINE_NAME!r}")

        pipeline = matches[0]
        latest = pipeline.get("latest_run") or {}
        if str(latest.get("status", "")).upper() == "RUNNING":
            print(f"Pipeline {PIPELINE_NAME} is already running; skipping this schedule.")
            return

        fs = _s3_filesystem()
        output_key = _object_key(INCIDENTS_URL)
        previous_signature = _object_signature(fs, output_key)
        started = _api_request(
            "POST",
            f"/api/v1/pipelines/{pipeline['id']}/runs",
            token=token,
        )
        if not isinstance(started, dict) or not started.get("workflow_id"):
            raise RuntimeError("The LogHawk API did not return a Workflow ID")
        workflow_id = started["workflow_id"]
        print(f"Started Pipeline {PIPELINE_NAME}: {workflow_id}", flush=True)
        _wait_for_completion(token, workflow_id)

        current_signature = _object_signature(fs, output_key)
        output_updated = current_signature is not None and current_signature != previous_signature
        csv_content = None
        row_count = 0
        if output_updated:
            csv_content, row_count = _csv_attachment(fs, output_key)
        _send_email(
            workflow_id=workflow_id,
            output_updated=output_updated,
            csv_content=csv_content,
            row_count=row_count,
        )
        print(f"Completion email sent to {_email_settings()['recipient']}")
    finally:
        try:
            _api_request("POST", "/api/v1/auth/logout", token=token)
        except Exception as exc:
            print(f"Warning: could not log out of the LogHawk API: {exc}", file=sys.stderr)


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"Pipeline cron job failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
