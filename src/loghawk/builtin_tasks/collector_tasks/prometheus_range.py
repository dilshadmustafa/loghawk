"""Bounded, read-only Prometheus range queries for historical evidence.

LogHawk supplies .env_tools values and LH_RUNNER_ARGUMENTS_JSON to this
standalone program. No LogHawk imports are required.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
import math
import os
import re
import sys
from urllib.parse import urlparse

import requests


MAX_SERIES = 10
MAX_EVALUATION_POINTS = 300
MAX_WINDOW_SECONDS = 24 * 60 * 60
# Keep complete JSON below lhrunner.MAX_OUTPUT_CHARS, including print's newline.
MAX_OUTPUT_CHARS = 99_999
TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})"
)


def _timestamp(value: object, name: str) -> datetime:
    if not isinstance(value, str) or not TIMESTAMP.fullmatch(value.strip()):
        raise ValueError(
            f"{name} must be an ISO timestamp with an explicit timezone, "
            "for example 2026-11-03T09:07:00Z or 2026-11-03T14:37:00+05:30. "
            "Do not infer a timezone from an incident timestamp without an offset."
        )
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        return parsed.astimezone(UTC)
    except (ValueError, OverflowError):
        raise ValueError(f"{name} is not a valid timestamp.") from None


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _encoded(result: dict) -> str:
    text = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(text) >= MAX_OUTPUT_CHARS:
        raise ValueError(
            "Prometheus range output exceeds the runner's safe JSON size. "
            "Use a narrower query, shorter time range, larger step, or PromQL aggregation. "
            "No partial JSON result was emitted."
        )
    return text


def _series_samples(series: dict) -> list[float]:
    if not isinstance(series, dict) or not isinstance(series.get("metric"), dict):
        raise RuntimeError("Prometheus returned an invalid range series.")
    timestamps = []
    for field in ("values", "histograms"):
        samples = series.get(field, [])
        if not isinstance(samples, list):
            raise RuntimeError("Prometheus returned an invalid range sample list.")
        for sample in samples:
            if (not isinstance(sample, list) or len(sample) != 2
                    or isinstance(sample[0], bool)
                    or not isinstance(sample[0], (int, float))
                    or not math.isfinite(sample[0])):
                raise RuntimeError("Prometheus returned an invalid sample timestamp.")
            timestamps.append(float(sample[0]))
    if len(timestamps) > MAX_EVALUATION_POINTS:
        raise RuntimeError("Prometheus returned more than 300 samples in one series; narrow the query.")
    return timestamps


def collect(arguments: dict) -> dict:
    if not isinstance(arguments, dict):
        raise ValueError("Collector arguments must be a JSON object.")
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip() or len(query) > 2000:
        raise ValueError("query must be a nonempty PromQL expression of at most 2000 characters.")
    start = _timestamp(arguments.get("start_time"), "start_time")
    end = _timestamp(arguments.get("end_time"), "end_time")
    duration = end - start
    if not timedelta(0) < duration <= timedelta(seconds=MAX_WINDOW_SECONDS):
        raise ValueError("end_time must be after start_time and the range must not exceed 24 hours.")
    step = arguments.get("step_seconds", 60)
    if isinstance(step, bool) or not isinstance(step, int) or not 15 <= step <= 3600:
        raise ValueError("step_seconds must be an integer between 15 and 3600.")
    evaluation_points = duration // timedelta(seconds=step) + 1
    if evaluation_points > MAX_EVALUATION_POINTS:
        suggested_step = max(15, math.ceil(duration.total_seconds() / (MAX_EVALUATION_POINTS - 1)))
        raise ValueError(
            f"This range requests {evaluation_points} evaluation points; the maximum is "
            f"{MAX_EVALUATION_POINTS}, including the start point. Set step_seconds to at "
            f"least {suggested_step} or shorten the range; the default step is 60 seconds."
        )

    base_url = os.environ.get("LH_TOOLS_PROMETHEUS_URL", "").strip().rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("Set LH_TOOLS_PROMETHEUS_URL to a reachable HTTP(S) Prometheus URL in .env_tools.")
    headers = {}
    token = os.environ.get("LH_TOOLS_PROMETHEUS_BEARER_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = requests.post(
            f"{base_url}/api/v1/query_range",
            data={"query": query, "start": _iso(start), "end": _iso(end),
                  "step": step, "limit": MAX_SERIES, "timeout": "15s"},
            headers=headers,
            timeout=20,
        )
    except requests.RequestException:
        # Avoid leaking endpoint credentials through a requests exception URL.
        raise RuntimeError("Prometheus range request failed or timed out; check connectivity and server logs.") from None
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    if not response.ok or payload.get("status") != "success":
        error_type = str(payload.get("errorType", "unknown"))[:100]
        detail = str(payload.get("error", "No structured Prometheus error details were returned."))[:1000]
        raise RuntimeError(f"Prometheus HTTP {response.status_code} — {error_type}: {detail}")
    data = payload.get("data")
    if (not isinstance(data, dict) or data.get("resultType") != "matrix"
            or not isinstance(data.get("result"), list)):
        raise RuntimeError("Prometheus did not return a range-query matrix.")
    returned = data["result"]
    included = returned[:MAX_SERIES]
    timestamps = []
    for series in included:
        timestamps.extend(_series_samples(series))
    try:
        actual_start = _iso(datetime.fromtimestamp(min(timestamps), UTC)) if timestamps else None
        actual_end = _iso(datetime.fromtimestamp(max(timestamps), UTC)) if timestamps else None
    except (ValueError, OverflowError, OSError):
        raise RuntimeError("Prometheus returned a timestamp outside the supported date range.") from None
    result = {
        "summary": (
            f"Prometheus returned {len(returned)} range result series; "
            f"{len(included)} series and {len(timestamps)} samples are included. "
            "This is the returned dataset, not a count of all matching series."
        ),
        "data": {
            "resultType": "matrix", "result": included,
            "coverage": {
                "requested_start_time": _iso(start), "requested_end_time": _iso(end),
                "step_seconds": step, "requested_evaluation_points": evaluation_points,
                "actual_sample_start_time": actual_start, "actual_sample_end_time": actual_end,
                "included_samples": len(timestamps),
                "returned_series": len(returned), "included_series": len(included),
                "omitted_series_from_response": max(0, len(returned) - len(included)),
                "server_series_limit": MAX_SERIES, "total_matching_series": None,
                "series_limit_reached": len(returned) >= MAX_SERIES,
                "notice": (
                    "Coverage is the union of included sample timestamps, not proof of continuous "
                    "coverage for every series. The server series limit may omit matching series. "
                    "An empty result does not establish zero errors or identify why samples are absent."
                ),
            },
            "warnings": payload.get("warnings", []),
            "infos": payload.get("infos", []),
        },
    }
    _encoded(result)  # Validate before returning, including when collect() is imported.
    return result


def main() -> int:
    try:
        arguments = json.loads(os.environ.get("LH_RUNNER_ARGUMENTS_JSON", "{}"))
        print(_encoded(collect(arguments)))
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
