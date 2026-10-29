"""UTC creation timestamps shared by database writers."""

from datetime import UTC, datetime


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")
