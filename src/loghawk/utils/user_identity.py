"""Cross-platform lookup for the current operating-system login name."""

from __future__ import annotations

import getpass
from contextvars import ContextVar


_userid_override: ContextVar[str | None] = ContextVar("loghawk_userid_override", default=None)


def set_userid_override(userid: str | None):
    """Set the current process-context identity for authenticated Console sessions."""
    return _userid_override.set(userid.strip() if userid and userid.strip() else None)


def reset_userid_override(token) -> None:
    _userid_override.reset(token)


def get_userid() -> str:
    """Return the current Windows, Linux, or macOS login name."""
    override = _userid_override.get()
    if override:
        return override
    userid = getpass.getuser().strip()
    if not userid:
        raise RuntimeError("Could not determine the current operating-system login name")
    return userid
