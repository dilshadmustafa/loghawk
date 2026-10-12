"""Cross-platform lookup for the current operating-system login name."""

from __future__ import annotations

import getpass


def get_userid() -> str:
    """Return the current Windows, Linux, or macOS login name."""
    userid = getpass.getuser().strip()
    if not userid:
        raise RuntimeError("Could not determine the current operating-system login name")
    return userid
