"""Command-line helper for local debugging of built-in Tasks.

Engineers can run this module directly to debug built-in General,
Collector, Remediation, and Rollback Tasks. For example, from Bash:

    echo '{"query":"up"}' | python -m loghawk.builtin_tasks.runner prometheus-query

Reads JSON arguments from standard input and delegates execution to
loghawk.lhrunner. Python scripts execute using the configured tools
environment.

This local debugging entry point does not enforce server-side approval
or enablement and does not create server execution audit records.
"""

from __future__ import annotations

import json
import sys

from loghawk.lhrunner import run_builtin_task


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python -m loghawk.builtin_tasks.runner <task-name>", file=sys.stderr)
        return 2
    name = sys.argv[1]
    try:
        arguments = json.load(sys.stdin)
        if not isinstance(arguments, dict):
            raise ValueError("Task arguments must be a JSON object.")
        result = run_builtin_task(name, arguments)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["status"] == "COMPLETED" else 1
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
