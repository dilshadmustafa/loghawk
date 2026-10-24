"""CLI entry point for executing a built-in collector in venv_tools."""

from __future__ import annotations

import json
import sys

from loghawk.collectors.builtins import run_builtin, validate_arguments


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python -m loghawk.collectors.runner <collector-name>", file=sys.stderr)
        return 2
    name = sys.argv[1]
    try:
        arguments = json.load(sys.stdin)
        if not isinstance(arguments, dict):
            raise ValueError("Collector arguments must be a JSON object.")
        result = run_builtin(name, validate_arguments(name, arguments))
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
