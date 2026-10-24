"""Controlled subprocess runner for API-approved LogHawk artifacts.

The approval gate lives in the API. This module only starts the selected artifact
and captures bounded output; it never interprets or rewrites the program.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from dotenv import dotenv_values

from loghawk.config import PROJECT_ROOT


MAX_OUTPUT_CHARS = 100_000
DEFAULT_TIMEOUT_SECONDS = 300


def tools_python_command() -> list[str]:
    """Resolve the tools Python, allowing an explicit .env_tools override."""
    values = dotenv_values(PROJECT_ROOT / ".env_tools")
    configured = os.environ.get("LH_TOOLS_PYTHON") or values.get("LH_TOOLS_PYTHON")
    if configured:
        executable = Path(str(configured).strip()).expanduser()
        if not executable.is_absolute():
            executable = PROJECT_ROOT / executable
        if not executable.is_file():
            raise FileNotFoundError(f"LH_TOOLS_PYTHON does not exist: {executable}")
        return [str(executable)]

    if os.name == "nt":
        candidate = PROJECT_ROOT / "venv_tools" / "Scripts" / "python.exe"
        if candidate.is_file():
            return [str(candidate)]
    else:
        candidates = [
            Path.home() / "miniforge3" / "envs" / "venv_tools" / "bin" / "python",
            Path.home() / "miniconda3" / "envs" / "venv_tools" / "bin" / "python",
            PROJECT_ROOT / "venv_tools" / "bin" / "python",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return [str(candidate)]
    conda = shutil.which("conda")
    if conda:
        return [conda, "run", "--no-capture-output", "-n", "venv_tools", "python"]
    raise FileNotFoundError(
        "Could not find the venv_tools Python. Run setup_tools.bat or setup_tools.sh, "
        "or set LH_TOOLS_PYTHON in .env_tools."
    )


def _bounded(value: str | None) -> str:
    value = value or ""
    if len(value) <= MAX_OUTPUT_CHARS:
        return value
    return value[:MAX_OUTPUT_CHARS] + "\n[output truncated]"


def run_artifact(
    artifact_type: str,
    content: str,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one Python/Bash/PowerShell/CMD artifact with captured output."""
    suffixes = {"python": ".py", "bash": ".sh", "powershell": ".ps1", "windows-cmd": ".cmd"}
    if artifact_type not in suffixes:
        raise ValueError(f"Unsupported executable artifact type: {artifact_type}")
    env = os.environ.copy()
    for key, value in dotenv_values(PROJECT_ROOT / ".env_tools").items():
        if value is not None:
            env.setdefault(key, value)
    env["LH_RUNNER_ARGUMENTS_JSON"] = json.dumps(arguments or {})
    with tempfile.TemporaryDirectory(prefix="loghawk-runner-") as temp_dir:
        script = Path(temp_dir) / f"artifact{suffixes[artifact_type]}"
        script.write_text(content, encoding="utf-8")
        if artifact_type == "python":
            command = [*tools_python_command(), str(script)]
        elif artifact_type == "bash":
            executable = shutil.which("bash")
            if not executable:
                raise RuntimeError("Bash is not installed or available on the API host.")
            command = [executable, str(script)]
        elif artifact_type == "powershell":
            executable = shutil.which("pwsh") or shutil.which("powershell")
            if not executable:
                raise RuntimeError("PowerShell is not installed or available on the API host.")
            command = [executable, "-NoProfile", "-File", str(script)]
        else:
            if os.name != "nt":
                raise RuntimeError("Windows CMD artifacts can only run on Windows API hosts.")
            command = [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", str(script)]
        try:
            result = subprocess.run(
                command, cwd=PROJECT_ROOT, env=env, capture_output=True,
                text=True, timeout=timeout, check=False,
            )
            return {
                "status": "COMPLETED" if result.returncode == 0 else "FAILED",
                "exit_code": result.returncode,
                "stdout": _bounded(result.stdout),
                "stderr": _bounded(result.stderr),
                "error_type": None,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "status": "FAILED", "exit_code": None,
                "stdout": _bounded(exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout),
                "stderr": _bounded(exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr),
                "error_type": "TimeoutExpired",
            }


def run_builtin_collector(name: str, arguments: dict[str, Any], *, timeout: int = 60) -> dict:
    """Run a built-in collector in venv_tools and return its JSON result."""
    command = [*tools_python_command(), "-m", "loghawk.collectors.runner", name]
    result = subprocess.run(
        command, cwd=PROJECT_ROOT, input=json.dumps(arguments), capture_output=True,
        text=True, timeout=timeout, check=False,
    )
    if result.returncode:
        message = _bounded(result.stderr or result.stdout)
        raise RuntimeError(message.strip() or f"Collector exited with code {result.returncode}.")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Collector runner returned invalid JSON.") from exc
