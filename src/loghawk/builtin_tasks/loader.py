"""Load packaged Task definitions without importing or executing their scripts."""
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
import re

from loghawk.builtin_tasks.contracts import validate_schema, validate_arguments as validate_schema_arguments

ROOT = Path(__file__).resolve().parent
KINDS = {"general", "collector", "remediation", "rollback"}
ARTIFACT_SUFFIXES = {"python": ".py", "bash": ".sh", "powershell": ".ps1", "windows-cmd": ".cmd"}
POLICIES = {"manual", "ai-assisted", "auto-approved", "disabled"}


def _source_path(definition):
    relative = PurePosixPath(definition["source_file"])
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative) or ":" in str(relative):
        raise ValueError("Built-in source must be a relative path inside builtin_tasks")
    if not relative.parts or relative.parts[0] != definition["task_kind"] + "_tasks":
        raise ValueError("Built-in source must be in its Task kind folder")
    path = (ROOT / str(relative)).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f"Missing or invalid built-in source: {relative}")
    if path.suffix != ARTIFACT_SUFFIXES[definition["artifact_type"]]:
        raise ValueError("Built-in source extension does not match artifact_type")
    return path


def load_definitions():
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("tasks"), list):
        raise ValueError("Built-in manifest must contain a tasks array")
    definitions = {}
    for item in manifest["tasks"]:
        if not isinstance(item, dict):
            raise ValueError("Each built-in Task must be an object")
        item = deepcopy(item)
        name = item.get("name", "")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name) or name in definitions:
            raise ValueError(f"Invalid or duplicate built-in Task name: {name}")
        if item.get("task_kind") not in KINDS or item.get("artifact_type") not in ARTIFACT_SUFFIXES:
            raise ValueError(f"Invalid Task kind or artifact type: {name}")
        for field in ("version", "display_name", "description", "source_file"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"Missing {field} for built-in Task {name}")
        item.setdefault("execution_policy", "manual")
        if item["execution_policy"] not in POLICIES:
            raise ValueError(f"Invalid execution policy: {name}")
        for field in ("approved", "enabled"):
            item.setdefault(field, False)
            if not isinstance(item[field], bool):
                raise ValueError(f"{field} must be boolean: {name}")
        item["arguments_schema"] = validate_schema(item.get("arguments_schema"))
        _source_path(item)
        definitions[name] = item
    return definitions


def get_definition(name):
    return load_definitions()[name]


def source_code(name):
    return _source_path(get_definition(name)).read_text(encoding="utf-8")


def validate_arguments(name, arguments):
    return validate_schema_arguments(get_definition(name)["arguments_schema"], arguments)
