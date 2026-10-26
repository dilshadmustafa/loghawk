"""Bounded JSON Schema subset for versioned collector input contracts.

Objects, arrays, scalar types, enums, defaults and simple size/range constraints
are supported. References, regexes and executable/custom schema extensions are
deliberately rejected. Validation never fetches a URL or imports collector code.
"""

from __future__ import annotations

import copy
import json
import math


def _json_size(value, limit: int) -> None:
    try:
        raw = json.dumps(value, allow_nan=False)
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValueError("Expected finite JSON values") from exc
    if len(raw) > limit:
        raise ValueError(f"JSON exceeds the {limit}-character limit")


def normalize_task_schema(schema: dict | None) -> dict:
    if schema is None or schema == {}:
        schema = {
            "type": "object", "properties": {}, "required": [],
            "additionalProperties": False,
        }
    return validate_schema(schema)


def validate_schema(schema: dict) -> dict:
    _json_size(schema, 16000)
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("Arguments schema must describe a JSON object")

    def visit(node, depth=0):
        if depth > 6 or not isinstance(node, dict):
            raise ValueError("Schema is invalid or nested too deeply")
        kind = node.get("type")
        fields = {
            "object": {"properties", "required", "additionalProperties"},
            "array": {"items", "minItems", "maxItems"},
            "string": {"minLength", "maxLength"},
            "integer": {"minimum", "maximum"},
            "number": {"minimum", "maximum"},
            "boolean": set(), "null": set(),
        }
        if not isinstance(kind, str) or kind not in fields:
            raise ValueError("Schema type must be object, array, string, integer, number, boolean or null")
        unknown = set(node) - fields[kind] - {"type", "description", "enum", "default"}
        if unknown:
            raise ValueError(f"Unsupported schema keyword(s): {', '.join(sorted(unknown))}")
        if "description" in node and not isinstance(node["description"], str):
            raise ValueError("Schema description must be text")
        for low, high in (("minItems", "maxItems"), ("minLength", "maxLength"), ("minimum", "maximum")):
            for key in (low, high):
                if key in node:
                    value = node[key]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not (isinstance(value, int) or math.isfinite(value)):
                        raise ValueError(f"{key} must be a finite number")
                    if key not in {"minimum", "maximum"} and (not isinstance(value, int) or value < 0):
                        raise ValueError(f"{key} must be a non-negative integer")
            if low in node and high in node and node[low] > node[high]:
                raise ValueError(f"{low} cannot exceed {high}")
        if kind == "object":
            props = node.get("properties", {})
            required = node.get("required", [])
            if not isinstance(props, dict) or len(props) > 50:
                raise ValueError("Schema properties must be an object with at most 50 fields")
            if node.get("additionalProperties") is not False:
                raise ValueError("Object schemas must set additionalProperties to false")
            if not isinstance(required, list) or any(not isinstance(x, str) or x not in props for x in required):
                raise ValueError("required must list declared property names")
            for child in props.values():
                visit(child, depth + 1)
        elif kind == "array":
            visit(node.get("items"), depth + 1)
        if "enum" in node:
            if not isinstance(node["enum"], list) or not node["enum"]:
                raise ValueError("enum must be a non-empty list")
            for value in node["enum"]:
                _validate_value(value, {k: v for k, v in node.items() if k != "enum"}, "enum")
        if "default" in node:
            _validate_value(node["default"], node, "default")

    visit(schema)
    return copy.deepcopy(schema)


def _validate_value(value, schema, path):
    kind = schema["type"]
    valid = {
        "object": isinstance(value, dict), "array": isinstance(value, list),
        "string": isinstance(value, str), "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "null": value is None,
    }[kind]
    if not valid:
        raise ValueError(f"{path} must have type {kind}")
    if "enum" in schema and not any(type(value) is type(item) and value == item for item in schema["enum"]):
        raise ValueError(f"{path} is not an allowed enum value")
    if kind == "object":
        props = schema.get("properties", {})
        if set(value) - set(props):
            raise ValueError(f"{path} contains undeclared arguments")
        result = dict(value)
        for key, spec in props.items():
            if key not in result and "default" in spec:
                result[key] = copy.deepcopy(spec["default"])
        for key in schema.get("required", []):
            if key not in result:
                raise ValueError(f"{path}.{key} is required")
        return {key: _validate_value(item, props[key], f"{path}.{key}") for key, item in result.items()}
    if kind in {"array", "string"}:
        low, high = ("minItems", "maxItems") if kind == "array" else ("minLength", "maxLength")
        if len(value) < schema.get(low, 0) or len(value) > schema.get(high, 32000):
            raise ValueError(f"{path} exceeds its size constraints")
        if kind == "array":
            return [_validate_value(item, schema["items"], f"{path}[]") for item in value]
    if kind in {"integer", "number"}:
        if not (isinstance(value, int) or math.isfinite(value)) or value < schema.get("minimum", -math.inf) or value > schema.get("maximum", math.inf):
            raise ValueError(f"{path} exceeds its numeric constraints")
    return value


def validate_arguments(schema: dict, arguments: dict) -> dict:
    validate_schema(schema)
    _json_size(arguments, 32000)
    return _validate_value(arguments, schema, "arguments")


def builtin_schema(fields: dict) -> dict:
    properties, required = {}, []
    for name, spec in fields.items():
        item = {"type": spec["type"]}
        if "description" in spec:
            item["description"] = spec["description"]
        for old, new in (("max_length", "maxLength"), ("min", "minimum"), ("max", "maximum"), ("default", "default")):
            if old in spec:
                item[new] = spec[old]
        if spec["type"] == "string":
            item["minLength"] = 1
        properties[name] = item
        if spec.get("required"):
            required.append(name)
    return validate_schema({"type": "object", "properties": properties, "required": required, "additionalProperties": False})
