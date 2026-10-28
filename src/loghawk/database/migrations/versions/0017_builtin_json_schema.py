"""Standardize stored built-in argument contracts without altering permissions."""

import json

from alembic import context, op
import sqlalchemy as sa

revision = "0017_builtin_json_schema"
down_revision = "0016_tool_origin_software"
branch_labels = None
depends_on = None


def _to_json_schema(fields):
    # Keep conversion frozen here so future application changes cannot alter it.
    if not isinstance(fields, dict):
        raise ValueError("Built-in argument definition must be an object.")
    if fields.get("type") == "object" and isinstance(fields.get("properties"), dict):
        return fields
    properties, required = {}, []
    for name, spec in fields.items():
        if not isinstance(spec, dict) or spec.get("type") not in {"string", "integer"}:
            raise ValueError("Unsupported legacy built-in argument definition.")
        if set(spec) - {"type", "required", "description", "max_length", "min", "max", "default"}:
            raise ValueError("Unknown legacy built-in schema fields; refusing to discard them.")
        item = {"type": spec["type"]}
        for old, new in (("description", "description"), ("max_length", "maxLength"),
                         ("min", "minimum"), ("max", "maximum"), ("default", "default")):
            if old in spec:
                item[new] = spec[old]
        if spec["type"] == "string":
            item["minLength"] = 1
        if spec.get("required"):
            required.append(name)
        properties[name] = item
    return {"type": "object", "properties": properties, "required": required,
            "additionalProperties": False}


def _to_legacy(schema):
    if schema.get("type") != "object":
        return schema
    if (set(schema) - {"type", "properties", "required", "additionalProperties"}
            or schema.get("additionalProperties") is not False):
        raise ValueError("Cannot downgrade an extended built-in schema without losing constraints.")
    fields = {}
    for name, spec in schema["properties"].items():
        if (spec.get("type") not in {"string", "integer"}
                or set(spec) - {"type", "description", "maxLength", "minLength", "minimum", "maximum", "default"}
                or (spec["type"] == "string" and spec.get("minLength") != 1)):
            raise ValueError("Cannot downgrade an extended built-in schema without losing constraints.")
        item = {"type": spec["type"], "required": name in schema.get("required", [])}
        for old, new in (("description", "description"), ("maxLength", "max_length"),
                         ("minimum", "min"), ("maximum", "max"), ("default", "default")):
            if old in spec:
                item[new] = spec[old]
        fields[name] = item
    return fields


def _convert(converter):
    if context.config.attributes.get("database_role", "server") != "server":
        return
    bind = op.get_bind()
    if "collector_registry" not in sa.inspect(bind).get_table_names():
        return
    table = sa.table("collector_registry", sa.column("name", sa.String()),
                     sa.column("arguments_schema_json", sa.Text()))
    rows = bind.execute(sa.select(table.c.name, table.c.arguments_schema_json)).mappings().all()
    updates = []
    for row in rows:
        original = json.loads(row["arguments_schema_json"])
        converted = converter(original)
        if original != converted:
            updates.append((row["name"], json.dumps(converted, sort_keys=True)))
    # Validate every conversion before issuing writes. Change only the schema column.
    for name, schema_json in updates:
        bind.execute(table.update().where(table.c.name == name).values(arguments_schema_json=schema_json))


def upgrade():
    _convert(_to_json_schema)


def downgrade():
    _convert(_to_legacy)
