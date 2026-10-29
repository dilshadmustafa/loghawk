"""Preview or explicitly delete old operational history, never business definitions.

Examples (preview is the default)::

    python -m loghawk.admin.database_retention --role server --older-than-years 5
    python -m loghawk.admin.database_retention --role console --before 2021-10-29T00:00:00Z

Add --execute to delete eligible records. Each batch commits separately; if a
later batch fails, earlier committed batches remain deleted. Rerunning is safe.
No startup hook, migration, S3 deletion, or LanceDB deletion is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
import sys

import sqlalchemy as sa

from loghawk.database.database_layer import maintenance_connection


SERVER_TARGETS = (
    "llm_tool_requests", "pipeline_runs", "task_runs", "collector_runs",
    "rag_ingestion_jobs", "console_llm_history", "console_command_history", "auth_sessions",
)
CONSOLE_TARGETS = ("cli_command_history", "llm_conversation_history")
CHILDREN = ("rca_evidence_views", "llm_tool_calls", "rca_llm_invocations")
RUNS = {"pipeline_runs", "task_runs", "collector_runs"}
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "CANCELED", "TERMINATED", "TIMED_OUT"}
CALL_TERMINAL = TERMINAL | {"REJECTED", "CONFIRMED", "AWAITING_CONFIRMATION",
                            "REQUEST_SETTINGS", "EXPLANATION_COMPLETED", "EXPLANATION_FAILED"}


def parse_date(value):
    """Unknown, malformed, and timezone-less legacy dates are never guessed."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.astimezone(UTC) if parsed.tzinfo is not None else None
    except (ValueError, OverflowError):
        return None


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return number


def cutoff_for(args, now):
    if args.before:
        cutoff = parse_date(args.before)
        if cutoff is None:
            raise ValueError("--before requires an ISO timestamp with a timezone.")
    else:
        year = now.year - args.older_than_years
        if year < 1:
            raise ValueError("The requested year is outside the supported date range.")
        try:
            cutoff = now.replace(year=year)
        except ValueError:  # February 29 in a non-leap target year.
            cutoff = now.replace(year=year, day=28)
    if cutoff > now:
        raise ValueError("The retention cutoff cannot be in the future.")
    return cutoff


def targets(role):
    return SERVER_TARGETS if role == "server" else CONSOLE_TARGETS


def schema(connection, role):
    """Fail closed before deletion if required fields or relationships differ."""
    names = targets(role) + (CHILDREN if role == "server" else ())
    inspector = sa.inspect(connection)
    if not set(names).issubset(inspector.get_table_names()):
        raise ValueError("Required retention tables are missing. Apply application migrations first.")
    metadata = sa.MetaData()
    tables = {name: sa.Table(name, metadata, autoload_with=connection) for name in names}
    for name, table in tables.items():
        required = {"created_at", "token_hash" if name == "auth_sessions" else "id"}
        if name in RUNS or name in {"llm_tool_calls", "rca_llm_invocations"}:
            required |= {"started_at", "completed_at", "status"}
        if name in {"llm_tool_requests", "rag_ingestion_jobs"}:
            required |= {"status", "updated_at"}
        if name == "rag_ingestion_jobs":
            required.add("completed_at")
        if name in CHILDREN:
            required.add("request_id")
        if name == "llm_tool_calls":
            required.add("run_id")
        if name == "rca_evidence_views":
            required |= {"tool_call_id", "invocation_id"}
        if name == "auth_sessions":
            required |= {"expires_at", "revoked_at"}
        if not required.issubset(table.c.keys()):
            raise ValueError(f"Required retention columns are missing in {name}; apply migrations first.")
    # Unknown incoming relationships must not silently cascade away other data.
    for name in inspector.get_table_names():
        for fk in inspector.get_foreign_keys(name):
            referred = fk["referred_table"]
            if referred not in names:
                continue
            allowed = (name in CHILDREN and referred == "llm_tool_requests") or (
                name == "rca_evidence_views" and referred in {"llm_tool_calls", "rca_llm_invocations"})
            if not allowed:
                raise ValueError("An unsupported incoming database relationship prevents retention cleanup.")
    return tables


def rows(connection, table, batch_size, condition=None):
    """Keyset paging keeps scans bounded even when many records are protected."""
    key = table.c.token_hash if table.name == "auth_sessions" else table.c.id
    last = None
    while True:
        # Do not load saved evidence, prompts, script output, or credentials.
        columns = [column for column in table.c if column.name in {
            "id", "token_hash", "status", "created_at", "updated_at", "started_at",
            "completed_at", "expires_at", "revoked_at", "request_id", "run_id"}]
        query = sa.select(*columns).order_by(key).limit(batch_size)
        if last is not None:
            query = query.where(key > last)
        if condition is not None:
            query = query.where(condition)
        page = connection.execute(query).mappings().all()
        if not page:
            return
        yield page
        last = page[-1][key.name]


def date_reason(row, cutoff, required):
    for field in required:
        value = parse_date(row.get(field))
        if value is None:
            return "unknown_or_invalid_date"
        if value >= cutoff:
            return "recent_activity"
    return None


def request_reason(connection, tables, row, cutoff, batch_size):
    if row["status"] not in TERMINAL | {"EXPIRED"}:
        return "active_or_unknown_status"
    reason = date_reason(row, cutoff, ("created_at", "updated_at"))
    if reason:
        return reason
    for name in CHILDREN:
        child = tables[name]
        for page in rows(connection, child, batch_size, child.c.request_id == row["id"]):
            for item in page:
                if name != "rca_evidence_views":
                    allowed = CALL_TERMINAL if name == "llm_tool_calls" else TERMINAL
                    if item["status"] not in allowed:
                        return "active_or_unknown_child_status"
                    fields = ("created_at", "started_at", "completed_at")
                else:
                    fields = ("created_at",)
                reason = date_reason(item, cutoff, fields)
                if reason:
                    return reason
    # A malformed cross-request evidence relationship must not cascade into a
    # retained request when its referenced invocation or tool call is removed.
    evidence = tables["rca_evidence_views"]
    for name, field in (("llm_tool_calls", "tool_call_id"),
                        ("rca_llm_invocations", "invocation_id")):
        child = tables[name]
        query = sa.select(evidence.c.id).join(child, evidence.c[field] == child.c.id).where(
            child.c.request_id == row["id"], evidence.c.request_id != row["id"]).limit(1)
        if connection.execute(query).first():
            return "cross_request_reference"
    return None


def reason_for(connection, tables, name, row, cutoff, now, batch_size, preview):
    if name == "llm_tool_requests":
        return request_reason(connection, tables, row, cutoff, batch_size)
    fields = ["created_at"]
    if name in RUNS or name == "rag_ingestion_jobs":
        if row["status"] not in TERMINAL:
            return "active_or_unknown_status"
        fields.append("completed_at")
        fields.append("updated_at" if name == "rag_ingestion_jobs" else "started_at")
    reason = date_reason(row, cutoff, fields)
    if reason:
        return reason
    if name == "auth_sessions":
        expires = parse_date(row["expires_at"])
        revoked = parse_date(row["revoked_at"]) if row["revoked_at"] is not None else None
        if expires is None or (row["revoked_at"] is not None and revoked is None):
            return "unknown_or_invalid_date"
        if expires > now and (revoked is None or revoked > now):
            return "active_session"
    if name in RUNS:
        calls, requests = tables["llm_tool_calls"], tables["llm_tool_requests"]
        refs = sa.select(calls.c.request_id).where(calls.c.run_id == row["id"]).distinct()
        for (request_id,) in connection.execute(refs):
            if not preview:
                return "retained_rca_reference"
            request = connection.execute(sa.select(requests.c.id, requests.c.status,
                requests.c.created_at, requests.c.updated_at).where(requests.c.id == request_id)).mappings().first()
            if request is None or request_reason(connection, tables, request, cutoff, batch_size):
                return "retained_rca_reference"
    return None


def lock_tables(connection, tables):
    if connection.dialect.name == "postgresql":
        # Prevent concurrent retries and new run references while eligibility is
        # rechecked and a batch is deleted. SQLite already holds BEGIN IMMEDIATE.
        quoted = ", ".join(connection.dialect.identifier_preparer.quote(name) for name in sorted(tables))
        connection.exec_driver_sql(f"LOCK TABLE {quoted} IN SHARE ROW EXCLUSIVE MODE")


def dependent_counts(connection, tables, request_id):
    return {name: connection.scalar(sa.select(sa.func.count()).select_from(tables[name]).where(
        tables[name].c.request_id == request_id)) for name in CHILDREN}


def preview(role, cutoff, now, batch_size):
    with maintenance_connection(role) as connection:
        tables = schema(connection, role)
        report = {"mode": "preview", "database": connection.info["maintenance_location"],
                  "cutoff_utc": cutoff.isoformat(), "tables": {}, "eligible_dependents": dict.fromkeys(CHILDREN, 0) if role == "server" else {}}
        for name in targets(role):
            counts = Counter()
            for page in rows(connection, tables[name], batch_size):
                for row in page:
                    reason = reason_for(connection, tables, name, row, cutoff, now, batch_size, True)
                    counts[reason or "eligible"] += 1
                    if reason is None and name == "llm_tool_requests":
                        for child, count in dependent_counts(connection, tables, row["id"]).items():
                            report["eligible_dependents"][child] += count
            report["tables"][name] = {"eligible": counts.pop("eligible", 0), "protected_or_skipped": dict(counts)}
        return report


def execute(role, cutoff, now, batch_size):
    report = {"mode": "execute", "cutoff_utc": cutoff.isoformat(), "tables": {}, "deleted_dependents": {}}
    # Requests are processed first. Remaining RCA references protect run records.
    for name in targets(role):
        last = None
        counts = Counter()
        while True:
            with maintenance_connection(role, writable=True) as connection:
                tables = schema(connection, role)
                lock_tables(connection, tables)
                report["database"] = connection.info["maintenance_location"]
                table = tables[name]
                key = table.c.token_hash if name == "auth_sessions" else table.c.id
                page = next(rows(connection, table, batch_size, key > last if last is not None else None), [])
                if not page:
                    break
                batch_counts = Counter()
                children_deleted = Counter()
                for row in page:
                    reason = reason_for(connection, tables, name, row, cutoff, now, batch_size, False)
                    if reason:
                        batch_counts[reason] += 1
                        continue
                    if name == "llm_tool_requests":
                        # Explicit deletion reports child counts and preserves FK order.
                        for child_name in CHILDREN:
                            child = tables[child_name]
                            result = connection.execute(child.delete().where(child.c.request_id == row["id"]))
                            children_deleted[child_name] += result.rowcount
                    result = connection.execute(table.delete().where(key == row[key.name]))
                    batch_counts["deleted"] += result.rowcount
                new_last = page[-1][key.name]
            # The connection context committed successfully; only now report counts.
            counts.update(batch_counts)
            for child, count in children_deleted.items():
                report["deleted_dependents"][child] = report["deleted_dependents"].get(child, 0) + count
            last = new_last
            print(json.dumps({"committed_table": name, "deleted_in_batch": batch_counts["deleted"],
                              "deleted_dependents_in_batch": dict(children_deleted)}), flush=True)
        report["tables"][name] = {"deleted": counts.pop("deleted", 0), "protected_or_skipped": dict(counts)}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", required=True, choices=("server", "console"))
    age = parser.add_mutually_exclusive_group(required=True)
    age.add_argument("--older-than-years", type=positive)
    age.add_argument("--before", help="Exclusive ISO cutoff, including timezone")
    parser.add_argument("--batch-size", type=positive, default=500)
    parser.add_argument("--execute", action="store_true", help="Delete eligible history; otherwise preview only")
    args = parser.parse_args(argv)
    try:
        now = datetime.now(UTC)
        cutoff = cutoff_for(args, now)
        report = (execute if args.execute else preview)(args.role, cutoff, now, args.batch_size)
        print(json.dumps(report, indent=2))
    except (ValueError, FileNotFoundError) as exc:
        # Database driver errors may contain credentials or SQL parameters.
        print(f"Retention stopped: {type(exc).__name__}. Check the cutoff, configured database, and migrated schema. "
              "No further batches will run; earlier committed batches, if any, remain deleted.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Retention stopped: {type(exc).__name__}. Check connectivity, permissions, locks, and schema. "
              "No further batches will run; earlier committed batches, if any, remain deleted.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
