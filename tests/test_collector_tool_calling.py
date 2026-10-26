"""Isolated collector-tool tests: no production databases, LLMs or collectors."""

import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
EMPTY = {"type": "object", "properties": {}, "additionalProperties": False}


def setUpModule():
    global temporary, environment, config, db, models, service, store, tasks, contracts, registry
    temporary = tempfile.TemporaryDirectory(prefix="loghawk-tool-tests-")
    values = {name: temporary.name for name in (
        "LH_DATA_DIR", "LH_SERVER_LANCEDB_FILE_PATH", "LH_CONSOLE_LANCEDB_FILE_PATH",
        "LH_DOCS_STORAGE_DIR_PATH", "LH_LOG_DIR", "LH_FEATURE_DIR",
        "LH_ANOMALY_DETECTION_DIR", "LH_MODEL_DIR")}
    values.update(LH_DUCKDB_FILE_PATH="", LH_SERVER_DATABASE_URL="sqlite:///:memory:",
                  LH_CONSOLE_DATABASE_URL="sqlite:///:memory:")
    environment = patch.dict(os.environ, values)
    environment.start()
    from loghawk import config
    from loghawk.database import database_layer as db, models
    from loghawk.api.services import collector_tool_service as service, collector_tool_store as store, task_store2 as tasks
    from loghawk.collectors import contracts, registry


def tearDownModule():
    environment.stop()
    temporary.cleanup()


class ToolTests(unittest.TestCase):
    def setUp(self):
        from sqlalchemy import create_engine
        from sqlalchemy.pool import StaticPool
        self.engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
        models.SERVER_METADATA.create_all(self.engine)
        self.addCleanup(self.engine.dispose)
        for name, value in {
            "LH_LLM_TOOL_CALLING_ENABLED": True, "LH_LLM_TOOL_CONTEXT_LENGTH": 16000,
            "LH_LLM_TOOL_MAX_TOKENS": 1024, "LH_LLM_TOOL_MAX_CALLS": 3,
            "LH_LLM_TOOL_TIMEOUT_SECONDS": 120, "LH_LLM_TOOL_MAX_RESULT_CHARS": 2000,
        }.items():
            self.patch(config, name, value)
        self.patch(db, "get_engine", lambda role: self.engine)
        self.patch(db, "initialize_database", lambda role: None)
        self.patch(tasks, "initialize_database", lambda role: None)
        self.patch(tasks, "get_userid", lambda: "engineer")
        with db.connect("server") as connection:
            connection.execute("INSERT INTO auth_users (userid,password_hash,role,created_at,updated_at) VALUES ('engineer','unused','User','now','now')")
        self.runner = self.patch(service.task_runner, "run_version", Mock(return_value={
            "run_id": "run-1", "status": "COMPLETED", "stdout": "healthy"}))
        self.model = Mock()
        self.patch(service, "get_llm_client", lambda: self.model)

    def patch(self, obj, name, value):
        mocker = patch.object(obj, name, value)
        mocker.start()
        self.addCleanup(mocker.stop)
        return value

    def collector(self, policy="auto-approved", schema=EMPTY, approved=True, enabled=True, kind="collector"):
        task = tasks.create_task({"name": "pods-" + os.urandom(3).hex(), "description": "Read pod health",
                                  "task_kind": kind, "execution_policy": policy, "artifact_type": "python"})
        artifact = tasks.save_artifact(task["id"], "print('healthy')", arguments_schema=schema)
        tasks.set_artifact_policy(task["id"], artifact["id"], approved=approved, enabled=enabled)
        return tasks.get_task(task["id"])

    def call(self, task, args=None, call_id="call-1"):
        return {"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "type": "function",
                "function": {"name": "collector_" + task["id"], "arguments": json.dumps(args or {})}}]}

    def start(self):
        return service.start([{"role": "user", "content": "Check pod health"}], "engineer")

    def pending(self, task):
        self.model.complete_tool_turn.side_effect = [self.call(task), {"role": "assistant", "content": "Healthy [run-1]"}]
        return self.start()

    def confirm(self, pending, approved=True, userid="engineer"):
        return service.resume(pending["request_id"], userid, pending["pending"]["confirmation_id"], approved)

    def test_catalog_requires_policy_approval_enabled_schema_and_kind(self):
        eligible = self.collector()
        for kwargs in ({"policy": "manual"}, {"policy": "disabled"}, {"approved": False},
                       {"enabled": False}, {"schema": None}, {"kind": "general"}):
            self.collector(**kwargs)
        self.assertEqual(list(service.catalog()), ["collector_" + eligible["id"]])

    def test_auto_approved_uses_exact_version_and_server_audit(self):
        task = self.collector()
        result = self.pending(task)
        self.assertEqual(result["status"], "COMPLETED")
        self.runner.assert_called_once()
        self.assertEqual(self.runner.call_args.args[1]["id"], task["latest_artifact"]["id"])
        with db.connect("server") as connection:
            row = connection.execute("SELECT * FROM llm_tool_calls").fetchone()
            self.assertEqual(row["origin_userid"], "engineer")
            self.assertEqual((row["origin"], row["run_id"], row["status"]), ("AI", "run-1", "COMPLETED"))
            self.assertEqual(connection.execute("SELECT state_json FROM llm_tool_requests").fetchone()["state_json"], "{}")

    def test_assisted_waits_and_rejects_wrong_user_id_and_replay(self):
        task = self.collector(policy="ai-assisted")
        pending = self.pending(task)
        self.assertEqual(pending["status"], "PENDING")
        self.runner.assert_not_called()
        with self.assertRaises(ValueError):
            self.confirm(pending, userid="other")
        with self.assertRaises(ValueError):
            service.resume(pending["request_id"], "engineer", "stale-confirmation", True)
        self.assertEqual(self.confirm(pending)["status"], "COMPLETED")
        with self.assertRaises(ValueError):
            self.confirm(pending)
        self.runner.assert_called_once()

    def test_cancel_does_not_execute_or_continue_model(self):
        pending = self.pending(self.collector(policy="ai-assisted"))
        self.assertEqual(self.confirm(pending, False)["status"], "CANCELLED")
        self.runner.assert_not_called()
        self.assertEqual(self.model.complete_tool_turn.call_count, 1)

    def test_expired_confirmation(self):
        pending = self.pending(self.collector(policy="ai-assisted"))
        with db.connect("server") as connection:
            connection.execute("UPDATE llm_tool_requests SET expires_at = '2000-01-01'")
        with self.assertRaises(ValueError):
            self.confirm(pending)
        self.runner.assert_not_called()

    def test_revocation_before_confirmation_prevents_execution(self):
        task = self.collector(policy="ai-assisted")
        pending = self.pending(task)
        tasks.set_artifact_policy(task["id"], task["latest_artifact"]["id"], enabled=False)
        self.confirm(pending)
        self.runner.assert_not_called()
        self.assertIn("no longer eligible", str(self.model.complete_tool_turn.call_args))

    def test_new_version_never_replaces_confirmed_version(self):
        task = self.collector(policy="ai-assisted")
        pending = self.pending(task)
        tasks.save_artifact(task["id"], "print('different')", arguments_schema=EMPTY)
        self.confirm(pending)
        self.assertEqual(self.runner.call_args.args[1]["id"], task["latest_artifact"]["id"])

    def test_deleted_version_cannot_run(self):
        task = self.collector(policy="ai-assisted")
        pending = self.pending(task)
        tasks.delete_artifact_version(task["id"], task["latest_artifact"]["id"])
        self.confirm(pending)
        self.runner.assert_not_called()

    def test_manual_policy_after_proposal_cannot_run(self):
        task = self.collector(policy="ai-assisted")
        pending = self.pending(task)
        tasks.set_task_execution_policy(task["id"], "manual")
        self.confirm(pending)
        self.runner.assert_not_called()

    def test_invalid_arguments_count_towards_limit(self):
        self.patch(config, "LH_LLM_TOOL_MAX_CALLS", 1)
        task = self.collector()
        self.model.complete_tool_turn.side_effect = [self.call(task, {"undeclared": "value"}), self.call(task)]
        result = self.start()
        self.assertEqual(result["status"], "FAILED")
        self.runner.assert_not_called()

    def test_excessive_parallel_calls_rejected_before_any_execution(self):
        self.patch(config, "LH_LLM_TOOL_MAX_CALLS", 1)
        task = self.collector()
        response = self.call(task)
        response["tool_calls"].extend(self.call(task, call_id="call-2")["tool_calls"])
        self.model.complete_tool_turn.return_value = response
        self.assertEqual(self.start()["status"], "FAILED")
        self.runner.assert_not_called()

    def test_timeout_after_model_prevents_execution(self):
        task = self.collector()
        clock = [0]
        self.patch(service.time, "monotonic", lambda: clock[0])
        def slow(*args, **kwargs):
            clock[0] = 121
            return self.call(task)
        self.model.complete_tool_turn.side_effect = slow
        self.assertEqual(self.start()["status"], "FAILED")
        self.runner.assert_not_called()

    def test_human_wait_does_not_consume_processing_budget(self):
        clock = [0]
        self.patch(service.time, "monotonic", lambda: clock[0])
        pending = self.pending(self.collector(policy="ai-assisted"))
        clock[0] = 5000
        self.confirm(pending)
        self.assertEqual(self.runner.call_args.kwargs["timeout"], 120)

    def test_multiple_proposals_do_not_accept_previous_confirmation(self):
        task = self.collector(policy="ai-assisted")
        response = self.call(task)
        response["tool_calls"].extend(self.call(task, call_id="call-2")["tool_calls"])
        self.model.complete_tool_turn.side_effect = [response, {"role": "assistant", "content": "Done"}]
        first = self.start()
        second = self.confirm(first)
        self.assertEqual(second["status"], "PENDING")
        with self.assertRaises(ValueError):
            self.confirm(first)
        self.assertEqual(self.runner.call_count, 1)
        self.confirm(second)
        self.assertEqual(self.runner.call_count, 2)

    def test_model_failure_does_not_retry_completed_collector(self):
        task = self.collector()
        self.model.complete_tool_turn.side_effect = [self.call(task), RuntimeError("Model unavailable")]
        self.assertEqual(self.start()["status"], "FAILED")
        self.runner.assert_called_once()

    def test_concurrent_confirmation_claims_only_once(self):
        from concurrent.futures import ThreadPoolExecutor
        from sqlalchemy import create_engine
        with tempfile.TemporaryDirectory() as directory:
            engine = create_engine("sqlite:///" + str(Path(directory) / "claims.sqlite3"),
                                   connect_args={"check_same_thread": False})
            try:
                models.SERVER_METADATA.create_all(engine)
                with patch.object(db, "get_engine", lambda role: engine):
                    with db.connect("server") as connection:
                        connection.execute("INSERT INTO auth_users (userid,password_hash,role,created_at,updated_at) VALUES ('engineer','unused','User','now','now')")
                    request_id = store.create("engineer")
                    store.save(request_id, {"pending": {"confirmation_id": "one"}}, "PENDING")
                    def claim(_):
                        try:
                            store.claim(request_id, "engineer", "one")
                            return True
                        except ValueError:
                            return False
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        self.assertEqual(sorted(pool.map(claim, range(2))), [False, True])
            finally:
                engine.dispose()

    def test_result_redaction_and_bound(self):
        self.patch(config, "LH_LLM_API_KEY", "private-example-key")
        self.runner.return_value = {"run_id": "run-1", "status": "COMPLETED", "password": "do-not-show",
                                    "stdout": "private-example-key " + "a" * 10000}
        self.pending(self.collector())
        messages = self.model.complete_tool_turn.call_args.args[0]
        text = next(m["content"] for m in messages if m["role"] == "tool")
        self.assertLessEqual(len(text), 2000)
        self.assertNotIn("private-example-key", text)
        self.assertNotIn("do-not-show", text)

    def test_builtin_default_manual_and_policy_enable(self):
        self.assertFalse(service.catalog())
        registry.set_execution_policy("prometheus-query", "auto-approved")
        self.assertIn("builtin_prometheus_query", service.catalog())

    def test_schema_defaults_ranges_and_unknown_fields(self):
        schema = {"type": "object", "properties": {"count": {"type": "integer", "minimum": 1,
                   "maximum": 10, "default": 2}}, "additionalProperties": False}
        self.assertEqual(contracts.validate_arguments(schema, {}), {"count": 2})
        for args in ({"count": True}, {"count": 11}, {"extra": 2}):
            with self.assertRaises(ValueError):
                contracts.validate_arguments(schema, args)
        with self.assertRaises(ValueError):
            contracts.validate_schema({**EMPTY, "$ref": "https://invalid.example"})

    def test_schema_change_creates_unapproved_version_preserving_old(self):
        from loghawk.api.services import task_service2
        task = self.collector()
        schema = {**EMPTY, "properties": {"namespace": {"type": "string"}}}
        updated = task_service2.save_artifact_version(task["id"], "print('healthy')", arguments_schema=schema)
        self.assertEqual(updated["latest_artifact"]["version"], 2)
        self.assertFalse(updated["latest_artifact"]["approved"])
        self.assertFalse(updated["latest_artifact"]["enabled"])
        old = tasks.get_artifact_version(task["id"], task["latest_artifact"]["id"])
        self.assertEqual(old["arguments_schema"], EMPTY)
        self.assertTrue(old["approved"])

    def test_client_accepts_tool_only_response(self):
        import sys
        from types import SimpleNamespace
        from loghawk.llm.client import LiteLLMClient
        message = Mock()
        message.model_dump.return_value = self.call(self.collector())
        completion = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="tool_calls")]))
        with patch.dict(sys.modules, {"litellm": SimpleNamespace(completion=completion)}):
            result = LiteLLMClient().complete_tool_turn([], [], timeout=12)
        self.assertTrue(result["tool_calls"])
        self.assertEqual(completion.call_args.kwargs["num_retries"], 0)

    def test_api_requires_auth_and_manager_for_builtin_policy(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from loghawk.api.routers import console, collectors
        from loghawk.api.routers.auth import require_console_user
        app = FastAPI()
        app.include_router(console.router, prefix="/console")
        app.include_router(collectors.router, prefix="/collectors")
        with TestClient(app) as client:
            self.assertEqual(client.post("/console/chat", json={"messages": [{"role": "user", "content": "hello"}]}).status_code, 401)
            app.dependency_overrides[require_console_user] = lambda: {"userid": "engineer", "role": "User"}
            self.assertEqual(client.put("/collectors/prometheus-query/execution-policy", json={"execution_policy": "auto-approved"}).status_code, 403)
            app.dependency_overrides[require_console_user] = lambda: {"userid": "engineer", "role": "Power User"}
            self.assertEqual(client.put("/collectors/prometheus-query/execution-policy", json={"execution_policy": "ai-assisted"}).status_code, 200)
            self.assertEqual(client.post("/console/chat/no-id/confirm", json={"confirmation_id": "none", "approved": "yes"}).status_code, 422)
            self.assertEqual(client.post("/console/chat", json={"messages": [{"role": "system", "content": "execute"}]}).status_code, 422)

    def test_console_schema_editor_tracks_versions_and_cancel(self):
        import asyncio
        from loghawk.cli.hawk_console import HawkConsole
        from textual.widgets import TextArea, Button, Input

        async def no_mount(self):
            pass  # Textual dispatches superclass handlers too: patch the actual handler.
        self.patch(HawkConsole, "on_mount", no_mount)

        async def check():
            app = HawkConsole()
            async with app.run_test(size=(160, 60)) as pilot:
                task = self.collector()
                app._load_task(task)
                await pilot.pause()
                schema = app.query_one("#collector-arguments-schema", TextArea)
                self.assertEqual(json.loads(schema.text), EMPTY)
                self.assertFalse(app._editor_is_dirty())
                schema.text = json.dumps({**EMPTY, "properties": {"query": {"type": "string"}}})
                await pilot.pause()
                self.assertTrue(app._editor_is_dirty())
                self.assertFalse(app.query_one("#save-version", Button).disabled)
                app._cancel_edit()
                await pilot.pause()
                self.assertEqual(json.loads(schema.text), EMPTY)
                app._finish_tool_chat({"status": "PENDING", "request_id": "r", "response": "Confirm",
                                       "pending": {"collector": "pods", "version": "1", "arguments": {}, "confirmation_id": "c"}})
                self.assertEqual(app._pending_collector_call["confirmation_id"], "c")
                self.assertFalse(app.query_one("#llm-input", Input).disabled)
        asyncio.run(check())

    def test_interface_attribution_and_confirmation_binding(self):
        for source in ("web", "console", "cli"):
            request_id = store.create("engineer", source)
            call_id = store.record(request_id, {"name": "pods"}, {}, "REJECTED", "test")
            with db.connect("server") as connection:
                row = connection.execute("SELECT origin, origin_userid, origin_software FROM llm_tool_calls WHERE id = ?", (call_id,)).fetchone()
                self.assertEqual((row["origin"], row["origin_userid"], row["origin_software"]), ("AI", "engineer", source))
            store.save(request_id, {"pending": {"confirmation_id": "token"}}, "PENDING")
            with self.assertRaises(ValueError):
                store.claim(request_id, "engineer", "token", "other")
            self.assertIn("pending", store.claim(request_id, "engineer", "token", source))
        with self.assertRaises(ValueError):
            store.create("engineer", "other")

    def test_web_rca_requires_auth_and_sets_source(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from loghawk.api.routers import rca
        from loghawk.api.routers.auth import require_webui_user
        app = FastAPI()
        app.include_router(rca.router, prefix="/rca")
        with TestClient(app) as client:
            self.assertIn(client.post("/rca/chat", json={"messages": [{"role": "user", "content": "hello"}]}).status_code, (401, 403))
            app.dependency_overrides[require_webui_user] = lambda: {"userid": "engineer"}
            with patch.object(rca.assistant_chat_service, "chat", return_value={"response": "ok"}) as chat:
                response = client.post("/rca/chat", json={"messages": [{"role": "user", "content": "hello"}], "origin_software": "console"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(chat.call_args.kwargs["origin_software"], "web")
                self.assertEqual(chat.call_args.args[1], "engineer")
            with patch.object(rca.collector_tool_service, "resume", return_value={"response": "cancelled"}) as resume:
                response = client.post("/rca/chat/r/confirm", json={"confirmation_id": "c", "approved": False})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(resume.call_args.kwargs["origin_software"], "web")

    def test_software_migration_preserves_records(self):
        from sqlalchemy import create_engine
        from alembic.migration import MigrationContext
        from alembic.operations import Operations
        from types import SimpleNamespace
        migration = importlib.import_module("loghawk.database.migrations.versions.0016_tool_origin_software")
        engine = create_engine("sqlite://")
        try:
            with engine.begin() as conn:
                for table in ("llm_tool_requests", "llm_tool_calls"):
                    conn.exec_driver_sql(f"CREATE TABLE {table} (id TEXT PRIMARY KEY)")
                    conn.exec_driver_sql(f"INSERT INTO {table} VALUES ('kept')")
                context = SimpleNamespace(config=SimpleNamespace(attributes={"database_role": "server"}))
                with patch.object(migration, "op", Operations(MigrationContext.configure(conn))), patch.object(migration, "context", context):
                    migration.upgrade()
                    migration.upgrade()
                for table in ("llm_tool_requests", "llm_tool_calls"):
                    self.assertEqual(conn.exec_driver_sql(f"SELECT id, origin_software FROM {table}").all(), [("kept", "console")])
            context.config.attributes["database_role"] = "console"
            with patch.object(migration, "context", context), patch.object(migration, "op") as operation:
                migration.upgrade()
                operation.get_bind.assert_not_called()
        finally:
            engine.dispose()

    def test_origin_userid_is_derived_from_request_owner(self):
        for userid in ("alice", "bob"):
            request_id = store.create(userid)
            call_id = store.record(request_id, {"name": "pods"}, {}, "REJECTED", "test")
            with db.connect("server") as connection:
                row = connection.execute("SELECT origin, origin_userid FROM llm_tool_calls WHERE id = ?", (call_id,)).fetchone()
                self.assertEqual((row["origin"], row["origin_userid"]), ("AI", userid))
        with self.assertRaises(ValueError):
            store.record("missing", {"name": "pods"}, {}, "REJECTED", "test")

    def test_origin_migration_backfills_without_losing_audit_rows(self):
        from sqlalchemy import create_engine, inspect
        from alembic.migration import MigrationContext
        from alembic.operations import Operations
        from types import SimpleNamespace
        migration = importlib.import_module("loghawk.database.migrations.versions.0015_tool_origin_userid")
        engine = create_engine("sqlite://")
        try:
            with engine.begin() as conn:
                conn.exec_driver_sql("CREATE TABLE llm_tool_requests (id TEXT PRIMARY KEY, userid TEXT NOT NULL)")
                conn.exec_driver_sql("CREATE TABLE llm_tool_calls (id TEXT PRIMARY KEY, request_id TEXT NOT NULL REFERENCES llm_tool_requests(id), origin TEXT NOT NULL DEFAULT 'ai', result_text TEXT)")
                for userid in ("alice", "bob"):
                    conn.exec_driver_sql("INSERT INTO llm_tool_requests VALUES (?, ?)", (userid, userid))
                    conn.exec_driver_sql("INSERT INTO llm_tool_calls VALUES (?, ?, 'ai', 'kept')", (userid, userid))
                context = SimpleNamespace(config=SimpleNamespace(attributes={"database_role": "server"}))
                with patch.object(migration, "op", Operations(MigrationContext.configure(conn))), patch.object(migration, "context", context):
                    migration.upgrade()
                rows = conn.exec_driver_sql("SELECT origin, origin_userid, result_text FROM llm_tool_calls ORDER BY id").all()
                self.assertEqual(rows, [("AI", "alice", "kept"), ("AI", "bob", "kept")])
                columns = {c["name"]: c for c in inspect(conn).get_columns("llm_tool_calls")}
                self.assertFalse(columns["origin_userid"]["nullable"])
                self.assertEqual(conn.exec_driver_sql("PRAGMA foreign_key_check").all(), [])
            context = SimpleNamespace(config=SimpleNamespace(attributes={"database_role": "console"}))
            with patch.object(migration, "context", context), patch.object(migration, "op") as operation:
                migration.upgrade()
                operation.get_bind.assert_not_called()
        finally:
            engine.dispose()

    def test_migration_preserves_existing_versions_for_both_roles(self):
        from sqlalchemy import create_engine, MetaData, inspect
        from alembic.migration import MigrationContext
        from alembic.operations import Operations
        from types import SimpleNamespace
        migration = importlib.import_module("loghawk.database.migrations.versions.0014_collector_tool_calling")
        for role in ("console", "server"):
            engine = create_engine("sqlite://")
            try:
                old = MetaData()
                for table in models.metadata_for_role(role).sorted_tables:
                    if table.name.startswith("llm_tool_"):
                        continue
                    clone = table.to_metadata(old)
                    if clone.name == "task_artifacts":
                        clone._columns.remove(clone.c.arguments_schema_json)
                    if clone.name == "collector_registry":
                        clone._columns.remove(clone.c.execution_policy)
                old.create_all(engine)
                with engine.begin() as conn:
                    conn.exec_driver_sql("INSERT INTO tasks (id,name,description,artifact_type,created_at,updated_at) VALUES ('t','kept','test','python','now','now')")
                    conn.exec_driver_sql("INSERT INTO task_artifacts (id,task_id,version,content,created_at) VALUES ('v','t',1,'print(42)','now')")
                    context = SimpleNamespace(config=SimpleNamespace(attributes={"database_role": role}))
                    with patch.object(migration, "op", Operations(MigrationContext.configure(conn))), patch.object(migration, "context", context):
                        migration.upgrade()
                    self.assertEqual(conn.exec_driver_sql("SELECT content FROM task_artifacts").scalar(), "print(42)")
                    self.assertIn("arguments_schema_json", {c["name"] for c in inspect(conn).get_columns("task_artifacts")})
                    self.assertEqual("llm_tool_requests" in inspect(conn).get_table_names(), role == "server")
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
