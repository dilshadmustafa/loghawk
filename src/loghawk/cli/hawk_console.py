"""Hawk Console: split-pane interface for LogHawk chat, commands, and Tasks."""

from __future__ import annotations

import asyncio
import argparse
import ast
import json
import os
import re
import shlex
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote
from uuid import uuid4
from pathlib import Path

from textual import events, work
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, RichLog, Select, Static, TextArea
from rich.table import Table

import loghawk.config as config
from loghawk import lhrunner
from loghawk.cli.command_terminal import CommandShellSession
from loghawk.cli.console_auth import ConsoleLoginApp
from loghawk.cli import command_history2 as command_history
from loghawk.cli import local_store2 as local_store
from loghawk.cli import task_workspace2 as local_task_workspace
from loghawk.cli.console_api_client import ConsoleApiClient
from loghawk.cli.console_api_task_workspace import ConsoleApiTaskWorkspace
from loghawk.llm import get_llm_client
from loghawk.rag import ingestion
from loghawk.utils import lancedbutils3
from loghawk.cli import console_rag_store
from loghawk.utils.user_identity import set_userid_override

_INPUT_BORDER_CSS = (
    "Input { border-left: ascii $border-blurred; }\n"
    "Input:focus { border-left: ascii $border; }\n"
    if config.LH_OS_FAMILY in {"LINUX", "MAC"}
    else ""
)


def _parse_api_run_command(command: str, expected: str) -> tuple[str, dict]:
    """Parse `/task|collect <name> [JSON arguments]` without altering JSON quotes."""
    match = re.fullmatch(
        r"\s*(/task|/collect)\s+(?:\"([^\"]+)\"|'([^']+)'|(\S+))(?:\s+(.*?))?\s*",
        command,
        flags=re.DOTALL,
    )
    if match is None or match.group(1).lower() != expected:
        raise ValueError(
            f'Usage: {expected} <name> [JSON arguments] (quote names containing spaces)'
        )
    name = match.group(2) or match.group(3) or match.group(4)
    raw_arguments = match.group(5)
    if not raw_arguments:
        return name, {}
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Arguments must be valid JSON: {exc}") from exc
    if not isinstance(arguments, dict):
        raise ValueError("Arguments must be a JSON object.")
    return name, arguments


def _format_api_run_result(result: Any) -> str:
    """Pretty-print JSON emitted by a program without escaping it as nested text."""
    if isinstance(result, dict):
        result = dict(result)
        for key in ("stdout", "stderr"):
            value = result.get(key)
            if isinstance(value, str):
                try:
                    result[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass
    return json.dumps(result, indent=2, ensure_ascii=False, default=str)


class TickCheckbox(Checkbox):
    BUTTON_INNER = "✓"

    @property
    def _button(self) -> Content:
        # Explicit brackets remain visible independently of panel shading.
        bracket_style = self.get_visual_style("toggle--label")
        mark_style = self.get_visual_style("toggle--button")
        return Content.assemble(
            ("[", bracket_style),
            (self.BUTTON_INNER if self.value else " ", mark_style),
            ("]", bracket_style),
        )


class CollectorArgumentReviewScreen(ModalScreen[dict]):
    """One editable input review; code approval remains a separate server policy."""

    CSS = """
    CollectorArgumentReviewScreen { align: center middle; }
    #collector-review-box { width: 90%; max-width: 110; height: 85%; padding: 1 2; border: ascii $primary; background: $surface; }
    #collector-review-arguments { height: 1fr; margin: 1 0; }
    #collector-review-error { height: auto; max-height: 6; overflow-y: auto; }
    #collector-review-actions { height: 3; }
    #collector-review-actions Button { margin-right: 1; }
    """

    def __init__(self, pending: dict) -> None:
        super().__init__()
        self.pending = pending
        self.blocked = False
        self.sending = False

    def compose(self) -> ComposeResult:
        review_arguments = self.pending.get("review_arguments", True)
        title = "Review input arguments" if review_arguments else "Confirm collector execution"
        with Container(id="collector-review-box"):
            yield Static(f"{title}: {self.pending['collector']} | Version {self.pending['version']}", markup=False)
            yield Static(
                "This reviews arguments only, not collector code approval. Leave masked values unchanged to retain them."
                if review_arguments else
                "The AI-assisted execution policy requires confirmation. Input arguments are read-only."
            )
            yield TextArea(json.dumps(self.pending["arguments"], indent=2, ensure_ascii=False),
                           id="collector-review-arguments", soft_wrap=True, read_only=not review_arguments)
            yield Static("", id="collector-review-error", markup=False)
            with Horizontal(id="collector-review-actions"):
                yield Button("Confirm & Execute", variant="primary", id="collector-review-confirm")
                yield Button("Cancel", id="collector-review-cancel")

    def set_sending(self, sending: bool) -> None:
        self.sending = sending
        self.query_one("#collector-review-arguments", TextArea).disabled = sending or self.blocked
        self.query_one("#collector-review-confirm", Button).disabled = sending or self.blocked
        self.query_one("#collector-review-cancel", Button).disabled = sending

    def show_error(self, error: str, correctable: bool) -> None:
        self.blocked = not correctable
        notice = error if correctable else error + "\nOutcome may be uncertain. Check execution audit records before starting another request."
        self.query_one("#collector-review-error", Static).update(notice)
        self.set_sending(False)
        if self.blocked:
            self.query_one("#collector-review-cancel", Button).label = "Close"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self.sending:
            return
        approved = event.button.id == "collector-review-confirm"
        if self.blocked:
            if not approved:
                self.dismiss({"response": "Argument review closed after an API error. Check execution audit records before retrying."})
            return
        arguments = None
        if approved:
            try:
                arguments = json.loads(self.query_one("#collector-review-arguments", TextArea).text)
                if not isinstance(arguments, dict):
                    raise ValueError("Arguments must be a JSON object.")
            except ValueError as exc:
                self.show_error(str(exc), True)
                return
        self.query_one("#collector-review-error", Static).update("Processing…")
        self.set_sending(True)
        self.app._confirm_collector_call(self.pending, approved, arguments, self)


class ConfirmDeleteScreen(ModalScreen[bool]):
    """Confirmation dialog for deleting a saved Task artifact version."""

    CSS = """
    ConfirmDeleteScreen { align: center middle; }
    #confirm-box { width: 64; height: auto; padding: 1 2; border: round $primary; background: $surface; }
    #confirm-box Button { margin: 1 1 0 0; }
    """

    def compose(self) -> ComposeResult:
        with Container(id="confirm-box"):
            yield Static("Delete this saved Task version? This action cannot be undone.")
            with Horizontal():
                yield Button("Delete version", variant="error", id="confirm-delete")
                yield Button("Cancel", id="cancel-delete")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm-delete")


class ConfirmDeleteTaskScreen(ModalScreen[bool]):
    """Confirmation dialog for deleting a Task and its saved data."""

    CSS = """
    ConfirmDeleteTaskScreen { align: center middle; }
    #confirm-task-box { width: 72; height: auto; padding: 1 2; border: round $primary; background: $surface; }
    #confirm-task-box Button { margin: 1 1 0 0; }
    """

    def __init__(self, task_name: str) -> None:
        super().__init__()
        self.task_name = task_name

    def compose(self) -> ComposeResult:
        with Container(id="confirm-task-box"):
            yield Static(
                f'Delete Task "{self.task_name}", its conversation, and all saved '
                "artifact versions? This cannot be undone."
            )
            with Horizontal():
                yield Button("Delete Task", variant="error", id="confirm-delete-task")
                yield Button("Cancel", id="cancel-delete-task")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm-delete-task")


class DownloadPathScreen(ModalScreen[str | None]):
    """Prompt for a destination path when downloading an artifact."""

    CSS = """
    DownloadPathScreen { align: center middle; }
    #download-box { width: 80; height: auto; padding: 1 2; border: round $primary; background: $surface; }
    #download-box Input { margin: 1 0; }
    #download-box Button { margin: 0 1 0 0; }
    """

    def __init__(self, suggested_path: str) -> None:
        super().__init__()
        self.suggested_path = suggested_path

    def compose(self) -> ComposeResult:
        with Container(id="download-box"):
            yield Static("Download artifact to:")
            yield Input(value=self.suggested_path, id="download-path")
            with Horizontal():
                yield Button("Download", variant="primary", id="confirm-download")
                yield Button("Cancel", id="cancel-download")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm-download":
            self.dismiss(self.query_one("#download-path", Input).value.strip())
        else:
            self.dismiss(None)


class HawkConsole(App[None]):
    """Three terminal panes alongside an editable Task artifact workspace."""

    TITLE = "Hawk Console"
    BINDINGS = [("ctrl+q", "quit", "Quit Hawk")]
    CSS = """
    Screen {
        layout: horizontal;
        padding: 0;
    }
    #terminal-column {
        layout: vertical;
        width: 1fr;
        height: 1fr;
    }
    .pane {
        border: round $primary;
        padding: 0 1;
        height: 1fr;
        width: 1fr;
    }
    .pane-title { text-style: bold; height: 1; }
    #editor-pane { width: 1.25fr; height: 1fr; }
    #rca-output-pane { height: 1fr; }
    #rca-output-text { height: 1fr; }
    #rca-output-toolbar { height: auto; min-height: 3; }
    #llm-chat-toolbar { height: auto; min-height: 3; }
    #new-chat { margin-right: 1; }
    #new-chat, #show-rca-output {
        background: #0078d4;
        color: #ffffff;
    }
    #new-chat:hover, #show-rca-output:hover {
        background: #106ebe;
    }
    #new-chat:disabled, #show-rca-output:disabled {
        background: #24445e;
        color: #a0aab4;
    }
    #llm-output, #command-one-output, #command-two-output { height: 1fr; }
    #editor-toolbar { height: auto; min-height: 3; }
    #task-create-pane { height: auto; padding: 0 1; }
    #task-create-fields { height: auto; }
    #task-chat-pane { height: 1fr; min-height: 8; overflow-y: auto; }
    #loaded-task-description-row { height: 5; }
    #loaded-task-description { width: 1fr; height: 5; }
    #save-loaded-task-description { width: auto; margin-left: 1; }
    #loaded-task-policy-row { height: auto; min-height: 3; }
    #loaded-task-execution-policy { width: 1fr; }
    #save-task-execution-policy { width: auto; margin-left: 1; }
    #loaded-task-version-actions { height: auto; min-height: 3; }
    #approve-loaded-version { margin-left: 1; }
    #loaded-version-status { height: auto; }
    #task-chat-pane > Static { height: auto; }
    #task-chat-output { min-height: 3; }
    #collector-schema-pane { height: auto; }
    #collector-arguments-schema { height: 6; }
    #task-chat-output { height: 1fr; }
    #task-message-row { height: auto; }
    #task-list-pane { height: 1fr; padding: 0 1; }
    #task-list-output { height: 1fr; }
    #command-editor-pane { height: 1fr; padding: 0 1; }
    #command-task-fields { height: auto; }
    .command-task-row { height: auto; }
    .command-task-field { height: auto; width: 1fr; padding-right: 1; }
    .command-task-action { height: auto; width: auto; }
    #command-editor { height: 1fr; border: round $secondary; }
    #command-editor-toolbar { height: auto; min-height: 3; }
    #version-select { width: 1fr; }
    #editor { height: 1fr; border: round $secondary; }
    #command-output-mirror { height: 1fr; }
    Button { min-width: 10; }
    """ + _INPUT_BORDER_CSS

    def __init__(self, api_token: str | None = None, api_user: dict | None = None) -> None:
        super().__init__()
        self._api_client = ConsoleApiClient(api_token) if api_token else None
        self._api_user = dict(api_user or {})
        self._task_workspace = (
            ConsoleApiTaskWorkspace(self._api_client)
            if self._api_client else local_task_workspace
        )
        self._pending_chat_history: list[dict[str, str]] = []
        self._pending_command_history: list[dict[str, str]] = []
        self.current_task: dict | None = None
        self.current_artifact: dict | None = None
        self._editor_baseline = ""
        self._schema_baseline = ""
        self._suppress_version_event = False
        self._has_generated_draft = False
        self._task_chat_busy = False
        self._pending_task_runs: dict[str, tuple[dict, dict]] = {}
        self._chat_history: list[dict[str, str]] = []
        self._chat_busy = False
        self._pending_llm_command: str | None = None
        self._pending_collector_call: dict | None = None
        self._awaiting_llm_command_final_confirmation = False
        self._command_editor_visibility: dict[str, bool] = {}
        self._sessions: dict[str, CommandShellSession] = {}
        self._active_command_outputs: set[str] = set()
        self._command_history = {"command-one": [], "command-two": []}
        self._command_history_index = {"command-one": 0, "command-two": 0}
        self._command_history_draft = {"command-one": "", "command-two": ""}
        self._help_previous_visibility: dict[str, bool] | None = None
        self._latest_rca_output = ""
        self._rca_retry_request_id = None
        self._rca_previous_visibility: dict[str, bool] | None = None

    def compose(self) -> ComposeResult:
        with Container(id="terminal-column"):
            with Container(id="llm-pane", classes="pane"):
                yield Static("LLM Terminal · /help · Ctrl+Q to quit, Shift+Ctrl+C/Shift+Ctrl+V to Copy/Paste", classes="pane-title")
                yield RichLog(id="llm-output", wrap=True, markup=False, auto_scroll=True)
                yield Input(placeholder="Chat with the configured LLM…", id="llm-input")
                with Horizontal(id="llm-chat-toolbar"):
                    yield Button("New Chat", id="new-chat")
                    yield Button("Show RCA Output", id="show-rca-output", disabled=True)

            with Container(id="command-one-pane", classes="pane"):
                yield Static("Command Terminal 1 · /help · /collect <name> · /collectrun <name> · /task <name> · /taskrun <name> · /tasklist · /tasknew · /taskedit <taskname>", classes="pane-title")
                yield RichLog(id="command-one-output", wrap=True, markup=False, auto_scroll=True)
                with Horizontal():
                    yield Input(placeholder="Shell command · /collect <name> · /collectrun <name> · /collectlist · /collectview · /task <name> · /taskrun <name> · /tasklist · /tasknew · /taskedit <taskname>", id="command-one-input")
                    yield Button("Stop", id="command-one-stop")

            with Container(id="command-two-pane", classes="pane"):
                yield Static("Command Terminal 2 · /help · /collect <name> · /collectrun <name> · /task <name> · /taskrun <name> · /tasklist · /tasknew · /taskedit <taskname>", classes="pane-title")
                yield RichLog(id="command-two-output", wrap=True, markup=False, auto_scroll=True)
                with Horizontal():
                    yield Input(placeholder="Shell command · /collect <name> · /collectrun <name> · /collectlist · /collectview · /task <name> · /taskrun <name> · /tasklist · /tasknew · /taskedit <taskname>", id="command-two-input")
                    yield Button("Stop", id="command-two-stop")

        with Container(id="editor-pane", classes="pane"):
            yield Static("Editor · /help · use /taskedit <name> or /tasknew", classes="pane-title")
            yield TickCheckbox("Review Input Args", value=True, id="review-input-args",
                           disabled=self._api_client is None)
            with Container(id="task-create-pane"):
                yield Static("Create Task", classes="pane-title")
                with Container(id="task-create-fields"):
                    yield Static("Task name")
                    yield Input(placeholder="Task name", id="task-name-input")
                    yield Static("Description · what this Task is for")
                    yield TextArea("", id="task-description-input", soft_wrap=True)
                    yield Static("Task Type")
                    yield Select(
                        [
                            ("Python", "python"),
                            ("Bash", "bash"),
                            ("PowerShell", "powershell"),
                            ("Windows CMD", "windows-cmd"),
                            ("n8n", "n8n"),
                        ],
                        value="python",
                        allow_blank=False,
                        id="task-type-select",
                    )
                    yield Static("Task Kind")
                    yield Select(
                        [("General Task", "general"), ("Remediation Task", "remediation"),
                         ("Rollback Task", "rollback"), ("Collector", "collector")],
                        value="general", allow_blank=False, id="task-kind-select",
                    )
                    yield Static("Execution Policy")
                    yield Select(
                        [("Manual", "manual"), ("AI-assisted", "ai-assisted"),
                         ("Auto-approved", "auto-approved"), ("Disabled", "disabled")],
                        value="manual", allow_blank=False, id="task-execution-policy-select",
                    )
                    yield Static("Auto-approved requires an Administrator or Power User in signed-in mode.")
                    yield Static("Category")
                    yield Input(placeholder="Enter a category", id="task-category-input")
                with Horizontal():
                    yield Button("Create Task", variant="primary", id="create-task")
                    yield Button("Cancel", id="cancel-create-task")
            with Container(id="task-chat-pane"):
                yield Static("Task Chat", classes="pane-title")
                yield Static("Task description")
                with Horizontal(id="loaded-task-description-row"):
                    yield TextArea("", id="loaded-task-description", soft_wrap=True)
                    yield Button("Save Description", id="save-loaded-task-description")
                yield Static("Execution Policy")
                with Horizontal(id="loaded-task-policy-row"):
                    yield Select(
                        [("Manual", "manual"), ("AI-assisted", "ai-assisted"),
                         ("Auto-approved", "auto-approved"), ("Disabled", "disabled")],
                        value="manual", allow_blank=False, id="loaded-task-execution-policy",
                    )
                    yield Button("Save Policy", id="save-task-execution-policy")
                yield Static("No saved version selected.", id="loaded-version-status")
                with Horizontal(id="loaded-task-version-actions"):
                    yield Button("Enable", id="enable-loaded-version", disabled=True)
                    yield Button("Approve", id="approve-loaded-version", disabled=True)
                with Container(id="collector-schema-pane"):
                    yield Static("Arguments schema JSON (optional; blank or {} means no arguments; saved with version)")
                    yield TextArea("", id="collector-arguments-schema", soft_wrap=True)
                yield RichLog(id="task-chat-output", wrap=True, markup=False, auto_scroll=True)
                with Horizontal(id="task-message-row"):
                    yield Input(placeholder="Describe the artifact you want…", id="task-message-input")
                    yield Button("Send", variant="primary", id="send-task-message")
            with Horizontal(id="editor-toolbar"):
                yield Select([], prompt="Task version", allow_blank=True, id="version-select")
                yield Button("Save Version", id="save-version")
                yield Button("Download", id="download-version")
                yield Button("Delete", variant="error", id="delete-version")
                yield Button("Cancel", id="cancel-edit")
            yield TextArea("", id="editor", soft_wrap=False)
            yield Static("No Task loaded.", id="editor-status")
            yield RichLog(
                id="command-output-mirror",
                wrap=True,
                markup=False,
                auto_scroll=True,
            )
            with Container(id="task-list-pane"):
                yield Static("Tasks", classes="pane-title")
                yield RichLog(id="task-list-output", wrap=True, markup=False, auto_scroll=True)
                yield Button("Close", id="close-task-list")
            with Container(id="command-editor-pane"):
                yield Static("Edit Proposed Command", classes="pane-title")
                with Container(id="command-task-fields"):
                    with Horizontal(classes="command-task-row"):
                        with Container(classes="command-task-field"):
                            yield Static("Task name")
                            yield Input(placeholder="Task name", id="command-task-name")
                        with Container(classes="command-task-field"):
                            yield Static("Type")
                            yield Select(
                                [
                                    ("Python", "python"),
                                    ("Bash", "bash"),
                                    ("PowerShell", "powershell"),
                                    ("Windows CMD", "windows-cmd"),
                                    ("n8n", "n8n"),
                                ],
                                prompt="Select task type",
                                allow_blank=True,
                                id="command-task-type",
                            )
                    with Container(classes="command-task-field"):
                        yield Static("Description · what this Task is for")
                        yield Input("", placeholder="Task description", id="command-task-description")
                    with Horizontal(classes="command-task-row"):
                        with Container(classes="command-task-field"):
                            yield Static("Category")
                            yield Input(placeholder="Enter a category", id="command-task-category")
                        with Container(classes="command-task-field"):
                            yield Static("Task kind")
                            yield Select(
                                [("General Task", "general"), ("Remediation Task", "remediation"),
                                 ("Rollback Task", "rollback"), ("Collector", "collector")],
                                value="general", allow_blank=False, id="command-task-kind",
                            )
                        with Container(classes="command-task-field"):
                            yield Static("Execution policy")
                            yield Select(
                                [("Manual", "manual"), ("AI-assisted", "ai-assisted"),
                                 ("Auto-approved", "auto-approved"), ("Disabled", "disabled")],
                                value="manual", allow_blank=False, id="command-task-execution-policy",
                            )
                        with Container(classes="command-task-action"):
                            yield Static("")
                            yield Button("Quick Save", id="quick-save-command-task")
                yield TextArea("", id="command-editor", soft_wrap=False)
                with Horizontal(id="command-editor-toolbar"):
                    yield Button("Confirm Changes", variant="primary", id="confirm-command-edits")
                    yield Button("Cancel", id="cancel-command-edits")
            with Container(id="rca-output-pane"):
                yield Static("RCA Output", classes="pane-title")
                yield TextArea("", id="rca-output-text", read_only=True, soft_wrap=True)
                with Horizontal(id="rca-output-toolbar"):
                    yield Button("Copy", id="copy-rca-output")
                    yield Button("Download", id="download-rca-output")
                    yield Button("Retry AI Explanation", id="retry-rca-explanation", disabled=True)
                    yield Button("Close", id="close-rca-output")
            with Container(id="help-pane"):
                yield Static("Hawk Console Command Help", classes="pane-title")
                yield RichLog(id="help-output", wrap=True, markup=False, auto_scroll=True)
                yield Button("Close Help", id="close-help")

    async def on_mount(self) -> None:
        if self._api_client is None:
            local_store.initialize_console_database()
        self.query_one("#llm-output", RichLog).write(
            f"Model: {config.LH_LLM_MODEL}\n"
            + (
                "Signed-in chat, RAG, Tasks, and history use the authenticated LogHawk API.\n"
                if self._api_client is not None
                else "Chat uses the configured LiteLLM provider and local LanceDB context when available.\n"
            )
            + "Signed-in chat can use approved collector tools when enabled on the API server. "
            "Prefix a question with /rag to search indexed documents.\n"
            "To request shell command(s), enter /command generate commands based on above discussion "
            "or /command <what to do>.\n"
            "Type YES or yes to run it, E or e to edit it in the Editor pane, "
            "or N or anything else to cancel.\n"
        )
        for terminal_id, output_id in (
            ("command-one", "command-one-output"),
            ("command-two", "command-two-output"),
        ):
            output = self.query_one(f"#{output_id}", RichLog)
            session = CommandShellSession(
                on_output=lambda text, terminal_id=terminal_id:
                    self._handle_command_output(terminal_id, text),
                cwd=str(config.PROJECT_ROOT),
            )
            self._sessions[terminal_id] = session
            try:
                if self._api_client is None:
                    self._command_history[terminal_id] = command_history.load_history(terminal_id)
                self._command_history_index[terminal_id] = len(
                    self._command_history[terminal_id]
                )
            except Exception as exc:
                output.write(f"Could not load command history: {exc}\n")
            try:
                await session.start()
            except Exception as exc:
                output.write(f"Could not start command shell: {exc}\n")
        self.query_one("#task-create-pane", Container).display = False
        self.query_one("#task-chat-pane", Container).display = False
        self.query_one("#task-list-pane", Container).display = False
        self.query_one("#command-output-mirror", RichLog).display = False
        self.query_one("#command-editor-pane", Container).display = False
        self.query_one("#help-pane", Container).display = False
        self.query_one("#rca-output-pane", Container).display = False
        self._load_chat_history()
        if self._api_client is not None:
            self.set_interval(300, self._start_history_flush)
        self._sync_editor_controls()

    def _start_history_flush(self) -> None:
        self.run_worker(self._flush_remote_history, thread=True, name="console-history-sync")

    def _load_chat_history(self) -> None:
        self._chat_history = []
        try:
            if self._api_client is not None:
                history = self._api_client.get_json("/api/v1/console/history")
                for terminal_id in ("command-one", "command-two"):
                    self._command_history[terminal_id] = [
                        item["command"] for item in history.get("commands", [])
                        if item.get("terminal_id") == terminal_id
                    ][-command_history.MAX_HISTORY_ENTRIES:]
                    self._command_history_index[terminal_id] = len(self._command_history[terminal_id])
            self.query_one("#llm-output", RichLog).write(
                "Started a fresh conversation. Saved chat history is not included in LLM context.\n"
            )
        except Exception as exc:
            self.query_one("#llm-output", RichLog).write(
                f"Could not load command history; conversation context remains empty: {exc}\n"
            )

    def _persist_chat_message(self, role: str, content: str) -> None:
        if self._api_client is None:
            local_store.save_conversation_message(role, content)
        else:
            self._pending_chat_history.append(
                {
                    "id": str(uuid4()), "role": role, "content": content,
                    "created_at": datetime.now(UTC).isoformat(),
                }
            )

    def _flush_remote_history(self, report_errors: bool = True) -> None:
        if self._api_client is None or not (
            self._pending_chat_history or self._pending_command_history
        ):
            return
        chats = list(self._pending_chat_history)
        commands = list(self._pending_command_history)
        try:
            self._api_client.post_json(
                "/api/v1/console/history",
                {"llm_messages": chats, "commands": commands},
            )
        except Exception as exc:
            if report_errors and self.is_mounted:
                self.app.call_from_thread(self._show_history_sync_error, str(exc))
            return
        chat_ids = {item["id"] for item in chats}
        command_ids = {item["id"] for item in commands}
        self._pending_chat_history = [item for item in self._pending_chat_history if item["id"] not in chat_ids]
        self._pending_command_history = [item for item in self._pending_command_history if item["id"] not in command_ids]

    def _show_history_sync_error(self, error: str) -> None:
        self.query_one("#llm-output", RichLog).write(
            f"Could not sync Console history; it will retry: {error}\n"
        )

    def _retrieve_context(self, prompt: str) -> tuple[str, str | None]:
        requested_names = self._document_names_in_prompt(prompt)
        database = lancedbutils3.connect_database(
            config.LH_CONSOLE_LANCEDB_FILE_PATH
        )
        table = lancedbutils3.open_table(
            database,
            config.LH_CONSOLE_LANCEDB_TABLE_NAME,
        )
        if table is None:
            if requested_names:
                return "", "No documents are indexed in Hawk Console RAG yet."
            return "", None

        matching_sources: list[str] = []
        for name in requested_names:
            matching_sources.extend(
                str(item["source_key"])
                for item in console_rag_store.find_document_by_name(name)
            )
        matching_sources = sorted(set(matching_sources))
        if requested_names and not matching_sources:
            return (
                "",
                "No indexed document matched: "
                + ", ".join(requested_names)
                + ". Check the filename or upload the document first.",
            )

        vector = lancedbutils3.embed_texts([prompt])[0]
        query = table.search(vector)
        if requested_names:
            source_values = ", ".join(
                "'" + source.replace("'", "''") + "'"
                for source in matching_sources
            )
            query = query.where(
                f"source IN ({source_values})",
                prefilter=True,
            )
        rows = query.limit(5).to_list()
        chunks = []
        for row in rows:
            if isinstance(row, dict):
                value = row.get("text") or row.get("content") or row.get("document")
                source = row.get("source")
            else:
                value = row
                source = None
            if value and str(value).strip():
                source_label = f"Source: {source}\n" if source else ""
                chunks.append(f"{source_label}{str(value).strip()}")
        if requested_names and not chunks:
            return (
                "",
                "The document was found in the index, but no text chunks "
                "could be retrieved for that question.",
            )
        return "\n\n".join(chunks), None

    @staticmethod
    def _document_names_in_prompt(prompt: str) -> list[str]:
        """Find supported document filenames or paths mentioned in a question."""
        candidates = re.findall(r"""['\"]([^'\"]+)['\"]|([^\s'\"<>|]+)""", prompt)
        names: list[str] = []
        for quoted, unquoted in candidates:
            candidate = (quoted or unquoted).strip(".,!?;:()[]{}<>")
            basename = candidate.replace("\\", "/").rsplit("/", 1)[-1]
            if (
                basename
                and Path(basename).suffix.lower() in lancedbutils3.SUPPORTED_SUFFIXES
                and basename.casefold()
                not in {name.casefold() for name in names}
            ):
                names.append(basename)
        return names

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""
        if event.input.id == "task-message-input":
            self._submit_task_message(text)
            return
        if event.input.id == "llm-input":
            if text.casefold() == "/help":
                self._show_help()
                return
            if self._pending_collector_call is not None:
                self.query_one("#llm-output", RichLog).write("Review the input arguments in the open dialog.")
                return
            if self._pending_llm_command is not None:
                self._confirm_llm_command(text)
                return
            if text.startswith("/command"):
                request = text[len("/command"):].strip()
                if not request:
                    self.query_one("#llm-output", RichLog).write(
                        "Usage: /command <what you want the shell to do>\n"
                    )
                    return
                if self._chat_busy:
                    self.query_one("#llm-output", RichLog).write(
                        "Wait for the current response to finish.\n"
                    )
                    return
                self._chat_busy = True
                event.input.disabled = True
                self.query_one("#llm-output", RichLog).write(
                    f"\nCommand request: {request}\nPreparing a command proposal…\n"
                )
                context_limit = config.LH_CONSOLE_COMMAND_CONTEXT_MESSAGES
                chat_history = (
                    list(self._chat_history[-context_limit:])
                    if context_limit
                    else []
                )
                self._propose_command(request, chat_history)
                return
            use_rag = text == "/rag" or text.startswith("/rag ")
            if use_rag:
                text = text[len("/rag"):].strip()
                if not text:
                    self.query_one("#llm-output", RichLog).write(
                        "Usage: /rag <question to answer using indexed documents>\n"
                    )
                    return
            if self._chat_busy:
                self.query_one("#llm-output", RichLog).write("Wait for the current response to finish.\n")
                return
            self._chat_busy = True
            event.input.disabled = True
            self.query_one("#llm-output", RichLog).write(
                f"\nYou: {text}\nAssistant:\n"
            )
            review_checkbox = self.query_one("#review-input-args", Checkbox)
            review_arguments = review_checkbox.value
            review_checkbox.disabled = True
            self._run_chat(text, use_rag=use_rag, review_arguments=review_arguments)
            return

        terminal_id = "command-one" if event.input.id == "command-one-input" else "command-two"
        output_id = f"{terminal_id}-output"
        output = self.query_one(f"#{output_id}", RichLog)
        if text.casefold() == "/help":
            self._show_help()
            output.write("Help displayed in the Editor pane.\n")
            return
        if terminal_id in self._pending_task_runs:
            self._confirm_task_run(terminal_id, text)
            return
        self._record_command_history(terminal_id, text)
        if text.startswith("/ragupload") or text.startswith("/ragscrape"):
            self._handle_console_rag_command(text, output, terminal_id)
            return
        if text.startswith("/collect"):
            self._handle_collector_command(text, output, terminal_id)
            return
        if text.startswith("/task"):
            self._handle_task_command(text, output, terminal_id)
            return
        output.write(f"$ {text}\n")
        self.run_worker(self._run_command(terminal_id, text), name=f"{terminal_id}-command")

    def _show_help(self) -> None:
        """Display the command reference without discarding the Editor's current view."""
        self._close_rca_output()
        widget_ids = (
            "task-create-pane",
            "task-chat-pane",
            "editor-toolbar",
            "editor",
            "editor-status",
            "task-list-pane",
            "command-output-mirror",
            "command-editor-pane",
        )
        if self._help_previous_visibility is None:
            self._help_previous_visibility = {
                widget_id: self.query_one(f"#{widget_id}").display
                for widget_id in widget_ids
            }
        for widget_id in widget_ids:
            self.query_one(f"#{widget_id}").display = False

        self.query_one("#help-pane", Container).display = True
        help_output = self.query_one("#help-output", RichLog)
        help_output.clear()

        table = Table(show_header=True, header_style="bold", show_lines=True, expand=True)
        for heading in ("Command", "Parameter", "Terminal", "Description"):
            table.add_column(heading)
        rows = (
            ("/help", "None", "All three", "Show this command reference in the Editor pane."),
            (
                "/rag",
                "<question>",
                "AI/LLM",
                "Ask a question using indexed documents. If mentioning filenames, put double quotes around a filename with spaces.",
            ),
            ("/command", "<request>", "AI/LLM", "Ask the LLM to propose shell command(s). Review, edit, then confirm before execution."),
            ("/collectlist", "None", "Command 1 or 2 · signed in", "List built-in collectors and custom Tasks with the Collector kind."),
            ("/collectpolicy", "<built-in-name> <policy>", "Command 1 or 2 - signed in", "Administrator/Power User: set manual, ai-assisted, auto-approved or disabled for a built-in collector."),
            ("/collectview", "<collector-name>", "Command 1 or 2", "Open a saved collector in the Task editor. Built-in details require sign-in."),
            ("/collect", "<collector-name> [JSON arguments]", "Command 1 or 2 · signed in", "Run a built-in or custom collector through the authenticated API; custom versions must be approved and enabled."),
            ("/collectrun", "<collector-name> [JSON arguments]", "Command 1 or 2", "Show saved collector code and input arguments, then ask Y/N before running locally. Built-ins also require Y/N. No approval or enabled status required. Saved schemas validate JSON arguments passed in LH_RUNNER_ARGUMENTS_JSON."),
            ("/collectenable", "<collector-name>", "Command 1 or 2 · signed in", "Enable a built-in collector or a custom collector Task version; Users may enable their own Tasks."),
            ("/collectdisable", "<collector-name>", "Command 1 or 2 · signed in", "Disable a built-in collector or a custom collector Task version; Users may disable their own Tasks."),
            (
                "/ragupload",
                "<file-or-folder-path>",
                "Command 1 or 2",
                "Index supported documents. Put double quotes around a file or folder path with spaces. Same path and unchanged content is skipped; matching filename and content at another path is skipped. A same-name file with different content is kept separately; changed content at the same path replaces its indexed chunks.",
            ),
            (
                "/ragscrape",
                "<url>",
                "Command 1 or 2",
                "Crawl same-site pages and linked supported documents. This command takes a URL, not a local filename; encode spaces in a URL as %20. Each normalized URL is visited once per crawl; unchanged content at the same final URL is skipped on later crawls, while changed content replaces its chunks. Identical content at different URLs is not deduplicated.",
            ),
            ("/tasknew", "None", "Command 1 or 2", "Open the Task creation form; choose artifact type, task kind, execution policy, and category."),
            ("/tasklist", "None", "Command 1 or 2", "List saved Tasks with their execution policy in the Editor pane."),
            ("/taskedit", "<taskname>", "Command 1 or 2", "Load a saved Task into the Editor pane to edit code and execution policy. Quote names containing spaces."),
            ("/taskview", "<taskname>", "Command 1 or 2", "Alias for /taskedit."),
            ("/task", "<taskname> [JSON arguments]", "Command 1 or 2 · signed in", "Run the latest saved Task version through the authenticated API; it must be approved and enabled. n8n workflows are not executed."),
            ("/taskrun", "<taskname> [JSON arguments]", "Command 1 or 2", "Load and locally run the latest saved version after confirmation. Validate JSON arguments against its saved schema and pass them in LH_RUNNER_ARGUMENTS_JSON. Local development runs do not require approval or enabled status; n8n is not executed."),
            ("/taskdelete", "<taskname>", "Command 1 or 2", "Delete a Task, its conversation, and saved versions after confirmation. Quote names containing spaces."),
            ("OS command", "<command>", "Command 1 or 2", "Run a native shell command in that terminal."),
        )
        for row in rows:
            table.add_row(*row)
        help_output.write(table)
        help_output.write(
            'Filenames or paths containing spaces: enclose them in double quotes.\n'
            '  /rag summarize "Science and Sports (Ch 5 to 8).pdf"\n'
            '  /ragupload "C:\\docs\\Science and Sports (Ch 5 to 8).pdf"\n'
        )
        help_output.write(
            "RAG content handling: document bytes are SHA-256 hashed before indexing. "
            "URL normalization lowercases the scheme and host, removes fragments, "
            "and normalizes default HTTP/HTTPS ports. Different URLs with identical "
            "content remain separate indexed sources."
        )

    def _handle_collector_command(
        self, command: str, output: RichLog, terminal_id: str
    ) -> None:
        command_name = command.split(maxsplit=1)[0].lower()
        if command_name in {"/collectview", "/collectrun"}:
            try:
                name, arguments = _parse_api_run_command(
                    "/collect " + command.split(maxsplit=1)[1], "/collect")
                if command_name == "/collectview" and arguments:
                    raise ValueError("Usage: /collectview <collector-name>")
                if self._task_chat_busy or self._sessions[terminal_id].busy:
                    raise ValueError("Wait for current Task generation or terminal execution to finish.")
                if self._editor_is_dirty() or self._has_generated_draft:
                    raise ValueError("Save or cancel the current Task edits first.")
                if (self.query_one("#task-create-pane", Container).display
                        or self.query_one("#command-editor-pane", Container).display):
                    raise ValueError("Finish or cancel the current editor form first.")
                if self.current_task and (
                    self.query_one("#loaded-task-description", TextArea).text != self.current_task.get("description", "")
                    or self.query_one("#loaded-task-execution-policy", Select).value != self.current_task.get("execution_policy", "manual")
                ):
                    raise ValueError("Save the current description and policy edits first.")
                task = self._task_workspace.find_task_by_name(name)
                if task is not None:
                    if task.get("task_kind") != "collector":
                        raise ValueError("This name belongs to a Task, not a Collector.")
                    if command_name == "/collectview":
                        self._load_task(task)
                        output.write(f"Loaded Collector: {task['name']} · User ID: {task['userid']} · Category: {task['category']}\n")
                        return
                    artifact = task.get("latest_artifact")
                    if not artifact:
                        raise ValueError("This collector has no saved version.")
                    if task["artifact_type"] not in {"python", "bash", "powershell", "windows-cmd"}:
                        raise ValueError("This collector type cannot run locally.")
                    if artifact.get("arguments_schema") is not None:
                        from loghawk.collectors.contracts import validate_arguments
                        arguments = validate_arguments(artifact["arguments_schema"], arguments)
                    self._load_task(task)
                    artifact = dict(artifact, _run_arguments=arguments, _run_command="/collectrun")
                    self._pending_task_runs[terminal_id] = (task, artifact)
                    output.write(f"Collector: {task['name']} · User ID: {task['userid']} · Version {artifact['version']}\n")
                elif command_name == "/collectview":
                    if self._api_client is None:
                        raise ValueError("Saved collector not found. Built-in details require sign-in.")
                    self._begin_command_output(terminal_id, command)
                    self._run_collector_request(terminal_id, command_name, name, {})
                    return
                else:
                    from loghawk.collectors.registry import _DEFINITIONS
                    if name not in _DEFINITIONS:
                        raise ValueError(f"Collector not found: {name}")
                    self._pending_task_runs[terminal_id] = (
                        {"name": name, "_builtin_collector": True},
                        {"_run_arguments": arguments},
                    )
                    output.write(f"Built-in collector: {name}\n")
                output.write("Input arguments: " + json.dumps(arguments, ensure_ascii=False) + "\nRun locally? (Y/N): ")
            except (ValueError, IndexError, RuntimeError) as exc:
                output.write(f"Collector request: {exc or 'Specify a collector name.'}\n")
            return
        if self._api_client is None and not command.lower().startswith("/collectrun"):
            output.write("Collector commands require a signed-in Hawk Console session; they are unavailable in --local mode.\n")
            return
        parts = command.split(maxsplit=2)
        command_name = parts[0].lower()
        if command_name == "/collect":
            try:
                collector_name, arguments = _parse_api_run_command(command, "/collect")
            except ValueError as exc:
                output.write(f"{exc}\n")
                return
            if self._sessions[terminal_id].busy:
                output.write("Wait for the current command to finish before running a collector.\n")
                return
            self._begin_command_output(terminal_id, command)
            self._run_collector_request(terminal_id, command_name, collector_name, arguments)
            return
        if command_name not in {
            "/collectlist", "/collectview", "/collectrun",
            "/collectenable", "/collectdisable", "/collectpolicy",
        }:
            output.write("Unknown collector command. Use /help.\n")
            return
        if command_name in {"/collectlist"}:
            if len(parts) > 1:
                output.write("Usage: /collectlist\n")
                return
            collector_name, arguments = None, {}
        elif command_name == "/collectpolicy":
            try:
                policy_parts = shlex.split(command)
            except ValueError as exc:
                output.write(f"Invalid collector policy command: {exc}\n")
                return
            if len(policy_parts) != 3 or policy_parts[2] not in {"manual", "ai-assisted", "auto-approved", "disabled"}:
                output.write("Usage: /collectpolicy <built-in-name> manual|ai-assisted|auto-approved|disabled\n")
                return
            collector_name, arguments = policy_parts[1], {"execution_policy": policy_parts[2]}
        elif command_name == "/collectrun":
            if len(parts) < 2:
                output.write('Usage: /collectrun <collector-name> [JSON arguments]\nExample: /collectrun prometheus-query {"query":"up"}\n')
                return
            collector_name = parts[1]
            try:
                arguments = json.loads(parts[2]) if len(parts) == 3 else {}
            except json.JSONDecodeError as exc:
                output.write(f"Arguments must be valid JSON: {exc}\n")
                return
            if not isinstance(arguments, dict):
                output.write("Collector arguments must be a JSON object.\n")
                return
        else:
            if len(parts) != 2:
                output.write(f"Usage: {command_name} <collector-name>\n")
                return
            collector_name, arguments = parts[1], {}

        self._begin_command_output(terminal_id, command)
        self._run_collector_request(terminal_id, command_name, collector_name, arguments)

    @work(thread=True, group="hawk-collectors")
    def _run_collector_request(
        self,
        terminal_id: str,
        command_name: str,
        collector_name: str | None,
        arguments: dict,
    ) -> None:
        try:
            if command_name == "/collectrun":
                # Saved collectors use the captured artifact through _run_task_artifact.
                result = lhrunner.run_builtin_collector(collector_name or "", arguments)
            elif self._api_client is None:
                raise RuntimeError("This collector command requires a signed-in Hawk Console session.")
            elif command_name == "/collectlist":
                result = self._api_client.get_json("/api/v1/console/collectors")
            elif command_name == "/collectpolicy":
                result = self._api_client.put_json(
                    f"/api/v1/console/collectors/{quote(collector_name or '', safe='')}/execution-policy",
                    arguments,
                )
            elif command_name == "/collectview":
                result = self._api_client.get_json(f"/api/v1/console/collectors/{collector_name}")
            elif command_name == "/collect":
                result = self._api_client.post_json(
                    f"/api/v1/console/collectors/{quote(collector_name or '', safe='')}/runs",
                    {"arguments": arguments},
                    timeout=330,
                )
            else:
                result = self._api_client.put_json(
                    f"/api/v1/console/collectors/{collector_name}",
                    {"enabled": command_name == "/collectenable"},
                )
            rendered = _format_api_run_result(result)
        except Exception as exc:
            rendered = f"Collector request failed: {exc}"
        self.app.call_from_thread(self._finish_api_run_request, terminal_id, rendered)

    @work(thread=True, group="hawk-task-runs")
    def _run_api_task(self, terminal_id: str, task_name: str, arguments: dict) -> None:
        try:
            task = self._task_workspace.find_task_by_name(task_name)
            if task is None:
                raise RuntimeError(f"Task not found: {task_name}")
            if task.get("artifact_type") == "n8n":
                raise RuntimeError(
                    "n8n workflows are not executed by LogHawk; import them into the customer-managed n8n instance."
                )
            result = self._api_client.post_json(
                f"/api/v1/tasks/{task['id']}/runs",
                {"arguments": arguments},
                timeout=330,
            )
            rendered = _format_api_run_result(result)
        except Exception as exc:
            rendered = f"Task API run failed: {exc}"
        self.app.call_from_thread(self._finish_api_run_request, terminal_id, rendered)

    def _finish_api_run_request(self, terminal_id: str, rendered: str) -> None:
        label = "Terminal 1" if terminal_id == "command-one" else "Terminal 2"
        for line in rendered.splitlines():
            self.query_one(f"#{terminal_id}-output", RichLog).write(line)
            self.query_one("#command-output-mirror", RichLog).write(f"[{label}] {line}")

    def _close_help(self) -> None:
        self.query_one("#help-pane", Container).display = False
        if self._help_previous_visibility is not None:
            for widget_id, was_visible in self._help_previous_visibility.items():
                self.query_one(f"#{widget_id}").display = was_visible
            self._help_previous_visibility = None

    def _handle_console_rag_command(
        self, command: str, output: RichLog, terminal_id: str
    ) -> None:
        try:
            parts = shlex.split(command, posix=os.name != "nt")
        except ValueError as exc:
            output.write(f"Invalid RAG command: {exc}\n")
            return
        command_name = parts[0].lower() if parts else ""
        if command_name == "/ragupload":
            if len(parts) != 2:
                output.write("Usage: /ragupload <file-or-folder-path>\n")
                return
            path_value = parts[1].strip('"')
            path = Path(path_value).expanduser()
            if not path.exists():
                output.write(f"RAG upload path does not exist: {path}\n")
                return
            if path.is_file():
                files = (
                    [path]
                    if path.suffix.lower() in lancedbutils3.SUPPORTED_SUFFIXES
                    else []
                )
            else:
                files = [
                    item for item in sorted(path.rglob("*"))
                    if item.is_file()
                    and item.suffix.lower() in lancedbutils3.SUPPORTED_SUFFIXES
                ]
            if not files:
                output.write("No supported documents found at that path.\n")
                return
            output.write(f"Indexing {len(files)} document(s) into Hawk Console RAG…\n")
            if self._api_client is not None:
                source_root = path if path.is_dir() else path.parent
                payload = [
                    (str(item), item.relative_to(source_root).as_posix())
                    for item in files
                ]
            else:
                payload = [(str(item), str(item.resolve())) for item in files]
            self._run_console_rag(
                terminal_id,
                "UPLOAD",
                payload,
            )
            return
        if command_name == "/ragscrape":
            if len(parts) != 2:
                output.write("Usage: /ragscrape <url>\n")
                return
            url = parts[1].strip('"')
            output.write(f"Crawling {url} into Hawk Console RAG…\n")
            self._run_console_rag(terminal_id, "WEB", url)
            return
        output.write("Unknown RAG command. Use /ragupload or /ragscrape.\n")

    @work(thread=True, exclusive=True, group="console-rag")
    def _run_console_rag(
        self, terminal_id: str, kind: str, payload: list[tuple[str, str]] | str
    ) -> None:
        try:
            if self._api_client is not None:
                if kind == "UPLOAD":
                    job = self._api_client.upload_rag_files(payload)
                else:
                    job = self._api_client.post_json(
                        "/api/v1/rag/ingestions/web", {"url": payload}
                    )
                job = self._api_client.wait_for_ingestion(job["id"])
                message = (
                    f"RAG {kind.lower()} {job['status'].lower()}: "
                    f"{job.get('files_processed', 0)} item(s), "
                    f"{job.get('chunks_indexed', 0)} chunk(s) indexed."
                )
                if job.get("error"):
                    message += f"\n{job['error']}"
                if job.get("warnings"):
                    message += "\n" + "\n".join(job["warnings"])
                self.app.call_from_thread(
                    self._write_console_rag_output, terminal_id, message
                )
                return
            elif kind == "UPLOAD":
                chunks, processed, warnings = ingestion.ingest_file_paths(
                    payload,
                    console_rag_store,
                    lancedb_path=config.LH_CONSOLE_LANCEDB_FILE_PATH,
                    table_name=config.LH_CONSOLE_LANCEDB_TABLE_NAME,
                )
            else:
                chunks, processed, warnings = ingestion.ingest_website(
                    payload,
                    console_rag_store,
                    lancedb_path=config.LH_CONSOLE_LANCEDB_FILE_PATH,
                    table_name=config.LH_CONSOLE_LANCEDB_TABLE_NAME,
                )
            message = (
                f"RAG {kind.lower()} completed: {processed} item(s), "
                f"{chunks} chunk(s) indexed."
            )
            if warnings:
                message += "\n" + "\n".join(warnings)
        except Exception as exc:
            message = f"RAG {kind.lower()} failed: {type(exc).__name__}: {exc}"
        self.app.call_from_thread(
            self._write_console_rag_output,
            terminal_id,
            message,
        )

    def _write_console_rag_output(self, terminal_id: str, message: str) -> None:
        self.query_one(f"#{terminal_id}-output", RichLog).write(message + "\n")

    def _record_command_history(self, terminal_id: str, command: str) -> None:
        history = self._command_history[terminal_id]
        history.append(command)
        del history[:-command_history.MAX_HISTORY_ENTRIES]
        self._command_history_index[terminal_id] = len(history)
        self._command_history_draft[terminal_id] = ""
        if self._api_client is not None:
            self._pending_command_history.append(
                {
                    "id": str(uuid4()), "terminal_id": terminal_id,
                    "command": command, "created_at": datetime.now(UTC).isoformat(),
                }
            )
        else:
            try:
                command_history.save_command(terminal_id, command)
            except Exception as exc:
                self.query_one(f"#{terminal_id}-output", RichLog).write(
                    f"Could not persist command history: {exc}\n"
                )

    def on_key(self, event: events.Key) -> None:
        if event.key not in {"up", "down"}:
            return
        focused = self.focused
        if not isinstance(focused, Input) or focused.id not in {
            "command-one-input",
            "command-two-input",
        }:
            return

        terminal_id = (
            "command-one" if focused.id == "command-one-input" else "command-two"
        )
        history = self._command_history[terminal_id]
        index = self._command_history_index[terminal_id]
        if event.key == "up":
            if index == len(history):
                self._command_history_draft[terminal_id] = focused.value
            if index > 0:
                index -= 1
                focused.value = history[index]
        elif index < len(history) - 1:
            index += 1
            focused.value = history[index]
        elif index == len(history) - 1 and history:
            index = len(history)
            focused.value = self._command_history_draft[terminal_id]

        self._command_history_index[terminal_id] = index
        focused.cursor_position = len(focused.value)
        event.stop()
        event.prevent_default()

    @work(thread=True, exclusive=True, group="hawk-command-assist")
    def _propose_command(
        self, request: str, chat_history: list[dict[str, str]]
    ) -> None:
        try:
            if self._api_client is not None:
                proposal = self._api_client.post_json(
                    "/api/v1/console/command",
                    {
                        "request": request,
                        "messages": chat_history,
                        "shell_name": "PowerShell" if os.name == "nt" else "Bash",
                    },
                    timeout=300,
                )
                command = proposal.get("command")
                if command is not None and (
                    not isinstance(command, str) or not command.strip()
                ):
                    raise ValueError("The model returned an invalid command.")
                self.app.call_from_thread(
                    self._finish_remote_command_proposal,
                    request,
                    proposal.get("explanation", ""),
                    command.strip() if command else None,
                )
                return
            shell_name = "PowerShell" if os.name == "nt" else "Bash"
            schema = {
                "type": "object",
                "properties": {
                    "explanation": {"type": "string"},
                    "command": {"type": ["string", "null"]},
                },
                "required": ["explanation", "command"],
                "additionalProperties": False,
            }
            content = get_llm_client().complete(
                [
                    {
                        "role": "system",
                        "content": (
                            f"Draft one {shell_name} command for the user's request. "
                            "Do not execute commands or claim they were executed. "
                            "Return a brief explanation and the exact command. "
                            "Set command to null if clarification is needed. "
                            "Return only data matching the response schema."
                        ),
                    },
                    *chat_history,
                    {"role": "user", "content": request},
                ],
                response_schema=schema,
                model=config.LH_LLM_CODEASSIST_MODEL or config.LH_LLM_MODEL,
                temperature=0.1,
                max_tokens=config.LH_LLM_CODEASSIST_MAX_TOKENS,
            )
            proposal = json.loads(content)
            command = proposal.get("command")
            if command is not None and (
                not isinstance(command, str) or not command.strip()
            ):
                raise ValueError("The model returned an invalid command.")
            self.app.call_from_thread(
                self._finish_command_proposal,
                proposal.get("explanation", ""),
                command.strip() if command else None,
                None,
            )
        except Exception as exc:
            self.app.call_from_thread(
                self._finish_command_proposal, "", None, str(exc)
            )

    def _finish_remote_command_proposal(
        self, request: str, explanation: str, command: str | None
    ) -> None:
        assistant_message = explanation
        if command:
            assistant_message += f"\nProposed command:\n{command}"
        self._chat_history.extend([
            {"role": "user", "content": request},
            {"role": "assistant", "content": assistant_message},
        ])
        self._persist_chat_message("user", request)
        self._persist_chat_message("assistant", assistant_message)
        self._finish_command_proposal(explanation, command, None)

    def _finish_command_proposal(
        self, explanation: str, command: str | None, error: str | None
    ) -> None:
        output = self.query_one("#llm-output", RichLog)
        if error:
            output.write(f"Could not prepare command: {error}\n")
        elif command is None:
            output.write(f"{explanation}\nNo command is ready to run.\n")
        else:
            self._pending_llm_command = command
            output.write(
                f"{explanation}\n"
                "Proposed command for Command Terminal 1:\n"
                f"{command}\n"
                "Type YES or yes to run it, E or e to edit it in the Editor pane, "
                "or N or anything else to cancel: "
            )
        self._chat_busy = False
        self.query_one("#llm-input", Input).disabled = False
        self.query_one("#llm-input", Input).focus()

    def _confirm_llm_command(self, answer: str) -> None:
        output = self.query_one("#llm-output", RichLog)
        response = answer.strip().casefold()
        if response == "e" and not self._awaiting_llm_command_final_confirmation:
            self._open_command_editor()
            return
        if response != "yes":
            self._pending_llm_command = None
            self._awaiting_llm_command_final_confirmation = False
            output.write("Command canceled.\n")
            return
        session = self._sessions["command-one"]
        if session.busy:
            output.write(
                "Command Terminal 1 is busy. The proposal is still pending; "
                "type YES or yes again after it finishes, or N or anything else to cancel.\n"
            )
            return
        command = self._pending_llm_command
        self._pending_llm_command = None
        self._awaiting_llm_command_final_confirmation = False
        self._record_command_history("command-one", command)
        output.write("Sending the approved command to Command Terminal 1.\n")
        self.query_one("#command-one-output", RichLog).write(f"$ {command}\n")
        self.run_worker(
            self._run_command("command-one", command),
            name="command-one-llm-approved",
        )

    def _open_command_editor(self) -> None:
        self._close_rca_output()
        if self._pending_llm_command is None:
            return
        editor_pane = self.query_one("#command-editor-pane", Container)
        if editor_pane.display:
            return
        widget_ids = (
            "task-create-pane",
            "task-chat-pane",
            "editor-toolbar",
            "editor",
            "editor-status",
            "command-output-mirror",
        )
        self._command_editor_visibility = {
            widget_id: self.query_one(f"#{widget_id}").display
            for widget_id in widget_ids
        }
        for widget_id in widget_ids:
            self.query_one(f"#{widget_id}").display = False
        self.query_one("#command-task-name", Input).value = ""
        self.query_one("#command-task-description", Input).value = ""
        self.query_one("#command-task-type", Select).clear()
        self.query_one("#command-task-category", Input).value = ""
        self.query_one("#command-task-kind", Select).value = "general"
        self.query_one("#command-task-execution-policy", Select).value = "manual"
        self.query_one("#command-editor", TextArea).text = self._pending_llm_command
        editor_pane.display = True
        self.query_one("#llm-input", Input).disabled = True
        self.query_one("#command-editor", TextArea).focus()

    def _close_command_editor(self, *, focus_llm: bool = True) -> None:
        self.query_one("#command-editor-pane", Container).display = False
        for widget_id, was_displayed in self._command_editor_visibility.items():
            self.query_one(f"#{widget_id}").display = was_displayed
        self._command_editor_visibility.clear()
        self.query_one("#llm-input", Input).disabled = False
        if focus_llm:
            self.query_one("#llm-input", Input).focus()

    def _confirm_command_edits(self) -> None:
        command = self.query_one("#command-editor", TextArea).text.strip()
        if not command:
            self.notify("Command cannot be empty.", severity="warning")
            self.query_one("#command-editor", TextArea).focus()
            return
        self._pending_llm_command = command
        self._awaiting_llm_command_final_confirmation = True
        self._close_command_editor()
        self.query_one("#llm-output", RichLog).write(
            "Updated command for Command Terminal 1:\n"
            f"{command}\n"
            "Type YES or yes to run it, or N or anything else to cancel: "
        )

    def _quick_save_command_task(self) -> None:
        name = self.query_one("#command-task-name", Input).value.strip()
        artifact_type_select = self.query_one("#command-task-type", Select)
        task_kind = str(self.query_one("#command-task-kind", Select).value)
        execution_policy = str(self.query_one("#command-task-execution-policy", Select).value)
        category = self.query_one("#command-task-category", Input).value.strip()
        description = self.query_one("#command-task-description", Input).value.strip()
        content = self.query_one("#command-editor", TextArea).text

        if not name:
            self.notify("Task name is required.", severity="warning")
            self.query_one("#command-task-name", Input).focus()
            return
        if not description:
            self.notify("Task description is required.", severity="warning")
            self.query_one("#command-task-description", Input).focus()
            return
        if artifact_type_select.is_blank():
            self.notify("Select a Task type.", severity="warning")
            artifact_type_select.focus()
            return
        if not category:
            self.notify("Task category is required.", severity="warning")
            self.query_one("#command-task-category", Input).focus()
            return
        if not content.strip():
            self.notify("Task code cannot be empty.", severity="warning")
            self.query_one("#command-editor", TextArea).focus()
            return

        artifact_type = str(artifact_type_select.value)
        if task_kind == "collector" and artifact_type == "n8n":
            self.notify("Collector Tasks must use Python, Bash, PowerShell, or Windows CMD.", severity="error")
            return
        if artifact_type == "python":
            try:
                ast.parse(content)
            except SyntaxError as exc:
                self.notify(f"Python syntax error: {exc}", severity="error")
                self.query_one("#command-editor", TextArea).focus()
                return

        try:
            task = self._task_workspace.find_task_by_name(name)
            if task is None:
                task = self._task_workspace.create_task(
                    name, description, category, artifact_type, task_kind, execution_policy
                )
            else:
                existing_category = task.get("category") or "Uncategorized"
                if task["artifact_type"] != artifact_type:
                    self.notify(
                        "This Task name already exists with a different type.",
                        severity="error",
                    )
                    return
                if existing_category.casefold() != category.casefold():
                    self.notify(
                        "This Task name already exists in a different category.",
                        severity="error",
                    )
                    return
                if task.get("task_kind", "general") != task_kind:
                    self.notify("This Task name already exists with a different task kind.", severity="error")
                    return
                task = self._task_workspace.set_description(task["id"], description)
                if task is None:
                    self.notify("Could not update Task description.", severity="error")
                    return
                task = self._task_workspace.set_execution_policy(task["id"], execution_policy)
                if task is None:
                    self.notify("Could not update Task execution policy.", severity="error")
                    return

            saved_task = self._task_workspace.save_artifact(task["id"], content, arguments_schema=(task.get("latest_artifact") or {}).get("arguments_schema"))
            if saved_task is None or saved_task.get("latest_artifact") is None:
                self.notify("Could not save the Task version.", severity="error")
                return
            version = saved_task["latest_artifact"]["version"]
            self.notify(f'Saved "{saved_task["name"]}" as Version {version}.')
        except Exception as exc:
            self.notify(f"Could not quick-save Task: {exc}", severity="error")

    def _cancel_command_edits(self) -> None:
        self._pending_llm_command = None
        self._awaiting_llm_command_final_confirmation = False
        self._close_command_editor()
        self.query_one("#llm-output", RichLog).write("Command canceled.\n")

    @work(thread=True, exclusive=True, group="hawk-llm")
    def _run_chat(self, prompt: str, use_rag: bool = False, review_arguments: bool = True) -> None:
        response_parts: list[str] = []
        try:
            if self._api_client is not None:
                if use_rag:
                    result = self._api_client.post_json(
                        "/api/v1/rag/chat", {"question": prompt}, timeout=300
                    )
                    response = result["answer"]
                else:
                    messages = [*self._chat_history, {"role": "user", "content": prompt}]
                    result = self._api_client.post_json(
                        "/api/v1/console/chat",
                        {"messages": messages, "review_arguments": review_arguments},
                        timeout=max(300, config.LH_LLM_TOOL_TIMEOUT_SECONDS
                                + config.LH_LLM_TOOL_EXPLANATION_TIMEOUT_SECONDS + 30)
                    )
                    response = result["response"]
                if not response:
                    raise RuntimeError("The configured LLM returned an empty response.")
                self._chat_history.extend([
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ])
                self._persist_chat_message("user", prompt)
                self._persist_chat_message("assistant", response)
                if not use_rag:
                    self.app.call_from_thread(self._finish_tool_chat, result)
                else:
                    self.app.call_from_thread(self._finish_chat, None, response)
                return
            try:
                context = ""
                if use_rag:
                    context, retrieval_notice = self._retrieve_context(prompt)
                    if retrieval_notice:
                        self.app.call_from_thread(
                            self._finish_chat, None, retrieval_notice
                        )
                        return
            except Exception as exc:
                context = ""
                self.app.call_from_thread(
                    self._write_chat_notice,
                    f"\n[RAG retrieval skipped: {exc}]\n",
                )
            user_content = prompt
            if context:
                user_content = (
                    "The following text came from indexed documents. Use it only "
                    "when it directly helps answer the question. If it is unrelated, "
                    "disregard it and answer from general knowledge.\n\n"
                    f"Context:\n{context}\n\nQuestion:\n{prompt}"
                )
            messages = [
                *self._chat_history,
                {"role": "user", "content": user_content},
            ]
            if not use_rag:
                matching_tasks = local_task_workspace.relevant_tasks(prompt)
                if matching_tasks:
                    task_context = "\n".join(
                        "- "
                        f"name={task['name']}; description={task.get('description', '')}; "
                        f"category={task.get('category', '')}; kind={task.get('task_kind', 'general')}; "
                        f"type={task.get('artifact_type', '')}; policy={task.get('execution_policy', 'manual')}; "
                        f"latest_version={(task.get('latest_artifact') or {}).get('version', 'none')}; "
                        f"approval_status={'approved' if (task.get('latest_artifact') or {}).get('approved') else 'not approved'}; "
                        f"enabled_status={'enabled' if (task.get('latest_artifact') or {}).get('enabled') else 'disabled'}"
                        for task in matching_tasks
                    )
                    messages.insert(0, {
                        "role": "system",
                        "content": (
                            "Potentially relevant saved LogHawk Tasks are listed below. Use them to answer "
                            "questions about existing tasks or recommend a suitable task. Do not claim "
                            "to run, approve, enable, or modify a task. State its execution policy and "
                            "latest-version approval/enabled status when relevant.\n" + task_context
                        ),
                    })
            client = get_llm_client()
            candidates = [config.LH_LLM_MODEL, *config.LH_LLM_FALLBACK_MODELS]
            last_error: Exception | None = None
            for model in dict.fromkeys(name for name in candidates if name):
                try:
                    for chunk in client.stream(messages, model=model):
                        response_parts.append(chunk)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if response_parts:
                        break
                    response_parts.clear()
            if last_error is not None:
                raise last_error
            response = "".join(response_parts)
            if not response:
                raise RuntimeError("The configured LLM returned an empty response.")
            self._chat_history.append({"role": "user", "content": user_content})
            self._chat_history.append({"role": "assistant", "content": response})
            self._persist_chat_message("user", prompt)
            self._persist_chat_message("assistant", response)
            self.app.call_from_thread(self._finish_chat, None, response)
        except Exception as exc:
            self.app.call_from_thread(self._finish_chat, str(exc), None)

    def _write_chat_notice(self, notice: str) -> None:
        self.query_one("#llm-output", RichLog).write(notice)

    def _finish_tool_chat(self, result: dict) -> None:
        if result.get("execution"):
            self._latest_rca_output = self._format_rca_output(result)
            self._rca_retry_request_id = (
                result.get("request_id") if (result.get("explanation") or {}).get("can_retry") else None
            )
            self.query_one("#show-rca-output", Button).disabled = False
            if self.query_one("#rca-output-pane", Container).display:
                self.query_one("#rca-output-text", TextArea).load_text(self._latest_rca_output)
        from rich.text import Text

        execution = result.get("execution")
        if execution:
            output = self.query_one("#llm-output", RichLog)
            lines = [
                "Execution status · reported by LogHawk server",
                str(execution.get("summary", "")),
                f"Discovery calls: {execution.get('discovery_calls', 0)} · "
                f"Request: {execution.get('request_id', result.get('request_id', ''))}",
            ]
            for event in execution.get("events", []):
                collector = event.get("collector", "unknown")
                label = "Tool request" if collector == "unknown" else collector
                version = event.get("version")
                if version:
                    label += f" · Version {version}"
                lines.append(f"- {label} — {event.get('status', 'UNKNOWN')}")
                if event.get("run_id"):
                    lines.append(f"  Run: {event['run_id']}")
                if event.get("reason"):
                    lines.append(f"  {event['reason']}")
            lines.append("The assistant's narrative below is not execution confirmation.")
            output.write(Text("\n".join(lines) + "\n"))
        self._finish_chat(None, result.get("response", ""))
        if result.get("status") == "PENDING":
            self.query_one("#review-input-args", Checkbox).disabled = True
            self._pending_collector_call = {
                "request_id": result["request_id"], **result["pending"],
            }
            self.query_one("#llm-input", Input).disabled = True
            self.push_screen(CollectorArgumentReviewScreen(self._pending_collector_call),
                             callback=self._collector_review_closed)

    @staticmethod
    def _format_rca_output(result: dict) -> str:
        execution = result.get("execution") or {}
        lines = ["RCA Output", "Execution status · reported by LogHawk server",
                 f"Request: {execution.get('request_id', result.get('request_id', ''))}",
                 str(execution.get("summary", "")),
                 f"Discovery calls: {execution.get('discovery_calls', 0)}"]
        for event in execution.get("events", []):
            lines.extend(["", f"Collector: {event.get('collector', 'unknown')}",
                          f"Version: {event.get('version', '')}",
                          f"Status: {event.get('status', 'UNKNOWN')}"])
            if event.get("run_id"):
                lines.append(f"Run: {event['run_id']}")
            if event.get("reason"):
                lines.append(str(event["reason"]))
            if event.get("arguments") is not None:
                lines.extend(["Input arguments (redacted):",
                              json.dumps(event["arguments"], indent=2, ensure_ascii=False)])
            if event.get("evidence"):
                evidence = event["evidence"]
                try:
                    evidence = json.dumps(json.loads(evidence), indent=2, ensure_ascii=False)
                except (ValueError, TypeError):
                    pass
                lines.extend(["", "Collector evidence (redacted; may be truncated):", evidence])
        lines.extend(["", "AI/LLM explanation (not execution confirmation):",
                      str(result.get("response", "")), "",
                      "Saved collector evidence is independent of the compact model input; source output may be limited."])
        return "\n".join(lines)

    def _new_chat(self) -> None:
        if (
            self._chat_busy
            or self._pending_collector_call is not None
            or self._pending_llm_command is not None
            or self._awaiting_llm_command_final_confirmation
        ):
            self.notify(
                "Wait for the current request or finish/cancel its confirmation before starting a new chat.",
                severity="warning",
            )
            return
        self._chat_history.clear()
        self._latest_rca_output = ""
        self._rca_retry_request_id = None
        self.query_one("#show-rca-output", Button).disabled = True
        self.query_one("#retry-rca-explanation", Button).disabled = True
        self._close_rca_output()
        output = self.query_one("#llm-output", RichLog)
        output.clear()
        output.write(
            "New chat started. Previous conversation context has been cleared.\n"
            "Saved history is retained. Messages awaiting synchronization are retained.\n"
        )
        # Preserve any question already typed by the user.
        self.query_one("#llm-input", Input).focus()

    def _show_rca_output(self) -> None:
        if not self._latest_rca_output:
            return
        if (self._task_chat_busy or self._pending_task_runs or self._pending_collector_call
                or self._pending_llm_command is not None
                or self.query_one("#command-editor-pane", Container).display
                or self.query_one("#task-create-pane", Container).display
                or self._editor_is_dirty() or self._has_generated_draft):
            self.notify("Finish, save, or cancel current edits or pending actions before viewing RCA output.",
                        severity="warning")
            return
        if self.current_task and (
            self.query_one("#loaded-task-description", TextArea).text != self.current_task.get("description", "")
            or self.query_one("#loaded-task-execution-policy", Select).value
                != self.current_task.get("execution_policy", "manual")
        ):
            self.notify("Save or cancel Task metadata edits before viewing RCA output.", severity="warning")
            return
        self._close_help()
        widget_ids = ("task-create-pane", "task-chat-pane", "editor-toolbar", "editor",
                      "editor-status", "task-list-pane", "command-output-mirror", "command-editor-pane")
        if self._rca_previous_visibility is None:
            self._rca_previous_visibility = {
                widget_id: self.query_one(f"#{widget_id}").display for widget_id in widget_ids
            }
        for widget_id in widget_ids:
            self.query_one(f"#{widget_id}").display = False
        self.query_one("#rca-output-text", TextArea).load_text(self._latest_rca_output)
        self.query_one("#rca-output-pane", Container).display = True

    def _retry_rca_explanation(self) -> None:
        if self._api_client is None or not self._rca_retry_request_id:
            return
        if (self._chat_busy or self._pending_collector_call or self._pending_llm_command
                or self._awaiting_llm_command_final_confirmation):
            self.notify("Finish the current chat or confirmation before retrying the explanation.", severity="warning")
            return
        self._chat_busy = True
        self.query_one("#llm-input", Input).disabled = True
        self.query_one("#review-input-args", Checkbox).disabled = True
        self.query_one("#retry-rca-explanation", Button).disabled = True
        self.query_one("#llm-output", RichLog).write("Retrying AI explanation from saved evidence. No collectors will run.\n")
        self._run_explanation_retry(self._rca_retry_request_id)

    @work(thread=True, group="hawk-explanation-retry")
    def _run_explanation_retry(self, request_id: str) -> None:
        try:
            result = self._api_client.post_json(
                f"/api/v1/console/chat/{request_id}/explanation/retry", {},
                timeout=config.LH_LLM_TOOL_EXPLANATION_TIMEOUT_SECONDS + 30,
            )
        except Exception as exc:
            self.app.call_from_thread(
                self._finish_chat, f"Explanation retry failed: {exc}. This retry does not execute collectors.", None,
            )
            return
        response = result.get("response", "")
        try:
            self._chat_history.append({"role": "assistant", "content": response})
            self._persist_chat_message("assistant", response)
        except Exception as exc:
            self.app.call_from_thread(self._write_chat_notice, f"Could not save explanation history: {exc}")
        self.app.call_from_thread(self._finish_tool_chat, result)

    def _close_rca_output(self) -> None:
        self.query_one("#rca-output-pane", Container).display = False
        self.query_one("#rca-output-text", TextArea).load_text("")
        if self._rca_previous_visibility is not None:
            for widget_id, was_visible in self._rca_previous_visibility.items():
                self.query_one(f"#{widget_id}").display = was_visible
            self._rca_previous_visibility = None

    def _download_rca_output(self) -> None:
        content = self.query_one("#rca-output-text", TextArea).text
        if content:
            self.push_screen(DownloadPathScreen("loghawk-rca-output.txt"),
                             callback=lambda destination: self._save_rca_output(destination, content))

    def _save_rca_output(self, destination: str | None, content: str) -> None:
        if not destination:
            return
        path = Path(destination).expanduser()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as stream:
                stream.write(content)
            self.notify(f"Downloaded RCA output to {path}")
        except OSError as exc:
            self.notify(f"Could not download RCA output: {exc}", severity="error")

    def _collector_review_closed(self, result: dict) -> None:
        self._pending_collector_call = None
        self._finish_tool_chat(result)

    @work(thread=True, group="hawk-tool-confirmation")
    def _confirm_collector_call(self, pending: dict, approved: bool,
                                arguments: dict | None, review: CollectorArgumentReviewScreen) -> None:
        try:
            result = self._api_client.post_json(
                f"/api/v1/console/chat/{pending['request_id']}/confirm",
                {"confirmation_id": pending["confirmation_id"], "approved": approved, "arguments": arguments},
                timeout=max(300, config.LH_LLM_TOOL_TIMEOUT_SECONDS
                                + config.LH_LLM_TOOL_EXPLANATION_TIMEOUT_SECONDS + 30),
            )
        except Exception as exc:
            # ConsoleApiClient prefixes HTTP validation errors with their status.
            correctable = isinstance(exc, RuntimeError) and str(exc).startswith("LogHawk API returned 422:")
            self.app.call_from_thread(review.show_error, str(exc), correctable)
            return
        action = ("Reviewed input arguments for" if pending.get("review_arguments", True)
                  else "Confirmed execution of") if approved else "Cancelled"
        decision = f"{action} collector {pending['collector']} version {pending['version']}"
        response = result.get("response", "")
        try:
            self._chat_history.extend([{"role": "user", "content": decision}, {"role": "assistant", "content": response}])
            self._persist_chat_message("user", decision)
            self._persist_chat_message("assistant", response)
        except Exception as exc:
            # Execution already returned: a history-save error must not offer execution again.
            self.app.call_from_thread(self._write_chat_notice, f"Could not save collector chat history: {exc}")
        self.app.call_from_thread(review.dismiss, result)

    def _finish_chat(self, error: str | None, response: str | None) -> None:
        self.query_one("#review-input-args", Checkbox).disabled = self._api_client is None
        self.query_one("#retry-rca-explanation", Button).disabled = (
            self._api_client is None or not self._rca_retry_request_id
        )
        output = self.query_one("#llm-output", RichLog)
        if response:
            output.write(response)
        if error:
            output.write(f"\nLLM chat failed: {error}")
        output.write("\n\n")
        self.query_one("#llm-input", Input).disabled = False
        self.query_one("#llm-input", Input).focus()
        self._chat_busy = False

    async def _run_command(self, terminal_id: str, command: str) -> None:
        session = self._sessions[terminal_id]
        self._begin_command_output(terminal_id, command)
        try:
            await session.submit(command)
        except Exception as exc:
            message = f"Command failed: {exc}"
            self.query_one(f"#{terminal_id}-output", RichLog).write(f"{message}\n")
            self.query_one("#command-output-mirror", RichLog).write(message)
            self._active_command_outputs.discard(terminal_id)

    def _begin_command_output(self, terminal_id: str, command: str) -> None:
        self._close_rca_output()
        if self.query_one("#command-editor-pane", Container).display:
            self._pending_llm_command = None
            self._awaiting_llm_command_final_confirmation = False
            self._close_command_editor(focus_llm=False)
            self.query_one("#llm-output", RichLog).write(
                "Pending /command proposal canceled because a Command Terminal command was run.\n"
            )
        mirror = self.query_one("#command-output-mirror", RichLog)
        if not mirror.display:
            mirror.clear()
        for widget_id in (
            "task-create-pane",
            "task-chat-pane",
            "editor-toolbar",
            "editor",
            "editor-status",
            "task-list-pane",
            "command-editor-pane",
        ):
            self.query_one(f"#{widget_id}").display = False
        mirror.display = True
        self._active_command_outputs.add(terminal_id)
        label = "Terminal 1" if terminal_id == "command-one" else "Terminal 2"
        mirror.write(f"[{label}] $ {command}")

    def _handle_command_output(self, terminal_id: str, text: str) -> None:
        self.query_one(f"#{terminal_id}-output", RichLog).write(text)
        if (
            terminal_id not in self._active_command_outputs
            or text.startswith("Shell ready:")
        ):
            return
        mirror = self.query_one("#command-output-mirror", RichLog)
        label = "Terminal 1" if terminal_id == "command-one" else "Terminal 2"
        for line in text.splitlines():
            mirror.write(f"[{label}] {line}")
        if "[command finished]" in text or "[ shell session ended ]" in text:
            self._active_command_outputs.discard(terminal_id)

    def _handle_task_command(
        self, command: str, output: RichLog, terminal_id: str
    ) -> None:
        try:
            parts = shlex.split(command)
        except ValueError as exc:
            output.write(f"Invalid /task command: {exc}\n")
            return
        command_name = parts[0].lower() if parts else ""
        if command_name == "/task":
            if self._api_client is None:
                output.write("/task requires a signed-in Hawk Console session; use /taskrun for local development.\n")
                return
            try:
                task_name, arguments = _parse_api_run_command(command, "/task")
            except ValueError as exc:
                output.write(f"{exc}\n")
                return
            if self._sessions[terminal_id].busy:
                output.write("Wait for the current command to finish before running a Task.\n")
                return
            self._begin_command_output(terminal_id, command)
            self._run_api_task(terminal_id, task_name, arguments)
            return
        if command_name == "/tasklist":
            if len(parts) != 1:
                output.write("Usage: /tasklist\n")
                return
            if self._task_chat_busy:
                output.write("Wait for the current Task assistant response before listing Tasks.\n")
                return
            if self.current_task and (
                self._editor_is_dirty() or self._has_generated_draft
            ):
                output.write("Save or cancel the current Task edits before displaying the Task list.\n")
                return
            self._show_task_list(output)
            return
        if command_name == "/tasknew":
            if len(parts) != 1:
                output.write("Usage: /tasknew\n")
                return
            if self.current_task and (
                self._editor_is_dirty() or self._has_generated_draft
            ):
                output.write("Save or cancel the current Task edits before starting a new Task.\n")
                return
            if self._task_chat_busy:
                output.write("Wait for the current Task assistant response before starting a new Task.\n")
                return
            self._begin_task_creation()
            output.write("Task creation form opened in the Editor pane.\n")
            return
        if command_name == "/taskdelete":
            if len(parts) != 2:
                output.write(
                    '/taskdelete <taskname> (quote names containing spaces)\n'
                )
                return
            if self._task_chat_busy:
                output.write(
                    "Wait for the current Task assistant response before deleting a Task.\n"
                )
                return
            if self.current_task and (
                self._editor_is_dirty() or self._has_generated_draft
            ):
                output.write(
                    "Save or cancel the current Task edits before deleting a Task.\n"
                )
                return
            task = self._task_workspace.find_task_by_name(parts[1])
            if task is None:
                output.write(f"Task not found: {parts[1]}\n")
                return
            self.push_screen(
                ConfirmDeleteTaskScreen(task["name"]),
                callback=lambda confirmed: self._delete_task(task, confirmed, output),
            )
            return
        if command_name == "/taskrun":
            try:
                task_name, arguments = _parse_api_run_command(
                    "/task " + command.split(maxsplit=1)[1], "/task")
            except (ValueError, IndexError) as exc:
                output.write(f'Usage: /taskrun <taskname> [JSON arguments] (quote names containing spaces): {exc}\n')
                return
            if self._sessions[terminal_id].busy:
                output.write("Wait for the current command to finish before running a Task.\n")
                return
            if self._task_chat_busy:
                output.write("Wait for the current Task assistant response before running a Task.\n")
                return
            if self.current_task and (
                self._editor_is_dirty() or self._has_generated_draft
            ):
                output.write("Save or cancel the current Task edits before running another Task.\n")
                return
            task = self._task_workspace.find_task_by_name(task_name)
            if task is None:
                output.write(f"Task not found: {task_name}\n")
                return
            artifact = task.get("latest_artifact")
            if artifact is None:
                output.write(f"Task has no saved artifact version: {task['name']}\n")
                return
            try:
                if artifact.get("arguments_schema") is not None:
                    from loghawk.collectors.contracts import validate_arguments
                    arguments = validate_arguments(artifact["arguments_schema"], arguments)
            except ValueError as exc:
                output.write(f"Invalid Task arguments: {exc}\n")
                return
            self._load_task(task)
            if task["artifact_type"] == "n8n":
                output.write(
                    f"Loaded {task['name']} · User ID: {task['userid']} · "
                    f"Category: {task['category']} · "
                    f"Version {artifact['version']} in the Editor pane. "
                    "n8n workflow JSON must be imported into the customer-managed n8n instance to run.\n"
                )
                return
            artifact = dict(artifact, _run_arguments=arguments)
            self._pending_task_runs[terminal_id] = (task, artifact)
            output.write("Input arguments: " + json.dumps(arguments, ensure_ascii=False) + "\n")
            output.write(
                f"Run {task['name']} · User ID: {task['userid']} · "
                f"Category: {task['category']} · "
                f"Version {artifact['version']}? (Y/N): "
            )
            return
        if len(parts) != 2 or command_name not in {"/taskedit", "/taskview"}:
            output.write('Usage: /taskedit <taskname> (or /taskview; quote names containing spaces)\n')
            return
        if self._task_chat_busy:
            output.write("Wait for the current Task assistant response before switching Tasks.\n")
            return
        if self.current_task and (
            self._editor_is_dirty() or self._has_generated_draft
        ):
            output.write("Save or discard the current editor changes before loading another Task.\n")
            return
        task = self._task_workspace.find_task_by_name(parts[1])
        if task is None:
            output.write(f"Task not found: {parts[1]}\n")
            return
        self._load_task(task)
        output.write(
            f"Loaded Task: {task['name']} · User ID: {task['userid']} · "
            f"Category: {task['category']} · "
            f"Type: {self._task_type_label(task)}\n"
        )

    def _delete_task(self, task: dict, confirmed: bool, output: RichLog) -> None:
        if not confirmed:
            output.write("Task deletion canceled.\n")
            return
        try:
            deleted = self._task_workspace.delete_task(task["id"])
        except Exception as exc:
            output.write(f"Could not delete Task {task['name']}: {exc}\n")
            return
        if not deleted:
            output.write(f"Task was not found: {task['name']}\n")
            return

        if self.current_task and self.current_task["id"] == task["id"]:
            self.current_task = None
            self.current_artifact = None
            self._has_generated_draft = False
            self.query_one("#task-chat-pane", Container).display = False
            self.query_one("#task-create-pane", Container).display = False
            self.query_one("#task-chat-output", RichLog).clear()
            selector = self.query_one("#version-select", Select)
            selector.set_options([])
            selector.clear()
            self._set_editor_text("")
            self.query_one("#editor-status", Static).update("No Task loaded.")
            self._sync_editor_controls()

        output.write(
            f"Deleted Task: {task['name']} · User ID: {task['userid']} "
            "(conversation and saved versions deleted).\n"
        )

    def _confirm_task_run(self, terminal_id: str, answer: str) -> None:
        output = self.query_one(f"#{terminal_id}-output", RichLog)
        normalized = answer.strip().casefold()
        if normalized in {"n", "no"}:
            self._pending_task_runs.pop(terminal_id, None)
            self.current_task = None
            self.current_artifact = None
            self._has_generated_draft = False
            self.query_one("#task-chat-pane", Container).display = False
            self.query_one("#task-chat-output", RichLog).clear()
            selector = self.query_one("#version-select", Select)
            selector.set_options([])
            selector.clear()
            self._set_editor_text("")
            self.query_one("#editor-status", Static).update("No Task loaded.")
            self._sync_editor_controls()
            output.write("Task run canceled.\n")
            return
        if normalized not in {"y", "yes"}:
            output.write("Please answer Y or N: ")
            return
        task, artifact = self._pending_task_runs.pop(terminal_id)
        if task.get("_builtin_collector"):
            self._begin_command_output(terminal_id, f"/collectrun {task['name']}")
            self._run_collector_request(terminal_id, "/collectrun", task["name"], artifact["_run_arguments"])
            return
        self._begin_command_output(
            terminal_id, f"{artifact.get('_run_command', '/taskrun')} {task['name']} · Version {artifact['version']}"
        )
        output.write(
            f"Confirmed. Running {task['name']} · User ID: {task['userid']} · "
            f"Category: {task['category']} · "
            f"Version {artifact['version']}…\n"
        )
        self.run_worker(
            self._run_task_artifact(terminal_id, task, artifact),
            name=f"task-run-{terminal_id}",
        )

    async def _run_task_artifact(
        self, terminal_id: str, task: dict, artifact: dict
    ) -> None:
        output = self.query_one(f"#{terminal_id}-output", RichLog)
        artifact_type = task["artifact_type"]
        suffix = {
            "python": ".py",
            "bash": ".sh",
            "powershell": ".ps1",
            "windows-cmd": ".cmd",
        }.get(artifact_type)
        if suffix is None:
            output.write(f"Unsupported Task type: {artifact_type}\n")
            return
        try:
            arguments = artifact.get("_run_arguments", {})
            if artifact.get("arguments_schema") is not None:
                from loghawk.collectors.contracts import validate_arguments
                arguments = validate_arguments(artifact["arguments_schema"], arguments)
            result = await asyncio.to_thread(
                lhrunner.run_artifact, artifact_type, artifact["content"], arguments=arguments
            )
            label = "Terminal 1" if terminal_id == "command-one" else "Terminal 2"
            for stream_name in ("stdout", "stderr"):
                stream = result.get(stream_name) or ""
                for line in stream.splitlines():
                    output.write(line)
                    self.query_one("#command-output-mirror", RichLog).write(f"[{label}] {line}")
            result_line = f"[Task exit code: {result.get('exit_code')}]"
            output.write(result_line)
            self.query_one("#command-output-mirror", RichLog).write(f"[{label}] {result_line}")
        except Exception as exc:
            output.write(f"Task run failed: {exc}\n")

    @staticmethod
    def _task_execution_command(artifact_type: str, artifact_path: Path) -> str:
        if os.name == "nt":
            quote = lambda value: "'" + str(value).replace("'", "''") + "'"
            path = quote(artifact_path)
            cleanup = f"Remove-Item -LiteralPath {path} -Force"
            if artifact_type == "python":
                executable = quote(sys.executable)
                return (
                    f"& {executable} {path}; $taskExitCode = $LASTEXITCODE; "
                    f"{cleanup}; Write-Output ('[Task exit code: ' + $taskExitCode + ']')"
                )
            if artifact_type == "bash":
                executable = shutil.which("bash")
                if not executable:
                    raise RuntimeError("Bash is not installed or not available on PATH.")
                return (
                    f"& {quote(executable)} {path}; $taskExitCode = $LASTEXITCODE; "
                    f"{cleanup}; Write-Output ('[Task exit code: ' + $taskExitCode + ']')"
                )
            if artifact_type == "windows-cmd":
                executable = quote(os.environ.get("COMSPEC", "cmd.exe"))
                return (
                    f"& {executable} /d /c ('\"\"' + {path} + '\"\"'); "
                    f"$taskExitCode = $LASTEXITCODE; {cleanup}; "
                    "Write-Output ('[Task exit code: ' + $taskExitCode + ']')"
                )
            executable = (
                shutil.which("pwsh")
                or shutil.which("powershell")
                or shutil.which("powershell.exe")
            )
            if not executable:
                raise RuntimeError("PowerShell is not available on PATH.")
            return (
                f"& {quote(executable)} -NoProfile -File {path}; "
                f"$taskExitCode = $LASTEXITCODE; {cleanup}; "
                "Write-Output ('[Task exit code: ' + $taskExitCode + ']')"
            )

        path = shlex.quote(str(artifact_path))
        cleanup = f"rm -f -- {path}"
        if artifact_type == "python":
            return (
                f"{shlex.quote(sys.executable)} {path}; task_exit_code=$?; "
                f"{cleanup}; printf '[Task exit code: %s]\\n' \"$task_exit_code\""
            )
        if artifact_type == "bash":
            executable = shutil.which("bash")
            if not executable:
                raise RuntimeError("Bash is not installed or not available on PATH.")
            return (
                f"{shlex.quote(executable)} {path}; task_exit_code=$?; {cleanup}; "
                "printf '[Task exit code: %s]\\n' \"$task_exit_code\""
            )
        if artifact_type == "windows-cmd":
            raise RuntimeError("Windows CMD Tasks can only run on Windows.")
        executable = shutil.which("pwsh")
        if not executable:
            raise RuntimeError("PowerShell (pwsh) is not installed or not available on PATH.")
        return (
            f"{shlex.quote(executable)} -NoProfile -File {path}; "
            f"task_exit_code=$?; {cleanup}; printf '[Task exit code: %s]\\n' \"$task_exit_code\""
        )

    def _begin_task_creation(self) -> None:
        self._close_rca_output()
        self._close_task_list()
        self.current_task = None
        self.current_artifact = None
        self._has_generated_draft = False
        self.query_one("#task-create-pane", Container).display = True
        self.query_one("#task-chat-pane", Container).display = False
        self.query_one("#command-output-mirror", RichLog).display = False
        self.query_one("#editor-toolbar", Horizontal).display = True
        self.query_one("#editor", TextArea).display = True
        self.query_one("#editor-status", Static).display = True
        self.query_one("#task-name-input", Input).value = ""
        self.query_one("#task-description-input", TextArea).text = ""
        self.query_one("#task-category-input", Input).value = ""
        self.query_one("#task-type-select", Select).value = "python"
        self.query_one("#task-kind-select", Select).value = "general"
        self.query_one("#task-execution-policy-select", Select).value = "manual"
        version_selector = self.query_one("#version-select", Select)
        version_selector.set_options([])
        version_selector.clear()
        self.query_one("#task-chat-output", RichLog).clear()
        self.query_one("#editor-status", Static).update(
            "Enter Task details, then create it to start the assistant chat."
        )
        self._set_editor_text("")
        self._sync_editor_controls()

    def _create_task(self) -> None:
        name = self.query_one("#task-name-input", Input).value.strip()
        description = self.query_one("#task-description-input", TextArea).text.strip()
        category = self.query_one("#task-category-input", Input).value.strip()
        if not name:
            self.notify("Task name is required.", severity="warning")
            self.query_one("#task-name-input", Input).focus()
            return
        if not description:
            self.notify("Task description is required.", severity="warning")
            self.query_one("#task-description-input", TextArea).focus()
            return
        if not category:
            self.notify("Task category is required.", severity="warning")
            self.query_one("#task-category-input", Input).focus()
            return
        artifact_type = str(self.query_one("#task-type-select", Select).value)
        task_kind = str(self.query_one("#task-kind-select", Select).value)
        execution_policy = str(self.query_one("#task-execution-policy-select", Select).value)
        if task_kind == "collector" and artifact_type == "n8n":
            self.notify("Collector Tasks must use Python, Bash, PowerShell, or Windows CMD.", severity="error")
            return
        try:
            task = self._task_workspace.create_task(
                name, description, category, artifact_type, task_kind, execution_policy
            )
        except Exception as exc:
            self.notify(f"Could not create Task: {exc}", severity="error")
            return
        self._load_task(task)
        self.query_one("#editor-status", Static).update(
            f"{task['name']} · {task['category']} · {self._task_type_label(task)} · "
            "describe the artifact in Task Chat"
        )

    def _show_task_list(self, terminal_output: RichLog) -> None:
        self._close_rca_output()
        pane = self.query_one("#task-list-pane", Container)
        for widget_id in (
            "task-create-pane",
            "task-chat-pane",
            "editor-toolbar",
            "editor",
            "editor-status",
            "command-output-mirror",
        ):
            self.query_one(f"#{widget_id}").display = False

        task_list_output = self.query_one("#task-list-output", RichLog)
        task_list_output.clear()
        tasks = self._task_workspace.list_tasks()
        if tasks:
            grouped: dict[str, list[dict]] = {}
            for task in tasks:
                category = task.get("category") or "Uncategorized"
                grouped.setdefault(category, []).append(task)
            for category in sorted(grouped, key=str.casefold):
                category_tasks = sorted(
                    grouped[category], key=lambda item: item["name"].casefold()
                )
                name_width = max(
                    len("Task Name"),
                    max(len(task["name"]) for task in category_tasks),
                )
                userid_width = max(
                    len("User ID"),
                    max(len(str(task.get("userid") or "")) for task in category_tasks),
                )
                task_list_output.write("")
                task_list_output.write(f"Category: {category}")
                kind_width = max(len("Task Kind"), max(len(task.get("task_kind", "general")) for task in category_tasks))
                task_list_output.write(
                    f"{'Task Name':<{name_width}}  {'User ID':<{userid_width}}  "
                    f"{'Task Type':<14}  {'Task Kind':<14}  Execution Policy"
                )
                task_list_output.write(
                    f"{'-' * name_width}  {'-' * userid_width}  {'-' * 14}  {'-' * kind_width}  ----------------"
                )
                for task in category_tasks:
                    task_list_output.write(
                        f"{task['name']:<{name_width}}  "
                        f"{str(task.get('userid') or ''):<{userid_width}}  "
                        f"{self._task_type_label(task):<14}  {task.get('task_kind', 'general'):<14}  "
                        f"{task.get('execution_policy', 'manual')}"
                    )
        else:
            task_list_output.write("No Tasks found.")
        pane.display = True
        terminal_output.write(f"Displayed {len(tasks)} Task(s) in the Editor pane.\n")

    def _close_task_list(self) -> None:
        pane = self.query_one("#task-list-pane", Container)
        if not pane.display:
            return
        pane.display = False
        self.current_task = None
        self.current_artifact = None
        self._has_generated_draft = False
        self.query_one("#task-create-pane", Container).display = False
        self.query_one("#task-chat-pane", Container).display = False
        self.query_one("#editor-toolbar", Horizontal).display = True
        self.query_one("#editor", TextArea).display = True
        self.query_one("#editor-status", Static).display = True
        self.query_one("#task-chat-output", RichLog).clear()
        self.query_one("#task-name-input", Input).value = ""
        self.query_one("#task-category-input", Input).value = ""
        selector = self.query_one("#version-select", Select)
        selector.set_options([])
        selector.clear()
        self._set_editor_text("")
        self.query_one("#editor-status", Static).update("No Task loaded.")
        self._sync_editor_controls()

    def _cancel_task_creation(self) -> None:
        self.query_one("#task-create-pane", Container).display = False
        self.query_one("#task-chat-pane", Container).display = False
        self.query_one("#task-description-input", TextArea).text = ""
        self.query_one("#task-category-input", Input).value = ""
        self._set_editor_text("")
        self._sync_editor_controls()
        self.query_one("#editor-status", Static).update(
            "Task creation canceled; no Task was saved."
        )

    def _submit_task_message(self, prompt: str) -> None:
        if not self.current_task:
            self.notify("Create or load a Task before chatting with its assistant.", severity="warning")
            return
        if self._task_chat_busy:
            self.notify("Wait for the current Task assistant response to finish.", severity="warning")
            return
        message_input = self.query_one("#task-message-input", Input)
        message_input.disabled = True
        self.query_one("#send-task-message", Button).disabled = True
        self._task_chat_busy = True
        self.query_one("#task-chat-output", RichLog).write(
            f"You: {prompt}\nAssistant: …\n"
        )
        self._run_task_message(self.current_task["id"], prompt)

    @work(thread=True, exclusive=True, group="hawk-task-assistant")
    def _run_task_message(self, task_id: str, prompt: str) -> None:
        try:
            task = self._task_workspace.respond_to_message(task_id, prompt)
            if task is None:
                raise RuntimeError("Task was deleted before the assistant could respond.")
            self.app.call_from_thread(self._finish_task_message, task, None)
        except Exception as exc:
            task = self._task_workspace.get_task(task_id)
            self.app.call_from_thread(self._finish_task_message, task, str(exc))

    def _finish_task_message(self, task: dict | None, error: str | None) -> None:
        self._task_chat_busy = False
        self.query_one("#task-message-input", Input).disabled = False
        self.query_one("#send-task-message", Button).disabled = False
        if task is not None:
            self.current_task = task
            self._render_task_messages(task)
            draft = task.get("draft_artifact")
            if draft:
                self._has_generated_draft = True
                self.query_one("#editor", TextArea).text = draft
                self.query_one("#editor-status", Static).update(
                    f"{task['name']} · generated draft · unsaved"
                )
        if error:
            self.query_one("#task-chat-output", RichLog).write(
                f"Task assistant failed: {error}\n"
            )
        self._sync_editor_controls()

    def _save_task_execution_policy(self) -> None:
        if not self.current_task:
            return
        policy = str(self.query_one("#loaded-task-execution-policy", Select).value)
        try:
            task = self._task_workspace.set_execution_policy(self.current_task["id"], policy)
            if task is None:
                self.notify("Could not update Task execution policy.", severity="error")
                return
            self.current_task["execution_policy"] = task["execution_policy"]
            self._sync_editor_controls()
            self.notify(f"Execution policy set to {policy}.")
        except Exception as exc:
            self.notify(f"Could not update Task execution policy: {exc}", severity="error")

    def _save_loaded_task_description(self) -> None:
        if not self.current_task:
            return
        description = self.query_one("#loaded-task-description", TextArea).text.strip()
        if not description:
            self.notify("Task description is required.", severity="warning")
            return
        try:
            task = self._task_workspace.set_description(self.current_task["id"], description)
            if task is None:
                self.notify("Could not update Task description.", severity="error")
                return
            self.current_task["description"] = task["description"]
            self.query_one("#loaded-task-description", TextArea).load_text(task["description"])
            self._sync_editor_controls()
            self.notify("Task description saved.")
        except Exception as exc:
            self.notify(f"Could not update Task description: {exc}", severity="error")

    def _set_selected_version_policy(self, *, approve: bool = False) -> None:
        if (self._api_client is None or not self.current_task or not self.current_artifact
                or self._task_chat_busy or self._editor_is_dirty() or self._has_generated_draft):
            self.notify("Save or cancel the artifact draft before changing a saved version.", severity="warning")
            return
        try:
            if approve:
                updated = self._task_workspace.approve_artifact(
                    self.current_task["id"], self.current_artifact["id"])
            else:
                updated = self._task_workspace.set_artifact_enabled(
                    self.current_task["id"], self.current_artifact["id"],
                    not self.current_artifact.get("enabled", False))
            if not updated or updated.get("id") != self.current_artifact["id"]:
                raise RuntimeError("The API did not return the selected artifact version.")
            self.current_artifact.update(updated)
            for artifact in self.current_task.get("artifacts", []):
                if artifact["id"] == updated["id"]:
                    artifact.update(updated)
            self._sync_editor_controls()
            self.notify("Selected version approved." if approve else "Selected version enabled status updated.")
        except Exception as exc:
            self.notify(f"Could not update selected version: {exc}", severity="error")

    def _render_task_messages(self, task: dict) -> None:
        output = self.query_one("#task-chat-output", RichLog)
        output.clear()
        for message in task.get("messages", []):
            label = "You" if message["role"] == "user" else "Assistant"
            output.write(f"{label}: {message['content']}\n")

    def _load_task(self, task: dict) -> None:
        self._close_rca_output()
        self._close_task_list()
        self.current_task = task
        self.query_one("#collector-schema-pane", Container).display = True
        self._set_schema_text(None)
        self.query_one("#loaded-task-description", TextArea).text = task.get("description", "")
        self._has_generated_draft = False
        self.query_one("#task-create-pane", Container).display = False
        self.query_one("#task-chat-pane", Container).display = True
        self.query_one("#loaded-task-execution-policy", Select).value = task.get("execution_policy", "manual")
        self.query_one("#command-output-mirror", RichLog).display = False
        self.query_one("#editor-toolbar", Horizontal).display = True
        self.query_one("#editor", TextArea).display = True
        self.query_one("#editor-status", Static).display = True
        self._render_task_messages(task)
        artifacts = task.get("artifacts", [])
        options = [
            (f"Version {item['version']} · {item['created_at']}", item["id"])
            for item in artifacts
        ]
        selector = self.query_one("#version-select", Select)
        self._suppress_version_event = True
        selector.set_options(options)
        if artifacts:
            self._load_artifact(artifacts[0])
            selector.value = artifacts[0]["id"]
        else:
            self.current_artifact = None
            self._set_editor_text("")
            self.query_one("#editor-status", Static).update(
                f"Task: {task['name']} · User ID: {task['userid']} · "
                f"Category: {task['category']} · Type: {self._task_type_label(task)} · "
                "no saved artifact versions yet"
            )
            selector.clear()
        self._suppress_version_event = False
        self._sync_editor_controls()

    def _load_artifact(self, artifact: dict) -> None:
        self.current_artifact = artifact
        self._set_schema_text(artifact.get("arguments_schema"))
        self._set_editor_text(artifact["content"])
        self.query_one("#editor-status", Static).update(
            f"Task: {self.current_task['name']} · "
            f"User ID: {self.current_task['userid']} · "
            f"Category: {self.current_task['category']} · "
            f"Type: {self._task_type_label(self.current_task)} · Version {artifact['version']}"
        )

    def _set_editor_text(self, text: str) -> None:
        editor = self.query_one("#editor", TextArea)
        editor.text = text
        self._editor_baseline = text

    def _set_schema_text(self, schema: dict | None) -> None:
        self._schema_baseline = json.dumps(schema, indent=2) if schema is not None else "{}"
        self.query_one("#collector-arguments-schema", TextArea).text = self._schema_baseline

    def _editor_is_dirty(self) -> bool:
        schema_dirty = bool(self.current_task
                            and self.query_one("#collector-arguments-schema", TextArea).text != self._schema_baseline)
        return self.query_one("#editor", TextArea).text != self._editor_baseline or schema_dirty

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "task-kind-select" and event.value == "collector":
            artifact_type = self.query_one("#task-type-select", Select)
            if artifact_type.value == "n8n":
                artifact_type.value = "python"
            return
        if event.select.id == "task-type-select" and event.value == "n8n":
            task_kind = self.query_one("#task-kind-select", Select)
            if task_kind.value == "collector":
                event.select.value = "python"
                self.notify("Collector Tasks cannot use n8n workflow artifacts.", severity="warning")
            return
        if event.select.id != "version-select" or self._suppress_version_event:
            return
        if event.select.is_blank or not self.current_task:
            return
        if self._editor_is_dirty() or self._has_generated_draft:
            self.notify("Save or cancel the current draft before selecting another version.", severity="warning")
            if self.current_artifact:
                event.select.value = self.current_artifact["id"]
            else:
                event.select.clear()
            return
        artifact = self._task_workspace.get_artifact(self.current_task["id"], str(event.value))
        if artifact is not None:
            self._load_artifact(artifact)
            self._sync_editor_controls()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id in {"editor", "collector-arguments-schema"}:
            self._sync_editor_controls()
        elif event.text_area.id == "task-description-input" and self.is_mounted:
            self._sync_editor_controls()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id in {"task-name-input", "task-category-input"} and self.is_mounted:
            self.query_one("#create-task", Button).disabled = not bool(
                self.query_one("#task-name-input", Input).value.strip()
                and self.query_one("#task-category-input", Input).value.strip()
            )

    def _sync_editor_controls(self) -> None:
        if not self.is_mounted:
            return
        editor = self.query_one("#editor", TextArea)
        has_task = self.current_task is not None
        dirty = self._editor_is_dirty()
        manager = self._api_user.get("role") in {"Administrator", "Power User"}
        can_manage = bool(has_task and (self._api_client is None or manager
                          or self.current_task.get("userid") == self._api_user.get("userid")))
        self.query_one("#loaded-task-description", TextArea).read_only = not can_manage
        self.query_one("#save-loaded-task-description", Button).disabled = not can_manage or self._task_chat_busy
        self.query_one("#loaded-task-execution-policy", Select).disabled = not can_manage or self._task_chat_busy
        self.query_one("#save-task-execution-policy", Button).disabled = not can_manage or self._task_chat_busy
        artifact = self.current_artifact if has_task else None
        ready = bool(self._api_client and artifact and not dirty
                     and not self._has_generated_draft and not self._task_chat_busy)
        enable = self.query_one("#enable-loaded-version", Button)
        enable.label = "Disable" if artifact and artifact.get("enabled") else "Enable"
        enable.disabled = not (ready and can_manage)
        self.query_one("#approve-loaded-version", Button).disabled = not (
            ready and manager and not artifact.get("approved"))
        status = "No saved version selected."
        if artifact:
            status = (f"Selected Version: {artifact['version']} · "
                      f"{'Approved' if artifact.get('approved') else 'Not approved'} · "
                      f"{'Enabled' if artifact.get('enabled') else 'Disabled'}")
        if self._api_client is None:
            status += " · Server approval controls require sign-in."
        self.query_one("#loaded-version-status", Static).update(status)
        self.query_one("#save-version", Button).disabled = not (
            has_task and (dirty or self._has_generated_draft) and bool(editor.text.strip())
        )
        self.query_one("#download-version", Button).disabled = not (
            has_task and bool(editor.text.strip())
        )
        self.query_one("#delete-version", Button).disabled = not (
            has_task and self.current_artifact is not None and not self._has_generated_draft
        )
        self.query_one("#cancel-edit", Button).disabled = not (
            has_task and (dirty or self._has_generated_draft)
        )
        self.query_one("#version-select", Select).disabled = not has_task
        self.query_one("#create-task", Button).disabled = not bool(
            self.query_one("#task-name-input", Input).value.strip()
            and self.query_one("#task-description-input", TextArea).text.strip()
            and self.query_one("#task-category-input", Input).value.strip()
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "new-chat":
            self._new_chat()
        elif event.button.id == "show-rca-output":
            self._show_rca_output()
        elif event.button.id == "copy-rca-output":
            self.copy_to_clipboard(self.query_one("#rca-output-text", TextArea).text)
            self.notify("Copy requested; clipboard availability depends on the terminal.")
        elif event.button.id == "download-rca-output":
            self._download_rca_output()
        elif event.button.id == "close-rca-output":
            self._close_rca_output()
        elif event.button.id == "retry-rca-explanation":
            self._retry_rca_explanation()
        elif event.button.id == "create-task":
            self._create_task()
        elif event.button.id == "cancel-create-task":
            self._cancel_task_creation()
        elif event.button.id == "close-task-list":
            self._close_task_list()
        elif event.button.id == "close-help":
            self._close_help()
        elif event.button.id == "send-task-message":
            message_input = self.query_one("#task-message-input", Input)
            prompt = message_input.value.strip()
            if prompt:
                message_input.value = ""
                self._submit_task_message(prompt)
        elif event.button.id == "enable-loaded-version":
            self._set_selected_version_policy()
        elif event.button.id == "approve-loaded-version":
            self._set_selected_version_policy(approve=True)
        elif event.button.id == "save-task-execution-policy":
            self._save_task_execution_policy()
        elif event.button.id == "save-loaded-task-description":
            self._save_loaded_task_description()
        elif event.button.id == "cancel-edit":
            self._cancel_edit()
        elif event.button.id == "save-version":
            self._save_version()
        elif event.button.id == "download-version":
            self._download_version()
        elif event.button.id == "delete-version":
            self._confirm_delete_version()
        elif event.button.id == "confirm-command-edits":
            self._confirm_command_edits()
        elif event.button.id == "quick-save-command-task":
            self._quick_save_command_task()
        elif event.button.id == "cancel-command-edits":
            self._cancel_command_edits()
        elif event.button.id in {"command-one-stop", "command-two-stop"}:
            terminal_id = event.button.id.removesuffix("-stop")
            self.run_worker(self._sessions[terminal_id].stop(), name=f"{terminal_id}-stop")

    def _save_version(self) -> None:
        if not self.current_task:
            return
        try:
            task = self._task_workspace.save_artifact(
                self.current_task["id"],
                self.query_one("#editor", TextArea).text,
                arguments_schema=(json.loads(self.query_one("#collector-arguments-schema", TextArea).text)
                                  if self.query_one("#collector-arguments-schema", TextArea).text.strip() else {}),
            )
            if task:
                self._load_task(task)
                self.query_one("#editor-status", Static).update(
                    f"{task['name']} · Version {task['latest_artifact']['version']} saved"
                )
        except Exception as exc:
            self.notify(f"Could not save version: {exc}", severity="error")

    def _cancel_edit(self) -> None:
        if not self.current_task:
            return
        self._has_generated_draft = False
        self._set_editor_text(self._editor_baseline)
        self.query_one("#collector-arguments-schema", TextArea).text = self._schema_baseline
        if self.current_artifact:
            status = f"{self.current_task['name']} · Version {self.current_artifact['version']}"
        else:
            status = f"{self.current_task['name']} · no saved artifact version selected"
        self.query_one("#editor-status", Static).update(status)
        self._sync_editor_controls()

    def _download_version(self) -> None:
        if not self.current_task:
            return
        content = self.query_one("#editor", TextArea).text
        if not content.strip():
            return
        version = (
            self.current_artifact["version"]
            if self.current_artifact and not self._has_generated_draft
            else None
        )
        filename = self._artifact_filename(
            self.current_task,
            version,
        )
        self.push_screen(
            DownloadPathScreen(filename),
            callback=lambda destination: self._download_to_path(
                destination,
                content,
            ),
        )

    def _download_to_path(self, destination: str | None, content: str) -> None:
        if not destination:
            return
        path = Path(destination).expanduser()
        try:
            if path.exists():
                raise FileExistsError(f"File already exists: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            self.query_one("#editor-status", Static).update(f"Downloaded to {path}")
        except OSError as exc:
            self.notify(f"Could not download artifact: {exc}", severity="error")

    @staticmethod
    def _artifact_filename(task: dict, version: int | None) -> str:
        suffix = {
            "n8n": ".json",
            "python": ".py",
            "bash": ".sh",
            "powershell": ".ps1",
            "windows-cmd": ".cmd",
        }.get(task["artifact_type"], ".txt")
        safe_name = "".join(
            character if character.isalnum() or character in "-_." else "-"
            for character in task["name"]
        ).strip("-.") or "loghawk-task"
        version_label = f"v{version}" if version is not None else "draft"
        return f"{safe_name}-{version_label}{suffix}"

    @staticmethod
    def _task_type_label(task: dict) -> str:
        return {
            "n8n": "n8n",
            "python": "Python",
            "bash": "Bash",
            "powershell": "PowerShell",
            "windows-cmd": "Windows CMD",
        }.get(task["artifact_type"], task["artifact_type"])

    def _confirm_delete_version(self) -> None:
        if not self.current_task or not self.current_artifact:
            return
        self.push_screen(ConfirmDeleteScreen(), callback=self._delete_version)

    def _delete_version(self, confirmed: bool) -> None:
        if not confirmed or not self.current_task or not self.current_artifact:
            return
        task_id = self.current_task["id"]
        artifact_id = self.current_artifact["id"]
        deleted_version = self.current_artifact["version"]
        if self._task_workspace.delete_artifact(task_id, artifact_id):
            task = self._task_workspace.get_task(task_id)
            if task:
                self._load_task(task)
                self.query_one("#editor-status", Static).update(
                    f"Deleted Version {deleted_version}"
                )
        else:
            self.notify("Artifact version was not found.", severity="error")

    async def on_unmount(self) -> None:
        if self._api_client is not None:
            self._flush_remote_history(report_errors=False)
        await asyncio.gather(
            *(session.close() for session in self._sessions.values()),
            return_exceptions=True,
        )


def main() -> None:
    """Launch the Hawk terminal UI."""
    parser = argparse.ArgumentParser(prog="hawk")
    parser.add_argument(
        "--local", action="store_true",
        help="Use Console-local SQLite, LanceDB, and RAG without signing in.",
    )
    args = parser.parse_args()
    if not args.local:
        session = ConsoleLoginApp().run()
        if not session:
            return
        set_userid_override(session["user"]["userid"])
        HawkConsole(api_token=session["access_token"], api_user=session["user"]).run()
    else:
        HawkConsole().run()


if __name__ == "__main__":
    main()
