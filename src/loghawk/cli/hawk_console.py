"""Hawk Console: split-pane interface for LogHawk chat, commands, and Tasks."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import shutil
import sys
import tempfile
from pathlib import Path

from textual import events, work
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Input, RichLog, Select, Static, TextArea
from rich.table import Table

import loghawk.config as config
from loghawk.cli.command_terminal import CommandShellSession
from loghawk.cli import command_history2 as command_history
from loghawk.cli import local_store2 as local_store
from loghawk.cli import task_workspace2 as task_workspace
from loghawk.llm import get_llm_client
from loghawk.rag import ingestion
from loghawk.utils import lancedbutils3
from loghawk.cli import console_rag_store


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
    #llm-output, #command-one-output, #command-two-output { height: 1fr; }
    #editor-toolbar { height: auto; min-height: 3; }
    #task-create-pane { height: auto; padding: 0 1; }
    #task-create-fields { height: auto; }
    #task-chat-pane { height: 1fr; min-height: 8; }
    #task-chat-output { height: 1fr; }
    #task-message-row { height: auto; }
    #task-list-pane { height: 1fr; padding: 0 1; }
    #task-list-output { height: 1fr; }
    #command-editor-pane { height: 1fr; padding: 0 1; }
    #command-editor { height: 1fr; border: round $secondary; }
    #command-editor-toolbar { height: auto; min-height: 3; }
    #version-select { width: 1fr; }
    #editor { height: 1fr; border: round $secondary; }
    #command-output-mirror { height: 1fr; }
    Button { min-width: 10; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.current_task: dict | None = None
        self.current_artifact: dict | None = None
        self._editor_baseline = ""
        self._suppress_version_event = False
        self._has_generated_draft = False
        self._task_chat_busy = False
        self._pending_task_runs: dict[str, tuple[dict, dict]] = {}
        self._chat_history: list[dict[str, str]] = []
        self._chat_busy = False
        self._pending_llm_command: str | None = None
        self._awaiting_llm_command_final_confirmation = False
        self._command_editor_visibility: dict[str, bool] = {}
        self._sessions: dict[str, CommandShellSession] = {}
        self._active_command_outputs: set[str] = set()
        self._command_history = {"command-one": [], "command-two": []}
        self._command_history_index = {"command-one": 0, "command-two": 0}
        self._command_history_draft = {"command-one": "", "command-two": ""}
        self._help_previous_visibility: dict[str, bool] | None = None

    def compose(self) -> ComposeResult:
        with Container(id="terminal-column"):
            with Container(id="llm-pane", classes="pane"):
                yield Static("LLM Terminal · /help · Ctrl+Q to quit, Shift+Ctrl+C/Shift+Ctrl+V to Copy/Paste", classes="pane-title")
                yield RichLog(id="llm-output", wrap=True, markup=False, auto_scroll=True)
                yield Input(placeholder="Chat with the configured LLM…", id="llm-input")

            with Container(id="command-one-pane", classes="pane"):
                yield Static("Command Terminal 1 · /help · /tasklist · /tasknew · /taskedit <taskname> · /taskview <taskname> (alias) · /taskrun <taskname> · /taskdelete <taskname>", classes="pane-title")
                yield RichLog(id="command-one-output", wrap=True, markup=False, auto_scroll=True)
                with Horizontal():
                    yield Input(placeholder="Shell command · /tasklist · /tasknew · /taskedit <taskname> · /taskview <taskname> (alias) · /taskrun <taskname> · /taskdelete <taskname>", id="command-one-input")
                    yield Button("Stop", id="command-one-stop")

            with Container(id="command-two-pane", classes="pane"):
                yield Static("Command Terminal 2 · /help · /tasklist · /tasknew · /taskedit <taskname> · /taskview <taskname> (alias) · /taskrun <taskname> · /taskdelete <taskname>", classes="pane-title")
                yield RichLog(id="command-two-output", wrap=True, markup=False, auto_scroll=True)
                with Horizontal():
                    yield Input(placeholder="Shell command · /tasklist · /tasknew · /taskedit <taskname> · /taskview <taskname> (alias) · /taskrun <taskname> · /taskdelete <taskname>", id="command-two-input")
                    yield Button("Stop", id="command-two-stop")

        with Container(id="editor-pane", classes="pane"):
            yield Static("Editor · /help · use /taskedit <name> or /tasknew", classes="pane-title")
            with Container(id="task-create-pane"):
                yield Static("Create Task", classes="pane-title")
                with Container(id="task-create-fields"):
                    yield Static("Task name")
                    yield Input(placeholder="Task name", id="task-name-input")
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
                    yield Static("Category")
                    yield Input(placeholder="Enter a category", id="task-category-input")
                with Horizontal():
                    yield Button("Create Task", variant="primary", id="create-task")
                    yield Button("Cancel", id="cancel-create-task")
            with Container(id="task-chat-pane"):
                yield Static("Task Chat", classes="pane-title")
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
                yield TextArea("", id="command-editor", soft_wrap=False)
                with Horizontal(id="command-editor-toolbar"):
                    yield Button("Confirm Changes", variant="primary", id="confirm-command-edits")
                    yield Button("Cancel", id="cancel-command-edits")
            with Container(id="help-pane"):
                yield Static("Hawk Console Command Help", classes="pane-title")
                yield RichLog(id="help-output", wrap=True, markup=False, auto_scroll=True)
                yield Button("Close Help", id="close-help")

    async def on_mount(self) -> None:
        local_store.initialize_console_database()
        self.query_one("#llm-output", RichLog).write(
            f"Model: {config.LH_LLM_MODEL}\n"
            "Chat uses the configured LiteLLM provider and indexed LanceDB context when available.\n"
            "Ordinary chat uses the LLM directly. Prefix a question with /rag to search LanceDB.\n"
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
                self._command_history[terminal_id] = (
                    command_history.load_history(terminal_id)
                )
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
        self._load_chat_history()
        self._sync_editor_controls()

    def _load_chat_history(self) -> None:
        try:
            self._chat_history = local_store.load_conversation_history()
            if self._chat_history:
                self.query_one("#llm-output", RichLog).write(
                    f"Loaded {len(self._chat_history)} prior chat messages from SQLite.\n"
                )
        except Exception as exc:
            self.query_one("#llm-output", RichLog).write(
                f"Could not load SQLite chat history: {exc}\n"
            )

    def _persist_chat_message(self, role: str, content: str) -> None:
        local_store.save_conversation_message(role, content)

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
            self._run_chat(text, use_rag=use_rag)
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
        if text.startswith("/task"):
            self._handle_task_command(text, output, terminal_id)
            return
        output.write(f"$ {text}\n")
        self.run_worker(self._run_command(terminal_id, text), name=f"{terminal_id}-command")

    def _show_help(self) -> None:
        """Display the command reference without discarding the Editor's current view."""
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
            ("/tasknew", "None", "Command 1 or 2", "Open the AI-assisted Task creation form in the Editor pane."),
            ("/tasklist", "None", "Command 1 or 2", "List saved Tasks in the Editor pane."),
            ("/taskedit", "<taskname>", "Command 1 or 2", "Load a saved Task into the editable Editor pane. Quote names containing spaces."),
            ("/taskview", "<taskname>", "Command 1 or 2", "Alias for /taskedit."),
            ("/taskrun", "<taskname>", "Command 1 or 2", "Load the latest saved version. Confirm with Y/yes to run supported code; n8n workflow artifacts are displayed but are not executed here."),
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
            self._run_console_rag(
                terminal_id,
                "UPLOAD",
                [(str(item), str(item.resolve())) for item in files],
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
            if kind == "UPLOAD":
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
        output.write("Sending the approved command to Command Terminal 1.\n")
        self.query_one("#command-one-output", RichLog).write(f"$ {command}\n")
        self.run_worker(
            self._run_command("command-one", command),
            name="command-one-llm-approved",
        )

    def _open_command_editor(self) -> None:
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
        self.query_one("#command-editor", TextArea).text = self._pending_llm_command
        editor_pane.display = True
        self.query_one("#llm-input", Input).disabled = True
        self.query_one("#command-editor", TextArea).focus()

    def _close_command_editor(self) -> None:
        self.query_one("#command-editor-pane", Container).display = False
        for widget_id, was_displayed in self._command_editor_visibility.items():
            self.query_one(f"#{widget_id}").display = was_displayed
        self._command_editor_visibility.clear()
        self.query_one("#llm-input", Input).disabled = False
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

    def _cancel_command_edits(self) -> None:
        self._pending_llm_command = None
        self._awaiting_llm_command_final_confirmation = False
        self._close_command_editor()
        self.query_one("#llm-output", RichLog).write("Command canceled.\n")

    @work(thread=True, exclusive=True, group="hawk-llm")
    def _run_chat(self, prompt: str, use_rag: bool = False) -> None:
        response_parts: list[str] = []
        try:
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

    def _finish_chat(self, error: str | None, response: str | None) -> None:
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
            task = task_workspace.find_task_by_name(parts[1])
            if task is None:
                output.write(f"Task not found: {parts[1]}\n")
                return
            self.push_screen(
                ConfirmDeleteTaskScreen(task["name"]),
                callback=lambda confirmed: self._delete_task(task, confirmed, output),
            )
            return
        if command_name == "/taskrun":
            if len(parts) != 2:
                output.write('Usage: /taskrun <taskname> (quote names containing spaces)\n')
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
            task = task_workspace.find_task_by_name(parts[1])
            if task is None:
                output.write(f"Task not found: {parts[1]}\n")
                return
            artifact = task.get("latest_artifact")
            if artifact is None:
                output.write(f"Task has no saved artifact version: {task['name']}\n")
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
            self._pending_task_runs[terminal_id] = (task, artifact)
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
        task = task_workspace.find_task_by_name(parts[1])
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
            deleted = task_workspace.delete_task(task["id"])
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
        self._begin_command_output(
            terminal_id, f"/taskrun {task['name']} · Version {artifact['version']}"
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
        session = self._sessions[terminal_id]

        try:
            with tempfile.TemporaryDirectory(prefix="hawk-task-") as temp_dir:
                artifact_path = Path(temp_dir) / f"task{suffix}"
                artifact_path.write_text(artifact["content"], encoding="utf-8")
                command = self._task_execution_command(artifact_type, artifact_path)
                if not await session.submit(command):
                    output.write("Could not start the Task in this command terminal.\n")
                    return
                while session.busy:
                    await asyncio.sleep(0.1)
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
        self.query_one("#task-category-input", Input).value = ""
        self.query_one("#task-type-select", Select).value = "python"
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
        category = self.query_one("#task-category-input", Input).value.strip()
        if not name:
            self.notify("Task name is required.", severity="warning")
            self.query_one("#task-name-input", Input).focus()
            return
        if not category:
            self.notify("Task category is required.", severity="warning")
            self.query_one("#task-category-input", Input).focus()
            return
        artifact_type = str(self.query_one("#task-type-select", Select).value)
        try:
            task = task_workspace.create_task(name, category, artifact_type)
        except Exception as exc:
            self.notify(f"Could not create Task: {exc}", severity="error")
            return
        self._load_task(task)
        self.query_one("#editor-status", Static).update(
            f"{task['name']} · {category} · {self._task_type_label(task)} · "
            "describe the artifact in Task Chat"
        )

    def _show_task_list(self, terminal_output: RichLog) -> None:
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
        tasks = task_workspace.list_tasks()
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
                task_list_output.write(
                    f"{'Task Name':<{name_width}}  "
                    f"{'User ID':<{userid_width}}  Task Type"
                )
                task_list_output.write(
                    f"{'-' * name_width}  {'-' * userid_width}  ---------"
                )
                for task in category_tasks:
                    task_list_output.write(
                        f"{task['name']:<{name_width}}  "
                        f"{str(task.get('userid') or ''):<{userid_width}}  "
                        f"{self._task_type_label(task)}"
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
            task = task_workspace.respond_to_message(task_id, prompt)
            if task is None:
                raise RuntimeError("Task was deleted before the assistant could respond.")
            self.app.call_from_thread(self._finish_task_message, task, None)
        except Exception as exc:
            task = task_workspace.get_task(task_id)
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

    def _render_task_messages(self, task: dict) -> None:
        output = self.query_one("#task-chat-output", RichLog)
        output.clear()
        for message in task.get("messages", []):
            label = "You" if message["role"] == "user" else "Assistant"
            output.write(f"{label}: {message['content']}\n")

    def _load_task(self, task: dict) -> None:
        self._close_task_list()
        self.current_task = task
        self._has_generated_draft = False
        self.query_one("#task-create-pane", Container).display = False
        self.query_one("#task-chat-pane", Container).display = True
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

    def _editor_is_dirty(self) -> bool:
        return self.query_one("#editor", TextArea).text != self._editor_baseline

    def on_select_changed(self, event: Select.Changed) -> None:
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
        artifact = task_workspace.get_artifact(self.current_task["id"], str(event.value))
        if artifact is not None:
            self._load_artifact(artifact)
            self._sync_editor_controls()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id == "editor":
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
        dirty = editor.text != self._editor_baseline
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
            and self.query_one("#task-category-input", Input).value.strip()
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "create-task":
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
        elif event.button.id == "cancel-command-edits":
            self._cancel_command_edits()
        elif event.button.id in {"command-one-stop", "command-two-stop"}:
            terminal_id = event.button.id.removesuffix("-stop")
            self.run_worker(self._sessions[terminal_id].stop(), name=f"{terminal_id}-stop")

    def _save_version(self) -> None:
        if not self.current_task:
            return
        try:
            task = task_workspace.save_artifact(
                self.current_task["id"],
                self.query_one("#editor", TextArea).text,
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
        if task_workspace.delete_artifact(task_id, artifact_id):
            task = task_workspace.get_task(task_id)
            if task:
                self._load_task(task)
                self.query_one("#editor-status", Static).update(
                    f"Deleted Version {deleted_version}"
                )
        else:
            self.notify("Artifact version was not found.", severity="error")

    async def on_unmount(self) -> None:
        await asyncio.gather(
            *(session.close() for session in self._sessions.values()),
            return_exceptions=True,
        )


def main() -> None:
    """Launch the Hawk terminal UI."""
    HawkConsole().run()


if __name__ == "__main__":
    main()
