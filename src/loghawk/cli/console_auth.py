"""Textual login screen for an API-authenticated Hawk Console session."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Button, Input, Static

import loghawk.config as config

_INPUT_BORDER_CSS = (
    "Input { border-left: ascii $border-blurred; }\n"
    "Input:focus { border-left: ascii $border; }\n"
    if config.LH_OS_FAMILY in {"LINUX", "MAC"}
    else ""
)


class ConsoleLoginApp(App[dict | None]):
    CSS = """
    Screen { align: center middle; }
    #login-box { width: 60; height: auto; padding: 1 2; border: round $primary; background: $surface; }
    #login-box Input { margin: 1 0; }
    #login-box Button { margin-top: 1; }
    .error { color: $error; }
    """ + _INPUT_BORDER_CSS

    TITLE = "Hawk Console Sign In"

    def compose(self) -> ComposeResult:
        with Container(id="login-box"):
            yield Static("Sign in to Hawk Console")
            yield Input(placeholder="User ID", id="userid")
            yield Input(placeholder="Password", password=True, id="password")
            yield Static("", id="login-error", classes="error")
            yield Button("Sign In", variant="primary", id="login")
            yield Button("Quit", id="quit")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit":
            self.exit(None)
            return
        userid = self.query_one("#userid", Input).value.strip()
        password = self.query_one("#password", Input).value
        if not userid or not password:
            self.query_one("#login-error", Static).update("Enter both user ID and password.")
            return
        request = Request(
            f"{config.LH_API_SERVER_URL}/api/v1/auth/login",
            data=json.dumps({"userid": userid, "password": password, "client": "console"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=15) as response:
                self.exit(json.loads(response.read().decode("utf-8")))
        except HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode("utf-8")).get("detail", "Sign in failed.")
            except Exception:
                detail = "Sign in failed."
            self.query_one("#login-error", Static).update(str(detail))
        except (URLError, TimeoutError, OSError) as exc:
            self.query_one("#login-error", Static).update(
                f"Could not reach LogHawk API at {config.LH_API_SERVER_URL}: {exc}"
            )
