"""Persistent, line-oriented shell sessions for Hawk CLI command panes."""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
import uuid
from collections.abc import Callable


class CommandShellSession:
    """A stateful shell session that streams text output to a callback.

    The shell has no TTY, so commands that require full-screen terminal control
    (for example, vim or htop) are not supported by this first implementation.
    """

    def __init__(self, on_output: Callable[[str], None], cwd: str) -> None:
        self.on_output = on_output
        self.cwd = cwd
        self.process: asyncio.subprocess.Process | None = None
        self.reader_task: asyncio.Task | None = None
        self.busy = False
        self._marker: str | None = None

    async def start(self) -> None:
        if self.process and self.process.returncode is None:
            return

        if os.name == "nt":
            executable = (
                shutil.which("pwsh")
                or shutil.which("powershell")
                or "powershell.exe"
            )
            self.process = await asyncio.create_subprocess_exec(
                executable,
                "-NoLogo",
                "-NoProfile",
                "-Command",
                "-",
                cwd=self.cwd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "TERM": "dumb"},
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
        else:
            shell = os.environ.get("SHELL")
            shell_name = os.path.basename(shell or "")
            if not shell or not os.path.isfile(shell) or shell_name not in {
                "bash", "zsh", "sh"
            }:
                shell = shutil.which("bash") or shutil.which("zsh") or "/bin/sh"
                shell_name = os.path.basename(shell)
            args = [shell]
            if shell_name == "bash":
                args.extend(["--noprofile", "--norc"])
            elif shell_name == "zsh":
                args.append("-f")
            args.append("-s")
            self.process = await asyncio.create_subprocess_exec(
                *args,
                cwd=self.cwd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "TERM": "dumb"},
                start_new_session=True,
            )

        self.on_output(f"Shell ready: {self._shell_label()}\n")
        self.reader_task = asyncio.create_task(self._read_output(self.process))

    def _shell_label(self) -> str:
        if os.name == "nt":
            return "PowerShell" if shutil.which("pwsh") or shutil.which("powershell") else "Windows shell"
        return os.path.basename(os.environ.get("SHELL", "/bin/sh"))

    async def submit(self, command: str) -> bool:
        if self.busy:
            self.on_output("A command is already running in this terminal.\n")
            return False
        await self.start()
        process = self.process
        if process is None or process.stdin is None or process.returncode is not None:
            self.on_output("Shell is not running; submit the command again to restart it.\n")
            return False

        marker = f"__HAWK_COMMAND_DONE_{uuid.uuid4().hex}__"
        self._marker = marker
        self.busy = True
        if os.name == "nt":
            payload = f"{command}\nWrite-Output '{marker}'\n"
        else:
            payload = f"{command}\nprintf '%s\\n' '{marker}'\n"
        process.stdin.write(payload.encode("utf-8", errors="replace"))
        await process.stdin.drain()
        return True

    async def _read_output(self, process: asyncio.subprocess.Process) -> None:
        if process.stdout is None:
            return
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace")
            if self._marker and self._marker in text:
                self._marker = None
                self.busy = False
                self.on_output("\n[command finished]\n")
            else:
                self.on_output(text)
        self.busy = False
        if self.process is process:
            self.process = None
        self.on_output("\n[ shell session ended ]\n")

    async def stop(self) -> None:
        process = self.process
        if process is None or process.returncode is not None:
            return
        self.on_output("\n[stopping shell session]\n")
        try:
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=3)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

    async def close(self) -> None:
        await self.stop()
        if self.reader_task and not self.reader_task.done():
            try:
                await asyncio.wait_for(self.reader_task, timeout=1)
            except asyncio.TimeoutError:
                self.reader_task.cancel()
