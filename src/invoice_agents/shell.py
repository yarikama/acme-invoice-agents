"""The interactive console in the terminal: `python main.py` with no file.

It starts the API and the web console (unless --no-serve), then reads
commands: /run a file, /dir a folder, /runs, /approve or /reject a run
waiting for the VP, /eval, and the rest in COMMANDS. A path typed or dragged
in on its own is run like /run, and any other sentence goes to the assistant
(assistant.py), which turns it into commands. Typing "/" opens a menu of
commands that Tab moves through; unknown commands get the closest match.

One event loop serves the whole session, so the in-process graph remembers
earlier invoices (duplicate detection) and Ctrl-C can stop one run without
leaving the console.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import difflib
import glob
import os
import shlex
import signal
import subprocess
import sys
import webbrowser
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from langchain_core.language_models import BaseChatModel
from prompt_toolkit import completion
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition, has_completions, is_searching
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent, merge_key_bindings
from prompt_toolkit.key_binding.defaults import load_key_bindings
from prompt_toolkit.layout import (
    ConditionalContainer,
    FormattedTextControl,
    HSplit,
    Layout,
    Window,
)
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.layout.processors import BeforeInput
from prompt_toolkit.styles import Style
from rich.markup import escape
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from invoice_agents import assistant, remote
from invoice_agents.cli import (
    Connection,
    LocalRunner,
    connect,
    console,
    invoice_files,
    process,
)
from invoice_agents.config import get_settings
from invoice_agents.inventory import Inventory
from invoice_agents.ledger import Ledger
from invoice_agents.llm import chat_model
from invoice_agents.payment import MockBank
from invoice_agents.terminal import OUTCOME_STYLE, OUTCOME_WORD, SIGNAL, money

LOGS = Path(".acme-logs")
VP_MODES = ("ask", "approve", "reject")
HISTORY = LOGS / "history"


@dataclass(frozen=True)
class Command:
    name: str
    usage: str
    help: str


COMMANDS = [
    Command("run", "/run <file>...", "Process invoices. A path on its own works too."),
    Command(
        "dir", "/dir [folder]", "Process every file in a folder (default: samples)."
    ),
    Command("runs", "/runs", "Recent runs and where each one ended."),
    Command("approve", "/approve <run> [note]", "Approve a run waiting for the VP."),
    Command("reject", "/reject <run> <reason>", "Reject a run waiting for the VP."),
    Command("vp", "/vp [ask|approve|reject]", "How to answer when a run needs the VP."),
    Command("serve", "/serve", "Start the API and the web console."),
    Command("open", "/open", "Open the web console in the browser."),
    Command("status", "/status", "Provider, models, and what is running."),
    Command("inventory", "/inventory", "Stock and list prices the agents check."),
    Command("eval", "/eval [set]", "Score the agents: samples, adversarial, held_out."),
    Command(
        "reset",
        "/reset",
        "Start over: forget past runs, so the samples aren't duplicates.",
    ),
    Command("clear", "/clear", "Clear the screen."),
    Command("help", "/help [command]", "All commands, or one."),
    Command("quit", "/quit", "Stop what this session started and leave (Ctrl-D)."),
]
BY_NAME = {c.name: c for c in COMMANDS}
ALIASES = {"exit": "quit", "q": "quit", "h": "help", "?": "help", "ls": "runs"}
# Commands whose argument is a path, for Tab completion.
PATH_ARGS = {"run", "dir"}


def suggest(word: str) -> str | None:
    """The command a mistyped word most likely meant."""
    names = [*BY_NAME, *ALIASES]
    match = difflib.get_close_matches(word, names, n=1, cutoff=0.5)
    if not match:
        match = [n for n in BY_NAME if n.startswith(word[:2])][:1] if word else []
    return ALIASES.get(match[0], match[0]) if match else None


def help_table(only: str | None = None) -> Padding:
    table = Table.grid(padding=(0, 3))
    table.add_column(no_wrap=True)
    table.add_column(style="#a6a69f")
    for c in COMMANDS:
        if only in (None, c.name):
            # Text, not markup: "[folder]" would read as a style tag.
            table.add_row(Text(c.usage, style=f"bold {SIGNAL}"), c.help)
    return Padding(table, (0, 0, 0, 2))


def complete(text: str, line: str) -> list[str]:
    """Candidates for `text`, the word being typed in `line`."""
    words = line.split()
    first = not words or (len(words) == 1 and not line.endswith(" "))
    command = words[0].lstrip("/") if words else ""
    if not first and command not in PATH_ARGS:
        choices = {"vp": VP_MODES, "help": tuple(BY_NAME)}.get(command, ())
        return [c for c in choices if c.startswith(text)]
    commands = (
        [f"/{n} " for n in BY_NAME if n.startswith(text[1:])]
        if first and text.startswith("/")
        else []
    )
    # "/ru" is a command being typed; a path has a second "/" ("/Users/...").
    if first and (not text or (commands and "/" not in text[1:])):
        return commands
    # "/Users/..." is a path too: a file dragged in starts with "/".
    found = sorted(glob.glob(str(Path(text).expanduser()) + "*"))
    return commands + [p + "/" if Path(p).is_dir() else p for p in found]


class Completer(completion.Completer):
    """The menu under the input: commands with what they do, then paths.

    It opens by itself for "/" and for paths; for a sentence to the
    assistant it stays shut unless Tab asks for it."""

    def get_completions(
        self, document: Document, complete_event: completion.CompleteEvent
    ) -> Iterator[completion.Completion]:
        line = document.text_before_cursor
        text = document.get_word_before_cursor(WORD=True)
        typing_a_path = "/" in text or text.startswith((".", "~"))
        if not (
            line.startswith("/") or typing_a_path or complete_event.completion_requested
        ):
            return
        for option in complete(text, line):
            name = option.strip().lstrip("/")
            meta = (
                BY_NAME[name].help if option.startswith("/") and name in BY_NAME else ""
            )
            yield completion.Completion(
                option,
                start_position=-len(text),
                display=option.strip(),
                display_meta=meta,
            )


class Prompt:
    """A chat-style input: a rule, the line, a rule, then hints. Typing "/"
    swaps the hints for a menu of commands; Tab and the arrows move in it."""

    def __init__(self, status: Callable[[], str]) -> None:
        HISTORY.parent.mkdir(parents=True, exist_ok=True)
        self.status = status
        self.buffer = Buffer(
            completer=Completer(),
            complete_while_typing=True,
            history=FileHistory(str(HISTORY)),
            multiline=False,
        )
        line = Window(
            BufferControl(
                self.buffer,
                input_processors=[BeforeInput("> ", style="class:arrow")],
            ),
            height=Dimension(min=1),
            dont_extend_height=True,
            wrap_lines=True,
        )
        rule = Window(char="─", height=1, style="class:rule")
        hints = Window(FormattedTextControl(self._hints), height=1)
        root = HSplit(
            [
                rule,
                line,
                rule,
                ConditionalContainer(hints, filter=~has_completions),
                CompletionsMenu(max_height=8, scroll_offset=1),
            ]
        )
        self.app: Application[str] = Application(
            layout=Layout(root, focused_element=line),
            key_bindings=merge_key_bindings([load_key_bindings(), self._keys()]),
            style=STYLE,
            erase_when_done=True,
        )

    def _keys(self) -> KeyBindings:
        keys = KeyBindings()

        # Not while searching history (Ctrl-R): there Enter takes the match.
        @keys.add("enter", filter=~is_searching)
        def _(event: KeyPressEvent) -> None:
            state = self.buffer.complete_state
            if state and state.current_completion:
                # Enter on a menu item takes it, and runs it when it needs
                # nothing more: /runs runs, /run waits for a file.
                self.buffer.apply_completion(state.current_completion)
                name = self.buffer.text.strip().lstrip("/")
                if name not in BY_NAME or " " in BY_NAME[name].usage:
                    return
            self.buffer.append_to_history()
            event.app.exit(result=self.buffer.text)

        @keys.add("s-tab")
        def _(_event: KeyPressEvent) -> None:
            self.buffer.complete_previous()

        # Not eager: Option+arrow and Option+Delete arrive as Esc then a key,
        # and an eager Esc would swallow them.
        @keys.add("escape", filter=has_completions)
        def _(_event: KeyPressEvent) -> None:
            self.buffer.cancel_completion()

        @keys.add("c-z", filter=Condition(lambda: hasattr(signal, "SIGTSTP")))
        def _(event: KeyPressEvent) -> None:
            event.app.suspend_to_background()

        @keys.add("c-c")
        def _(event: KeyPressEvent) -> None:
            event.app.exit(exception=KeyboardInterrupt())

        @keys.add("c-d")
        def _(event: KeyPressEvent) -> None:
            if not self.buffer.text:
                event.app.exit(exception=EOFError())

        return keys

    def _hints(self) -> StyleAndTextTuples:
        hints = [("/", " commands"), ("tab", " next"), ("↑", " history")]
        hints += [("ctrl-c", " stop"), ("ctrl-d", " quit")]
        line: StyleAndTextTuples = []
        for i, (key, what) in enumerate(hints):
            line += [("", "  " if i == 0 else "   "), ("class:key", key)]
            line.append(("class:hint", what))
        line.append(("class:hint", "   ·   " + self.status()))
        return line

    def read(self) -> str:
        self.buffer.reset()
        text = self.app.run()
        # The box is erased; what was typed stays, like a sent message.
        console.print(Text.assemble(("> ", f"bold {SIGNAL}"), (text, "#c3c2b7")))
        return text


STYLE = Style.from_dict(
    {
        "rule": "#3a3a37",
        "arrow": f"bold {SIGNAL}",
        "key": "#c3c2b7",
        "hint": "#7c7c75",
        "completion-menu": "bg:default #c3c2b7",
        "completion-menu.completion": "bg:default #c3c2b7",
        "completion-menu.completion.current": f"bg:default bold {SIGNAL}",
        "completion-menu.meta.completion": "bg:default #7c7c75",
        "completion-menu.meta.completion.current": "bg:default #a6a69f",
        "scrollbar.background": "bg:default",
        "scrollbar.button": "bg:#3a3a37",
    }
)


def _fancy_prompt(status: Callable[[], str]) -> Prompt | None:
    """The input box, or None for plain input(): piped input, a dumb terminal,
    or a terminal prompt_toolkit can't drive."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    if os.environ.get("TERM") in ("dumb", "unknown"):
        return None
    try:
        return Prompt(status)
    except Exception as exc:  # pragma: no cover - depends on the terminal
        console.print(f"  [bold]Plain input instead:[/] {escape(str(exc))}")
        return None


class Shell:
    def __init__(self, args: argparse.Namespace, loop: asyncio.Runner) -> None:
        self.args = args
        self.loop = loop
        self.client = httpx.AsyncClient()
        self.vp_mode: str = args.vp
        self.started: list[subprocess.Popen[bytes]] = []
        self.local: LocalRunner | None = None
        self.conn: Connection | None = None
        # Results of runs in this process, for /runs when there is no API.
        self.history: list[dict[str, Any]] = []
        self.llm: BaseChatModel | None = None
        self.handlers: dict[str, Callable[[list[str]], Any]] = {
            "run": self.run,
            "dir": self.dir,
            "runs": self.runs,
            "approve": lambda a: self.decide(a, approved=True),
            "reject": lambda a: self.decide(a, approved=False),
            "vp": self.vp,
            "serve": self.serve,
            "open": self.open,
            "status": self.status,
            "inventory": self.inventory,
            "eval": self.eval,
            "reset": self.reset,
            "clear": lambda _a: console.clear(),
            "help": self.help,
        }

    # --- the loop -------------------------------------------------------------

    def start(self) -> None:
        if not self.args.no_serve:
            self._serve(quiet=True)
        self._connect()
        assert self.conn is not None
        console.print(self.conn.banner(self.args.web, start="/serve"))
        console.print(
            Text.assemble(
                "  Type ",
                ("/help", f"bold {SIGNAL}"),
                " for commands, a path to run an invoice, or ask in plain words.\n",
                style="#a6a69f",
            )
        )

    def loop_forever(self) -> int:
        prompt = _fancy_prompt(self._status_line)
        while True:
            try:
                line = prompt.read() if prompt else input()
            except EOFError:
                console.print()
                return 0
            except KeyboardInterrupt:
                console.print()
                continue
            try:
                if not self.handle(line):
                    return 0
            except KeyboardInterrupt:
                # Rich's live view is gone; the API (if any) keeps the run.
                console.print(
                    "\n  Stopped. The run keeps going in the API if it was there."
                )

    def handle(self, line: str) -> bool:
        """Run one line. False means leave."""
        line = line.strip()
        if not line:
            return True
        try:
            words = shlex.split(line)
        except ValueError:
            # "what's waiting?" has an open quote; only a sentence does that.
            words = []
        paths = [Path(w).expanduser() for w in words]
        # A file dragged in arrives as an absolute path, starting with "/".
        if paths and all(p.exists() for p in paths):
            head = words[0].lstrip("/").lower()
            if not (words[0].startswith("/") and head in {*BY_NAME, *ALIASES}):
                self.run(words)
                return True
        if not line.startswith("/") or not words:
            self.ask(line)
            return True
        head, rest = words[0], words[1:]
        name = ALIASES.get(head[1:].lower(), head[1:].lower())
        if name == "quit":
            return False
        handler = self.handlers.get(name)
        if handler is None:
            hint = suggest(name)
            message = f"  [bold]Unknown command {escape(head)}.[/]"
            if hint:
                message += f" Did you mean [bold {SIGNAL}]/{hint}[/]?"
            console.print(message + " Type /help for all commands.")
            return True
        handler(rest)
        return True

    def close(self) -> None:
        # Servers first: when the terminal was closed, printing fails, and
        # that must not leave the API and Vite running.
        if self.started:
            with contextlib.suppress(OSError):
                console.print("  Stopping the API and the web console…")
        remote.stop(self.started)
        with contextlib.suppress(Exception):
            self.loop.run(self.client.aclose())

    # --- commands -------------------------------------------------------------

    def run(self, args: list[str]) -> None:
        if not args:
            self._usage("run")
            return
        targets = [Path(a).expanduser() for a in args]
        missing = [t for t in targets if not t.exists()]
        if missing:
            names = ", ".join(escape(str(t)) for t in missing)
            console.print(f"  [bold]No such file:[/] {names}")
            return
        paths = [p for t in targets for p in invoice_files(t)]
        self._process(paths)

    def dir(self, args: list[str]) -> None:
        if len(args) > 1:
            self._usage("dir")
            return
        folder = Path(args[0]).expanduser() if args else get_settings().samples_dir
        if not folder.is_dir():
            console.print(f"  [bold]Not a folder:[/] {escape(str(folder))}")
            return
        self._process(invoice_files(folder))

    def runs(self, _args: list[str]) -> None:
        self._connect()
        assert self.conn is not None
        if self.conn.api_up:
            rows = self.loop.run(self._api_runs())
        else:
            rows = [
                {
                    "id": "",
                    "source": r.get("file"),
                    "status": r.get("status"),
                    **(r.get("invoice") or {}),
                }
                for r in reversed(self.history)
            ]
        if not rows:
            console.print("  No runs yet. Try /dir to run the samples.")
            return
        table = Table.grid(padding=(0, 2))
        for _ in range(5):
            table.add_column()
        for r in rows[:20]:
            status = str(r.get("status") or "")
            table.add_row(
                Text(str(r.get("id") or "")[:6], style="#7c7c75"),
                Text(
                    f" {OUTCOME_WORD.get(status, status)} ",
                    style=OUTCOME_STYLE.get(status, "reverse"),
                ),
                Text(str(r.get("invoice_number") or r.get("source") or ""), "bold"),
                Text(str(r.get("vendor") or "")),
                Text(
                    money(r.get("total"), r.get("currency")) if r.get("total") else ""
                ),
            )
        console.print(table)
        if any(r.get("status") == "awaiting_vp" for r in rows):
            console.print(
                "\n  Answer a waiting run with /approve <run> or /reject <run>.",
                style="#a6a69f",
            )

    def decide(self, args: list[str], *, approved: bool) -> None:
        verb = "approve" if approved else "reject"
        # A rejection goes to the vendor and the audit log: it needs a reason.
        if not args or (not approved and len(args) < 2):
            self._usage(verb)
            return
        self._connect()
        assert self.conn is not None
        if not self.conn.api_up:
            console.print(
                "  Runs in this process are answered when they pause (see /vp)."
                " /approve and /reject answer runs waiting in the web console;"
                " start it with /serve."
            )
            return
        waiting = [
            r
            for r in self.loop.run(self._api_runs())
            if r["status"] == "awaiting_vp" and r["id"].startswith(args[0])
        ]
        if len(waiting) != 1:
            console.print(
                f"  [bold]{'No' if not waiting else 'More than one'} run waiting"
                f" for the VP starts with {escape(args[0])}.[/] See /runs."
            )
            return
        run = waiting[0]
        note = " ".join(args[1:]) or "decided in the CLI"
        api = remote.Api(self.args.api, self.client)
        try:
            self.loop.run(api.decide(run["id"], approved, note))
        except httpx.HTTPStatusError as exc:
            reason = exc.response.json().get("detail", exc.response.text)
            console.print(f"  [bold]The API refused:[/] {escape(str(reason))}")
            return
        console.print(
            f"  {'Approved' if approved else 'Rejected'}"
            f" {run.get('invoice_number') or run['source']}. See /runs."
        )

    def vp(self, args: list[str]) -> None:
        if not args:
            console.print(f"  When a run needs the VP: [bold]{self.vp_mode}[/]")
            return
        if args[0] not in VP_MODES:
            self._usage("vp")
            return
        self.vp_mode = args[0]
        console.print(f"  When a run needs the VP: [bold]{self.vp_mode}[/]")

    def serve(self, _args: list[str]) -> None:
        self._serve(quiet=False)
        self._connect()
        assert self.conn is not None
        console.print(self.conn.banner(self.args.web, start="/serve"))

    def open(self, _args: list[str]) -> None:
        self._connect()
        assert self.conn is not None
        if not self.conn.web_up:
            console.print("  The web console isn't running. Start it with /serve.")
            return
        webbrowser.open(self.args.web)
        console.print(f"  Opened {self.args.web}")

    def status(self, _args: list[str]) -> None:
        self._connect()
        assert self.conn is not None
        s = get_settings()
        table = Table.grid(padding=(0, 3))
        table.add_column(style="#a6a69f")
        table.add_column()
        table.add_row("Provider", s.llm_provider)
        table.add_row("Extraction", s.extract_model)
        table.add_row("Approver", s.review_model)
        table.add_row("Critic", s.critic_model or s.review_model)
        table.add_row("Triage", s.triage_model or "off")
        table.add_row("VP answers", self.vp_mode)
        table.add_row("API", self.args.api if self.conn.api_up else "not running")
        table.add_row(
            "Web console", self.args.web if self.conn.web_up else "not running"
        )
        console.print(table)

    def reset(self, _args: list[str]) -> None:
        self._connect()
        assert self.conn is not None
        if self.conn.api_up:
            refused = self.loop.run(remote.Api(self.args.api, self.client).reset())
            if refused:
                console.print(f"  [bold]Not reset:[/] {escape(refused)}")
                return
        else:
            s = get_settings()
            Ledger(s.db_path, s.decisions_log).clear()
            MockBank(s.db_path).clear()
        self.history.clear()
        console.print(
            "  Started over: no past runs, and no invoice counts as received."
        )

    def inventory(self, _args: list[str]) -> None:
        table = Table(box=None, padding=(0, 3), header_style="#a6a69f")
        table.add_column("Item", style="bold")
        table.add_column("In stock", justify="right")
        table.add_column("List price", justify="right")
        db = Inventory(get_settings().db_path)
        db.seed()  # creates the database on a first run
        for r in db.records():
            table.add_row(r.item, str(r.stock), money(r.unit_price, "USD"))
        console.print(table)
        vendors = db.vendors()
        console.print(
            f"\n  Vendor master ({len(vendors)}): " + escape(", ".join(vendors)),
            style="#a6a69f",
        )

    def eval(self, args: list[str]) -> None:
        from invoice_agents import evaluation

        which = args[0] if args else "samples"
        if which not in evaluation.SETS:
            console.print(
                f"  [bold]No set {escape(which)}.[/] Sets: {', '.join(evaluation.SETS)}"
            )
            return
        # The evaluation runs its own event loop; ours is idle between commands.
        evaluation.main(["--set", which, *args[1:]])

    def help(self, args: list[str]) -> None:
        name = ALIASES.get(args[0].lstrip("/"), args[0].lstrip("/")) if args else None
        if name is not None and name not in BY_NAME:
            hint = suggest(name)
            console.print(
                f"  [bold]No command /{escape(name)}.[/]"
                + (f" Did you mean /{hint}?" if hint else "")
            )
            return
        console.print(help_table(name))
        if name is None:
            console.print(
                "\n  Ctrl-C stops a run, Ctrl-D leaves. Runs show in the web console"
                " too when the API is up.",
                style="#7c7c75",
            )

    def ask(self, request: str) -> None:
        """Hand a sentence to the assistant, then run what it chose."""
        s = get_settings()
        if s.llm_provider not in ("openrouter", "xai"):
            console.print(
                "  [bold]Plain-language requests need a live model.[/] Set"
                " LLM_PROVIDER=openrouter and OPENROUTER_API_KEY, or use a /command:"
                " type / to see them."
            )
            return
        if self.llm is None:
            self.llm = chat_model(s, s.extract_model, "low")
        try:
            with console.status(f"Asking {s.extract_model}…"):
                answer, tokens = self.loop.run(assistant.plan(self.llm, request))
        except Exception as exc:  # the provider, the network or a bad answer
            console.print(
                f"  [bold]The assistant couldn't answer:[/] {escape(str(exc))}"
            )
            return
        steps, dropped = assistant.checked(answer)
        used = sum(u["input_tokens"] + u["output_tokens"] for u in tokens.values())
        head = Text("  ◆ ", style=SIGNAL)
        head.append(s.extract_model, style="bold")
        head.append(f"  {used:,} tokens" if used else "", style="#7c7c75")
        console.print(head)
        console.print(Padding(Text(answer.reply), (0, 0, 0, 4)))
        for reason in dropped:
            console.print(Text(f"    ✕ Not run: {reason}", style="#a6a69f"))
        for step in steps:
            command = shlex.join([f"/{step.command}", *step.args])
            console.print(Text.assemble("    → ", (command, f"bold {SIGNAL}")))
        if steps:
            console.print()
        for step in steps:
            self.handlers[step.command](step.args)

    def _status_line(self) -> str:
        s = get_settings()
        api = "API up" if self.conn and self.conn.api_up else "local"
        return f"{s.llm_provider} · {api} · VP: {self.vp_mode}"

    # --- helpers --------------------------------------------------------------

    def _usage(self, name: str) -> None:
        c = BY_NAME[name]
        console.print(
            Text.assemble("  Usage: ", (c.usage, f"bold {SIGNAL}"), f"  {c.help}")
        )

    def _connect(self) -> None:
        self.conn = self.loop.run(connect(self.args, self.client, self.local))
        if isinstance(self.conn.runner, LocalRunner):
            self.local = self.conn.runner

    def _serve(self, *, quiet: bool) -> None:
        try:
            with console.status("Starting the API and the web console…"):
                self.started += self.loop.run(
                    remote.serve(self.args.api, self.args.web, LOGS)
                )
        except RuntimeError as exc:
            console.print(f"  [bold]{escape(str(exc))}[/] Running here instead.")
            return
        if not quiet:
            console.print("  The API and the web console are up.")

    def _process(self, paths: list[Path]) -> None:
        if not paths:
            console.print("  [bold]No invoice files there.[/]")
            return
        self._connect()
        assert self.conn is not None
        results = self.loop.run(process(self.conn.runner, paths, self.vp_mode))
        self.history += results

    async def _api_runs(self) -> list[dict[str, Any]]:
        r = await self.client.get(f"{self.args.api}/api/runs", timeout=10)
        r.raise_for_status()
        return r.json()


def run_shell(args: argparse.Namespace) -> int:
    with asyncio.Runner() as loop:
        shell = Shell(args, loop)
        try:
            shell.start()
            target = args.invoice_path or args.dir
            if target is not None:
                shell.run([str(target)])
            return shell.loop_forever()
        except KeyboardInterrupt:
            console.print("\n  Stopped.")
            return 130
        finally:
            shell.close()
