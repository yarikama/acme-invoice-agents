"""What the CLI draws: the banner, a live view of each run, the VP card and
the batch summary. The same story the web console tells, in a terminal.

A run is fed events in the shape the API streams them:
  {"type": "trace", "node": ..., "message": ..., "tool": ..., "at": seconds}
  {"type": "progress", "stage": ..., "step": ...}
so the view works the same whether the graph runs here or behind the API.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from rich.console import Group, RenderableType
from rich.padding import Padding
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

SIGNAL = "#ff5a1f"
VOICE = {"model": "#6aa8ee", "code": "#5fd38a", "human": "#ff7a2f"}
OUTCOME_STYLE = {
    "paid": "bold #0b0b0b on #5fd38a",
    "rejected": "bold #f0efec on #3987e5",
    "awaiting_vp": f"bold #111111 on {SIGNAL}",
    "skipped": "bold #111111 on #c3c2b7",
    "failed": "bold #f0efec on #8a2b2b",
}
OUTCOME_WORD = {
    "paid": "Paid",
    "rejected": "Rejected",
    "awaiting_vp": "Waiting for a VP",
    "skipped": "Skipped",
    "failed": "Failed",
}
DECISION_STYLE = {
    "approve": "bold #0b0b0b on #5fd38a",
    "reject": "bold #f0efec on #3987e5",
    "escalate": f"bold #111111 on {SIGNAL}",
}


# --- Banner -------------------------------------------------------------------

MARK = [
    ("▐████▌ ", "●"),
    ("▐██████▌", ""),
    ("▐██████▌", ""),
    ("▝▀▀▀▀▀▀▘", ""),
]
WORDMARK = [
    "▄▀█ █▀▀ █▀▄▀█ █▀▀   ▄▀█ █▀█",
    "█▀█ █▄▄ █ ▀ █ ██▄   █▀█ █▀▀",
]


def banner(
    console_url: str | None, *, live: bool, mirrored: bool, mode: str
) -> RenderableType:
    """The Acme mark and name, then where to watch the run in a browser.

    `mirrored` says whether runs go through the API, and so show in the console.
    """
    art = Text()
    lines = [*WORDMARK, "invoice agents: Grok judges, code decides", ""]
    for i, (sheet, dot) in enumerate(MARK):
        art.append("  " + sheet, style="bold #edede9")
        art.append(dot.ljust(2), style=f"bold {SIGNAL}")
        art.append("   ")
        style = "bold" if i < 2 else "#a6a69f"
        art.append(lines[i], style=style)
        art.append("\n")
    where = Text("  ")
    if console_url and live:
        where.append("● ", style="#5fd38a")
        where.append("Web console  ", style="bold")
        where.append(console_url, style=f"underline {SIGNAL} link {console_url}")
        where.append(
            "  open it to watch these runs there too"
            if mirrored
            else "  runs from --local happen here only, so they won't show there",
            style="#a6a69f",
        )
    else:
        where.append("○ ", style="#a6a69f")
        where.append("Web console offline. ", style="bold")
        where.append("Run ", style="#a6a69f")
        where.append("python main.py --serve", style="bold")
        where.append(" to start it, then open ", style="#a6a69f")
        where.append(console_url or "http://localhost:5173", style="underline")
    how = Text(f"  {mode}", style="#7c7c75")
    return Group(art, where, how, Text())


# --- The run as a conversation -------------------------------------------------

DECISION = r"(approve|reject|escalate)"


def sentences(text: str) -> list[str]:
    """Prose as sentences; "1.08" and "$4,455.00" stay whole."""
    # Ends after . ! ? or after one of them inside a closing bracket or quote.
    parts = re.split(r"(?:(?<=[.!?])|(?<=[.!?][)\"']))\s+(?=[A-Z(~$€£\"'])", text)
    return [p.strip() for p in parts if p.strip()]


@dataclass
class Turn:
    kind: str
    node: str
    took: float | None = None
    data: dict[str, Any] = field(default_factory=dict)
    # Tokens and cost of the turn's model calls, from its trace line.
    usage: dict[str, Any] | None = None


def turns(events: list[dict[str, Any]], started: float | None) -> list[Turn]:
    """Trace events grouped into one turn per actor, tool calls inside the
    approver's turn, each timed since the turn before it."""
    out: list[Turn] = []
    tools: list[tuple[str, str]] = []
    sendbacks: list[str] = []
    last = started

    def timed(at: float | None) -> float | None:
        nonlocal last
        took = at - last if at is not None and last is not None else None
        if at is not None:
            last = at
        return None if took is None else max(took, 0.0)

    def flush(at: float | None) -> None:
        nonlocal tools, sendbacks
        if tools or sendbacks:
            out.append(
                Turn(
                    "approver",
                    "review",
                    timed(at),
                    {"tools": tools, "sendbacks": sendbacks},
                )
            )
            tools, sendbacks = [], []

    for e in events:
        if e.get("type") != "trace":
            continue
        node, msg, at = e["node"], e["message"], e.get("at")
        tool = e.get("tool")
        if tool and tool != "guardrail":
            call, _, result = msg.partition("): ")
            tools.append((call + (")" if result else ""), result))
            continue
        if tool == "guardrail":
            sendbacks.append(re.sub(r"^Sent the decision back\.\s*", "", msg))
            continue
        if node == "review":
            m = re.match(rf"^Draft (\d+): {DECISION}\.\s*(.*)$", msg, re.S)
            data: dict[str, Any] = {"tools": tools, "sendbacks": sendbacks}
            if m:
                data |= {
                    "round": int(m[1]),
                    "decision": m[2],
                    "rationale": m[3].strip(),
                }
            else:
                data |= {"rationale": msg}
            out.append(Turn("approver", "review", timed(at), data, e.get("usage")))
            tools, sendbacks = [], []
            continue
        flush(at)
        before = len(out)
        if node == "validate":
            m = re.match(
                r"^(\d+) errors?, (\d+) warnings?; "
                r"policy route: (\w+)(?: \((.*)\))?\.?$",
                msg,
            )
            data = (
                {
                    "errors": int(m[1]),
                    "warnings": int(m[2]),
                    "route": m[3],
                    "reason": m[4],
                }
                if m
                else {"text": msg}
            )
            out.append(Turn("validate", node, timed(at), data))
        elif node == "triage":
            m = re.search(rf"says {DECISION} \((\d+)% sure\), suspicion (\d+)%", msg)
            data = {"text": msg, "settled": "approver was skipped" in msg}
            if m:
                data |= {"decision": m[1], "sure": int(m[2]), "suspicion": int(m[3])}
            out.append(Turn("triage", node, timed(at), data))
        elif node == "critique":
            outcome = re.search(r"Outcome: (\w+)\.", msg)
            body = re.sub(r"\s*Outcome: \w+\.\s*$", "", msg)
            agrees = body.startswith("Critic agrees")
            out.append(
                Turn(
                    "critic",
                    node,
                    timed(at),
                    {
                        "agrees": agrees,
                        "issues": ""
                        if agrees
                        else re.sub(r"^Critic objects:\s*", "", body),
                        "outcome": outcome[1] if outcome else None,
                    },
                )
            )
        elif node == "human_review":
            m = re.match(
                r"^VP (approved|rejected)(?: \(([^)]*)\))?\.\s*(?:Note:\s*)?(.*)$",
                msg,
                re.S,
            )
            out.append(
                Turn(
                    "human",
                    node,
                    timed(at),
                    {
                        "approved": bool(m and m[1] == "approved"),
                        "by": m[2] if m else None,
                        "note": (m[3].strip() if m else ""),
                    },
                )
            )
        else:
            out.append(Turn("step", node, timed(at), {"text": msg}))
        if len(out) > before:
            out[-1].usage = e.get("usage")
    if events:
        flush(None)
    return out


def short_model(name: str) -> str:
    """ "x-ai/grok-4.7" -> "grok-4.7"; several models are kept, joined."""
    return ", ".join(n.strip().rsplit("/", 1)[-1] for n in name.split(","))


def spent(usage: dict[str, Any] | None) -> str:
    """ "2.1k tokens · $0.0042", or "" for a step with no model calls."""
    if not usage:
        return ""
    n = int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
    words = f"{n / 1000:.1f}k tokens" if n >= 1000 else f"{n} tokens"
    usd = usage.get("cost_usd")
    if usd is not None:
        words += f" · ${usd:.4f}" if usd < 0.01 else f" · ${usd:.2f}"
    return words


def run_usage(all_turns: list[Turn]) -> dict[str, Any] | None:
    """The run's tokens, and its cost if every step was priced."""
    used = [t.usage for t in all_turns if t.usage]
    if not used:
        return None
    costs = [u.get("cost_usd") for u in used]
    return {
        "input_tokens": sum(int(u.get("input_tokens", 0)) for u in used),
        "output_tokens": sum(int(u.get("output_tokens", 0)) for u in used),
        "cost_usd": None if None in costs else sum(c or 0.0 for c in costs),
    }


def _took(seconds: float | None) -> str:
    if seconds is None:
        return ""
    return f"{seconds * 1000:.0f}ms" if seconds < 1 else f"{seconds:.1f}s"


def _pill(word: str, style: str) -> Text:
    """A word on a coloured block; text appended after it stays unstyled."""
    pill = Text()
    pill.append(f" {word} ", style=style)
    return pill


GLYPH = {
    "ingest": "▤",
    "validate": "✓",
    "triage": "ϟ",
    "review": "✎",
    "critique": "⌕",
    "human_review": "◉",
    "pay": "$",
    "reject": "✕",
    "error": "!",
}
NAME = {
    "ingest": "Ingest",
    "pay": "Payment",
    "reject": "Not paid",
    "error": "Run failed",
}


def _header(
    node: str,
    name: str,
    role: str,
    voice: str,
    took: float | None,
    usage: dict[str, Any] | None = None,
) -> Table:
    row = Table.grid(expand=True)
    row.add_column()
    row.add_column(justify="right")
    left = Text()
    left.append(f" {GLYPH.get(node, '•')} ", style=f"bold #111111 on {VOICE[voice]}")
    left.append(f"  {name}", style="bold")
    # A model's turn names the model that served it, when the trace says.
    if usage and usage.get("model") and voice == "model":
        role = short_model(str(usage["model"]))
    left.append(f"  {role}", style=VOICE[voice])
    right = "  ".join(x for x in (spent(usage), _took(took)) if x)
    row.add_row(left, Text(right, style="#7c7c75"))
    return row


def render_turn(t: Turn) -> RenderableType:
    d = t.data
    body: list[RenderableType] = []
    if t.kind == "validate":
        head = _header("validate", "Validation", "code", "code", t.took, t.usage)
        if "errors" in d:
            line = Text()
            line.append(
                f"{d['errors']} error{'s' if d['errors'] != 1 else ''}", style="bold"
            )
            line.append("  ·  ")
            line.append(
                f"{d['warnings']} warning{'s' if d['warnings'] != 1 else ''}",
                style="bold",
            )
            line.append(f"  ·  route: {d['route']}")
            if d.get("reason"):
                line.append(f"   {d['reason']}", style="#a6a69f")
            body.append(line)
        else:
            body.append(Text(d["text"]))
    elif t.kind == "triage":
        head = _header("triage", "Triage", "decision model", "model", t.took, t.usage)
        if "decision" in d:
            line = _pill(d["decision"].capitalize(), DECISION_STYLE[d["decision"]])
            line.append(f"  {d['sure']}% sure  ·  suspicion {d['suspicion']}%  ")
            line.append(
                "code confirmed it, so the approver was skipped"
                if d["settled"]
                else "handed to the approver",
                style="#a6a69f",
            )
            body.append(line)
        else:
            body.append(Text(d["text"]))
    elif t.kind == "approver":
        head = _header("review", "Approver", "model", "model", t.took, t.usage)
        for call, result in d.get("tools", []):
            line = Text("⚒ ", style=VOICE["code"])
            line.append(call, style="bold")
            if result:
                line.append(f"\n  ↳ {result}", style="#a6a69f")
            body.append(line)
        body.extend(
            Text(f"⛨ Code sent the decision back. {s}", style=SIGNAL)
            for s in d.get("sendbacks", [])
        )
        if d.get("decision"):
            line = Text(f"Draft {d['round']}  ", style="#a6a69f")
            line.append_text(
                _pill(d["decision"].capitalize(), DECISION_STYLE[d["decision"]])
            )
            body.append(line)
        body.extend(Text(f"▪ {s}") for s in sentences(d.get("rationale", "")))
    elif t.kind == "critic":
        head = _header("critique", "Critic", "model", "model", t.took, t.usage)
        line = _pill(
            "Agrees" if d["agrees"] else "Objects",
            DECISION_STYLE["approve" if d["agrees"] else "reject"],
        )
        if d.get("outcome"):
            line.append(f"  outcome: {d['outcome']}")
        body.append(line)
        if d.get("issues"):
            body.append(Text(d["issues"]))
    elif t.kind == "human":
        name = d.get("by") or "VP"
        head = _header("human_review", name, "human", "human", t.took, t.usage)
        line = _pill(
            "Approved" if d["approved"] else "Rejected",
            DECISION_STYLE["approve" if d["approved"] else "reject"],
        )
        if d.get("note"):
            line.append(f"  {d['note']}")
        body.append(line)
    else:
        head = _header(
            t.node,
            NAME.get(t.node, t.node.capitalize()),
            "code",
            "code",
            t.took,
            t.usage,
        )
        body.append(Text(d["text"]))
    return Group(head, Padding(Group(*body), (0, 0, 1, 6)))


# --- Pipeline lights --------------------------------------------------------------

SEGMENTS = ["ingest", "validate", "draft", "critique", "vp", "pay"]
NODE_SEGMENT = {
    "ingest": "ingest",
    "validate": "validate",
    "triage": "draft",
    "review": "draft",
    "critique": "critique",
    "human_review": "vp",
    "pay": "pay",
    "reject": "pay",
}
LIGHT = {
    "off": ("□", "#5a5a55"),
    "done": ("■", "bold #edede9"),
    "flag": ("▨", "bold #edede9"),
    "block": ("■", "bold #3987e5"),
    "wait": ("■", f"bold {SIGNAL}"),
    "run": ("◆", f"bold {SIGNAL}"),
    "skip": ("·", "#5a5a55"),
    "fail": ("✕", "bold #e66767"),
}


def lights(events: list[dict[str, Any]], status: str) -> dict[str, str]:
    reached = {
        NODE_SEGMENT[e["node"]]
        for e in events
        if e.get("type") == "trace" and e["node"] in NODE_SEGMENT
    }
    states = {s: ("done" if s in reached else "off") for s in SEGMENTS}
    for e in events:
        if e.get("type") == "trace" and e["node"] == "validate":
            m = re.match(r"^(\d+) errors?, (\d+) warnings?", e["message"])
            if m and int(m[1]):
                states["validate"] = "block"
            elif m and int(m[2]):
                states["validate"] = "flag"
    if status == "awaiting_vp":
        states["vp"] = "wait"
    if status == "paid":
        states["pay"] = "done"
    if status == "rejected":
        states["pay"] = "skip"
    if status in ("paid", "rejected") and states["vp"] == "off":
        states["vp"] = "skip"
    if status == "running":
        progress = [e for e in events if e.get("type") == "progress"]
        if progress:
            p = progress[-1]
            seg = {"ingest": "ingest", "validate": "validate", "pay": "pay"}.get(
                p["stage"]
            )
            if p["stage"] == "approve":
                seg = "critique" if p.get("step") == "critique" else "draft"
            if seg and states.get(seg) == "off":
                states[seg] = "run"
    if status == "failed":
        first_off = next((s for s in SEGMENTS if states[s] == "off"), None)
        if first_off:
            states[first_off] = "fail"
    return states


def render_lights(states: dict[str, str]) -> Text:
    line = Text("  ")
    for i, seg in enumerate(SEGMENTS):
        glyph, style = LIGHT[states[seg]]
        line.append(f"{glyph} ", style=style)
        line.append(
            seg, style="bold" if states[seg] not in ("off", "skip") else "#7c7c75"
        )
        if i < len(SEGMENTS) - 1:
            line.append("  ─  ", style="#3a3a37")
    return line


ACTIVITY = {
    "ingest": "Reading the invoice",
    "validate": "Checking the rules",
    "draft": "The approver is reviewing",
    "critique": "The critic is checking the draft",
    "pay": "Paying",
}


class RunView:
    """A live view of one run, redrawn as its events arrive."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.events: list[dict[str, Any]] = []
        self.status = "running"
        self.started = time.time()
        # Turns already printed before a VP pause, not drawn again after it.
        self.skip = 0
        self._spinner = Spinner("dots", style=SIGNAL)

    def add(self, event: dict[str, Any]) -> None:
        if "at" not in event:
            event = {**event, "at": time.time()}
        self.events.append(event)

    def resumed(self) -> RunView:
        """The view after a VP pause: same run, drawing only what comes next."""
        nxt = RunView(self.name)
        nxt.events = list(self.events)
        nxt.started = self.started
        nxt.skip = len(turns(self.events, self.started))
        return nxt

    def __rich__(self) -> RenderableType:
        states = lights(self.events, self.status)
        parts: list[RenderableType] = [render_lights(states)]
        drawn = turns(self.events, self.started)
        total = run_usage(drawn)
        if total:
            parts.append(Text(f"  this run so far: {spent(total)}", style="#7c7c75"))
        parts.append(Text())
        parts += [render_turn(t) for t in drawn[self.skip :]]
        if self.status == "running":
            now = next((s for s in SEGMENTS if states[s] == "run"), None)
            row = Table.grid()
            row.add_row(
                self._spinner,
                Text(f" {ACTIVITY.get(now or '', 'Working')}…", style="italic #a6a69f"),
            )
            parts.append(Padding(row, (0, 0, 0, 2)))
        return Group(*parts)


# --- After a run ------------------------------------------------------------------


def money(amount: float | None, currency: str | None) -> str:
    if amount is None:
        return "no amount"
    sym = {"USD": "$", "EUR": "€", "GBP": "£"}.get(currency or "USD")
    return f"{sym}{amount:,.2f}" if sym else f"{currency} {amount:,.2f}"


def result_line(file: str, result: dict[str, Any]) -> Text:
    status = str(result.get("status") or "failed")
    line = _pill(OUTCOME_WORD.get(status, status), OUTCOME_STYLE.get(status, "reverse"))
    inv = result.get("invoice") or {}
    line.append(f"  {inv.get('invoice_number') or file}", style="bold")
    if inv.get("vendor"):
        line.append(f"  {inv['vendor']}")
    if inv:
        line.append(f"  {money(inv.get('total'), inv.get('currency'))}", style="bold")
    reason = (
        result.get("route_reason") if status == "awaiting_vp" else result.get("reason")
    )
    if reason:
        line.append(f"\n  {reason}", style="#a6a69f")
    return line


def vp_card(summary: dict[str, Any]) -> Panel:
    """The decision a VP needs, in sections like the console's card."""
    inv = summary.get("invoice") or {}
    draft = summary.get("draft") or {}
    parts: list[RenderableType] = []
    head = Text()
    head.append(f"{inv.get('invoice_number') or 'Invoice'}  ", style="bold")
    head.append(f"{inv.get('vendor') or ''}  ")
    head.append(money(inv.get("total"), inv.get("currency")), style="bold")
    parts.append(head)
    if draft.get("decision"):
        line = Text("Agent recommends  ", style="#a6a69f")
        line.append_text(
            _pill(draft["decision"].capitalize(), DECISION_STYLE[draft["decision"]])
        )
        parts += [Text(), line]
    if summary.get("route_reason"):
        reason = summary["route_reason"]
        parts += [
            Text(),
            Text("Why it needs a VP", style="#a6a69f"),
            Text(reason[:1].upper() + reason[1:] + ".", style="bold"),
        ]
    if draft.get("rationale"):
        parts += [Text(), Text("The agent's reasoning", style="#a6a69f")]
        parts += [Text(f"▪ {s}") for s in sentences(draft["rationale"])]
    flags = [f for f in summary.get("findings", []) if f["severity"] != "info"]
    if flags:
        parts += [Text(), Text("Flags to weigh", style="#a6a69f")]
        for f in flags:
            line = Text(f" {f['code']} ", style="bold #f0efec on #2a2a28")
            line.append(f"  {f['detail']}")
            parts.append(line)
    return Panel(
        Group(*parts),
        title=Text(" VP decision needed ", style=f"bold #111111 on {SIGNAL}"),
        title_align="left",
        border_style=SIGNAL,
        padding=(1, 2),
    )


def batch_summary(results: list[dict[str, Any]], seconds: float) -> RenderableType:
    """Totals and where the invoices went, as a text bar chart."""
    counts = Counter(r.get("status", "failed") for r in results)
    total = len(results)
    table = Table.grid(padding=(0, 2))
    table.add_column()
    table.add_column(justify="right")
    table.add_column()
    width = 30
    for status in ("paid", "awaiting_vp", "rejected", "skipped", "failed"):
        n = counts.get(status, 0)
        if not n:
            continue
        bar = "█" * max(1, round(width * n / max(total, 1)))
        table.add_row(
            _pill(OUTCOME_WORD[status], OUTCOME_STYLE[status]),
            Text(str(n), style="bold"),
            Text(bar, style=OUTCOME_STYLE[status].split(" on ")[-1]),
        )
    by_vp = sum(
        1
        for r in results
        if any(t.get("message", "").startswith("VP ") for t in r.get("trace", []))
    )
    alone = counts["paid"] + counts["rejected"] - by_vp
    head = Text()
    head.append(f"{total} invoices in {_took(seconds)}", style="bold")
    to_vp = counts["awaiting_vp"] + by_vp
    head.append(
        f"   {alone} settled with no person, {to_vp} sent to a VP", style="#a6a69f"
    )
    return Panel(
        Group(head, Text(), table),
        title=Text(" Batch ", style="bold"),
        title_align="left",
        border_style="#3a3a37",
        padding=(1, 2),
    )
