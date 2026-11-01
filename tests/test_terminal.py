"""What the CLI draws, rendered to plain text."""

from rich.console import Console, Group

from invoice_agents.terminal import (
    RunView,
    banner,
    batch_summary,
    impact_line,
    lights,
    money,
    render_turn,
    result_card,
    sentences,
    turns,
    vp_card,
)


def text(renderable: object) -> str:
    console = Console(width=120, record=True, color_system=None, force_terminal=False)
    console.print(renderable)
    return console.export_text()


def shown(view: RunView) -> str:
    """What the terminal shows: the turns printed so far, then the live status."""
    return text(Group(*view.take(), view))


def trace(node: str, message: str, at: float, tool: str | None = None) -> dict:
    return {"type": "trace", "node": node, "message": message, "tool": tool, "at": at}


EVENTS = [
    trace("ingest", "Parsed XML in code.", 100.0),
    trace(
        "validate", "0 errors, 1 warning; policy route: human (Billed in EUR).", 100.5
    ),
    trace(
        "triage",
        "m says escalate (98% sure), suspicion 11%. Needs the full review.",
        101.0,
    ),
    trace(
        "review",
        "convert_to_usd(4125.0, EUR): EUR 4,125.00 is $4,455.00.",
        109.0,
        "convert_to_usd",
    ),
    trace(
        "review",
        "Sent the decision back. Missing: FOREIGN_CURRENCY.",
        109.0,
        "guardrail",
    ),
    trace("review", "Draft 1: approve. Converted 4,125.00 at 1.08. Harmless.", 109.0),
    trace("critique", "Critic agrees. Outcome: human.", 112.0),
]


def test_sentences_keep_numbers_whole() -> None:
    assert sentences("Converted 1.08 to $4,455.00. Fine! (Checked.) Done") == [
        "Converted 1.08 to $4,455.00.",
        "Fine!",
        "(Checked.)",
        "Done",
    ]


def test_turns_group_tools_into_the_approver_and_time_each() -> None:
    got = turns(EVENTS, started=99.0)
    assert [t.kind for t in got] == ["step", "validate", "triage", "approver", "critic"]
    assert [round(t.took or 0, 1) for t in got] == [1.0, 0.5, 0.5, 8.0, 3.0]
    approver = got[3].data
    assert approver["tools"] == [
        ("convert_to_usd(4125.0, EUR)", "EUR 4,125.00 is $4,455.00.")
    ]
    assert approver["sendbacks"] == ["Missing: FOREIGN_CURRENCY."]
    assert (approver["round"], approver["decision"]) == (1, "approve")
    assert got[1].data == {
        "errors": 0,
        "warnings": 1,
        "route": "human",
        "reason": "Billed in EUR",
    }
    assert got[2].data["decision"] == "escalate" and not got[2].data["settled"]
    assert got[4].data == {"agrees": True, "issues": "", "outcome": "human"}


def test_turns_handle_odd_lines() -> None:
    got = turns(
        [
            trace("review", "check_inventory(X): not found.", 1, "check_inventory"),
            trace("validate", "something else", 2),
            trace("triage", "unreadable", 3),
            trace("review", "free text", 4),
            trace("critique", "Critic objects: ignores X.", 5),
            trace("human_review", "VP rejected. Note: no", 6),
            trace("error", "boom", None),  # type: ignore[arg-type]
        ],
        started=None,
    )
    assert [t.kind for t in got] == [
        "approver",
        "validate",
        "triage",
        "approver",
        "critic",
        "human",
        "step",
    ]
    assert got[1].data == {"text": "something else"}
    assert got[4].data["issues"] == "ignores X." and not got[4].data["agrees"]
    assert got[5].data == {"approved": False, "by": None, "note": "no"}
    # Timing starts at the first stamped line; the turn before it has no time.
    assert got[0].took is None and got[1].took == 0


def test_lights_follow_the_run() -> None:
    running = lights(
        [*EVENTS[:2], {"type": "progress", "stage": "approve", "step": "draft"}],
        "running",
    )
    assert running["validate"] == "flag" and running["draft"] == "run"
    assert lights(EVENTS, "awaiting_vp")["vp"] == "wait"
    paid = lights(
        [*EVENTS, trace("pay", "mock_payment returned success.", 113)], "paid"
    )
    assert paid["pay"] == "done" and paid["vp"] == "skip"
    blocked = lights(
        [trace("validate", "2 errors, 0 warnings; policy route: reject.", 1)],
        "rejected",
    )
    assert blocked["validate"] == "block" and blocked["pay"] == "skip"
    assert lights([], "failed")["ingest"] == "fail"
    critique = lights(
        [*EVENTS[:6], {"type": "progress", "stage": "approve", "step": "critique"}],
        "running",
    )
    assert critique["critique"] == "run"


def test_run_view_draws_turns_and_resumes_after_the_pause() -> None:
    view = RunView("invoice_1014.xml")
    view.started = 99.0
    for e in EVENTS[:6]:
        view.add(e)
    view.add({"type": "progress", "stage": "approve", "step": "critique"})
    out = shown(view)
    assert "The critic is checking the draft" in out
    view.add(EVENTS[6])
    out += shown(view)
    for words in (
        "Validation",
        "Triage",
        "convert_to_usd(4125.0, EUR)",
        "Code sent the decision back",
        "Agrees",
        "8.0s",
        "500ms",
    ):
        assert words in out

    view.status = "awaiting_vp"
    after = view.resumed()
    after.add(
        {"type": "trace", "node": "human_review", "message": "VP approved. Note: ok"}
    )
    after.status = "paid"
    drawn = shown(after)
    assert "VP" in drawn and "Approved" in drawn
    assert "Triage" not in drawn  # already printed before the pause


def test_only_the_status_is_redrawn() -> None:
    """Finished turns are printed once; the live part stays a few lines tall,
    so it never outgrows the screen and gets cropped or repainted whole."""
    view = RunView("invoice_1014.xml")
    view.started = 99.0
    tool_calls = [e for e in EVENTS if e.get("tool") and e["tool"] != "guardrail"]
    first_tool = EVENTS.index(tool_calls[0])
    for e in EVENTS[: first_tool + 1]:
        view.add(e)
    printed = text(Group(*view.take()))
    assert "convert_to_usd" not in printed  # the approver's turn isn't done
    status = text(view)
    assert "1 tool call so far" in status
    # The call that just came back shows live, on one line.
    assert "⚒ convert_to_usd(4125.0, EUR)  ↳ EUR 4,125.00 is $4,455.00." in status
    assert len(status.strip().splitlines()) <= 4

    for e in EVENTS[first_tool + 1 :]:
        view.add(e)
    printed = text(Group(*view.take()))
    assert "convert_to_usd" in printed and "Triage" not in printed
    assert view.take() == []  # each turn is printed once
    assert "Triage" not in text(view) and len(text(view).strip().splitlines()) <= 3


def test_banner_says_where_runs_show() -> None:
    assert "watch these runs there too" in text(
        banner("http://x", live=True, mirrored=True, mode="m")
    )
    assert "won't show there" in text(
        banner("http://x", live=True, mirrored=False, mode="m")
    )
    assert "python main.py --serve" in text(
        banner(None, live=False, mirrored=False, mode="m")
    )


SUMMARY = {
    "invoice": {
        "invoice_number": "INV-1014",
        "vendor": "TechParts",
        "total": 4125.0,
        "currency": "EUR",
    },
    "draft": {"decision": "approve", "rationale": "Converted at 1.08. Harmless."},
    "route_reason": "billed in EUR",
    "findings": [
        {
            "code": "FOREIGN_CURRENCY",
            "severity": "warning",
            "title": "t",
            "detail": "Payments run in USD.",
        },
        {"code": "NOTE", "severity": "info", "title": "t", "detail": "ignored"},
    ],
}


def test_vp_card_has_each_section() -> None:
    out = text(vp_card(SUMMARY))
    for words in (
        "VP decision needed",
        "€4,125.00",
        "Approve",
        "Billed in EUR.",
        "Harmless.",
        "FOREIGN_CURRENCY",
    ):
        assert words in out
    assert "ignored" not in out


def test_result_card_says_who_decided_and_why() -> None:
    result = {
        **SUMMARY,
        "status": "paid",
        "payment": {
            "status": "success",
            "vendor": "TechParts",
            "amount": 4455.0,
            "currency": "USD",
        },
        "trace": [
            {
                "stage": "approve",
                "message": "Draft 1: approve. Fine.",
                "usage": {"input_tokens": 900, "output_tokens": 100, "cost_usd": 0.004},
            },
            {
                "stage": "approve",
                "message": "Critic agrees. Outcome: human.",
                "usage": {"input_tokens": 400, "output_tokens": 50, "cost_usd": 0.001},
            },
            {"stage": "approve", "message": "VP approved (Dana). Note: ok"},
        ],
    }
    out = text(result_card("f", result))
    for words in (
        "Paid",
        "INV-1014",
        "€4,125.00",
        "Approver  Approve",
        "Critic  Agrees",
        "VP  Approved",
        "▪ Converted at 1.08.",
        "FOREIGN_CURRENCY",
        "Paid $4,455.00 to TechParts",
        "1.4k tokens · $0.0050 across 2 model steps",
    ):
        assert words in out
    assert "NOTE" not in out


def test_result_card_for_runs_without_a_model() -> None:
    waiting = text(
        result_card("f", {**SUMMARY, "status": "awaiting_vp", "draft": None})
    )
    assert "Waiting for a VP" in waiting and "Billed in EUR." in waiting
    failed = text(result_card("broken.pdf", {"status": "failed", "reason": "boom"}))
    assert "broken.pdf" in failed and "▪ boom" in failed
    objected = text(
        result_card(
            "f",
            {
                "status": "rejected",
                "invoice": {"total": -250.0, "currency": "CAD"},
                "trace": [
                    {"message": "Critic objects: no."},
                    {"message": "VP rejected."},
                ],
            },
        )
    )
    assert "no vendor" in objected and "CAD -250.00" in objected
    assert "Critic  Objects" in objected and "VP  Rejected" in objected


def test_money() -> None:
    assert money(-250, "USD") == "-$250.00"
    assert money(1, "GBP") == "£1.00"
    assert money(None, "USD") == "no amount"


def test_turns_show_what_each_step_produced() -> None:
    invoice = {
        "invoice_number": "INV-1009",
        "vendor": None,
        "total": -250.0,
        "currency": "USD",
        "due_date": None,
        "line_items": [
            {"item": "WidgetA", "quantity": -5, "unit_price": 250, "amount": None},
            {"item": "WidgetB", "quantity": 2.5, "unit_price": 500, "amount": 1250},
        ],
    }
    findings = [
        {"code": "NOTE", "severity": "info", "title": "t", "detail": "fyi"},
        {
            "code": "INVALID_QUANTITY",
            "severity": "error",
            "title": "t",
            "detail": "WidgetA has quantity -5.",
        },
    ]
    events = [
        {
            **trace("ingest", "Attempt 1: extracted with the LLM.", 1.0),
            "data": {"invoice": invoice, "corrections": ["read 2O26 as 2026"]},
        },
        {
            **trace("validate", "1 error, 0 warnings; policy route: reject.", 1.1),
            "data": {"findings": findings},
        },
        trace("reject", "Not paid. One. Two. Three. Four.", 1.2),
    ]
    out = text(Group(*(render_turn(t) for t in turns(events, 1.0))))
    for words in (
        "INV-1009  no vendor  -$250.00  due not stated",
        "WidgetA",
        "-$1,250.00",
        "2.5",
        "$1,250.00",
        "✎ read 2O26 as 2026",
        "✕ INVALID_QUANTITY  WidgetA has quantity -5.",
        "· NOTE  fyi",
        "Not paid. One.",
    ):
        assert words in out, words
    assert "Four." not in out
    # Errors come before notes.
    assert out.index("INVALID_QUANTITY") < out.index("NOTE")

    clean = [
        {
            **trace("validate", "0 errors, 0 warnings; policy route: auto.", 1.0),
            "data": {"findings": []},
        },
        {
            **trace("pay", "mock_payment returned success.", 1.1),
            "data": {
                "payment": {
                    "status": "success",
                    "vendor": "W",
                    "amount": 10.0,
                    "currency": "USD",
                }
            },
        },
    ]
    out = text(Group(*(render_turn(t) for t in turns(clean, 1.0))))
    assert "✓ Every check passed" in out
    assert "Paid $10.00 to W  via mock_payment" in out


def test_batch_summary_counts_vp_decisions_as_a_person() -> None:
    results = [
        {"status": "paid", "trace": []},
        {"status": "paid", "trace": [{"message": "VP approved."}]},
        {"status": "rejected", "trace": []},
        {"status": "awaiting_vp", "trace": []},
        {"status": "skipped"},
    ]
    out = text(batch_summary(results, 12.3))
    assert "5 invoices in 12.3s" in out
    assert "in 40ms" in text(batch_summary(results, 0.04))
    assert "2 settled with no person, 2 sent to a VP" in out
    assert "Skipped" in out and "Failed" not in out


def test_usage_shows_beside_each_turn_and_as_a_run_total() -> None:
    from invoice_agents.terminal import run_usage, spent

    priced = {"input_tokens": 1800, "output_tokens": 300, "cost_usd": 0.0042}
    events = [
        trace("ingest", "Parsed XML in code.", 1.0),
        {**trace("review", "Draft 1: approve. Fine.", 9.0), "usage": priced},
        {
            **trace("critique", "Critic agrees. Outcome: pay.", 12.0),
            "usage": {"input_tokens": 500, "output_tokens": 0, "cost_usd": None},
        },
    ]
    got = turns(events, started=0.0)
    assert got[1].usage == priced and got[0].usage is None
    assert spent(priced) == "2.1k tokens · $0.0042"
    assert (
        spent({"input_tokens": 40, "output_tokens": 0, "cost_usd": 0.5})
        == "40 tokens · $0.50"
    )
    assert spent(None) == ""
    assert run_usage(got) == {
        "input_tokens": 2300,
        "output_tokens": 300,
        "cost_usd": None,
    }
    assert run_usage(got[:2]) == {
        "input_tokens": 1800,
        "output_tokens": 300,
        "cost_usd": 0.0042,
    }
    assert run_usage(got[:1]) is None

    view = RunView("x")
    view.started = 0.0
    for e in events[:2]:
        view.add(e)
    out = shown(view)
    assert "2.1k tokens · $0.0042" in out and "this run so far: 2.1k tokens" in out


def test_model_turns_name_their_model() -> None:
    from invoice_agents.terminal import short_model

    assert short_model("x-ai/grok-4.7") == "grok-4.7"
    assert (
        short_model("x-ai/grok-4.7, anthropic/claude-haiku-5.5")
        == "grok-4.7, claude-haiku-5.5"
    )
    usage = {
        "model": "anthropic/claude-haiku-5.5",
        "input_tokens": 10,
        "output_tokens": 1,
        "cost_usd": None,
    }
    view = RunView("x")
    view.started = 0.0
    view.add({**trace("critique", "Critic agrees. Outcome: pay.", 1.0), "usage": usage})
    view.add(
        {
            **trace("validate", "0 errors, 0 warnings; policy route: auto.", 2.0),
            "usage": usage,
        }
    )
    out = shown(view)
    assert "Critic  claude-haiku-5.5" in out
    assert "Validation  code" in out  # code steps keep their role


def test_a_vp_name_with_brackets_keeps_the_verdict() -> None:
    [turn] = turns(
        [
            trace(
                "human_review",
                "VP approved (Dana Whitfield (Finance)). Note: ok (fx).",
                1,
            )
        ],
        started=0,
    )
    assert turn.data == {
        "approved": True,
        "by": "Dana Whitfield (Finance)",
        "note": "ok (fx).",
    }


def test_a_duplicate_says_how_to_start_over() -> None:
    dup = {
        "status": "rejected",
        "findings": [
            {"code": "DUPLICATE", "severity": "error", "title": "t", "detail": "d"}
        ],
    }
    assert "python main.py --reset" in text(result_card("f", dup))
    assert "--reset" not in text(result_card("f", {"status": "paid"}))


def test_a_step_that_called_a_model_is_labelled_with_it() -> None:
    usage = {"input_tokens": 1500, "output_tokens": 400, "model": "x-ai/grok-4.3"}
    llm = {**trace("ingest", "Attempt 1: extracted with the LLM.", 1.0), "usage": usage}
    code = trace("ingest", "Parsed JSON in code.", 1.0)
    [read] = turns([llm], 1.0)
    [parsed] = turns([code], 1.0)
    assert "Ingest  grok-4.3" in text(render_turn(read))
    assert "Ingest  code" in text(render_turn(parsed))


def test_a_repeated_payment_says_no_money_moved() -> None:
    paid = {"status": "success", "vendor": "W", "amount": 10.0, "currency": "USD"}
    again = {**paid, "duplicate": True, "idempotency_key": "w:inv1:"}
    events = [
        {
            **trace("pay", "Already paid under key w:inv1:.", 1.0),
            "data": {"payment": again},
        }
    ]
    out = text(Group(*(render_turn(t) for t in turns(events, 1.0))))
    assert "Already paid $10.00 to W; key w:inv1: was used before" in out
    assert "Already paid $10.00" in text(
        result_card("f", {"status": "paid", "payment": again})
    )


def test_the_batch_is_measured_against_the_manual_process() -> None:
    usage = {"input_tokens": 100, "output_tokens": 10, "cost_usd": 0.02}
    results = [
        {"status": "paid", "seconds": 4.0, "trace": [{"message": "x", "usage": usage}]},
        {
            "status": "rejected",
            "seconds": 9.0,
            "invoice": {"total": 5000.0, "currency": "USD"},
            "trace": [{"message": "y", "usage": usage}],
        },
        {
            "status": "rejected",
            "seconds": 6.0,
            "invoice": {"total": 300.0, "currency": "EUR"},  # not summed as dollars
        },
    ]
    out = text(impact_line(results))
    assert "6.0s per invoice (median; was 5 days by hand)" in out
    assert "$5,000.00 stopped before paying (2 rejected)" in out
    assert "$0.013 model cost per invoice" in out
    assert "replayed answers return at once" in text(impact_line(results, True))


def test_money_held_back_counts_each_invoice_once() -> None:
    copy = {
        "vendor": "Atlas",
        "invoice_number": "INV-1013",
        "total": 100.0,
        "currency": "USD",
    }
    results = [
        {"status": "rejected", "invoice": copy, "file": "a.pdf"},
        {
            "status": "rejected",
            "invoice": {**copy, "vendor": "ATLAS"},
            "file": "a.json",
        },
        {
            "status": "rejected",
            "invoice": {**copy, "vendor": "Gadgets"},
            "file": "b.json",
        },
    ]
    assert "$200.00 stopped before paying" in text(impact_line(results))


def test_steps_say_when_they_were_replayed_or_decided_by_rules() -> None:
    replayed = {"input_tokens": 9, "output_tokens": 1, "model": "x-ai/grok-4.7"}
    replayed["replayed"] = True
    events = [
        {**trace("review", "Draft 1: approve. Fine.", 1.0), "usage": replayed},
        trace("critique", "Critic agrees. Outcome: pay.", 1.1),
        trace(
            "review",
            "Draft 1: reject. Not in the recording, so the rule-based approver"
            " decided: Hard failures.",
            1.2,
        ),
    ]
    out = text(Group(*(render_turn(t) for t in turns(events, 1.0))))
    assert "Approver  grok-4.7 · replayed" in out
    assert "Critic  rules" in out
    assert "Approver  rules, not in the recording" in out
