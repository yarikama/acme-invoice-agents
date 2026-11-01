"""Stand-ins shared by tests."""

from typing import Any


class FakePlanner:
    """A chat model for the assistant: answers each prompt with the next plan
    and keeps the prompts it was given."""

    def __init__(self, *plans: Any) -> None:
        from langchain_core.messages import AIMessage

        self.plans = list(plans)
        self.prompts: list[list[Any]] = []
        self.schema: Any = None
        self.reply = AIMessage(
            "{}",
            usage_metadata={
                "input_tokens": 900,
                "output_tokens": 40,
                "total_tokens": 940,
            },
            response_metadata={"model_name": "x-ai/grok-4.3"},
        )

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        from langchain_core.runnables import RunnableLambda

        self.schema = (schema, kwargs)

        async def answer(prompt: Any) -> dict[str, Any]:
            self.prompts.append(prompt.to_messages())
            parsed = self.plans.pop(0)
            error = "bad json" if parsed is None else None
            return {"raw": self.reply, "parsed": parsed, "parsing_error": error}

        return RunnableLambda(lambda _p: None, afunc=answer)
