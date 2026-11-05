"""
The language model memory asks, for consolidation, sleep and gathering — never for adding or searching. One structured
ask: a system text, a prompt and a JSON Schema the answer must follow (strict structured output).

``openai_ask`` speaks the OpenAI Responses API, so any provider or local server that serves it works (``base_url``).
Bring any other model by passing your own ``Ask``: a function of an ``LlmRequest`` that returns an answer — plain or
``async``, either works. Every call's tokens can be counted (``on_usage``).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from ._http import using
from ._js import flatten_ws, js_trim, num_str, stringify, u16slice
from .core.redact import redact_string


@dataclass
class LlmRequest:
    """One structured ask."""

    #: What this is for, in a word (``memory``).
    purpose: str
    system: str
    prompt: str
    #: The answer's JSON Schema.
    schema: dict[str, Any]
    #: What the structured answer is called.
    name: str | None = None
    model: str | None = None
    max_tokens: int | None = None
    timeout_ms: float | None = None
    #: What a failure cost, said after its reason ("so no summary was written").
    consequence: str | None = None


#: ``{"answer": {...}, "model"?: str}``; ``{"unavailable": "why, in words a person can read"}``; or ``{"busy": True}``
#: (one for this purpose is already being asked).
LlmAnswer = dict[str, Any]
Ask = Callable[[LlmRequest], LlmAnswer | Awaitable[LlmAnswer]]


@dataclass
class LlmUsage:
    model: str
    name: str | None
    input_tokens: int
    cached_tokens: int
    output_tokens: int


async def ask_with(ask: Ask, req: LlmRequest) -> LlmAnswer:
    """An ``Ask`` asked: an ``async`` one awaited, a plain one run on a thread of its own (memory asks several at
    once, and a plain function would hold every other ask up)."""
    if inspect.iscoroutinefunction(ask) or inspect.iscoroutinefunction(getattr(ask, "__call__", None)):  # noqa: B004
        got: Any = await ask(req)  # type: ignore[misc]
    else:
        got = await asyncio.to_thread(ask, req)
        if inspect.isawaitable(got):
            got = await got
    return got if isinstance(got, dict) else {"unavailable": "the model's answer could not be read (not an answer)"}


_OPENAI = "https://api.openai.com/v1"


def _why(reason: str, req: LlmRequest) -> str:
    return f"{reason}, {req.consequence}" if req.consequence else reason


def _env(key: str) -> str | None:
    v = js_trim(os.environ.get(key) or "")
    return v or None


class OpenAiAsk:
    """An ``Ask`` over the OpenAI Responses API with strict JSON Schema output, retrying rate limits, server errors and
    dropped connections.

    ``api_key``: default $MEMTRAILS_LLM_API_KEY, else $OPENAI_API_KEY — the latter only for OpenAI's own address.
    ``model``: default $MEMTRAILS_MODEL. ``base_url``: default $MEMTRAILS_LLM_BASE_URL, else https://api.openai.com/v1.
    ``reasoning_effort`` is sent only when given. ``retries``: tries after a failure (default 5), waiting longer each
    time. ``http``: the client to call it with (one for the call when none is given)."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None,
        on_usage: Callable[[LlmUsage], None] | None = None,
        retries: int = 5,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.reasoning_effort = reasoning_effort
        self.on_usage = on_usage
        self.retries = retries
        self._http = http

    async def __call__(self, req: LlmRequest) -> LlmAnswer:
        model = req.model or self.model or _env("MEMTRAILS_MODEL")
        base = (self.base_url or _env("MEMTRAILS_LLM_BASE_URL") or _OPENAI).rstrip("/")
        # An OpenAI key goes to OpenAI only: with another address set, it would hand the key to whoever serves that.
        key = self.api_key or _env("MEMTRAILS_LLM_API_KEY") or (_env("OPENAI_API_KEY") if base == _OPENAI else None)
        if not model:
            return {"unavailable": _why("no model is set (the model option, or MEMTRAILS_MODEL)", req)}
        if not key:
            also = ", or OPENAI_API_KEY" if base == _OPENAI else ""
            return {"unavailable": _why(f"no API key is set (the api_key option or MEMTRAILS_LLM_API_KEY{also})", req)}
        payload: dict[str, Any] = {
            "model": model,
            "instructions": req.system,
            "input": req.prompt,
            "max_output_tokens": req.max_tokens if req.max_tokens is not None else 12_000,
        }
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        payload["text"] = {"format": {"type": "json_schema", "name": req.name or "memory", "strict": True, "schema": req.schema}}
        body = stringify(payload).encode("utf-8")
        tries = max(1, self.retries + 1)
        last = ""
        timeout = (req.timeout_ms if req.timeout_ms is not None else 180_000) / 1000
        async with using(self._http) as http:
            for attempt in range(tries):
                if attempt:
                    await asyncio.sleep(min(60.0, 2 ** (attempt - 1)) * (0.5 + random.random()))
                try:
                    # The whole call in its time, the answer read too (the reference's AbortSignal.timeout): httpx's own
                    # timeout is per phase, and an answer that trickles in would never run out of it.
                    res = await asyncio.wait_for(
                        http.post(
                            f"{base}/responses",
                            content=body,
                            timeout=timeout,
                            headers={"authorization": f"Bearer {key}", "content-type": "application/json"},
                        ),
                        timeout,
                    )
                except asyncio.TimeoutError:
                    last = f"no answer within {num_str(timeout)} s"
                    continue
                except Exception as err:  # noqa: BLE001 - any failure to call (an address httpx refuses too) is tried again, as fetch's are
                    last = str(err) or type(err).__name__
                    continue
                if res.status_code == 429 or res.status_code >= 500:
                    last = f"HTTP {res.status_code}"
                    continue
                if not res.is_success:
                    # What the provider said (a wrong model name, a schema it refuses) — not only its status.
                    said = u16slice(redact_string(js_trim(flatten_ws(res.text))), 0, 300)
                    return {"unavailable": _why(f"the model answered HTTP {res.status_code}{f': {said}' if said else ''}", req)}
                try:
                    r = res.json()
                    usage = r.get("usage") or {}
                    if self.on_usage:
                        self.on_usage(
                            LlmUsage(
                                model=r.get("model") or model,
                                name=req.name,
                                input_tokens=usage.get("input_tokens") or 0,
                                cached_tokens=(usage.get("input_tokens_details") or {}).get("cached_tokens") or 0,
                                output_tokens=usage.get("output_tokens") or 0,
                            )
                        )
                    text = "".join(c.get("text") or "" for x in r.get("output") or [] for c in x.get("content") or [])
                    return {"model": r.get("model") or model, "answer": json.loads(text)}
                except Exception as err:  # noqa: BLE001 - an answer that cannot be read, or an on_usage that raises, pauses the caller
                    return {"unavailable": _why(f"the model's answer could not be read ({err})", req)}
        return {"unavailable": _why(f"the model did not answer ({last})", req)}


def openai_ask(**options: Any) -> OpenAiAsk:
    """An ``Ask`` over the OpenAI Responses API (see OpenAiAsk for the options)."""
    return OpenAiAsk(**options)


async def default_ask(req: LlmRequest) -> LlmAnswer:
    """The model the environment names (MEMTRAILS_MODEL and a key), asked when no ``ask`` is given."""
    return await OpenAiAsk()(req)
