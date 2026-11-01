"""Model adapters for the LLM strategy generator (llm_generator.py).

The generator needs one thing from a model: given a system prompt and the user/assistant
messages so far, return the text of the next assistant message. That is ChatModel.complete.
Each provider gets a small adapter that keeps its quirks inside (parameter names, thinking
blocks, refusals, retries); the generator never imports a vendor SDK.

Included:
  ClaudeModel            Anthropic Messages API (the default model of the generator);
  OpenAICompatibleModel  any /v1/chat/completions endpoint: OpenAI, OpenRouter or a local
                         server (llama.cpp, vLLM, Ollama, LM Studio);
  GeminiModel            Google Gemini via the google-genai SDK;
  RecordedModel          wraps any model, logs every call to JSONL and replays a run exactly.

A new adapter is a subclass with a `name` and a `complete` method, for example:

    class MyModel(ChatModel):
        name = "my-model"
        def complete(self, system, messages, seed=None):
            text = my_client.generate(system=system, turns=[(m.role, m.content) for m in messages])
            return Completion(text)
"""
from __future__ import annotations

import abc
import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class Message:
    role: str            # "user" or "assistant"
    content: str


@dataclass
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    stop_reason: str = ""


class ChatModel(abc.ABC):
    """What the generator calls. Messages alternate user/assistant and end with a user turn."""

    name = "model"

    @abc.abstractmethod
    def complete(self, system: str, messages: Sequence[Message], seed: int | None = None) -> Completion:
        """Return the next assistant message. `seed` is a hint for providers that support
        repeatable sampling; adapters whose API has no seed ignore it."""


class ClaudeModel(ChatModel):
    """Claude through the Anthropic Messages API: pip install anthropic, set ANTHROPIC_API_KEY.

    Current Claude models reject temperature, top_p and top_k, so variety comes from the
    prompts and from repeated runs, and a run is reproduced from its log (RecordedModel), not
    from a seed. `effort` sets how much the model thinks; thinking tokens count toward
    max_tokens and are billed as output. The system prompt is the same in every call of a
    run, so it is marked for prompt caching."""

    def __init__(self, model="claude-sonnet-5-5", effort="low", max_tokens=16_000, cache_system=True,
                 max_retries=5, client=None):
        if client is None:
            import anthropic
            client = anthropic.Anthropic(max_retries=max_retries)
        self.client, self.name = client, model
        self.effort, self.max_tokens, self.cache_system = effort, max_tokens, cache_system

    def complete(self, system, messages, seed=None):
        block = {"type": "text", "text": system}
        if self.cache_system:
            block["cache_control"] = {"type": "ephemeral"}
        resp = self.client.messages.create(
            model=self.name, max_tokens=self.max_tokens, system=[block],
            messages=[{"role": m.role, "content": m.content} for m in messages],
            extra_body={"output_config": {"effort": self.effort}} if self.effort else None)
        text = "".join(b.text for b in resp.content if b.type == "text")   # thinking blocks may come first
        usage = resp.usage
        return Completion(text, usage.input_tokens or 0, usage.output_tokens or 0,
                          getattr(usage, "cache_read_input_tokens", 0) or 0,
                          getattr(usage, "cache_creation_input_tokens", 0) or 0,
                          resp.stop_reason or "")              # "refusal" comes back with no text


class OpenAICompatibleModel(ChatModel):
    """Any /v1/chat/completions endpoint (pip install openai): OpenAI, OpenRouter, or a local
    llama.cpp, vLLM, Ollama or LM Studio server given by base_url. Local servers honour
    temperature and seed, so a run repeats on the same machine. Set a parameter to None if
    the server rejects it; extra_body carries server-specific options."""

    def __init__(self, model, base_url=None, api_key=None, temperature=0.7, max_tokens=4000,
                 send_seed=True, extra_body=None, max_retries=5, client=None):
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY") or "not-needed",
                            max_retries=max_retries)
        self.client, self.name = client, model
        self.temperature, self.max_tokens, self.send_seed, self.extra_body = temperature, max_tokens, send_seed, extra_body

    def complete(self, system, messages, seed=None):
        options = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.max_tokens is not None:
            options["max_tokens"] = self.max_tokens
        if self.send_seed and seed is not None:
            options["seed"] = seed
        if self.extra_body:
            options["extra_body"] = self.extra_body
        resp = self.client.chat.completions.create(
            model=self.name, messages=[{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages], **options)
        choice, usage = resp.choices[0], resp.usage
        return Completion(choice.message.content or "",
                          (usage.prompt_tokens or 0) if usage else 0,
                          (usage.completion_tokens or 0) if usage else 0,
                          stop_reason=choice.finish_reason or "")


class GeminiModel(ChatModel):
    """Google Gemini via the google-genai SDK: pip install google-genai, set GEMINI_API_KEY.

    Gemini uses "user"/"model" roles; "assistant" is mapped to "model" automatically.
    The system prompt is passed as system_instruction. Seed is not supported by the API and
    is silently ignored."""

    def __init__(self, model="gemini-2.0-flash", api_key=None, temperature=1.0,
                 max_tokens=8192, client=None):
        if client is None:
            from google import genai
            client = genai.Client(
                api_key=api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        self.client, self.name = client, model
        self.temperature, self.max_tokens = temperature, max_tokens

    def complete(self, system, messages, seed=None):
        from google.genai import types
        contents = [
            types.Content(role="model" if m.role == "assistant" else m.role,
                          parts=[types.Part(text=m.content)])
            for m in messages
        ]
        resp = self.client.models.generate_content(
            model=self.name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )
        text = resp.text or ""
        usage = resp.usage_metadata
        candidate = resp.candidates[0] if resp.candidates else None
        stop_reason = candidate.finish_reason.name if candidate and candidate.finish_reason else ""
        return Completion(
            text,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            stop_reason=stop_reason,
        )


class ReplayMismatch(RuntimeError):
    pass


class RecordedModel(ChatModel):
    """Wraps a model and logs every call to a JSONL file: request hash, seed, messages, reply.

    mode="record"  start a new log;
    mode="replay"  answer every call from the log in order, without the model (pass None);
                   a request that differs from the logged one raises ReplayMismatch;
    mode="resume"  replay what the log holds, then call the model and append, so a crashed
                   run continues where it stopped.
    The generator is deterministic given the replies, so a replayed run is identical."""

    def __init__(self, model, path, mode="record"):
        if mode not in ("record", "replay", "resume"):
            raise ValueError(f"unknown mode {mode!r}")
        self.model, self.path, self.mode = model, Path(path), mode
        self.name = getattr(model, "name", "replay")
        if mode == "replay" and not self.path.exists():
            raise FileNotFoundError(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if mode == "record" or not self.path.exists():
            self.path.write_text("", encoding="utf-8")
        self.log = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.position = 0
        self._system = next((r["system"] for r in reversed(self.log) if "system" in r), None)

    def complete(self, system, messages, seed=None):
        turns = [[m.role, m.content] for m in messages]
        request = hashlib.sha256(json.dumps([system, turns, seed], ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
        if self.position < len(self.log):
            rec = self.log[self.position]
            self.position += 1
            if rec["request"] != request:
                raise ReplayMismatch(f"call {self.position}: the request differs from the logged one")
            return Completion(**rec["reply"])
        if self.mode == "replay":
            raise ReplayMismatch(f"the log ends after {len(self.log)} calls")
        reply = self.model.complete(system, messages, seed=seed)
        rec = {"request": request, "seed": seed, "messages": turns, "reply": dataclasses.asdict(reply)}
        if system != self._system:            # the system prompt is stored once per change
            rec["system"] = self._system = system
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.log.append(rec)
        self.position += 1
        return reply
