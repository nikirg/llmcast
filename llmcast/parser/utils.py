import random
import tomllib
from dataclasses import dataclass
from typing import Literal, Type

import yaml
from loguru import logger
from pydantic import BaseModel, ValidationError

from llmcast.template import BaseTemplate, ResultFormat

Role = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    """A single chat turn passed to the parser as conversation history."""

    role: Role
    content: str


@dataclass
class RetryPolicy:
    n_tries: int = 1
    backoff: float = 1.0  # initial delay in seconds
    multiplier: float = 2.0  # exponential factor per attempt
    max_backoff: float = 60.0  # upper bound on delay
    jitter: bool = True  # randomize delay to avoid thundering herd

    def delay_for(self, attempt: int) -> float:
        """Return sleep duration (seconds) before the given attempt (0-indexed)."""
        if attempt == 0:
            return 0.0
        delay = min(self.backoff * (self.multiplier ** (attempt - 1)), self.max_backoff)
        if self.jitter:
            delay *= 0.5 + random.random() * 0.5
        return delay


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    @classmethod
    def from_completion(cls, completion) -> "TokenUsage":
        usage = getattr(completion, "usage", None)
        if not usage:
            return cls()
        return cls(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )


@dataclass(frozen=True)
class SamplingParams:
    """Decoder-level generation controls passed through to the OpenAI-compatible API.

    Provider-neutral fields map to standard chat-completions parameters. vLLM (and
    other backends) specific knobs — ``top_k``, ``min_p``, ``repetition_penalty``,
    ``guided_json`` / ``guided_grammar`` / ``guided_choice`` / ``guided_regex`` —
    go through ``extra_body``. ``n`` requests multiple samples in a single call.
    """

    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    seed: int | None = None
    max_tokens: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    logit_bias: dict[int, float] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    extra_body: dict | None = None

    def to_kwargs(self) -> dict:
        """Build chat-completions kwargs, omitting unset (``None``) fields."""
        fields = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "logit_bias": self.logit_bias,
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs,
            "extra_body": self.extra_body,
        }
        return {key: value for key, value in fields.items() if value is not None}


def build_messages(
    prompt: BaseTemplate,
    query: str | None = None,
    history: list[Message] | None = None,
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": str(prompt)}]
    if history:
        messages += [{"role": m.role, "content": m.content} for m in history]
    if query:
        messages.append({"role": "user", "content": query})
    return messages


def parse_content[TResult: BaseModel](
    content: str,
    result_schema: Type[TResult],
    result_format: ResultFormat = "json",
) -> TResult | None:
    content = content.replace(f"```{result_format}", "").replace("```", "").strip()
    try:
        match result_format:
            case "json":
                return result_schema.model_validate_json(content)
            case "yaml":
                return result_schema.model_validate(yaml.safe_load(content))
            case "toml":
                return result_schema.model_validate(tomllib.loads(content))
    except (ValidationError, yaml.YAMLError, tomllib.TOMLDecodeError) as e:
        logger.error(e)
        logger.debug(content)
        return None
