import time
from typing import Type

import openai
from openai import OpenAI
from pydantic import BaseModel
from loguru import logger

from llmcast.parser.utils import (
    Message,
    RetryPolicy,
    TokenUsage,
    build_messages,
    parse_content,
)
from llmcast.template import BaseTemplate

_RETRYABLE = (openai.RateLimitError, openai.APITimeoutError, openai.InternalServerError)


class SyncLLMParser[TResult: BaseModel]:
    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        timeout: int = 10,
        retry_policy: RetryPolicy | None = None,
        structured_output: bool = True,
    ):
        self._client = client
        self._model_name = model_name
        self._timeout = timeout
        self._retry_policy = retry_policy or RetryPolicy()
        self._structured_output = structured_output

    def _complete(self, messages: list[dict]) -> tuple[str | None, TokenUsage]:
        completion = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,  # type: ignore
            timeout=self._timeout,
        )
        usage = TokenUsage(
            prompt_tokens=completion.usage.prompt_tokens if completion.usage else 0,
            completion_tokens=(
                completion.usage.completion_tokens if completion.usage else 0
            ),
            total_tokens=completion.usage.total_tokens if completion.usage else 0,
        )
        return completion.choices[0].message.content, usage

    def _parse(
        self, messages: list[dict], result_schema: Type[TResult]
    ) -> tuple[TResult | None, TokenUsage]:
        completion = self._client.chat.completions.parse(
            model=self._model_name,
            messages=messages,  # type: ignore
            response_format=result_schema,
            timeout=self._timeout,
        )
        usage = TokenUsage(
            prompt_tokens=completion.usage.prompt_tokens if completion.usage else 0,
            completion_tokens=(
                completion.usage.completion_tokens if completion.usage else 0
            ),
            total_tokens=completion.usage.total_tokens if completion.usage else 0,
        )
        return completion.choices[0].message.parsed, usage

    def parse(
        self,
        prompt: BaseTemplate,
        result_schema: Type[TResult],
        query: str | None = None,
        retry_policy: RetryPolicy | None = None,
        history: list[Message] | None = None,
    ) -> tuple[TResult, TokenUsage] | None:
        if self._structured_output:
            assert prompt.output_format == "json", (
                "Structured output only supported for JSON"
            )

        messages = build_messages(prompt, query, history)
        policy = retry_policy or self._retry_policy
        total_usage = TokenUsage()

        for attempt in range(policy.n_tries):
            if (delay := policy.delay_for(attempt)) > 0:
                time.sleep(delay)
            try:
                if self._structured_output:
                    result, usage = self._parse(messages, result_schema)
                    total_usage = total_usage + usage
                    if result is not None:
                        return result, total_usage
                else:
                    content, usage = self._complete(messages)
                    total_usage = total_usage + usage
                    if content and (
                        result := parse_content(
                            content, result_schema, result_format=prompt.output_format
                        )
                    ):
                        return result, total_usage
            except _RETRYABLE as e:
                if attempt == policy.n_tries - 1:
                    raise
                logger.warning(
                    "Retryable API error (attempt {}/{}): {}",
                    attempt + 1,
                    policy.n_tries,
                    e,
                )
                continue

            logger.debug(
                "Parse failed on attempt {}/{}, retrying", attempt + 1, policy.n_tries
            )

        return None
