import asyncio
from dataclasses import replace
from typing import Type

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel
from loguru import logger

from llmcast.parser.utils import (
    Message,
    RetryPolicy,
    SamplingParams,
    TokenUsage,
    build_messages,
    parse_content,
)
from llmcast.template import BaseTemplate

_RETRYABLE = (openai.RateLimitError, openai.APITimeoutError, openai.InternalServerError)


class AsyncLLMParser:
    def __init__(
        self,
        client: AsyncOpenAI,
        model_name: str,
        timeout: int = 10,
        concurrency_limit: int = 10,
        retry_policy: RetryPolicy | None = None,
        structured_output: bool = True,
    ):
        self._client = client
        self._model_name = model_name
        self._timeout = timeout
        self._retry_policy = retry_policy or RetryPolicy()
        self._structured_output = structured_output
        self._semaphore = asyncio.Semaphore(concurrency_limit)

    async def _complete_choices(
        self, messages: list[dict], sampling: SamplingParams | None = None
    ) -> tuple[list[str | None], TokenUsage]:
        async with self._semaphore:
            completion = await self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,  # type: ignore
                timeout=self._timeout,
                **(sampling.to_kwargs() if sampling else {}),
            )
        return (
            [choice.message.content for choice in completion.choices],
            TokenUsage.from_completion(completion),
        )

    async def _parse_choices[TResult: BaseModel](
        self,
        messages: list[dict],
        result_schema: Type[TResult],
        sampling: SamplingParams | None = None,
    ) -> tuple[list[TResult | None], TokenUsage]:
        async with self._semaphore:
            completion = await self._client.chat.completions.parse(
                model=self._model_name,
                messages=messages,  # type: ignore
                response_format=result_schema,
                timeout=self._timeout,
                **(sampling.to_kwargs() if sampling else {}),
            )
        return (
            [choice.message.parsed for choice in completion.choices],
            TokenUsage.from_completion(completion),
        )

    async def _complete(
        self, messages: list[dict], sampling: SamplingParams | None = None
    ) -> tuple[str | None, TokenUsage]:
        contents, usage = await self._complete_choices(messages, sampling)
        return (contents[0] if contents else None), usage

    async def _parse[TResult: BaseModel](
        self,
        messages: list[dict],
        result_schema: Type[TResult],
        sampling: SamplingParams | None = None,
    ) -> tuple[TResult | None, TokenUsage]:
        results, usage = await self._parse_choices(messages, result_schema, sampling)
        return (results[0] if results else None), usage

    async def parse[TResult: BaseModel](
        self,
        prompt: BaseTemplate,
        result_schema: Type[TResult],
        query: str | None = None,
        retry_policy: RetryPolicy | None = None,
        history: list[Message] | None = None,
        sampling: SamplingParams | None = None,
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
                await asyncio.sleep(delay)
            try:
                if self._structured_output:
                    result, usage = await self._parse(messages, result_schema, sampling)
                    total_usage = total_usage + usage
                    if result is not None:
                        return result, total_usage
                else:
                    content, usage = await self._complete(messages, sampling)
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

    async def parse_many[TResult: BaseModel](
        self,
        prompt: BaseTemplate,
        result_schema: Type[TResult],
        n: int,
        query: str | None = None,
        retry_policy: RetryPolicy | None = None,
        history: list[Message] | None = None,
        sampling: SamplingParams | None = None,
    ) -> tuple[list[TResult], TokenUsage] | None:
        """Draw ``n`` independent samples in a single request.

        Returns every successfully parsed sample (length ``<= n``; choices that fail
        to parse are dropped). Retries only when no sample parses on an attempt.
        Deduplication is the caller's responsibility.
        """
        if self._structured_output:
            assert prompt.output_format == "json", (
                "Structured output only supported for JSON"
            )

        messages = build_messages(prompt, query, history)
        policy = retry_policy or self._retry_policy
        sampling = replace(sampling or SamplingParams(), n=n)
        total_usage = TokenUsage()

        for attempt in range(policy.n_tries):
            if (delay := policy.delay_for(attempt)) > 0:
                await asyncio.sleep(delay)
            try:
                if self._structured_output:
                    raw, usage = await self._parse_choices(
                        messages, result_schema, sampling
                    )
                    total_usage = total_usage + usage
                    results = [item for item in raw if item is not None]
                else:
                    contents, usage = await self._complete_choices(messages, sampling)
                    total_usage = total_usage + usage
                    results = [
                        parsed
                        for content in contents
                        if content
                        and (
                            parsed := parse_content(
                                content,
                                result_schema,
                                result_format=prompt.output_format,
                            )
                        )
                        is not None
                    ]
                if results:
                    return results, total_usage
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
                "parse_many produced no valid samples on attempt {}/{}, retrying",
                attempt + 1,
                policy.n_tries,
            )

        return None
