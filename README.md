# llmcast

A lightweight Python library for structured LLM output parsing. Define prompts as typed Pydantic templates, get back validated Python objects — with retries, token tracking, and support for both sync and async workflows.

**Requirements:** Python 3.13+

## Installation

```bash
pip install llmcast
```

## Core concepts

### `BaseTemplate` — typed prompt

A prompt is a Pydantic model whose docstring is a [Jinja2](https://jinja.palletsprojects.com/) template. Fields become template variables and are automatically rendered on `str()`. The full Jinja2 syntax is available — conditionals, loops, filters, etc.

```python
from llmcast.template import BaseTemplate

class ExtractCompanyInfo(BaseTemplate):
    """
    Extract company information from the text below.
    Respond in {{ output_format }} format.

    Text: {{ text }}
    """
    text: str
```

The built-in `output_format` field (`"json"` by default) can be referenced in the template to instruct the model which format to use.

### `SyncLLMParser` / `AsyncLLMParser` — parsers

Wrap an OpenAI-compatible client and call `.parse()` to get a validated Pydantic object back.

```python
from openai import OpenAI
from pydantic import BaseModel
from llmcast.parser.sync import SyncLLMParser

class CompanyInfo(BaseModel):
    name: str
    founded: int
    employees: int

client = OpenAI()
parser = SyncLLMParser(client, "gpt-4o")

result = parser.parse(
    prompt=ExtractCompanyInfo(text="Anthropic was founded in 2021 and has ~500 employees."),
    result_schema=CompanyInfo,
)

if result:
    company, usage = result
    print(company.name)          # Anthropic
    print(usage.total_tokens)    # 142
```

Async usage with concurrency control:

```python
import asyncio
from openai import AsyncOpenAI
from llmcast.parser.async_ import AsyncLLMParser

parser = AsyncLLMParser(AsyncOpenAI(), "gpt-4o", concurrency_limit=5)

results = await asyncio.gather(*[
    parser.parse(ExtractCompanyInfo(text=text), CompanyInfo)
    for text in texts
])
```

## Structured output vs. text parsing

By default (`structured_output=True`) the parser uses OpenAI's native structured output API (`response_format`), which guarantees schema-valid JSON. For providers or models that don't support this, set `structured_output=False` — the parser will instead extract and validate output from the raw text response.

```python
# Provider doesn't support structured output — parse from text
parser = SyncLLMParser(client, "mistral-large-latest", structured_output=False)
```

When `structured_output=False`, the library only strips code fences and validates the result against the schema — it does **not** instruct the model how to respond. You are fully responsible for crafting a prompt that reliably produces output in the expected format. The `output_format` field is provided as a convenience variable you can reference in your template, but it has no effect unless you explicitly use it.

```python
class SummaryPrompt(BaseTemplate):
    """
    Summarize the following. Reply in {{ output_format }}.

    {{ text }}
    """
    text: str
    output_format: str = "yaml"
```

## Retry policy

Configure retries with exponential backoff and jitter via `RetryPolicy`:

```python
from llmcast.parser.utils import RetryPolicy

policy = RetryPolicy(
    n_tries=5,
    backoff=1.0,       # initial delay in seconds
    multiplier=2.0,    # exponential factor
    max_backoff=30.0,  # cap
    jitter=True,       # randomize to avoid thundering herd
)

parser = SyncLLMParser(client, "gpt-4o", retry_policy=policy)
```

Rate limit errors, timeouts, and server errors (`429`, `408`, `5xx`) are retried with backoff. Parse validation failures are also retried. Non-retryable errors (auth, bad request) are raised immediately.

A per-call policy can override the instance default:

```python
result = parser.parse(prompt, MySchema, retry_policy=RetryPolicy(n_tries=1))
```

## Token usage

Every `.parse()` call returns a `(result, TokenUsage)` tuple. If multiple attempts were needed, usage is summed across all of them.

```python
result, usage = result
print(usage.prompt_tokens)      # 98
print(usage.completion_tokens)  # 44
print(usage.total_tokens)       # 142
```

## License

MIT — see [LICENSE](LICENSE).
