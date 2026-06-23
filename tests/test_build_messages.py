import pytest
from pydantic import ValidationError

from llmcast import BaseTemplate, Message
from llmcast.parser.utils import build_messages


class GreetTemplate(BaseTemplate):
    """Hello, {{ name }}!"""

    name: str


def test_system_only():
    msgs = build_messages(GreetTemplate(name="x"))
    assert msgs == [{"role": "system", "content": "Hello, x!"}]


def test_system_and_query():
    msgs = build_messages(GreetTemplate(name="x"), query="do it")
    assert msgs == [
        {"role": "system", "content": "Hello, x!"},
        {"role": "user", "content": "do it"},
    ]


def test_backward_compatible_without_history():
    # history defaults to None -> output identical to the pre-0.2.0 behavior
    prompt = GreetTemplate(name="x")
    assert build_messages(prompt, "q") == build_messages(prompt, "q", None)


def test_history_inserted_between_system_and_query():
    history = [
        Message(role="assistant", content="a1"),
        Message(role="user", content="o1"),
    ]
    msgs = build_messages(GreetTemplate(name="x"), query="q", history=history)
    assert msgs == [
        {"role": "system", "content": "Hello, x!"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "o1"},
        {"role": "user", "content": "q"},
    ]


def test_history_without_query():
    history = [Message(role="user", content="task")]
    msgs = build_messages(GreetTemplate(name="x"), history=history)
    assert msgs == [
        {"role": "system", "content": "Hello, x!"},
        {"role": "user", "content": "task"},
    ]


def test_message_validates_role():
    with pytest.raises(ValidationError):
        Message(role="robot", content="x")  # type: ignore[arg-type]
