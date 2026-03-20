from llmcast import TokenUsage


def test_defaults_to_zero():
    usage = TokenUsage()
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0


def test_addition():
    a = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    b = TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30)
    c = a + b
    assert c.prompt_tokens == 30
    assert c.completion_tokens == 15
    assert c.total_tokens == 45


def test_addition_with_zero():
    a = TokenUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8)
    b = TokenUsage()
    assert (a + b) == a


def test_addition_is_not_mutating():
    a = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    b = TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    _ = a + b
    assert a.prompt_tokens == 10
