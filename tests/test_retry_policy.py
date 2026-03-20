from llmcast import RetryPolicy


def test_first_attempt_no_delay():
    policy = RetryPolicy()
    assert policy.delay_for(0) == 0.0


def test_exponential_backoff():
    policy = RetryPolicy(backoff=1.0, multiplier=2.0, jitter=False)
    assert policy.delay_for(1) == 1.0
    assert policy.delay_for(2) == 2.0
    assert policy.delay_for(3) == 4.0


def test_max_backoff_capped():
    policy = RetryPolicy(backoff=1.0, multiplier=2.0, max_backoff=3.0, jitter=False)
    assert policy.delay_for(5) == 3.0


def test_jitter_within_range():
    policy = RetryPolicy(backoff=1.0, multiplier=2.0, jitter=True)
    for _ in range(20):
        delay = policy.delay_for(1)
        assert 0.5 <= delay <= 1.0


def test_default_single_try():
    policy = RetryPolicy()
    assert policy.n_tries == 1
