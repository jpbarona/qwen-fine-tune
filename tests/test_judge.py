"""Tests for the OpenCode Go judge client. No network, no real key."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from rlvr.config import load_config, default_config_path
from rlvr.experiment import JudgeResult, Prompt
from rlvr.judge import (
    DEFAULT_CONCURRENCY,
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    KEY_ENV,
    OUTPUT_RUBRIC_VERSION,
    PROCESS_RUBRIC_VERSION,
    REASONING_EFFORT,
    SESSION_VALUE,
    USER_AGENT,
    HTTPResponse,
    JudgeCache,
    JudgeCacheEntry,
    JudgeClient,
    JudgeError,
    JudgeRequest,
    JudgeResponseError,
    JudgeRetriesExhausted,
    MissingAPIKeyError,
    TransportError,
    UnrecoverableJudgeError,
    cache_key,
    output_messages,
    parse_score_payload,
    process_messages,
    require_api_key,
)

# A fake credential. The tests assert it never lands on disk or in an error.
FAKE_KEY = "test-key-never-store-this"
ENDPOINT = "https://opencode.ai/inference/go/openai/v1/chat/completions"

PROBLEM = "Return the sum of all integers from 0 up to n."
CODE = "def solution(n):\n    return sum(range(n))\n"
REASONING = "I will loop from zero to n. I check the empty case."
TEST_SENTINEL = "assert solution(3) == 3  # hidden test code"


def prompt() -> Prompt:
    return Prompt(prompt_id="mbpp-1", problem=PROBLEM)


def completion(score: float | int, reason: str = "looks right") -> bytes:
    payload = {
        "choices": [
            {"message": {"role": "assistant", "content": json.dumps(
                {"score": score, "reason": reason}
            )}}
        ]
    }
    return json.dumps(payload).encode("utf-8")


class FakeTransport:
    """A transport that answers from a handler and records every request."""

    def __init__(self, handler=None) -> None:
        self.requests: list[dict] = []
        self.handler = handler or (lambda request: HTTPResponse(200, completion(0.5)))

    def post(self, url, headers, body, timeout):
        request = {
            "url": url,
            "headers": dict(headers),
            "body": body,
            "timeout": timeout,
        }
        self.requests.append(request)
        return self.handler(request)

    def body_json(self, index: int = 0) -> dict:
        return json.loads(self.requests[index]["body"].decode("utf-8"))


class ScriptedTransport:
    """Returns queued responses, raising an error object when one is queued."""

    def __init__(self, *responses) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []

    def post(self, url, headers, body, timeout):
        self.requests.append({"url": url, "headers": dict(headers), "body": body})
        if not self.responses:
            raise AssertionError("no scripted response left")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def make_client(transport, tmp_path, **kwargs) -> JudgeClient:
    kwargs.setdefault("cache", JudgeCache(tmp_path / "cache"))
    kwargs.setdefault("sleep", lambda delay: None)
    return JudgeClient(
        api_key=FAKE_KEY,
        transport=transport,
        **kwargs,
    )


# --- request shape -------------------------------------------------------


def test_request_goes_to_the_opencode_go_endpoint_with_the_frozen_model(tmp_path):
    transport = FakeTransport()
    client = make_client(transport, tmp_path)
    result = client.score_output(prompt(), CODE)

    assert result.score == 0.5
    request = transport.requests[0]
    assert request["url"] == ENDPOINT == DEFAULT_ENDPOINT
    assert "/inference/go/openai/v1/chat/completions" in request["url"]
    body = transport.body_json()
    assert body["model"] == DEFAULT_MODEL == "deepseek-v4.1-flash"
    assert body["temperature"] == 0.0
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["role"] == "user"


def test_auth_header_carries_the_key_and_nothing_else_exposes_it(tmp_path):
    transport = FakeTransport()
    client = make_client(transport, tmp_path)
    client.score_output(prompt(), CODE)

    headers = transport.requests[0]["headers"]
    assert headers["Authorization"] == f"Bearer {FAKE_KEY}"
    assert headers["Content-Type"] == "application/json"
    assert FAKE_KEY not in transport.requests[0]["body"].decode("utf-8")


def test_cloudflare_headers_are_sent_on_every_request(tmp_path):
    transport = FakeTransport()
    client = make_client(transport, tmp_path)
    client.score_output(prompt(), CODE)
    client.score_process(prompt(), REASONING)

    assert len(transport.requests) == 2
    for request in transport.requests:
        headers = request["headers"]
        assert headers["User-Agent"] == USER_AGENT == "rlvr/0.1 (research judge)"
        assert headers["x-opencode-session"] == SESSION_VALUE == "rlvr-judge"


def test_reasoning_effort_is_sent_explicitly(tmp_path):
    transport = FakeTransport()
    make_client(transport, tmp_path).score_output(prompt(), CODE)
    body = transport.body_json()

    assert body["reasoning_effort"] == REASONING_EFFORT == "medium"


# --- prompts -------------------------------------------------------------


def test_output_prompt_has_problem_and_code_and_no_tests():
    system, user = output_messages(prompt(), CODE)
    text = system + "\n" + user

    assert PROBLEM in text
    assert CODE in text
    assert TEST_SENTINEL not in text
    assert OUTPUT_RUBRIC_VERSION in text


def test_process_prompt_has_problem_and_trace_and_no_code_or_tests():
    system, user = process_messages(prompt(), REASONING)
    text = system + "\n" + user

    assert PROBLEM in text
    assert REASONING in text
    assert CODE not in text
    assert TEST_SENTINEL not in text
    assert PROCESS_RUBRIC_VERSION in text


def test_the_two_rubric_versions_differ():
    assert OUTPUT_RUBRIC_VERSION != PROCESS_RUBRIC_VERSION


# --- parsing -------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(-1.0, 0.0), (0.0, 0.0), (0.42, 0.42), (1, 1.0), (1.5, 1.0)],
)
def test_score_is_clamped_into_the_unit_interval(raw, expected):
    score, _reason = parse_score_payload(json.dumps({"score": raw, "reason": "x"}))
    assert score == expected


@pytest.mark.parametrize(
    "payload",
    [
        {"reason": "no score"},
        {"score": "0.5", "reason": "a string score"},
        {"score": None, "reason": "null"},
        {"score": True, "reason": "a bool"},
    ],
)
def test_missing_or_non_numeric_score_is_an_error(payload):
    with pytest.raises(JudgeResponseError):
        parse_score_payload(json.dumps(payload))


def test_a_missing_reason_is_an_error():
    with pytest.raises(JudgeResponseError):
        parse_score_payload(json.dumps({"score": 0.5}))


def test_invalid_json_is_an_error():
    with pytest.raises(JudgeResponseError):
        parse_score_payload("not json")


@pytest.mark.parametrize("score", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_scores_are_an_error(score):
    with pytest.raises(JudgeResponseError):
        parse_score_payload('{"score": ' + score + ', "reason": "x"}')


def test_reason_is_trimmed_to_a_short_string():
    score, reason = parse_score_payload(
        json.dumps({"score": 0.5, "reason": "  a   long\nreason  "})
    )
    assert score == 0.5
    assert reason == "a long reason"


def test_a_malformed_completion_body_is_an_error(tmp_path):
    transport = FakeTransport(handler=lambda request: HTTPResponse(200, b"{}"))
    client = make_client(transport, tmp_path)
    with pytest.raises(JudgeResponseError):
        client.score_output(prompt(), CODE)


# --- missing key ---------------------------------------------------------


def test_require_api_key_raises_when_the_variable_is_absent():
    with pytest.raises(MissingAPIKeyError) as caught:
        require_api_key({})
    assert KEY_ENV in str(caught.value)


def test_require_api_key_returns_the_value_without_printing_it():
    assert require_api_key({KEY_ENV: FAKE_KEY}) == FAKE_KEY


def test_client_construction_fails_fast_without_a_key(tmp_path):
    with pytest.raises(MissingAPIKeyError):
        JudgeClient(api_key="", transport=FakeTransport(), cache=JudgeCache(tmp_path))


def test_from_config_fails_before_any_request_when_the_key_is_missing(
    tmp_path, write_config
):
    transport = FakeTransport()
    config = load_config(write_config())
    with pytest.raises(MissingAPIKeyError):
        JudgeClient.from_config(
            config,
            environ={},
            transport=transport,
            cache=JudgeCache(tmp_path / "cache"),
        )
    assert transport.requests == []


def test_missing_key_error_does_not_contain_a_key():
    with pytest.raises(MissingAPIKeyError) as caught:
        require_api_key({})
    assert "oc_sk_" not in str(caught.value)


def test_from_config_reads_the_key_from_the_environment(
    tmp_path, write_config, monkeypatch
):
    monkeypatch.setenv(KEY_ENV, FAKE_KEY)
    client = JudgeClient.from_config(
        load_config(write_config()),
        transport=FakeTransport(),
        cache=JudgeCache(tmp_path / "cache"),
    )
    assert client.score_output(prompt(), CODE).score == 0.5


# --- cache ---------------------------------------------------------------


def test_cache_key_changes_with_every_input():
    base = cache_key("model-a", "output", "prompt text", "rubric-v1")
    assert base == cache_key("model-a", "output", "prompt text", "rubric-v1")
    assert base != cache_key("model-b", "output", "prompt text", "rubric-v1")
    assert base != cache_key("model-a", "process", "prompt text", "rubric-v1")
    assert base != cache_key("model-a", "output", "other text", "rubric-v1")
    assert base != cache_key("model-a", "output", "prompt text", "rubric-v2")
    assert len(base) == 64


def test_identical_requests_hit_the_cache(tmp_path):
    transport = FakeTransport()
    cache = JudgeCache(tmp_path / "cache")
    client = make_client(transport, tmp_path, cache=cache)

    first = client.score_output(prompt(), CODE)
    second = client.score_output(prompt(), CODE)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.score == first.score
    assert len(transport.requests) == 1
    assert len(list(cache.root.glob("*.json"))) == 1


def test_output_and_process_requests_do_not_share_a_cache_entry(tmp_path):
    transport = FakeTransport(
        handler=lambda request: HTTPResponse(
            200,
            completion(
                0.8
                if b"Reasoning trace" in request["body"]
                else 0.2
            ),
        )
    )
    client = make_client(transport, tmp_path)

    output = client.score_output(prompt(), CODE)
    process = client.score_process(prompt(), REASONING)
    assert output.score == 0.2
    assert process.score == 0.8
    assert len(transport.requests) == 2


def test_cached_reason_is_returned_and_the_key_is_not_on_disk(tmp_path):
    cache = JudgeCache(tmp_path / "cache")
    client = make_client(FakeTransport(), tmp_path, cache=cache)

    result = client.score_output(prompt(), CODE)

    assert result.reason == "looks right"
    for path in cache.root.glob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert FAKE_KEY not in text
        assert "Authorization" not in text


def test_a_corrupt_cache_file_is_a_miss(tmp_path):
    cache = JudgeCache(tmp_path / "cache")
    key = cache_key(DEFAULT_MODEL, "output", "x", OUTPUT_RUBRIC_VERSION)
    cache.root.mkdir(parents=True, exist_ok=True)
    cache.path_for(key).write_text("{not json", encoding="utf-8")

    assert cache.get(key) is None


def test_cache_writes_are_atomic_and_leave_no_temp_files(tmp_path):
    cache = JudgeCache(tmp_path / "cache")
    entry = JudgeCacheEntry(score=0.5, reason="x")

    cache.put("a" * 64, entry)
    cache.put("a" * 64, entry)

    files = list(cache.root.iterdir())
    assert [path.name for path in files] == ["a" * 64 + ".json"]
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload == {"score": 0.5, "reason": "x"}


def test_the_cache_stores_only_score_and_reason(tmp_path):
    cache = JudgeCache(tmp_path / "cache")
    make_client(FakeTransport(), tmp_path, cache=cache).score_output(prompt(), CODE)
    payload = json.loads(next(cache.root.glob("*.json")).read_text(encoding="utf-8"))
    assert set(payload) == {"score", "reason"}


# --- retries -------------------------------------------------------------


def test_retries_a_429_then_succeeds(tmp_path):
    sleeps: list[float] = []
    transport = ScriptedTransport(
        HTTPResponse(429, b"slow down"),
        HTTPResponse(200, completion(0.7)),
    )
    client = make_client(transport, tmp_path, sleep=sleeps.append)

    result = client.score_output(prompt(), CODE)

    assert result.score == 0.7
    assert len(transport.requests) == 2
    assert len(sleeps) == 1
    assert sleeps[0] >= 0.0


def test_retries_a_5xx_then_succeeds(tmp_path):
    transport = ScriptedTransport(
        HTTPResponse(503, b"unavailable"),
        HTTPResponse(200, completion(0.3)),
    )
    client = make_client(transport, tmp_path, sleep=lambda delay: None)

    assert client.score_output(prompt(), CODE).score == 0.3
    assert len(transport.requests) == 2


def test_retries_a_connection_error_then_succeeds(tmp_path):
    transport = ScriptedTransport(
        TransportError("connection reset"),
        HTTPResponse(200, completion(0.4)),
    )
    client = make_client(transport, tmp_path, sleep=lambda delay: None)

    assert client.score_output(prompt(), CODE).score == 0.4
    assert len(transport.requests) == 2


def test_retries_stop_at_the_configured_cap(tmp_path):
    sleeps: list[float] = []
    transport = ScriptedTransport(*[HTTPResponse(500, b"boom") for _ in range(10)])
    client = make_client(
        transport, tmp_path, max_attempts=3, sleep=sleeps.append
    )

    with pytest.raises(JudgeRetriesExhausted) as caught:
        client.score_output(prompt(), CODE)

    assert len(transport.requests) == 3
    assert len(sleeps) == 2
    assert FAKE_KEY not in str(caught.value)


def test_a_non_retryable_4xx_fails_immediately(tmp_path):
    transport = ScriptedTransport(HTTPResponse(400, b"bad request"))
    client = make_client(transport, tmp_path, max_attempts=4)

    with pytest.raises(UnrecoverableJudgeError) as caught:
        client.score_output(prompt(), CODE)

    assert len(transport.requests) == 1
    assert "400" in str(caught.value)
    assert FAKE_KEY not in str(caught.value)


def test_all_judge_errors_share_a_base_class():
    assert issubclass(MissingAPIKeyError, JudgeError)
    assert issubclass(JudgeResponseError, JudgeError)
    assert issubclass(JudgeRetriesExhausted, JudgeError)
    assert issubclass(UnrecoverableJudgeError, JudgeError)


def test_backoff_delays_stay_under_the_exponential_cap(tmp_path):
    sleeps: list[float] = []
    transport = ScriptedTransport(*[HTTPResponse(500, b"boom") for _ in range(10)])
    client = make_client(
        transport,
        tmp_path,
        max_attempts=4,
        backoff_base=0.5,
        backoff_max=2.0,
        sleep=sleeps.append,
    )

    with pytest.raises(JudgeRetriesExhausted):
        client.score_output(prompt(), CODE)

    assert len(sleeps) == 3
    assert all(0.0 <= delay <= 2.0 for delay in sleeps)


# --- batch ---------------------------------------------------------------


def test_batch_preserves_request_order_and_returns_one_result_each(tmp_path):
    def handler(request):
        body = json.loads(request["body"])
        user = body["messages"][1]["content"]
        value = float(user.rsplit("score-me:", 1)[1].splitlines()[0])
        return HTTPResponse(200, completion(value))

    transport = FakeTransport(handler=handler)
    client = make_client(transport, tmp_path)
    requests = [
        JudgeRequest("output", prompt(), f"score-me:{index / 10}") for index in range(6)
    ]

    results = client.score_batch(requests)

    assert [round(result.score, 2) for result in results] == [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
    ]
    assert len(transport.requests) == 6


def test_batch_uses_the_bounded_executor(tmp_path):
    active = 0
    peak = 0
    lock = threading.Lock()

    def handler(request):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return HTTPResponse(200, completion(0.5))

    transport = FakeTransport(handler=handler)
    client = make_client(transport, tmp_path, max_workers=3)
    requests = [
        JudgeRequest("output", prompt(), f"candidate {index}") for index in range(12)
    ]

    results = client.score_batch(requests)

    assert len(results) == 12
    assert 1 < peak <= 3


def test_batch_raises_when_one_response_is_unrecoverable(tmp_path):
    def handler(request):
        if b"candidate 2" in request["body"]:
            return HTTPResponse(401, b"denied")
        return HTTPResponse(200, completion(0.5))

    client = make_client(FakeTransport(handler=handler), tmp_path)
    requests = [
        JudgeRequest("output", prompt(), f"candidate {index}") for index in range(4)
    ]

    with pytest.raises(UnrecoverableJudgeError):
        client.score_batch(requests)


def test_an_empty_batch_makes_no_calls(tmp_path):
    transport = FakeTransport()
    client = make_client(transport, tmp_path)
    assert client.score_batch([]) == []
    assert transport.requests == []


# --- config --------------------------------------------------------------


def test_from_config_reads_the_judge_settings(tmp_path, write_config):
    transport = FakeTransport()
    config = load_config(
        write_config(
            """\
[model]
name = "Qwen/Qwen3-0.6B"
revision = "base"

[judge]
model = "deepseek-v4.1-flash"
temperature = 0.0
concurrency = 5
max_attempts = 2
timeout_seconds = 12.5
cache_dir = "some-cache"

[data]
split_seed = 1

[fake]
steps = 1
prompts_per_step = 1
"""
        )
    )
    client = JudgeClient.from_config(
        config,
        api_key=FAKE_KEY,
        transport=transport,
        sleep=lambda delay: None,
    )

    assert client.model == "deepseek-v4.1-flash"
    assert client.max_workers == 5
    assert client.max_attempts == 2
    assert client.timeout == 12.5
    # A relative cache directory is relative to the working directory, not the
    # config file, so the packaged default config does not write inside the
    # package.
    assert client.cache.root == Path("some-cache")


def test_the_packaged_config_has_the_concurrency_and_retry_caps():
    config = load_config(default_config_path())
    judge = config.raw["judge"]

    assert judge["concurrency"] == DEFAULT_CONCURRENCY == 32
    assert judge["max_attempts"] >= 1
    assert judge["timeout_seconds"] > 0
    assert judge["cache_dir"] == "judge-cache"
    assert judge["temperature"] == 0.0


def test_judge_result_keeps_the_frozen_fields():
    result = JudgeResult(score=0.5, cache_hit=False, reason="ok")
    assert result.score == 0.5
    assert result.cache_hit is False
    assert result.reason == "ok"
