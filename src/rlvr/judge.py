"""The OpenCode Go judge client.

Two judges share one transport and one file cache:

* the output judge sees the problem and the candidate code
* the process judge sees the problem and the reasoning trace

Neither prompt has a parameter for task tests, so test code cannot reach the
judge. The rubric version is part of the cache key, so editing a rubric starts a
new cache generation.

Endpoint, from the OpenCode Go docs:

    POST https://opencode.ai/inference/openai/v1/chat/completions
    Authorization: Bearer <OPENCODE_GO_API_KEY>

The request uses the OpenAI Chat Completions shape. The reasoning effort
parameter is deliberately omitted. The docs do not define a "normal" value, so
the provider default is used.

The key is read from the environment by `require_api_key`. It is only placed in
the Authorization header. It is never written to the cache, a log, an exception
message, or `meta.json`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import socket
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .config import Config, ConfigError
from .experiment import JudgeResult, Prompt

__all__ = [
    "DEFAULT_BACKOFF_BASE",
    "DEFAULT_BACKOFF_MAX",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_CONCURRENCY",
    "DEFAULT_ENDPOINT",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MODEL",
    "DEFAULT_TIMEOUT_SECONDS",
    "KEY_ENV",
    "OUTPUT_RUBRIC_VERSION",
    "PROCESS_RUBRIC_VERSION",
    "HTTPResponse",
    "JudgeCache",
    "JudgeCacheEntry",
    "JudgeClient",
    "JudgeError",
    "JudgeHTTPError",
    "JudgeRequest",
    "JudgeResponseError",
    "JudgeRetriesExhausted",
    "MissingAPIKeyError",
    "Transport",
    "TransportError",
    "UnrecoverableJudgeError",
    "UrllibTransport",
    "cache_key",
    "output_messages",
    "parse_score_payload",
    "process_messages",
    "require_api_key",
]

#: The OpenAI-compatible endpoint from the OpenCode Go docs.
DEFAULT_ENDPOINT = "https://opencode.ai/inference/openai/v1/chat/completions"

#: The judge model fixed by SPEC.md.
DEFAULT_MODEL = "deepseek-v4.1-flash"

#: Environment variable that carries the key.
KEY_ENV = "OPENCODE_GO_API_KEY"

#: Bounded in-flight calls. The number is fixed by SPEC.md.
DEFAULT_CONCURRENCY = 32

#: Total attempts, including the first one.
DEFAULT_MAX_ATTEMPTS = 4

#: Per-attempt HTTP timeout in seconds.
DEFAULT_TIMEOUT_SECONDS = 60.0

#: Relative cache directory used when the config does not name one.
DEFAULT_CACHE_DIR = "judge-cache"

#: Backoff start and cap in seconds. Jitter is applied inside these bounds.
DEFAULT_BACKOFF_BASE = 1.0
DEFAULT_BACKOFF_MAX = 30.0

#: Rubric versions. A change here invalidates cached scores.
OUTPUT_RUBRIC_VERSION = "output-v1"
PROCESS_RUBRIC_VERSION = "process-v1"

#: Longest reason kept from a judge response.
REASON_LIMIT = 300

#: Longest response body snippet kept in an error message.
_ERROR_SNIPPET_LIMIT = 200


class JudgeError(Exception):
    """Base class for every judge failure."""


class MissingAPIKeyError(JudgeError):
    """The environment does not hold the judge key."""


class TransportError(JudgeError):
    """A timeout or connection failure. Retryable by the client."""


class JudgeHTTPError(JudgeError):
    """A response with an unexpected status."""


class JudgeRetriesExhausted(JudgeError):
    """Every allowed attempt failed with a retryable error."""


class UnrecoverableJudgeError(JudgeError):
    """A non-retryable 4xx or redirect. Fails loudly."""


class JudgeResponseError(JudgeError):
    """The response is not a usable JSON score object."""


@dataclass(frozen=True)
class HTTPResponse:
    """What a transport returns. Status is not raised, so retries can decide."""

    status: int
    body: bytes


class Transport(Protocol):
    def post(
        self,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout: float,
    ) -> HTTPResponse: ...


class UrllibTransport:
    """Stdlib HTTP transport. No third-party dependency."""

    def post(
        self,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout: float,
    ) -> HTTPResponse:
        request = urllib.request.Request(
            url,
            data=body,
            headers=dict(headers),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return HTTPResponse(status=response.status, body=response.read())
        except urllib.error.HTTPError as error:
            # A 4xx or 5xx still has a body. Let the client classify it.
            try:
                payload = error.read()
            except OSError:
                payload = b""
            return HTTPResponse(status=error.code, body=payload)
        except urllib.error.URLError as error:
            raise TransportError(f"connection failed: {_reason(error.reason)}") from None
        except (TimeoutError, socket.timeout):
            raise TransportError("request timed out") from None
        except OSError as error:
            raise TransportError(f"connection failed: {type(error).__name__}") from None


def _reason(value: Any) -> str:
    text = str(value)
    if "timed out" in text.lower():
        return "request timed out"
    return text.replace("\n", " ")[:200]


@dataclass(frozen=True)
class JudgeCacheEntry:
    """One cached judge answer. Only the score and reason are stored."""

    score: float
    reason: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise ValueError(f"cached score must be numeric, got {self.score!r}")
        value = float(self.score)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"cached score must be in [0, 1], got {self.score!r}")
        if not isinstance(self.reason, str):
            raise ValueError(f"cached reason must be a string, got {type(self.reason).__name__}")


class JudgeCache:
    """A file cache keyed by a SHA-256 digest. Writes use `os.replace`."""

    def __init__(self, root: str | Path = DEFAULT_CACHE_DIR) -> None:
        self.root = Path(root)

    def path_for(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> JudgeCacheEntry | None:
        """Return the entry, or None when absent or unreadable."""
        try:
            with self.path_for(key).open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        score = payload.get("score")
        reason = payload.get("reason", "")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            return None
        value = float(score)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            return None
        if not isinstance(reason, str):
            return None
        return JudgeCacheEntry(score=value, reason=reason)

    def put(self, key: str, entry: JudgeCacheEntry) -> None:
        """Write one entry atomically, so a crash cannot leave half a file."""
        self.root.mkdir(parents=True, exist_ok=True)
        handle = None
        temp_name = None
        try:
            handle, temp_name = tempfile.mkstemp(
                dir=self.root, prefix=f".{key}.", suffix=".tmp"
            )
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                handle = None  # The context manager owns the descriptor now.
                json.dump(
                    {"score": entry.score, "reason": entry.reason},
                    stream,
                    ensure_ascii=False,
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, self.path_for(key))
            temp_name = None
        finally:
            if handle is not None:
                os.close(handle)
            if temp_name is not None:
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass


def _system_prompt(kind: str) -> str:
    subject = "Python code" if kind == "output" else "a reasoning trace"
    return (
        f"You are a strict evaluator of {subject}. "
        "Return only a JSON object with the keys score and reason."
    )


_OUTPUT_TEMPLATE = """\
Problem:
{problem}

Candidate code:
```python
{code}
```

Rubric (output rubric {version}):
- 1.0: the code solves the problem correctly for every reasonable input.
- 0.5: the code is on the right track but has a real defect or an edge case failure.
- 0.0: the code is wrong, incomplete, or does not run.

Judge only from the problem and the code above. Reply with only this JSON object:
{{"score": <number from 0 to 1>, "reason": "<one short sentence>"}}
"""

_PROCESS_TEMPLATE = """\
Problem:
{problem}

Reasoning trace:
{reasoning}

Rubric (process rubric {version}):
- Planning: the trace states a plan before writing code.
- Debugging: the trace checks the approach and looks for mistakes.
- Honest reporting: the trace does not claim results it did not verify.
A longer trace is not better. Do not reward length, repetition, or restating
the problem. Score 0.0 for no real reasoning, 0.5 for partial reasoning, and 1.0
for clear planning, checking, and honest reporting.

Judge only from the problem and the reasoning trace above. Reply with only this
JSON object:
{{"score": <number from 0 to 1>, "reason": "<one short sentence>"}}
"""


def output_messages(prompt: Prompt, code: str) -> tuple[str, str]:
    """The output judge system and user messages. Tests are not an argument."""
    user = _OUTPUT_TEMPLATE.format(
        problem=prompt.problem,
        code=code,
        version=OUTPUT_RUBRIC_VERSION,
    )
    return _system_prompt("output"), user


def process_messages(prompt: Prompt, reasoning: str) -> tuple[str, str]:
    """The process judge system and user messages. Code and tests are absent."""
    user = _PROCESS_TEMPLATE.format(
        problem=prompt.problem,
        reasoning=reasoning,
        version=PROCESS_RUBRIC_VERSION,
    )
    return _system_prompt("process"), user


def cache_key(model: str, kind: str, prompt_text: str, rubric_version: str) -> str:
    """SHA-256 over the model, judge kind, prompt text, and rubric version."""
    material = "\x1f".join((model, kind, prompt_text, rubric_version))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _prompt_text(system: str, user: str) -> str:
    return system + "\x1e" + user


def _short_reason(reason: str) -> str:
    return " ".join(reason.split())[:REASON_LIMIT]


def parse_score_payload(content: str) -> tuple[float, str]:
    """Parse the score JSON, clamp the score, and shorten the reason.

    A missing or non-numeric score is an error. It is never turned into zero.
    """
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise JudgeResponseError(f"judge content is not JSON: {error}") from error
    if not isinstance(payload, dict):
        raise JudgeResponseError("judge content must be a JSON object")
    if "score" not in payload:
        raise JudgeResponseError("judge content has no score")
    if "reason" not in payload:
        raise JudgeResponseError("judge content has no reason")

    raw_score = payload["score"]
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        raise JudgeResponseError(
            f"judge score must be a number, got {type(raw_score).__name__}"
        )
    score = float(raw_score)
    if not math.isfinite(score):
        raise JudgeResponseError("judge score must be finite")

    reason = payload["reason"]
    if not isinstance(reason, str):
        raise JudgeResponseError(
            f"judge reason must be a string, got {type(reason).__name__}"
        )

    return min(1.0, max(0.0, score)), _short_reason(reason)


def _completion_content(body: bytes) -> str:
    """Pull `choices[0].message.content` out of a chat completion response."""
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise JudgeResponseError("judge response body is not JSON") from error
    if not isinstance(payload, dict):
        raise JudgeResponseError("judge response body must be a JSON object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise JudgeResponseError("judge response has no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise JudgeResponseError("judge response has no message content")
    return content


def _snippet(body: bytes) -> str:
    text = body.decode("utf-8", errors="replace")
    return " ".join(text.split())[:_ERROR_SNIPPET_LIMIT]


def require_api_key(environ: Mapping[str, str] | None = None) -> str:
    """Read the key from the environment, or fail before any work starts."""
    source = os.environ if environ is None else environ
    value = source.get(KEY_ENV)
    if not isinstance(value, str) or not value.strip():
        raise MissingAPIKeyError(
            f"{KEY_ENV} is not set. Export it before starting a run."
        )
    return value.strip()


@dataclass(frozen=True)
class JudgeRequest:
    """One item for the batch API. `text` is code or a reasoning trace."""

    kind: str
    prompt: Prompt
    text: str

    def __post_init__(self) -> None:
        if self.kind not in ("output", "process"):
            raise ValueError(f"judge kind must be 'output' or 'process', got {self.kind!r}")


def _int_setting(table: Mapping[str, Any], name: str, default: int) -> int:
    value = table.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"config: 'judge.{name}' must be an integer, got {value!r}")
    if value < 1:
        raise ConfigError(f"config: 'judge.{name}' must be at least 1, got {value}")
    return value


def _float_setting(
    table: Mapping[str, Any], name: str, default: float, *, allow_zero: bool
) -> float:
    value = table.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"config: 'judge.{name}' must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"config: 'judge.{name}' must be finite, got {value!r}")
    if allow_zero and result < 0.0:
        raise ConfigError(f"config: 'judge.{name}' must not be negative, got {value!r}")
    if not allow_zero and result <= 0.0:
        raise ConfigError(f"config: 'judge.{name}' must be positive, got {value!r}")
    return result


class JudgeClient:
    """The concrete Judge seam for OpenCode Go.

    `score_output` and `score_process` are the frozen one-at-a-time calls. The
    training loop may instead call `score_batch`, which runs the same calls on a
    bounded pool of `max_workers` threads and keeps the input order.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        endpoint: str = DEFAULT_ENDPOINT,
        temperature: float = 0.0,
        cache: JudgeCache | None = None,
        transport: Transport | None = None,
        max_workers: int = DEFAULT_CONCURRENCY,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        backoff_max: float = DEFAULT_BACKOFF_MAX,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise MissingAPIKeyError(
                f"{KEY_ENV} is not set. Export it before starting a run."
            )
        if max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {max_workers}")
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")
        if backoff_base < 0:
            raise ValueError(f"backoff_base must not be negative, got {backoff_base}")
        if backoff_max < backoff_base:
            raise ValueError(
                f"backoff_max must be at least backoff_base, got {backoff_max}"
            )

        self.model = model
        self.endpoint = endpoint
        self.temperature = float(temperature)
        self.cache = cache if cache is not None else JudgeCache()
        self.transport = transport if transport is not None else UrllibTransport()
        self.max_workers = max_workers
        self.max_attempts = max_attempts
        self.timeout = float(timeout)
        self.backoff_base = float(backoff_base)
        self.backoff_max = float(backoff_max)
        self._sleep = sleep
        self._rng = rng if rng is not None else random.Random()
        self._api_key = api_key.strip()

    @classmethod
    def from_config(
        cls,
        config: Config,
        *,
        api_key: str | None = None,
        environ: Mapping[str, str] | None = None,
        cache: JudgeCache | None = None,
        transport: Transport | None = None,
        sleep: Callable[[float], None] | None = None,
        rng: random.Random | None = None,
    ) -> "JudgeClient":
        """Build a client from the `[judge]` table.

        The key is read from the environment unless one is passed. A missing key
        raises here, before the training loop starts.
        """
        table = config.raw.get("judge", {})
        if not isinstance(table, dict):
            raise ConfigError("config: 'judge' must be a table")

        key = api_key if api_key is not None else require_api_key(environ)
        cache_dir = table.get("cache_dir", DEFAULT_CACHE_DIR)
        if not isinstance(cache_dir, str) or not cache_dir.strip():
            raise ConfigError(
                f"config: 'judge.cache_dir' must be a non-empty string, got {cache_dir!r}"
            )
        endpoint = table.get("endpoint", DEFAULT_ENDPOINT)
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ConfigError(
                f"config: 'judge.endpoint' must be a non-empty string, got {endpoint!r}"
            )

        return cls(
            api_key=key,
            model=config.judge_model,
            endpoint=endpoint,
            temperature=_float_setting(table, "temperature", 0.0, allow_zero=True),
            cache=cache if cache is not None else JudgeCache(cache_dir),
            transport=transport,
            max_workers=_int_setting(table, "concurrency", DEFAULT_CONCURRENCY),
            max_attempts=_int_setting(table, "max_attempts", DEFAULT_MAX_ATTEMPTS),
            timeout=_float_setting(
                table, "timeout_seconds", DEFAULT_TIMEOUT_SECONDS, allow_zero=False
            ),
            sleep=sleep if sleep is not None else time.sleep,
            rng=rng,
        )

    # --- the frozen seam -------------------------------------------------

    def score_output(self, prompt: Prompt, code: str) -> JudgeResult:
        return self._score("output", prompt, code)

    def score_process(self, prompt: Prompt, reasoning: str) -> JudgeResult:
        return self._score("process", prompt, reasoning)

    # --- batch API -------------------------------------------------------

    def score_batch(
        self,
        requests: Sequence[JudgeRequest],
        *,
        max_workers: int | None = None,
    ) -> list[JudgeResult]:
        """Score many requests with bounded concurrency, preserving order."""
        items = list(requests)
        if not items:
            return []
        if max_workers is not None and max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {max_workers}")
        workers = min(max_workers or self.max_workers, len(items))
        if workers == 1:
            return [self._score_request(item) for item in items]
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="rlvr-judge"
        ) as pool:
            return list(pool.map(self._score_request, items))

    # --- internals -------------------------------------------------------

    def _score_request(self, request: JudgeRequest) -> JudgeResult:
        return self._score(request.kind, request.prompt, request.text)

    def _score(self, kind: str, prompt: Prompt, text: str) -> JudgeResult:
        if kind == "output":
            system, user = output_messages(prompt, text)
            rubric_version = OUTPUT_RUBRIC_VERSION
        else:
            system, user = process_messages(prompt, text)
            rubric_version = PROCESS_RUBRIC_VERSION

        key = cache_key(
            self.model,
            kind,
            _prompt_text(system, user),
            rubric_version,
        )
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                return JudgeResult(
                    score=cached.score,
                    cache_hit=True,
                    reason=cached.reason,
                )

        content = self._complete(system, user)
        score, reason = parse_score_payload(content)
        if self.cache is not None:
            self.cache.put(key, JudgeCacheEntry(score=score, reason=reason))
        return JudgeResult(score=score, cache_hit=False, reason=reason)

    def _complete(self, system: str, user: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": self.temperature,
                "response_format": {"type": "json_object"},
            },
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        last: JudgeError | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.transport.post(
                    self.endpoint, headers, body, self.timeout
                )
            except TransportError as error:
                last = error
            else:
                if 200 <= response.status < 300:
                    return _completion_content(response.body)
                if response.status == 408 or response.status == 429 or (
                    500 <= response.status < 600
                ):
                    last = JudgeHTTPError(
                        f"judge request failed with status {response.status}: "
                        f"{_snippet(response.body)}"
                    )
                else:
                    raise UnrecoverableJudgeError(
                        f"judge request failed with status {response.status}: "
                        f"{_snippet(response.body)}"
                    )

            if attempt == self.max_attempts:
                break
            self._sleep(self._delay(attempt))

        detail = str(last) if last is not None else "unknown error"
        raise JudgeRetriesExhausted(
            f"judge request failed after {self.max_attempts} attempts: {detail}"
        )

    def _delay(self, attempt: int) -> float:
        """Full jitter on an exponential backoff, capped at backoff_max."""
        ceiling = min(self.backoff_max, self.backoff_base * (2 ** (attempt - 1)))
        return self._rng.uniform(0.0, ceiling)
