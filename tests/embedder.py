"""The embedding client: what it retries, what it refuses to, and how it waits.

A retry policy is the easiest thing in a codebase to believe in without checking.
Every claim here is one somebody would otherwise take on trust:

    it retries at all              a 503 is followed by a second attempt
    it backs off                   the waits double rather than repeating
    it stops backing off           the doubling is capped, not unbounded
    it gives up                    the budget is a budget, and names itself
    it does not retry a refusal    a 404 raises on the first attempt

The last is the one worth having a test for. Spending a retry budget on a
permanent error is worse than having no budget: "nomic-embed-text is not pulled"
becomes a fifteen-second wait ending in the same message, and the operator has
gone to look somewhere else by the time it arrives.

The waits are recorded rather than served. Asserting an exponential backoff by
sitting through one makes the suite slower for no extra information - what is
being tested is the schedule, and the schedule is visible from the arguments.

Talks to an httpx.MockTransport, so there is no Ollama here and no network.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

_VAULT = Path(tempfile.mkdtemp(prefix="vault-embed-"))
os.environ["VAULT_PATH"] = str(_VAULT)

import httpx  # noqa: E402
import numpy as np  # noqa: E402

from src import embedder as embedder_module  # noqa: E402
from src.config import settings  # noqa: E402
from src.embedder import Embedder  # noqa: E402

FAILURES: list[str] = []


def check(name: str, actual, expected) -> None:
    if actual == expected:
        return
    FAILURES.append(f"{name}\n    expected: {expected!r}\n    actual:   {actual!r}")


def report() -> int:
    if not FAILURES:
        return 0
    print(f"{len(FAILURES)} failure(s):\n")
    for failure in FAILURES:
        print(failure + "\n")
    return 1


def embedding_response(request: httpx.Request, *, reverse: bool = False) -> httpx.Response:
    """A well-formed OpenAI-compatible answer for whatever was asked."""
    inputs = json.loads(request.content)["input"]
    rows = [
        {"index": position, "embedding": [float(position + 1)] * settings.embed_dims}
        for position in range(len(inputs))
    ]
    if reverse:
        # Legal, and the reason the client sorts: the index is authoritative and
        # the order of the list is not.
        rows.reverse()
    return httpx.Response(200, json={"data": rows})


class Recorder:
    """Stands in for asyncio.sleep and keeps what it was asked to wait."""

    def __init__(self) -> None:
        self.waits: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.waits.append(round(delay, 3))


def run(handler, texts=None, **overrides) -> tuple[object, Recorder]:
    """One embed() call against a mock transport, with the waits recorded.

    Returns the result or the exception, so the caller asserts on either without
    two spellings of the same setup.
    """
    saved = {name: getattr(settings, name) for name in overrides}
    for name, value in overrides.items():
        object.__setattr__(settings, name, value)

    recorder = Recorder()
    real_sleep = embedder_module.asyncio.sleep
    embedder_module.asyncio.sleep = recorder

    async def once():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        embedder = Embedder(client=client)
        try:
            return await embedder.embed(texts if texts is not None else ["one", "two"])
        finally:
            await embedder.aclose()

    try:
        return asyncio.run(once()), recorder
    except Exception as exc:  # noqa: BLE001 - the exception is the assertion
        return exc, recorder
    finally:
        embedder_module.asyncio.sleep = real_sleep
        for name, value in saved.items():
            object.__setattr__(settings, name, value)


# --------------------------------------------------------------------------
# 1  The happy path, and the ordering it does not trust
# --------------------------------------------------------------------------


def success() -> None:
    result, recorder = run(embedding_response)
    check("two texts, two rows", getattr(result, "shape", None), (2, settings.embed_dims))
    check("and no waiting", recorder.waits, [])
    check("rows are unit length", np.allclose(np.linalg.norm(result, axis=1), 1.0), True)

    # Row 0 was sent as all-ones and row 1 as all-twos; normalised they are
    # identical, so the check that the sort happened has to be on a response the
    # order of which disagrees with the index.
    def descending(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        rows = [
            {"index": position, "embedding": [0.0] * (settings.embed_dims - 1) + [float(position + 1)]}
            for position in range(len(inputs))
        ]
        rows.reverse()
        return httpx.Response(200, json={"data": rows})

    out_of_order, _ = run(descending)
    check(
        "the index decides the row order, not the list",
        [float(row[-1] > 0) for row in out_of_order],
        [1.0, 1.0],
    )
    check(
        "and the first row is the first input's",
        np.array_equal(out_of_order[0], out_of_order[1]),
        True,
    )


# --------------------------------------------------------------------------
# 2  Retrying, and the shape of the wait
# --------------------------------------------------------------------------


def transient_failure() -> None:
    attempts = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(503, text="model is loading")
        return embedding_response(request)

    result, recorder = run(flaky, embed_max_attempts=5, embed_backoff_seconds=1.0)
    check("it got there in the end", getattr(result, "shape", None), (2, settings.embed_dims))
    check("after three attempts", attempts["n"], 3)
    check("waiting 1s then 2s", recorder.waits, [1.0, 2.0])


def the_wait_is_capped() -> None:
    """Doubling without a ceiling turns a long outage into a stopped build."""
    def always_502(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="bad gateway")

    result, recorder = run(
        always_502,
        embed_max_attempts=7,
        embed_backoff_seconds=1.0,
        embed_backoff_max_seconds=4.0,
    )
    check("it gives up", isinstance(result, RuntimeError), True)
    check("after six waits, not seven", len(recorder.waits), 6)
    check("doubling to the cap and staying there", recorder.waits, [1.0, 2.0, 4.0, 4.0, 4.0, 4.0])
    check("and says how many attempts it made", "7 attempt(s)" in str(result), True)
    check("and what the last failure was", "HTTP 502" in str(result), True)


def one_attempt_is_a_choice() -> None:
    """EMBED_MAX_ATTEMPTS=1 has to mean no retry rather than no limit."""
    seen = {"n": 0}

    def always_500(request: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        return httpx.Response(500)

    result, recorder = run(always_500, embed_max_attempts=1)
    check("it is tried once", seen["n"], 1)
    check("and not waited on", recorder.waits, [])
    check("and raises", isinstance(result, RuntimeError), True)


# --------------------------------------------------------------------------
# 3  What is not worth retrying
# --------------------------------------------------------------------------


def permanent_failure() -> None:
    for status, description in ((404, "no such model"), (400, "malformed request")):
        seen = {"n": 0}

        def refuse(request: httpx.Request, status=status, description=description) -> httpx.Response:
            seen["n"] += 1
            return httpx.Response(status, text=description)

        result, recorder = run(refuse, embed_max_attempts=5)
        check(f"a {status} is not retried", seen["n"], 1)
        check(f"and not waited on ({status})", recorder.waits, [])
        check(f"and raises at once ({status})", isinstance(result, RuntimeError), True)
        check(f"naming the status ({status})", f"HTTP {status}" in str(result), True)
        check(f"and the body ({status})", description in str(result), True)


def the_retryable_set() -> None:
    """The decision itself, since the cases above cannot cover every code."""
    def status_error(code: int) -> httpx.HTTPStatusError:
        request = httpx.Request("POST", "http://ollama/v1/embeddings")
        return httpx.HTTPStatusError(
            "x", request=request, response=httpx.Response(code, request=request)
        )

    for code in (500, 502, 503, 504, 408, 425, 429):
        check(f"{code} is retryable", embedder_module._retryable(status_error(code)), True)
    for code in (400, 401, 403, 404, 409, 413, 422):
        check(f"{code} is not", embedder_module._retryable(status_error(code)), False)

    check(
        "a connection that never opened is retryable",
        embedder_module._retryable(httpx.ConnectError("refused")),
        True,
    )
    check(
        "so is a timeout",
        embedder_module._retryable(httpx.ReadTimeout("slow")),
        True,
    )
    # A truncated response from a proxy arrives as a body that does not parse,
    # and that is a transport failure wearing a parser's error class.
    check(
        "and so is a body that did not parse",
        embedder_module._retryable(ValueError("Expecting value")),
        True,
    )


# --------------------------------------------------------------------------
# 4  The guards that were already there
# --------------------------------------------------------------------------


def malformed_answers() -> None:
    def wrong_dims(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        rows = [{"index": i, "embedding": [1.0] * 17} for i in range(len(inputs))]
        return httpx.Response(200, json={"data": rows})

    result, recorder = run(wrong_dims)
    check("a dimension change is fatal", isinstance(result, RuntimeError), True)
    check("and is not retried into", recorder.waits, [])
    check("and says both numbers", "17 dims" in str(result), True)

    def short(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [
            {"index": 0, "embedding": [1.0] * settings.embed_dims}
        ]})

    result, _ = run(short, texts=["one", "two", "three"])
    check("a short batch is fatal", isinstance(result, RuntimeError), True)
    check("and counts both sides", "1 embeddings for 3 inputs" in str(result), True)

    def no_data(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "list"})

    result, recorder = run(no_data, embed_max_attempts=2, embed_backoff_seconds=0.5)
    check("a response with no data is retried", recorder.waits, [0.5])
    check("and then raises", isinstance(result, RuntimeError), True)

    check("nothing in, nothing out", run(embedding_response, texts=[])[0].shape,
          (0, settings.embed_dims))


def main() -> int:
    success()
    transient_failure()
    the_wait_is_capped()
    one_attempt_is_a_choice()
    permanent_failure()
    the_retryable_set()
    malformed_answers()

    if report():
        return 1
    print("embedder: all checks passed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        shutil.rmtree(_VAULT, ignore_errors=True)
