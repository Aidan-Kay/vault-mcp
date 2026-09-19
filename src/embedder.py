"""Batched embedding against Ollama, with L2 normalisation on receipt.

Retried with exponential backoff, because the failures this sees are nearly all
timing. The first call after an idle period races a model load when
OLLAMA_KEEP_ALIVE is unset; a restarted Ollama refuses connections for a second
or two; a cold build asks for thirty-five batches in a row and only needs one of
them to land badly. One fixed 2s retry covered the first of those and nothing
else.

What it does *not* retry is the other half of the same decision, and the more
useful half: a 4xx that is not a timing answer is a statement about the request,
and the request will be identical next time.
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx
import numpy as np

from .config import settings

log = logging.getLogger(__name__)

QUERY_PREFIX = "search_query: "

# Status codes worth trying again. Everything 5xx is the endpoint failing to do
# something it agreed to; 429 is it asking us to wait; 408 and 425 are the two
# timing answers that are not a refusal. Every other 4xx is a statement about the
# request, and the request will be identical next time.
RETRYABLE_STATUS = frozenset({408, 425, 429})


def _retryable(exc: Exception) -> bool:
    """Whether trying the same call again could plausibly answer differently.

    A retry budget spent on a permanent error is worse than no budget at all: it
    turns "nomic-embed-text is not pulled" from an error into a wait, and the log
    line that says so arrives after the operator has gone looking elsewhere.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status >= 500 or status in RETRYABLE_STATUS
    # A transport error, a timeout, or a body that did not parse - which is what
    # a truncated response from a proxy looks like from here. All transient
    # shapes, so all retried.
    return True


def _describe(exc: Exception | None) -> str:
    """One line, with the status code on it when there is one.

    httpx's own str() for a status error is three lines of URL and a link to its
    documentation, which is not what belongs in a log this is read from.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        body = exc.response.text.strip().replace("\n", " ")
        return f"HTTP {exc.response.status_code} from {exc.request.url}: {body[:200]}"
    if exc is None:
        return "no error recorded"
    return f"{type(exc).__name__}: {exc}"


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Unit-length rows, so cosine similarity collapses to a dot product."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)  # a zero vector must not produce NaN
    return (matrix / norms).astype(np.float32, copy=False)


class Embedder:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
        self._dims: int | None = None

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, inputs: list[str]) -> list[list[float]]:
        payload = {"model": settings.embed_model, "input": inputs}
        url = f"{settings.ollama_url}/embeddings"

        attempts = max(1, settings.embed_max_attempts)
        started = time.monotonic()
        last_error: Exception | None = None

        for attempt in range(attempts):
            try:
                response = await self._client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()["data"]
                # OpenAI-compatible responses carry an index; do not trust order.
                data.sort(key=lambda row: row.get("index", 0))
                if attempt:
                    log.info("embed batch succeeded on attempt %d", attempt + 1)
                return [row["embedding"] for row in data]
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                last_error = exc
                if not _retryable(exc):
                    # A 404 for a model nobody pulled will be a 404 in thirty
                    # seconds too, and spending the whole budget on it turns a
                    # clear message into a slow one.
                    raise RuntimeError(f"embedding failed: {_describe(exc)}") from exc
                if attempt == attempts - 1:
                    break
                delay = min(
                    settings.embed_backoff_max_seconds,
                    settings.embed_backoff_seconds * 2**attempt,
                )
                log.warning(
                    "embed batch failed (%s); attempt %d of %d, retrying in %.1fs",
                    _describe(exc), attempt + 1, attempts, delay,
                )
                await asyncio.sleep(delay)

        raise RuntimeError(
            f"embedding failed after {attempts} attempt(s) over "
            f"{time.monotonic() - started:.1f}s: {_describe(last_error)}"
        ) from last_error

    async def embed(self, texts: list[str]) -> np.ndarray:
        """Embed pre-scaffolded texts. Returns an (N, dims) normalised float32 array."""
        if not texts:
            return np.zeros((0, settings.embed_dims), dtype=np.float32)

        vectors: list[list[float]] = []
        for start in range(0, len(texts), settings.embed_batch_size):
            batch = texts[start : start + settings.embed_batch_size]
            embeddings = await self._post(batch)
            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"Ollama returned {len(embeddings)} embeddings for {len(batch)} inputs"
                )
            if self._dims is None:
                self._dims = len(embeddings[0])
                if self._dims != settings.embed_dims:
                    # A silent dimension change would corrupt the whole index.
                    raise RuntimeError(
                        f"{settings.embed_model} returned {self._dims} dims, "
                        f"expected {settings.embed_dims}"
                    )
                log.info("embedding model %s: %d dims", settings.embed_model, self._dims)
            vectors.extend(embeddings)

        return l2_normalise(np.asarray(vectors, dtype=np.float32))

    async def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query with the asymmetric prefix nomic requires."""
        matrix = await self.embed([QUERY_PREFIX + query])
        return matrix[0]
