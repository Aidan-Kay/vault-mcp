"""Hybrid retrieval: dense vectors and BM25, fused with reciprocal rank fusion.

Hybrid is not optional here. The vault is dense with exact tokens - reg plates,
boiler model numbers, policy references, postcodes - where dense retrieval
underperforms and BM25 is decisive.

Fusion is not the last step. The fused scores are put on a scale that means the
same thing for every query, and the top k is then chosen by maximal marginal
relevance under a per-note cap, because a long note that chunks six ways could
otherwise occupy every slot and hide every other note that answered.
"""

from __future__ import annotations

import math

import numpy as np

from .embedder import Embedder
from .index import VaultIndex, tokenize

RRF_K = 60
CANDIDATES = 50

# BM25 is down-weighted relative to dense. Measured on this vault: at equal
# weight the lexical list pulls topically-wrong chunks to position one when a
# query happens to share a common term with an unrelated note ("renew" matches
# certbot renewal as readily as insurance renewal).
SPARSE_WEIGHT = 0.5

# If the query's terms appear in no more than this many chunks corpus-wide, the
# query is a lookup rather than a question - a reg plate, a licence key, a part
# code - and the lexical hit is the answer. See _is_lookup.
LOOKUP_MAX_MATCHES = 5

# The largest weighted RRF sum on offer: rank one in both lists. Dividing by it
# puts every fused score in (0, 1] on a scale that reads the same for every
# query - 1.0 is "both arms ranked it first", 0.67 is "dense alone did", 0.33 is
# "BM25 alone did". The raw sums were not that: the top score was about 0.025
# and comparable with nothing. MMR is why this has to happen first - it adds a
# relevance term to a diversity penalty, so the two have to share a scale.
RRF_MAX = (1.0 + SPARSE_WEIGHT) / (RRF_K + 1)

# The mix of relevance and diversity: a chunk from a note already represented
# has to beat the best unseen note by a quarter of the scale to take a second
# slot. Higher than textbook MMR, and deliberately - the usual 0.5 to 0.7 is
# quoted for a *graded* penalty, the cosine distance to what is already chosen.
# Ours is binary, applied in full to any repeat however different the two chunks
# are, so the same lambda bites far harder. Measured on the fixture: at 0.7 the
# penalty alone held every query to one chunk per note and the cap below never
# bound, which makes the guarantee the cap is supposed to give unreadable.
MMR_LAMBDA = 0.8

# No one note may hold more than this share of the result set. Upstream caps at
# a constant three, which at our default k=5 still lets one note take three of
# five slots - the failure the cap exists to stop. Derived from k instead, so it
# keeps its meaning when a caller asks for twenty.
MAX_PER_PATH_SHARE = 3


def _top_indices(scores: np.ndarray, limit: int) -> list[int]:
    if scores.size == 0:
        return []
    limit = min(limit, scores.size)
    # argpartition is O(n) - the full sort only touches the candidate window.
    partitioned = np.argpartition(-scores, limit - 1)[:limit]
    return partitioned[np.argsort(-scores[partitioned])].tolist()


def fuse(dense: list[int], sparse: list[int]) -> list[tuple[int, float]]:
    """Weighted reciprocal rank fusion, normalised to (0, 1].

    score(d) = w / (RRF_K + rank(d)), summed over the lists containing d, over
    the most that sum could have been.
    """
    scores: dict[int, float] = {}
    for weight, ranking in ((1.0, dense), (SPARSE_WEIGHT, sparse)):
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + weight / (RRF_K + rank)
    return sorted(((doc, s / RRF_MAX) for doc, s in scores.items()), key=lambda item: -item[1])


def _is_lookup(lexical_matches: int) -> bool:
    """True when the query's terms are near-unique in the corpus.

    Rank fusion alone cannot handle this case. A document found by only one of
    the two rankers scores 1/(RRF_K+1) whichever ranker found it, so BM25's
    correct rank-one hit for an exact identifier ties with the dense list's
    rank-one hit and loses the tie-break. Measured on this vault, dense
    retrieval scores 0/40 on exact identifiers where BM25 scores 40/40, so the
    tie must be resolved in favour of the lexical hit - but only when the match
    is genuinely near-unique, or the same rule would promote an incidental
    keyword hit over a correct semantic one.
    """
    return 0 < lexical_matches <= LOOKUP_MAX_MATCHES


def per_path_cap(k: int) -> int:
    """How many of k slots one note may hold. Two at the default k of five."""
    return max(1, math.ceil(k / MAX_PER_PATH_SHARE))


def diversify(
    ranked: list[tuple[int, float]],
    chunks: list[dict],
    k: int,
    pinned: int | None = None,
) -> list[tuple[int, float]]:
    """Choose k results by maximal marginal relevance over the source note.

    Diversity is measured on the chunk's path rather than on its vector. The
    failure being fixed is named in terms of paths - "six chunks of one file,
    hiding other relevant files" - and a path comparison is exact, costs
    nothing, and stays meaningful under the stub embedder the relevance suite
    runs with, where a vector comparison would not.

    `pinned` is the lookup override's hit. It is taken first and exempt from
    diversification: if the query is a policy number, one exact hit is the
    answer and diversity is noise. Its note still counts against the cap, so a
    pin does not buy that note a second slot it would not otherwise have had.

    It keeps its *own* fused score. It used to be handed ranked[0][1] - another
    chunk's score, a fused value belonging to a document that had just lost its
    place to it. Carrying its own means the returned scores no longer descend
    monotonically when the override fires, which is the accurate picture: the
    override moved a chunk on evidence the fusion does not hold, and saying so
    beats restating the fused ranking in an order that contradicts it.
    """
    cap = per_path_cap(k)
    scores = dict(ranked)
    chosen: list[tuple[int, float]] = []
    taken: dict[str, int] = {}
    used: set[int] = set()

    def take(doc: int, score: float) -> None:
        chosen.append((doc, score))
        used.add(doc)
        path = chunks[doc]["path"]
        taken[path] = taken.get(path, 0) + 1

    if pinned is not None and pinned in scores:
        take(pinned, scores[pinned])

    while len(chosen) < k:
        best: tuple[int, float] | None = None
        best_score = -math.inf
        for doc, score in ranked:
            if doc in used:
                continue
            path = chunks[doc]["path"]
            if taken.get(path, 0) >= cap:
                continue
            marginal = MMR_LAMBDA * score - (1 - MMR_LAMBDA) * (1.0 if path in taken else 0.0)
            if marginal > best_score:
                best, best_score = (doc, score), marginal
        if best is None:
            break
        take(*best)

    # The cap reorders; it does not truncate. If every candidate left belongs to
    # a note already at its cap then there are no other notes being hidden,
    # which is the only thing the cap was protecting, and the caller asked for k.
    if len(chosen) < k:
        for doc, score in ranked:
            if doc not in used:
                take(doc, score)
                if len(chosen) == k:
                    break
    return chosen


async def search(index: VaultIndex, embedder: Embedder, query: str, k: int) -> list[dict]:
    if index.size == 0:
        return []

    query_vector = await embedder.embed_query(query)
    dense_ranking = _top_indices(index.matrix @ query_vector, CANDIDATES)

    sparse_ranking: list[int] = []
    lexical_matches = 0
    if index.bm25 is not None:
        tokens = tokenize(query)
        if tokens:
            sparse_scores = np.asarray(index.bm25.get_scores(tokens))
            lexical_matches = int((sparse_scores > 0).sum())
            # Drop non-matching candidates. argpartition returns a full window
            # regardless of score, so without this a query matching nothing
            # lexically contributes 50 arbitrary votes to the fusion.
            sparse_ranking = [
                i for i in _top_indices(sparse_scores, CANDIDATES) if sparse_scores[i] > 0
            ]

    ranked = fuse(dense_ranking, sparse_ranking)
    pinned = sparse_ranking[0] if sparse_ranking and _is_lookup(lexical_matches) else None

    results: list[dict] = []
    for doc, score in diversify(ranked, index.chunks, k, pinned):
        chunk = index.chunks[doc]
        results.append(
            {
                "path": chunk["path"],
                "title": chunk["title"],
                "breadcrumb": chunk["breadcrumb"],
                "line": chunk["line"],
                "score": round(score, 4),
                "text": chunk["text"],
            }
        )
    return results


def format_results(query: str, results: list[dict]) -> str:
    """Markdown, with the source path as a heading above each chunk, so the
    model can cite it without a second call."""
    if not results:
        return f'No vault matches for "{query}".'

    blocks = [f'{len(results)} result(s) for "{query}":\n']
    for position, result in enumerate(results, start=1):
        location = result["path"]
        if result["breadcrumb"]:
            location += f" > {result['breadcrumb']}"
        blocks.append(
            f"### {position}. {location}\n"
            f"*score {result['score']} - line {result['line']}*\n\n"
            f"{result['text'].strip()}\n"
        )
    return "\n".join(blocks)
