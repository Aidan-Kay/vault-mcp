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
from .index import VaultIndex, chunk_terms, tokenize

RRF_K = 60
CANDIDATES = 50

# BM25 is down-weighted relative to dense. Measured on this vault: at equal
# weight the lexical list pulls topically-wrong chunks to position one when a
# query happens to share a common term with an unrelated note ("renew" matches
# certbot renewal as readily as insurance renewal).
SPARSE_WEIGHT = 0.5

# If the *rarest* of the query's terms appears in no more than this many chunks,
# the query is a lookup rather than a question - a reg plate, a licence key, a
# part code - and the lexical hit is the answer. See _is_lookup.
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


def rarest_term_frequency(doc_freqs: dict[str, int], tokens: list[str]) -> int:
    """How many chunks hold the least common of the query's terms.

    Terms the corpus has never seen are skipped rather than counted as zero, so
    that pairing an identifier with a word the vault does not contain still
    reads as a lookup for the identifier.
    """
    present = [doc_freqs[token] for token in tokens if doc_freqs.get(token)]
    return min(present) if present else 0


def _is_lookup(rarest_matches: int, any_chunk_holds_every_term: bool) -> bool:
    """True when the query names something near-unique that one chunk holds.

    Rank fusion alone cannot handle this case. A document found by only one of
    the two rankers scores 1/(RRF_K+1) whichever ranker found it, so BM25's
    correct rank-one hit for an exact identifier ties with the dense list's
    rank-one hit and loses the tie-break. Measured on this vault, dense
    retrieval scores 0/40 on exact identifiers where BM25 scores 40/40, so the
    tie must be resolved in favour of the lexical hit - but only when the match
    is genuinely near-unique, or the same rule would promote an incidental
    keyword hit over a correct semantic one.

    This asks about the rarest term, where it used to ask about the union over
    all of them - how many chunks matched *any* query term. The union is a
    different quantity, and it made the rule fail in both directions at once.
    An identifier that tokenises into one rare piece and one common one - a MAC
    address, a spec code, a hyphenated part number - would push the union into
    the hundreds and the override would not fire, which is how two provable
    identifier lookups missed entirely against the real vault: their rarest
    piece appeared in exactly one chunk, their commonest in 203 and 621. In the
    other direction the union was small for any short question the vault mostly
    did not answer, so 'what has gone wrong with the car' pinned the car
    insurance note and a query nothing answered still pinned something.

    The rarest term alone is not enough either, and the fixture said so before
    this shipped: it fired on 35 of 36 queries, because "appears in at most five
    chunks" is a different claim in a 59-chunk corpus than in a 2102-chunk one,
    where it is most content words rather than few. Lowering the threshold does
    not separate them - 'escape of water claim' and a query the vault cannot
    answer at all both contain a term appearing exactly once.

    So the second half: the chunk about to be pinned must contain *every* term
    of the query. That is the difference between naming a thing and sharing a
    word with one. An identifier's pieces all live in the one chunk that holds
    it, however common any single piece is; five words scattered across five
    notes are a question, and the fusion should answer it.
    """
    return any_chunk_holds_every_term and 0 < rarest_matches <= LOOKUP_MAX_MATCHES


def per_path_cap(k: int) -> int:
    """How many of k slots one note may hold. Two at the default k of five."""
    return max(1, math.ceil(k / MAX_PER_PATH_SHARE))


def lookup_hits(
    chunks: list[dict],
    term_sets: list[frozenset[str]],
    tokens: list[str],
    sparse_ranking: list[int],
) -> list[int]:
    """Every chunk that holds the whole query, best first, one per file.

    One per file rather than all of them, and the difference is what filing
    documents made visible. A 27-page statement can name an account number on
    every page; returning five chunks of it answers the question five times and
    hides the note that owns it. One chunk per path says "here is every file
    that names this thing", which is what a lookup is actually asking.

    It is bounded without needing a bound: _is_lookup has already established
    that the rarest term appears in at most LOOKUP_MAX_MATCHES chunks, so there
    are at most that many paths for this to find.
    """
    wanted = set(tokens)
    hits: list[int] = []
    seen: set[str] = set()
    for doc in sparse_ranking:
        if not wanted <= term_sets[doc]:
            continue
        path = chunks[doc]["path"]
        if path in seen:
            continue
        seen.add(path)
        hits.append(doc)
        if len(seen) >= LOOKUP_MAX_MATCHES:
            # There cannot be more: the caller has already established that the
            # rarest term appears in at most this many chunks, so this many
            # distinct paths is every path there is.
            break
    return hits


def diversify(
    ranked: list[tuple[int, float]],
    chunks: list[dict],
    k: int,
    pinned: list[int] | None = None,
) -> list[tuple[int, float]]:
    """Choose k results by maximal marginal relevance over the source note.

    Diversity is measured on the chunk's path rather than on its vector. The
    failure being fixed is named in terms of paths - "six chunks of one file,
    hiding other relevant files" - and a path comparison is exact, costs
    nothing, and stays meaningful under the stub embedder the relevance suite
    runs with, where a vector comparison would not.

    `pinned` is the lookup override's hits, best first. They are taken first and
    exempt from diversification: if the query is a policy number, the exact hits
    are the answer and diversity is noise. Their notes still count against the
    cap, so a pin does not buy a note a second slot it would not otherwise have
    had.

    It became a list when documents joined the index, because the case it could
    not express turned out to be the ordinary one: an account number lives in
    the note *and* in the bill filed beside it. Pinning only BM25's best of the
    two left the other to compete on fused score alone, and a provable
    identifier lookup went from rank one to missing entirely - the eval caught
    it on the fixture before any of this was deployed.

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

    # Taken in the order given, which is BM25's. Re-sorting them by fused score
    # was tried and is worse: 'Poly1305' appears in two notes, the one that
    # defines it scores 7.57 on BM25 and 0.333 fused, the one that mentions it
    # in passing 5.93 and 0.790. The override fires on lexical evidence, so the
    # set it pins is ordered by lexical evidence - handing that ordering back to
    # the fusion asks the ranker that could not tell them apart to arbitrate
    # between the two it was overruled on.
    for doc in pinned or ():
        if len(chosen) >= k:
            break
        if doc in scores and doc not in used:
            take(doc, scores[doc])

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
    tokens: list[str] = []
    rarest = 0
    if index.bm25 is not None:
        tokens = tokenize(query)
        if tokens:
            sparse_scores = np.asarray(index.bm25.get_scores(tokens))
            rarest = rarest_term_frequency(index.doc_freqs, tokens)
            # Drop non-matching candidates. argpartition returns a full window
            # regardless of score, so without this a query matching nothing
            # lexically contributes 50 arbitrary votes to the fusion.
            sparse_ranking = [
                i for i in _top_indices(sparse_scores, CANDIDATES) if sparse_scores[i] > 0
            ]

    ranked = fuse(dense_ranking, sparse_ranking)

    pinned: list[int] = []
    # The rarest-term test first, because it is two dict lookups and the
    # containment test is a scan. Most queries are questions whose commonest
    # term is everywhere, so they fail here and never pay for the scan.
    if sparse_ranking and tokens and 0 < rarest <= LOOKUP_MAX_MATCHES:
        holders = lookup_hits(index.chunks, index.term_sets, tokens, sparse_ranking)
        if _is_lookup(rarest, bool(holders)):
            pinned = holders

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
