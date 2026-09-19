# Vault MCP

An MCP server and REST API over a local [Obsidian](https://obsidian.md) vault. It
indexes the vault for hybrid search and serves read and write access to it through
two interfaces backed by one implementation.

## Why

Two problems with the plugin it replaces:

- **Heading targets had to be exact.** It keys every heading by its full ancestor
  path and does a single lookup, so anything short of the complete path from the H1
  down matches nothing. 86% of this vault's notes are wrapped in one H1, which makes
  nearly every useful target a two- or three-segment path the model has to guess up
  front. Here a bare leaf name works whenever it is unique, and when it is not, the
  error names the ancestors to prepend.
- **No retrieval.** Finding a note meant knowing its path. `vault_search` is hybrid —
  dense vectors *and* BM25, fused with reciprocal rank fusion. Hybrid is not optional
  for this corpus: it is dense with exact tokens (reg plates, boiler model numbers,
  policy references, postcodes) where dense retrieval alone underperforms.

## Interfaces

**MCP** at `/mcp` — eleven tools:

| Read | Write |
| --- | --- |
| `vault_search` — hybrid search | `vault_patch` — replace a section |
| `vault_read` — whole note, one `section=`, or a document's extracted text | `vault_append` — add to the end |
| `vault_list` — browse a folder | `vault_write` — create or overwrite |
| `vault_map` — heading tree as `::` paths | `vault_set_body` — replace the prose, keep the block |
| | `vault_set_frontmatter` — set or delete a key |
| | `vault_delete` — remove a note or a document |
| | `vault_move` — move, rewriting inbound links |

`vault_map` emits `::`-joined paths rather than an indented tree, because the output
is meant to be pasted straight back as a patch target.

**REST** at `/vault/<path>` — `GET`, `PUT`, `POST`, `PATCH`, `DELETE`, mirroring the
shape `obsidian-local-rest-api` used. n8n's HTTP Request nodes speak plain REST and
cannot easily build a JSON-RPC envelope, so migrating a node is a find-and-replace on
the URL and the auth header rather than a rewrite into JSON-RPC.

| Call | Does |
| --- | --- |
| `GET /vault/<path>` | The note's markdown. `?section=` narrows it to one heading. |
| `GET /vault/<path>` with `Accept: application/json` | `{path, content, body, frontmatter}` |
| `PUT /vault/<path>` | Create or replace, body is the note. A document suffix takes the binary branch instead: the body is stored byte for byte and the answer is JSON. |
| `POST /vault/<path>` | Append, creating the note if it is absent |
| `PATCH /vault/<path>` | `Target:` a heading, or a frontmatter key with `Target-Type: frontmatter`, or the prose with `Target-Type: body`. `Operation: delete` removes a frontmatter key. |
| `DELETE /vault/<path>` | Remove the note |
| `GET /frontmatter?key=&value=` | Notes whose field holds that exact value, as `[{"filename": …}]`. `&dir=` narrows the walk to one folder. |
| `GET /maintenance` | Runs the vault's nine checkers and answers what each printed, as JSON and as one markdown block. |
| `GET /callouts` | Every open callout in the vault, as JSON: one object per callout with its note, line, type, severity, title and body. |
| `GET /healthz` | `ok`, once the port is bound. Depends on nothing, and is the one route with no bearer token. |
| `GET /readyz` | Index, `index.md` and watcher state as JSON, 200 when all three are good and 503 with the same body when any is not. |

On the structured read, `content` is the file byte for byte and `body` is the same text
with the frontmatter block removed — so a caller wanting the prose does not carry its
own YAML regex, and a note whose block does not parse still comes back whole in
`content`.

A frontmatter `PATCH` takes a **JSON** body, so `"approved"` needs its quotes and `2`
does not: the value is decoded rather than copied, because writing the quotes into the
YAML would change what every comparison downstream sees. Removing a field is
`Operation: delete` with no body rather than a `null` value, because a null is refused —
there is no text that reads back as one. A `Target-Type: body` `PATCH`
replaces the prose and leaves the frontmatter block exactly as it was, for notes whose
text is regenerated on a schedule but whose metadata is written once.

`/maintenance` and `/callouts` are the two read-only routes that touch no note. Both
run a stdlib script out of the vault's own `.scripts/`, because n8n's container has
neither Python nor the vault mounted and this one has both. Both take no parameters —
every argv is a constant here, so nothing a caller sends can reach a command line — and
both accept `POST` as well as `GET`, so a workflow that only sends `POST`s needs no
special case. `/maintenance` asks whether the vault is well-formed; `/callouts` asks what
it is still carrying, and hands back the list a workflow raises from. Findings are a 200,
however many there are: only the run failing outright is a 500, which is what lets a
workflow branch on "the report is missing" without parsing it.

`/frontmatter` walks the filesystem and never the semantic index — `Workflows/` is
in `SEARCH_EXCLUDE_DIRS` and so is absent from search entirely, which is exactly
where the notes it is asked about live.

Two deliberate differences from the plugin it replaces:

- **The structured read answers `application/json`**, never the vendor
  `application/vnd.olrapi.note+json`, even when that is what was requested. n8n does
  not recognise the vendor type as JSON and hands the body to the workflow as a string
  in `$json.data`; answering real JSON makes a node written against that shape throw
  rather than silently succeed with nothing.
- **A missing note is `404`**, where every other rejected path, target or body is
  `400`. "No such note" is the one error a caller routes on rather than logs.

Both surfaces call `src/operations.py`, so the resolver and the vault conventions are
applied once regardless of how the caller arrived. Every write bumps the note's
`timestamp`, or reports why it could not.

## Filed documents

A vault holds more than notes. A contract, a bill, an insurance schedule — the note
beside it can summarise what it says, but the document is the thing that actually says
it. `PUT`ting one to a path with an allowlisted suffix stores the bytes exactly as sent,
extracts the text, and indexes it under the document's own path, so a search result can
read `Home/Utilities/Files/2026-09-17 Kestrel Energy - Contract Confirmation.pdf`.

**A document may only be written directly inside a folder named `Files`.** That is the
containment control, and it is stricter than a suffix allowlist on its own: a compromised
pipeline token that can carry raw bytes still cannot drop a file anywhere a note lives.
The `Files/` folder is created on demand, so filing the first document beside a note
needs no separate folder-creation capability. The rule is the vault's own filing
convention doing double duty as enforcement, which is what makes it checkable by a human
reading the vault as well as by the server.

```
PUT /vault/Home/Utilities/Files/2026-09-17 Kestrel Energy - Contract Confirmation.pdf
→ {"path": "…", "sha256": "…", "size": 80157, "status": "filed",
   "extraction": "extracted", "pages": 1, "has_text_layer": true,
   "extracted_chars": 802, "detail": ""}
```

`status` says what happened to the file and `extraction` what happened to its text. They
are separate because the interesting case is where they disagree: a scan files perfectly
and extracts to nothing.

- **The branch is chosen by suffix, never by `Content-Type`.** n8n forwards whatever the
  upstream mail server labelled an attachment, and the one thing the caller is reliably
  sure of is the name it chose to file it under. No header should decide how bytes are
  stored.
- **Re-sending identical bytes is a satisfied `200`, not a conflict.** The same attachment
  *will* arrive twice — a thread reprocessed, a statement forwarded on, a run retried —
  and a pipeline should not have to interpret an error to discover that what it wanted is
  already true. Different bytes at the same path are a real collision and need
  `?overwrite=true`.
- **A document is opaque and move-only.** It can be read, filed, renamed and deleted; it
  cannot be patched, appended to, or given frontmatter, because there is no markdown
  inside it for those verbs to address. Moving one preserves every byte — no timestamp
  bump, no re-encode — while inbound links are still repointed, since the note holding
  the link is the thing being rewritten.
- **Extraction is the server's own read.** The workflow that uploads a PDF has usually
  read it already to decide where to file it. That read serves the workflow's decisions;
  this one decides what search contains, and folding them together would make the vault's
  contents depend on which workflow happened to deliver the file.
- **A document that yields no text is reported, not smoothed over.** By the time
  extraction runs the file is already in the vault, so refusing it is not on offer —
  `extraction` says `needs_ocr` or `no_text` and `extracted_chars` is `0`. The vault's
  `check_documents.py` turns that into a finding, alongside a document in a `Files/`
  folder that no note links to, which is the failure the two-step filing workflow
  creates when the upload succeeds and the `## Documents` row does not.

PDF text is recovered with PyMuPDF4LLM — inferred headings, tables as markdown tables,
and page boundaries as the structural unit where a document has no headings of its own,
which a one-page bill usually does not. Scans have no text layer and are OCR'd through
Tesseract where it is installed; the image ships it, and `DOC_OCR=false` turns it off.

Uploading is REST-only. MCP arguments are JSON, and a 10 MB PDF would be 13 MB of base64
emitted a token at a time.

## How a search result is chosen

Fusing the two arms is not the last step, and the three steps after it are the ones
that decide what a caller actually sees.

- **Fused scores are normalised to `(0, 1]`.** A weighted RRF sum has no meaning on
  its own — the top score used to be about `0.025` and was comparable with nothing, not
  even with the same query run yesterday. Each sum is now divided by the largest one
  available, which is rank one in both arms, so `1.0` reads as "both arms ranked it
  first", `0.67` as "dense alone did" and `0.33` as "BM25 alone did". It is still not a
  probability and should not be read as one, but it is at least the same scale twice.
- **No note may hold more than a third of the results.** A note that chunks six ways
  could take every slot at the default `k` and hide every other note that answered;
  results are now picked by maximal marginal relevance under a cap of `ceil(k/3)`. The
  cap is a ceiling rather than a quota — when other notes are competitive the diversity
  term spends the slots on them and the cap never binds — and it reorders rather than
  truncates, so a query whose only answers live in one note still gets `k` of them.
- **A query that names one thing is answered by that thing.** When the rarest of a
  query's terms appears in at most five chunks *and* one chunk holds every term, BM25's
  top hit is pinned to rank one and exempted from diversification: if the query is a
  policy number, one exact hit is the answer and diversity is noise. Both halves are
  load-bearing. Asking about the rarest term rather than the union over all of them is
  what lets a hyphenated part number match when one of its pieces is common — measured
  against the real vault, that alone moved five identifier lookups from missing to rank
  one. Requiring one chunk to hold every term is what stops a question that merely
  contains an unusual word pinning whichever note happens to use it. The pinned hit
  carries its own fused score rather than the score of the chunk it displaced, so the
  returned scores do not always descend — the honest picture, since the override moved a
  chunk on evidence the fusion does not hold.

  It pins one chunk per *file*, not one chunk. An account number lives in the note and
  in the statement filed beside it, and both are answers; pinning only BM25's best of the
  two left the other to compete on fused score alone, which took a provable identifier
  lookup from rank one to missing outright. One per file is also what stops a 27-page
  policy booklet that names the number on every page answering the question five times
  and hiding the note that owns it. The set is bounded without needing a bound, since the
  rule has already established the rarest term appears in at most five chunks.

Two things feed it that are worth knowing about:

- **The lexical arm stems.** `tokenize()` lowercases, splits on anything outside
  `[a-z0-9]`, drops stop words and runs Porter2, so "readings" finds a note that only
  ever writes "reading" and "renewing" finds one that writes "renewal". The deliberate
  part stays: `nomic-embed-text` is still three tokens, and the stemmer leaves `nomic`
  and every identifier alone.
- **A section that is one flat list chunks per item.** A log, an inbox or a list of
  twelve unrelated bullets under one heading would otherwise become one embedding
  averaging twelve subjects. Most lists are not that, so the rule asks for seven
  items, one list rather than two, bullets rather than numbers, a median item of at
  least eight tokens, and something left over once the links are stripped out. Each of
  those is a shape this vault holds: a recipe method is a sequence and step four
  answers nothing alone, a list of film titles is a register of names rather than of
  subjects, `## Related notes` is four pointers, and a `## Account` block is four
  fields. All of them stay whole; the house log does not.

## The index is generated

The vault's root `index.md` is one line per note — its title, a link, and its
`description` — under headings that mirror the folder tree. Nothing in it is a
judgement call, so `src/indexdoc.py` derives it rather than asking a model to remember
to update it. The rationale, and every fallback, is in that module's docstring.

It is rebuilt from the **filesystem watcher**, not from the write path, so it does not
matter how the change arrived — an MCP tool call, a REST `PUT` from n8n, or someone
typing in Obsidian all reach it the same way. A full scan runs once at startup; each
change after that re-reads a single note.

Two properties are deliberate:

- **It only writes when the rendered body differs**, so a note edit that changes nothing
  the index displays leaves `index.md` — and the vault's git history — alone.
- **`index.md` is protected from every writer.** A write to it is futile rather than
  dangerous, and a tool that accepts one teaches the caller the edit worked. Fix a wrong
  line by fixing the note's `title` or `description`. It stays readable.

Generated note series — the folders in `INDEX_EXCLUDE_DIRS` — are not indexed note by
note; the approvals folder alone would swamp the document. Each gets one line in its
parent section saying so, and only when the folder actually exists.

## Restarts, and what one costs

Two routes, answering two questions that must not be conflated. An unhealthy
container is one something restarts, and restarting to recover a single broken
subsystem takes out every working one with it — so **`/healthz` depends on nothing**.
It returns `ok` once the port is bound, reads no state at all, and is what the image's
`HEALTHCHECK` calls. It is also the one route with no bearer token, so that the
healthcheck command does not carry the key and `docker inspect` does not show it.

`/readyz` is the diagnosis, and is never acted on automatically. It reports the index
(built, building or failed, with note and chunk counts and the last build time),
`index.md`, the watcher, and the cache below — **the same body shape on 200 and on
503**, so a failure is read from the fields a success was read from rather than from a
second schema. A vault whose watch thread has died still answers every read correctly
from a slightly older index: worth reporting, not worth a restart.

It also names the running build, which is what AGPL section 13 asks of a service with
no UI. The commit is written into the image at build time; a checkout reads `.git`.

### The index cache

A restart used to re-read and re-embed the whole vault. It now reuses whatever has not
changed, keyed by content, in a single file outside the vault:

| Start | Real vault, 208 files, 2223 chunks |
| --- | --- |
| cold | **40 s** |
| warm | **7.8 s** |
| after a chunker change | **17.3 s** |

Measured from a development machine reading the vault over a network mount, where the
walk and the per-file `resolve()` alone are 2.4 s. On the server the same vault warms in
**3.3 s**; the ratio is the part that transfers, not the absolute figures.

Two stores rather than one, and that is the whole design:

    chunks    (vault path, sha256 of the file's bytes)  ->  the chunks it made
    vectors   sha256 of a chunk's embed_text            ->  its embedding

Each invalidation then costs what it should. A **chunker or extractor change** drops
the chunks and keeps every vector, because an embedding is a function of the text
alone — so re-chunking a vault is the ten seconds of chunking and PDF extraction and
not the forty of a cold build. A **model change** drops the vectors and keeps the
chunks. A **note edit** drops one file's chunks and only the vectors whose text moved.
A **rename** drops the chunks, since the path is in them, and keeps the vectors.

The chunk store's key includes a digest of the *source* of the modules that produce
chunks, so a chunker change invalidates it without anybody remembering to say so.
Editing a comment invalidates it too; that costs ten seconds once, where a missed
invalidation serves stale chunks until somebody notices search has gone strange.

It is an optimisation and never more: a warm build produces the same chunks in the same
order and the same matrix row for row — asserted in `tests/cache.py`, and confirmed
bit-for-bit against `nomic-embed-text` on the real vault. Every failure is contained:
a corrupt, unreadable or unwritable cache costs a slower start and nothing else.

It holds the vault's text — finances, insurance, addresses — outside the vault, so the
directory is `0700` and the file `0600`. `INDEX_CACHE_PATH=` empty turns it off.

What it does not cache is tokenising, which is what a warm start now spends 5.3 of its
7.8 seconds on. The same tokenising used to be repeated for the *whole corpus* every
time one note was saved, which is 5.4 seconds of the event loop for a one-file change;
the index now carries its token lists forward, and a note edit costs **133 ms**.

## Scoped writes

`/mcp/only/<path>` is the same MCP surface with this request's writes confined to one
note (`/mcp/only/Workflows/Approvals/x.md`) or one folder
(`/mcp/only/Workflows/Approvals`). Reads are never scoped: an agent confined to one note
still has to read the conventions and whatever that note refers to. `vault_move` is
refused outright while a scope is set, because rewriting inbound links touches every
note that points at the source.

It rides on the URL rather than a header because that is the part a caller can vary per
call — n8n's MCP Client node takes its auth from a static credential but its endpoint
from an expression — so one agent with one tool list can be handed a different remit per
invocation, with no second copy of the workflow to keep in step. The scope cannot
outlive its request: the transport is stateless, and a `ContextVar` keeps concurrent
requests from seeing each other's.

This exists because an agent told in prose to "carry nothing out" replaced a section of
the vault's root `index.md` while revising an unrelated note. A sentence in a prompt is
not a guard.

## Configuration

All configuration is environment variables. `VAULT_MCP_API_KEY` is required — the
server refuses to start without it rather than treating an empty key as "auth off".

| Variable | Default | Purpose |
| --- | --- | --- |
| `VAULT_MCP_API_KEY` | — | Bearer token. **Required.** |
| `VAULT_PATH` | `/vault` | Vault root inside the container |
| `MCP_ALLOWED_HOSTS` | `vault-mcp:8080,127.0.0.1:8090` | Host-header allowlist |
| `OLLAMA_URL` | `http://ollama:11434/v1` | OpenAI-compatible embedding endpoint |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `EMBED_DIMS` | `768` | Embedding dimensions |
| `EMBED_BATCH_SIZE` | `64` | Embedding requests per batch |
| `EMBED_MAX_ATTEMPTS` | `5` | Attempts per embedding batch before giving up |
| `EMBED_BACKOFF_SECONDS` | `1.0` | First wait between attempts; it doubles from there |
| `EMBED_BACKOFF_MAX_SECONDS` | `30.0` | Ceiling on that doubling |
| `INDEX_CACHE_PATH` | `/cache/index.npz` in the image, `$XDG_CACHE_HOME/vault-mcp/index.npz` otherwise | Chunk-and-vector cache. Empty disables it. |
| `INDEX_CACHE_FLUSH_SECONDS` | `60.0` | How often an edited cache is written back. `0` writes at build and shutdown only. |
| `SEARCH_EXCLUDE_DIRS` | `Workflows,Reports,.obsidian` | Folder *names*, left out of the search index |
| `INDEX_EXCLUDE_DIRS` | the six generated series | Folder *paths*, left out of `index.md` |
| `CHUNK_TARGET_TOKENS` | `400` | Target chunk size |
| `CHUNK_OVERLAP_TOKENS` | `60` | Overlap between chunks |
| `CHUNK_MIN_TOKENS` | `120` | Below this, a chunk merges into its neighbour |
| `SEARCH_DEFAULT_K` | `6` | Default result count |
| `DOC_SUFFIXES` | `.pdf` | Binary document types the vault will carry. Each needs an extractor, so adding one is a code change. |
| `DOC_FILES_DIR` | `Files` | The folder name a document upload must land directly inside |
| `DOC_OCR` | `true` | OCR a document with no text layer, where Tesseract is installed |
| `WATCH_DEBOUNCE_SECONDS` | `2.0` | Filesystem-watch debounce before reindexing |
| `BIND_HOST` / `BIND_PORT` | `0.0.0.0` / `8080` | Listen address |

`SEARCH_EXCLUDE_DIRS` and `INDEX_EXCLUDE_DIRS` read as a pair, but they are not
interchangeable and must not be merged — they differ in **what they ask** and in **what
they accept**.

The first drops `Workflows/` and `Reports/` from search wholesale; the second cannot,
because curated notes live inside both — `Workflows/Email Triage/Rules.md` and the
`Reports/PC/` reports are navigated even though they are not searched.

They also take different value shapes, which the names do not show:

| | Value | Matched |
| --- | --- | --- |
| `SEARCH_EXCLUDE_DIRS` | bare folder names — `Workflows` | against every part of a path, at any depth |
| `INDEX_EXCLUDE_DIRS` | root-relative paths — `Workflows/Approvals` | as a prefix, from the vault root |

So `Approvals` on its own excludes nothing from `index.md`, and `Workflows/Approvals`
excludes nothing from search. Neither is an error; both silently do nothing.

`INDEX_EXCLUDE_DIRS` mirrors the "Excluded folders" table in the vault's
`Meta/Conventions.md`; that table, this variable, `.scripts/check_frontmatter.py` and
the vault's `.gitignore` are four copies of one list and have to move together.

The search index is built at startup and kept current by a filesystem watcher, so an
edit made in Obsidian is searchable a moment later without a restart. The watcher
feeds `index.md` too, and does so independently: search needs Ollama and can be slow
or unavailable, while the navigation document needs neither and must not stop updating
because an embedding endpoint is down.

## Running

The image clones this repository at build time, so the build context holds only the
Dockerfile:

```bash
docker build -t vault-mcp .
docker run --rm \
  -e VAULT_MCP_API_KEY=<token> \
  -v /path/to/vault:/vault \
  -p 8080:8080 \
  vault-mcp
```

Docker caches the clone layer on the URL alone, so a new commit on `main` does not
invalidate it — rebuild with `--no-cache` to pick one up.

The image carries a `HEALTHCHECK` against `/healthz`, and writes its index cache to
`/cache`. Mount a volume there to keep it across `up --force-recreate`; without one it
survives a restart and no more. A *named* volume inherits the image's ownership and
works as it is, where a bind mount arrives owned by root and needs chowning to uid
1000 — an unwritable cache is logged once and then costs only a slower start.

## Security

- **Bearer auth on both surfaces**, failing closed on an unset key.
- **Path containment** in `safe_resolve()` — the single control on where writes land,
  since the vault is mounted read-write. Encoded traversal, `.git`, `index.md` and
  non-`.md` writes are all rejected.
- **Symlinks are refused outright**, checked on the *unresolved* path so `resolve()`
  cannot follow one first. The vault has none and is not going to, and they are
  creatable on this mount — so this fails loudly rather than reasoning about the window
  between resolving a path and replacing a file.
- **Host-header allowlist**, so the MCP transport is not reachable by DNS rebinding.
- **Per-request write scoping** on `/mcp/only/<path>`, above.
- **One unauthenticated route**, `/healthz`, matched by equality rather than by prefix
  so nothing that merely starts with it is exempt. It reads no state and returns a
  constant, which tells a caller who reached the port nothing they did not have.
- **The index cache holds the vault's text outside the vault**, which is the one place
  this server puts it. `0700` on the directory and `0600` on the file;
  `INDEX_CACHE_PATH=` empty if that trade is not wanted.

## Tests

```bash
python -m tests.run                 # everything, against the committed fixture vault
python -m tests.run --real-vault    # the vault readers against VAULT_PATH instead
python -m tests.run rest indexdoc   # just these
```

Each script still runs on its own, which is what you want when one of them fails:

```bash
python -m tests.primitives
python -m tests.resolve_all
python -m tests.resolve_leaves
python -m tests.write_scope
python -m tests.documents
python -m tests.chunker
python -m tests.retrieval
python -m tests.indexdoc
python -m tests.rest
python -m tests.cache
python -m tests.embedder
python -m tests.relevance.eval
```

`tests.run` gives each script its own subprocess rather than importing them together.
That is not tidiness: `src.config` resolves settings at import and `tests.indexdoc`
points `VAULT_PATH` at a temp tree before importing `src`, so two scripts wanting two
different vaults cannot share an interpreter.

`write_scope`, `documents`, `indexdoc`, `rest` and `cache` build their own temp vault. `chunker`,
`retrieval` and `embedder` need no vault at all — they test functions that take text rather than
paths — and point `VAULT_PATH` at an empty temp tree only because `src.config`
refuses to resolve without one. `primitives`,
`resolve_all` and `resolve_leaves` read a vault and assert against what is in it —
which used to mean the real vault, and now means
[`tests/fixtures/vault`](tests/fixtures/vault) unless you pass `--real-vault`. The
fixture reproduces this vault's *shapes* rather than its contents: an H1-wrapped note
and an unwrapped one, a note carrying the same leaf heading under two parents and
another carrying the same full path twice, a flat-list note of twelve unrelated
bullets, a note long enough to chunk six ways, identifier-dense notes, generated
series under `Workflows/` and `Reports/` that must stay out of the index, and three
PDFs under `Files/` — two with a text layer and one deliberately without, which is the
scanned-bill case the extractor has to report rather than quietly return nothing for.

`primitives` asserts POSIX file modes and symlink refusal, so three of its checks fail
on Windows for want of privileges rather than for want of correctness. It passes on
Linux and in CI.

### Retrieval relevance

```bash
python -m tests.relevance.eval                       # fixture, offline, deterministic
python -m tests.relevance.eval --update-baseline     # record a deliberate change
python -m tests.relevance.eval --vault /media/Share/Vault \
    --queries tests/relevance/private.json --embedder ollama
```

Recall@k, MRR and per-note concentration over a committed query set, compared against
a committed baseline: a query that hit at rank 3 may not start missing, and an
improvement is reported rather than failed. This is the harness the measured claims in
[Why](#why) belong in, and the gate for any later change to `SPARSE_WEIGHT`,
`LOOKUP_MAX_MATCHES`, the tokeniser or the chunker.

It runs without Ollama by hashing tokens into `EMBED_DIMS` buckets for the dense arm —
deterministic on every machine, and honest about what that costs: the stub has no
semantics, so queries needing them are tagged `dense` and reported without being
scored. What the fixture run does measure is the lexical arm, the tokeniser, the
fusion, the lookup override, document retrieval and single-source concentration. The
real vault and the real embedder are a local run against a query set that stays out of
git, because the queries name real accounts — see
[`tests/relevance/private.example.json`](tests/relevance/private.example.json).

Four of the fixture's queries can only be answered by a filed PDF — a tariff name, a
supply address, a cooling-off period and an employment notice period, none of which any
note contains. They are there because "documents are indexed" is a claim about
retrieval, and the only way to hold it is a query that fails when indexing them stops.

`tests.chunker` and `tests.retrieval` are the unit half of the retrieval work: the
relevance suite says whether retrieval got better, these say why. They carry the cases
the fixture corpus cannot reach — 27 notes never exhaust their candidates, so the cap's
backfill branch never runs there, and a branch nobody has seen run is not known to work.
Both were checked against six mutations of the code they cover; each one failed them.

`tests.rest` drives the REST surface through the real app — the structured read, the
frontmatter `PATCH` that is the claim in claim-before-act and the `delete` that removes a
field, the body `PATCH` that leaves the block alone, the frontmatter query, that a
missing note is a 404 where a refused one is a 400, and that `/maintenance` and
`/callouts` build their argv from constants whatever the query string says. `tests.indexdoc` covers the generated document: coverage, folder-derived
headings, incremental updates on create, edit, move and delete, that `index.md` is
refused to every writer and still readable, and that an edit changing nothing the index
displays does not rewrite it. `tests.primitives` covers the write traps that are silent
corruption rather than errors — the values that must round-trip through YAML unchanged,
and the ones that are refused because no spelling of them would.

## Licence

AGPL-3.0-or-later. See [LICENSE](LICENSE).

Copyright (C) 2026 Aidan Kay.

The copyleft is deliberate rather than inherited. The source-document work planned for
this project will extract PDFs with [PyMuPDF](https://pymupdf.readthedocs.io/), which
Artifex dual-licenses under AGPL-3.0 or a commercial licence; taking the AGPL half means
this project takes it too. That was
the occasion for licensing the repo at all, rather than leaving it public and
all-rights-reserved, which is what it was before.

The practical consequence for anyone running it: if you let other people interact with
your instance over a network, AGPL section 13 obliges you to offer them the source of
the version you are running. No image is published anywhere, so every deployment is
someone's own build from this repo.
