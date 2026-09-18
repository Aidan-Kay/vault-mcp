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
| `vault_read` — whole note or one `section=` | `vault_append` — add to the end |
| `vault_list` — browse a folder | `vault_write` — create or overwrite |
| `vault_map` — heading tree as `::` paths | `vault_set_body` — replace the prose, keep the block |
| | `vault_set_frontmatter` — set or delete a key |
| | `vault_delete` — remove a note |
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
| `PUT /vault/<path>` | Create or replace, body is the note |
| `POST /vault/<path>` | Append, creating the note if it is absent |
| `PATCH /vault/<path>` | `Target:` a heading, or a frontmatter key with `Target-Type: frontmatter`, or the prose with `Target-Type: body`. `Operation: delete` removes a frontmatter key. |
| `DELETE /vault/<path>` | Remove the note |
| `GET /frontmatter?key=&value=` | Notes whose field holds that exact value, as `[{"filename": …}]`. `&dir=` narrows the walk to one folder. |
| `GET /maintenance` | Runs the vault's eight checkers and answers what each printed, as JSON and as one markdown block. |
| `GET /callouts` | Every open callout in the vault, as JSON: one object per callout with its note, line, type, severity, title and body. |

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
| `SEARCH_EXCLUDE_DIRS` | `Workflows,Reports,.obsidian` | Folder *names*, left out of the search index |
| `INDEX_EXCLUDE_DIRS` | the six generated series | Folder *paths*, left out of `index.md` |
| `CHUNK_TARGET_TOKENS` | `400` | Target chunk size |
| `CHUNK_OVERLAP_TOKENS` | `60` | Overlap between chunks |
| `CHUNK_MIN_TOKENS` | `120` | Below this, a chunk merges into its neighbour |
| `SEARCH_DEFAULT_K` | `5` | Default result count |
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

## Tests

Standalone scripts, no test runner:

```bash
python -m tests.primitives
python -m tests.resolve_all
python -m tests.resolve_leaves
python -m tests.write_scope
python -m tests.indexdoc
python -m tests.rest
```

The last three write, so they build their own temp vault rather than touching the real
one; the first three read the real vault, so they need it mounted.

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

The copyleft is deliberate rather than inherited. The source-document work in
[docs/plan-of-action.md](docs/plan-of-action.md) will extract PDFs with
[PyMuPDF](https://pymupdf.readthedocs.io/), which Artifex dual-licenses under AGPL-3.0
or a commercial licence; taking the AGPL half means this project takes it too. That was
the occasion for licensing the repo at all, rather than leaving it public and
all-rights-reserved, which is what it was before.

The practical consequence for anyone running it: if you let other people interact with
your instance over a network, AGPL section 13 obliges you to offer them the source of
the version you are running. No image is published anywhere, so every deployment is
someone's own build from this repo.
