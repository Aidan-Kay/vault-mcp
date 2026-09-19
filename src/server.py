"""MCP server: tool surface, REST surface, auth, transport security, startup.

Two interfaces over one implementation. Lyra speaks MCP; n8n's HTTP Request
nodes speak plain REST and cannot easily build a JSON-RPC envelope, so /vault/*
mirrors the shape obsidian-local-rest-api used. Both call src.operations, so the
resolver and every convention are handled once.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import time
import urllib.parse
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from . import callouts
from . import documents
from . import maintenance
from . import operations
from . import search as search_module
from . import target as target_mod
from . import vault
from .config import settings
from .embedder import Embedder
from .index import VaultIndex
from .indexdoc import IndexDoc
from .watcher import VaultWatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("vault-mcp")


# --------------------------------------------------------------------------
# Mutable server state
#
# _index is rebound, never mutated. Rebinding is atomic under the GIL, and
# readers capture the reference once per request, so an in-flight search always
# completes against a consistent snapshot.
# --------------------------------------------------------------------------

_index: VaultIndex = VaultIndex.empty()
_embedder: Embedder | None = None
_build_started: float = 0.0
_build_error: str | None = None
_ready = False
_reindex_lock = asyncio.Lock()

_indexdoc: IndexDoc = IndexDoc(entries={})
_indexdoc_lock = asyncio.Lock()
_indexdoc_ready = False


def _index_status() -> str:
    if _build_error:
        return f"The vault index failed to build: {_build_error}"
    elapsed = time.monotonic() - _build_started if _build_started else 0
    return (
        f"The vault index is still building ({elapsed:.0f}s elapsed). "
        "Retry shortly, or use vault_list / vault_read, which do not need it."
    )


async def _build_index() -> None:
    global _index, _build_error, _ready
    assert _embedder is not None
    try:
        _index = await VaultIndex.build(_embedder)
        _ready = True
    except Exception as exc:
        _build_error = str(exc)
        log.exception("index build failed")


async def _reindex(path: Path) -> None:
    global _index
    assert _embedder is not None
    async with _reindex_lock:  # serialise rebuilds; each reads the live index
        started = time.perf_counter()
        _index = await _index.replace_note(_embedder, path)
        log.info(
            "reindex %s -> %d chunks in %.0f ms",
            vault.relpath(path),
            _index.size,
            (time.perf_counter() - started) * 1000,
        )


async def _scan_index_doc() -> None:
    """Read every note once and reconcile index.md against them.

    Runs at startup, so whatever was edited in Obsidian while the container was
    down is picked up without anyone asking. Costs a few seconds across the
    Samba mount, which is why it happens once and every later change swaps a
    single entry instead.
    """
    global _indexdoc, _indexdoc_ready
    async with _indexdoc_lock:
        try:
            # to_thread: blocking reads over the mount, and the loop is already
            # serving vault_read and vault_list while this runs.
            _indexdoc = await asyncio.to_thread(IndexDoc.build)
        except Exception:
            # _indexdoc_ready stays False, so no later change writes index.md
            # from a half-built document. Stale beats truncated: the file on
            # disk is still the last good one.
            log.exception("index.md scan failed; it will not be updated until a restart")
            return
        _indexdoc_ready = True
        wrote = await asyncio.to_thread(_indexdoc.write_if_changed)
        log.info("index.md: %s", wrote or f"already current ({len(_indexdoc.entries)} entries)")


async def _reconcile_index_doc() -> None:
    """Re-read every note on a timer and correct index.md if it has drifted.

    The incremental path is only as complete as the event stream feeding it.
    A directory moved in whole carries no watch and no events, which is what
    lost AI/Prompts/Server; watcher.py now handles that one, but "the event
    never came" has more shapes than the one that has bitten us, and without
    this any of them costs a restart to notice.

    Cheap to be wrong about: the scan is a few seconds of reads, and
    write_if_changed compares bodies, so a pass that finds nothing touches
    neither index.md nor git.
    """
    interval = settings.index_reconcile_seconds
    if interval <= 0:
        return

    global _indexdoc
    while True:
        await asyncio.sleep(interval)
        try:
            async with _indexdoc_lock:
                if not _indexdoc_ready:
                    continue  # the startup scan is about to do this anyway
                rebuilt = await asyncio.to_thread(IndexDoc.build)
                if rebuilt.entries == _indexdoc.entries:
                    continue
                missing = set(rebuilt.entries) - set(_indexdoc.entries)
                extra = set(_indexdoc.entries) - set(rebuilt.entries)
                log.warning(
                    "reconcile: index.md had drifted (%d missing, %d stale); "
                    "events were lost for %s",
                    len(missing),
                    len(extra),
                    ", ".join(sorted(missing | extra)[:5]) or "changed entries",
                )
                _indexdoc = rebuilt
                wrote = await asyncio.to_thread(_indexdoc.write_if_changed)
                log.info("reconcile: %s", wrote or "entries changed but index.md body did not")
        except asyncio.CancelledError:
            raise
        except Exception:
            # A failed pass is not fatal; the next one tries again.
            log.exception("reconcile pass failed")


async def _refresh_index_doc(path: Path) -> None:
    """Bring index.md back in step after one note changed.

    Held apart from _reindex and its lock on purpose. Search needs Ollama and
    can be slow or unavailable; this needs neither, and the navigation document
    should not stop updating because an embedding endpoint is down.
    """
    global _indexdoc
    async with _indexdoc_lock:
        if not _indexdoc_ready:
            # The startup scan has not run yet. It reads the live filesystem, so
            # it will see this write itself - doing an incremental update against
            # an empty document here would render an index.md with one entry in it.
            return

        previous = _indexdoc
        _indexdoc = await asyncio.to_thread(previous.replace_note, path)
        if _indexdoc is previous:
            return  # not an indexed note, so index.md cannot have changed
        wrote = await asyncio.to_thread(_indexdoc.write_if_changed)
        if wrote:
            log.info("%s", wrote)


async def _on_change(path: Path) -> None:
    """One settled filesystem event, fanned out to both consumers.

    index.md goes first and is awaited separately: it is the cheap one, and a
    failed or slow re-embed must not leave the navigation document stale.
    """
    try:
        await _refresh_index_doc(path)
    except Exception:
        log.exception("index.md refresh failed for %s", vault.relpath(path))

    if _ready and not vault.is_search_excluded(path.resolve().relative_to(vault.ROOT)):
        await _reindex(path)


@asynccontextmanager
async def lifespan(_server: MCPServer) -> AsyncIterator[None]:
    global _embedder, _build_started

    log.info("vault=%s exclude=%s", vault.ROOT, sorted(settings.search_exclude_dirs))
    _embedder = Embedder()
    _build_started = time.monotonic()

    # Built in the background so vault_read / vault_list / vault_map serve
    # immediately and do not depend on Ollama being up.
    build_task = asyncio.create_task(_build_index(), name="index-build")
    scan_task = asyncio.create_task(_scan_index_doc(), name="indexdoc-scan")
    reconcile_task = asyncio.create_task(_reconcile_index_doc(), name="indexdoc-reconcile")

    # Started straight away, no longer behind the embedding build. It used to
    # wait for it and give up if it failed, which was tolerable when search was
    # all it fed; now it also keeps index.md current, and that must not stop
    # because Ollama is down. _on_change skips the re-embed until _ready.
    watcher = VaultWatcher(_on_change)
    await watcher.start()

    try:
        yield
    finally:
        for task in (reconcile_task, scan_task, build_task):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await watcher.stop()
        if _embedder is not None:
            await _embedder.aclose()


mcp = MCPServer(
    "vault-mcp",
    instructions=(
        "Semantic and keyword search, reading and writing over the Obsidian "
        "vault. Prefer vault_search to locate information, then vault_read with "
        "section= to pull only the heading you need. To edit, call vault_map "
        "first and patch the '::' path it gives you - a bare heading name works "
        "whenever it is unique, and the error tells you what to prepend when it "
        "is not. Writes bump the note's timestamp for you, and the root index.md "
        "is generated from every note's title and description - never edit it, "
        "and never add an entry to it. Fix a wrong line there by fixing the "
        "note's frontmatter. Never probe for a note's existence before writing - "
        "the write tools take the missing case as an argument "
        "(vault_append's create_if_missing, vault_write's overwrite), so a read "
        "or list first only buys a round trip."
    ),
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

def _do(fn, *args, **kwargs) -> str:
    """Run a vault operation, surfacing its error message to the model.

    The MCP runtime masks an arbitrary exception as a bare "Error executing tool
    <name>" and only lets a ToolError's message through. Every VaultError here is
    written to be acted on - the ambiguity error lists the exact paths to retry
    with - so masking it would throw away the entire point of the resolver.
    """
    try:
        return fn(*args, **kwargs)
    except vault.VaultError as exc:
        raise ToolError(str(exc)) from exc



@mcp.tool()
async def vault_search(query: str, k: int | None = None) -> str:
    """Search the vault for a topic, question, or exact term.

    Hybrid semantic + keyword search. Returns ranked excerpts with their source
    path and heading, suitable for citing directly.

    Args:
        query: Natural language question or exact term (a model number, reg
            plate or policy reference all work).
        k: Number of excerpts to return. Defaults to 5.
    """
    index = _index  # snapshot
    if not _ready:
        return _index_status()
    assert _embedder is not None
    limit = max(1, min(k or settings.search_default_k, 20))
    results = await search_module.search(index, _embedder, query, limit)
    return search_module.format_results(query, results)


@mcp.tool()
def vault_read(path: str, section: str | None = None) -> str:
    """Read a note, or the text of a filed document.

    A document - a PDF under a `Files/` folder - comes back as the markdown
    extracted from it, not as bytes. It has no sections to ask for, because its
    headings are inferred during extraction rather than written by anyone.

    Args:
        path: Vault-relative path, e.g. "Pets/Levi.md" or
            "Home/Utilities/Files/2026-09-17 Kestrel Energy - Contract.pdf".
        section: Optional heading name, for notes only. Returns just that
            heading's content, down to the next heading of equal or shallower
            depth. Use this instead of reading whole notes.
    """
    return _do(vault.read_note, path, section)


@mcp.tool()
def vault_list(path: str = "") -> str:
    """List the contents of a vault directory.

    Args:
        path: Vault-relative directory. Defaults to the vault root.
    """
    entries = _do(vault.list_dir, path)
    if not entries:
        return f"{path or '/'} is empty."
    lines = [f"{len(entries)} entr(ies) in {path or '/'}:"]
    for entry in entries:
        if entry["type"] == "dir":
            lines.append(f"  {entry['name']}/")
        else:
            lines.append(f"  {entry['name']}  ({entry['size']} B, {entry['modified']})")
    return "\n".join(lines)


@mcp.tool()
def vault_map(path: str) -> str:
    """Show a note's frontmatter and heading structure without its content.

    Use this to decide which section to request from vault_read.

    Args:
        path: Vault-relative path, e.g. "Pets/Levi.md".
    """
    parsed = _do(vault.parse_note, path)
    if parsed["document"]:
        extraction = parsed["extraction"]
        searchable = "yes" if extraction["extracted_chars"] else "NO"
        return "\n".join(
            [
                f"# {parsed['path']}",
                "",
                "A filed document, not a note. It can be read, moved and deleted;",
                "it cannot be patched, appended to, or given frontmatter, because",
                "there is no markdown source inside it to address.",
                "",
                "## Extraction",
                "",
                f"- pages: {extraction['pages']}",
                f"- text layer: {'yes' if extraction['has_text_layer'] else 'no'}",
                f"- extracted characters: {extraction['extracted_chars']}",
                f"- searchable: {searchable}",
                f"- status: {extraction['status']}"
                + (f" ({extraction['detail']})" if extraction["detail"] else ""),
            ]
        )
    lines = [f"# {parsed['path']}", "", "## Frontmatter"]
    if parsed["frontmatter"]:
        for key, value in parsed["frontmatter"].items():
            lines.append(f"- {key}: {json.dumps(value, default=str, ensure_ascii=False)}")
    else:
        lines.append("- (none)")
    lines += ["", "## Headings", "", "Patch targets. A trailing segment on its own works when it is"]
    lines += ["unique in this note; prepend ancestors with '::' when it is not.", ""]
    text = _do(lambda p: vault.read_text(vault.safe_resolve(p)), path)
    outline = target_mod.outline(text)
    lines += [f"- {entry}" for entry in outline] or ["- (none)"]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Write tools
#
# Thin wrappers: every one delegates to src.operations, which the REST routes
# below call too. Nothing here holds logic of its own.
# --------------------------------------------------------------------------


@mcp.tool()
def vault_patch(
    path: str,
    target: str,
    operation: str = "replace",
    content: str = "",
    target_scope: str = "content",
) -> str:
    """Edit one section of a note, addressed by its heading.

    Args:
        path: Vault-relative path, e.g. "Pets/Levi.md".
        target: Heading to act on. A bare name works whenever it is unique in
            the note; otherwise join ancestors with "::", as
            "Cottage Pie::Mash::Method". Call vault_map to see the paths. If
            the target is ambiguous the error lists exactly which paths to
            choose between - re-call with one of them.
        operation: "replace", "prepend" or "append".
        content: The markdown to write.
        target_scope: "content" (default, the section body), "marker" (the
            heading line only) or "markerAndContent" (both).
    """
    return _do(operations.patch, path, target, operation, content, target_scope)


@mcp.tool()
def vault_append(path: str, content: str, create_if_missing: bool = False) -> str:
    """Append a block to the end of a note, creating it if the path is absent.

    Do not read or list first to find out whether the note is there - pass
    create_if_missing=True and this one call covers both cases. A probe
    beforehand costs a whole round trip to answer a question this tool already
    takes as an argument.

    Args:
        path: Vault-relative path.
        content: The markdown to append.
        create_if_missing: Create the note instead of failing when the path is
            absent. Nothing is invented for you, so put frontmatter in `content`
            when the note should carry any - its timestamp is then bumped for
            you, exactly as on every other write.
    """
    return _do(operations.append, path, content, create_if_missing)


@mcp.tool()
def vault_write(path: str, content: str, overwrite: bool = False) -> str:
    """Create a note, or replace one wholesale.

    Include frontmatter: type, title, description, tags, timestamp. Prefer
    vault_patch for editing part of an existing note, and vault_append when you
    only want to add to the end - neither needs the note looked up first.

    Args:
        path: Vault-relative path. Parent directories are created as needed.
        content: The complete note.
        overwrite: Required to replace an existing note. Without it an existing
            path is an error, so a create can never silently clobber. Set it
            from your intent rather than from a lookup: pass it when you mean
            "create or replace", leave it off when the note must be new.
    """
    return _do(operations.write, path, content, overwrite)


@mcp.tool()
def vault_set_body(path: str, content: str) -> str:
    """Replace a note's prose, leaving its frontmatter exactly as it is.

    For a note whose text is rewritten wholesale but whose metadata is written
    once - a summary regenerated on a schedule. vault_write is the wrong tool
    for that: it replaces the file, so the block goes with it and the note stops
    being able to describe itself.

    Args:
        path: Vault-relative path.
        content: The complete new body, without frontmatter. The existing block
            is carried across as the bytes it already had, so a block this
            server could not parse survives too. A note that has no frontmatter
            has no prefix to keep, and this is then the same as vault_write.
    """
    return _do(operations.set_body, path, content)


@mcp.tool()
def vault_set_frontmatter(path: str, key: str, value: str | list | None = None, delete: bool = False) -> str:
    """Set or remove one frontmatter field, leaving the rest of the block alone.

    Args:
        path: Vault-relative path.
        key: Field name, e.g. "description" or "tags". A field that does not
            exist yet is inserted in the order Conventions mandates.
        value: The value. A list for "tags"; for "expires", a list of
            {"date": "YYYY-MM-DD", "what": "..."} entries - pass every entry,
            as the whole block is replaced and a dropped date stops being
            checked silently.
        delete: Remove the field instead of setting it.
    """
    return _do(operations.set_frontmatter, path, key, value, delete)


@mcp.tool()
def vault_delete(path: str) -> str:
    """Delete a note or a filed document. There is no trash; git is the undo.

    Deleting a document leaves the `## Documents` row in the note that owned it
    pointing at nothing, so remove that row too.

    Args:
        path: Vault-relative path.
    """
    return _do(operations.delete, path)


@mcp.tool()
def vault_move(source: str, destination: str, update_links: bool = True) -> str:
    """Move or rename a note or a filed document, repointing every link to it.

    Moving notes changes the vault's structure, so confirm with the user first.
    index.md follows the move on its own - its headings are the folder tree.

    A document is moved byte for byte and keeps its timestamp; the notes linking
    to it are what get rewritten. Refiling one - a document filed under the wrong
    note, or a `Files/` folder that has grown a subject level - is what this is
    for. A move may not change a file between a note and a document.

    Args:
        source: Current vault-relative path.
        destination: New vault-relative path. Parent directories are created.
            Keep a document inside a folder named `Files`: unlike an upload this
            is not enforced here, and the vault's checker reports one that is
            not as misfiled.
        update_links: Rewrite internal links pointing at the old path.
    """
    return _do(operations.move, source, destination, update_links)


# --------------------------------------------------------------------------
# ASGI app: transport security, then auth
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# REST surface
#
# n8n's HTTP Request nodes send a raw markdown body to a path-shaped URL. Route
# shapes mirror obsidian-local-rest-api so migrating a node is a find-and-replace
# on the URL and the auth header, not a rewrite into JSON-RPC.
#
# The point of the mirroring is to close the second door. n8n reached the vault
# through the plugin, which enforced no containment, no protection for the
# generated index.md and no write scoping - so those were guarantees about one
# API rather than about the vault. Everything the plugin offered that n8n
# actually used is answerable here, which is what lets it be switched off.
# --------------------------------------------------------------------------


async def _body(request: Request) -> str:
    return (await request.body()).decode("utf-8")


def _flag(request: Request, name: str) -> bool:
    """A boolean query parameter, present-but-empty counting as true."""
    if name not in request.query_params:
        return False
    raw = request.query_params[name].strip().lower()
    return raw in {"", "1", "true", "yes", "on"}


def _failed(exc: vault.VaultError) -> PlainTextResponse:
    """One VaultError, as the status code a caller can branch on.

    404 for a note that is not there, 400 for everything else. The distinction
    exists because "no such note" is the one error a caller routinely *routes
    on* rather than logs - a missing proposal is an ordinary outcome of
    reconciliation, a malformed target is a bug - and a caller that only ever
    sees 400 has to read the message to tell them apart.

    Never 500: every one of these is the caller's path, target or body, and the
    message is written to be acted on rather than to diagnose the server.
    """
    return PlainTextResponse(
        str(exc), status_code=404 if isinstance(exc, vault.NotFound) else 400
    )


class VaultJSON(JSONResponse):
    """JSONResponse that cannot be stopped by whatever YAML resolved a field to.

    vault.metadata() already renders dates back to the text the note carries, so
    this should never fire on a note in this vault. It is here because the
    alternative when it does is a 500 on an otherwise valid read - a note with
    one unusual frontmatter value should not become an unreadable note.
    """

    def render(self, content) -> bytes:
        return json.dumps(
            content, default=str, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")


# obsidian-local-rest-api signalled the structured read with its own media type.
# It is accepted here so a migrating node needs nothing but its URL changed, but
# the response is always application/json - see _structured_read. This constant
# can go once nothing sends it.
OLRAPI_NOTE_JSON = "application/vnd.olrapi.note+json"


def _structured_read(request: Request, path: str) -> JSONResponse | PlainTextResponse:
    """GET a note as JSON when asked for it, as markdown otherwise.

    Answered as `application/json`, never as the vendor type, even when the
    vendor type is what was requested. That is the one place this differs from
    the plugin and it is deliberate: n8n does not recognise the vendor type as
    JSON, so it hands the body to the workflow as a *string* in `$json.data`,
    and a node written against that shape reads `undefined` from real JSON.
    Answering real JSON makes such a node throw on its next run rather than
    silently succeed with nothing, which is the failure anyone would rather
    have.
    """
    section = request.query_params.get("section")
    accept = request.headers.get("accept", "")
    if OLRAPI_NOTE_JSON not in accept and "application/json" not in accept:
        return PlainTextResponse(vault.read_note(path, section))
    return VaultJSON(vault.note_json(path, section))


def _patch_frontmatter(path: str, key: str, operation: str, body: str) -> PlainTextResponse:
    """PATCH with `Target-Type: frontmatter` - set or remove one key.

    The body is JSON-*decoded*, not taken as written: `rev` arrives as the
    number 2 and a status as the quoted string "approved". Writing those quotes
    into the YAML would change what every status comparison downstream sees,
    which is the kind of break that surfaces three workflows away from its
    cause.

    Removal is `Operation: delete`, with no body, rather than a `null` value.
    A null is refused - there is no text that reads back as one - and the
    refusal tells the caller to delete the field instead, which until now was
    something only an MCP caller could do.
    """
    if operation == "delete":
        # No body to decode: a delete names the key and nothing else.
        return PlainTextResponse(operations.set_frontmatter(path, key, delete=True))

    if operation != "replace":
        # set_frontmatter replaces the key outright. Accepting "append" here
        # would quietly discard the rest of a list rather than add to it, and a
        # refusal is the only honest answer until there is a caller to build for.
        raise vault.VaultError(
            f"Operation {operation!r} is not supported on frontmatter; only "
            "'replace' and 'delete' are. Read the key and replace it with the "
            "value you want."
        )
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise vault.VaultError(
            "a frontmatter PATCH body must be JSON, so a string needs its "
            f"quotes: got {body[:80]!r}, expected something like \"approved\" "
            f"or 2 ({exc})"
        ) from exc
    return PlainTextResponse(operations.set_frontmatter(path, key, value))


async def vault_endpoint(request: Request) -> JSONResponse | PlainTextResponse:
    path = request.path_params["path"]
    method = request.method

    try:
        if method == "GET":
            return _structured_read(request, path)
        if method == "PUT":
            # Dispatched on the suffix, not on Content-Type. n8n sends whatever
            # the upstream mail server labelled the attachment - and the one
            # thing the caller is always sure of is the name it chose to file it
            # under. A misdeclared content type would otherwise decide how the
            # bytes are stored, which is a decision no header should make.
            if documents.is_document(path):
                overwrite = _flag(request, "overwrite")
                return VaultJSON(
                    operations.upload(path, await request.body(), overwrite=overwrite)
                )
            return PlainTextResponse(
                operations.write(path, await _body(request), overwrite=True)
            )
        if method == "POST":
            return PlainTextResponse(
                operations.append(path, await _body(request), create_if_missing=True)
            )
        if method == "PATCH":
            target_type = request.headers.get("target-type", "heading").strip().lower()
            operation = request.headers.get("operation", "replace")

            # Checked before Target, because the body is the one target that
            # does not need naming - there is exactly one of it.
            if target_type == "body":
                if operation != "replace":
                    raise vault.VaultError(
                        f"Operation {operation!r} is not supported on the body; "
                        "only 'replace' is. POST to the note to add to the end "
                        "of it."
                    )
                return PlainTextResponse(
                    operations.set_body(path, await _body(request))
                )

            target = request.headers.get("target")
            if not target:
                raise vault.VaultError(
                    "PATCH needs a Target header naming the heading, or naming "
                    "the frontmatter key when Target-Type is 'frontmatter'. "
                    "Target-Type 'body' needs no Target."
                )

            if target_type == "frontmatter":
                return _patch_frontmatter(path, target, operation, await _body(request))
            if target_type != "heading":
                raise vault.VaultError(
                    f"unsupported Target-Type {target_type!r}; use 'heading' "
                    "(the default), 'frontmatter' or 'body'"
                )
            return PlainTextResponse(
                operations.patch(
                    path,
                    target,
                    operation,
                    await _body(request),
                    request.headers.get("target-scope", "content"),
                )
            )
        if method == "DELETE":
            return PlainTextResponse(operations.delete(path))
    except vault.VaultError as exc:
        return _failed(exc)

    return PlainTextResponse(f"{method} not supported on /vault", status_code=405)


async def frontmatter_endpoint(request: Request) -> JSONResponse | PlainTextResponse:
    """Find notes by an exact frontmatter value: /frontmatter?key=&value=&dir=

    A narrow replacement for the plugin's jsonlogic search, which every caller
    used to ask the same single question - one field, one exact value. Answers a
    list of {"filename": "<vault-relative path>"}, which is all any consumer
    read out of it.

    Never touches the semantic index: Workflows/ is in SEARCH_EXCLUDE_DIRS and so is
    absent from search entirely, and that is exactly where the notes this finds
    live.
    """
    key = request.query_params.get("key", "").strip()
    value = request.query_params.get("value")
    try:
        if not key or value is None:
            raise vault.VaultError(
                "/frontmatter needs key= and value=, as "
                "/frontmatter?key=status&value=pending. Add dir= to walk one "
                "folder instead of the whole vault."
            )
        matches = vault.find_by_frontmatter(key, value, request.query_params.get("dir"))
    except vault.VaultError as exc:
        return _failed(exc)
    return VaultJSON([{"filename": name} for name in matches])


async def maintenance_endpoint(request: Request) -> JSONResponse:
    """Run the vault's own checkers: /maintenance

    Read-only, and the only route here that does not touch a note. It exists
    for the weekly maintenance workflow: n8n calls it, hands the JSON to Lyra,
    and Lyra writes the report. See src/maintenance.py for why the work happens
    in this container rather than over SSH on the host.

    Answers 200 whenever the suite ran, however many findings it produced - a
    vault with broken links is a successful check, not a failed request. Only
    the suite failing to run at all is a 500, which is what lets the caller
    branch on "the report is missing" without parsing it.
    """
    log.info("running the vault maintenance checks")
    try:
        result = await asyncio.to_thread(maintenance.run_all)
    except Exception as exc:  # noqa: BLE001 - the route must not 502 silently
        log.exception("maintenance run failed")
        return VaultJSON({"error": f"the maintenance run failed: {exc}"}, status_code=500)
    log.info(
        "maintenance checks done in %d ms: %d/%d exited zero",
        result["duration_ms"],
        result["summary"]["exit_zero"],
        result["summary"]["total"],
    )
    return VaultJSON(result)


async def callouts_endpoint(request: Request) -> JSONResponse:
    """Extract every open callout in the vault: /callouts

    Read-only, and a sibling of /maintenance rather than one of its checks:
    that route asks whether the vault is well-formed, this one asks what it is
    still carrying. n8n calls it, hands the list to Lyra, and Lyra raises what
    is outstanding. See src/callouts.py for why it runs here and why it takes
    no parameters.

    Answers 200 with however many callouts were found, including none - an
    empty list is a real answer about a vault with nothing open. Only the
    extractor failing to run is a 500, which is what lets the caller branch on
    "the list is missing" without inspecting it.
    """
    log.info("extracting the vault callouts")
    try:
        result = await asyncio.to_thread(callouts.run)
    except callouts.ExtractionError as exc:
        log.error("callout extraction failed: %s", exc)
        return VaultJSON({"error": f"the callout extraction failed: {exc}"}, status_code=500)
    except Exception as exc:  # noqa: BLE001 - the route must not 502 silently
        log.exception("callout extraction failed")
        return VaultJSON({"error": f"the callout extraction failed: {exc}"}, status_code=500)
    log.info(
        "extracted %d callouts from %d notes in %d ms",
        result["counts"]["callouts"],
        result["counts"]["notes_with_callouts"],
        result["duration_ms"],
    )
    return VaultJSON(result)


rest_app = Starlette(
    routes=[
        Route(
            "/vault/{path:path}",
            vault_endpoint,
            methods=["GET", "PUT", "POST", "PATCH", "DELETE"],
        ),
        Route("/frontmatter", frontmatter_endpoint, methods=["GET"]),
        # GET because it changes nothing; POST too, so a caller that only
        # sends POSTs does not need a special case.
        Route("/maintenance", maintenance_endpoint, methods=["GET", "POST"]),
        Route("/callouts", callouts_endpoint, methods=["GET", "POST"]),
    ]
)


REST_PREFIXES = ("/vault", "/frontmatter", "/maintenance", "/callouts")


class VaultRoutes:
    """Serve the REST prefixes from the REST app, everything else from MCP.

    A wrapper rather than a parent Starlette app so the MCP app keeps owning the
    lifespan that starts the index, the watcher and the session manager. Only
    http scopes are diverted; lifespan and everything else pass straight through.
    """

    def __init__(self, mcp_app, rest) -> None:
        self.mcp_app = mcp_app
        self.rest = rest

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http" and scope["path"].startswith(REST_PREFIXES):
            return await self.rest(scope, receive, send)
        return await self.mcp_app(scope, receive, send)


WRITE_SCOPE_PREFIX = "/mcp/only/"


class ScopedWrites:
    """Confine one request's writes, from the path it arrived on.

        /mcp                                   writes anywhere (unchanged)
        /mcp/only/Workflows/Approvals/x.md     writes only to that note
        /mcp/only/Workflows/Approvals          writes only under that folder

    **Why the URL and not a header.** The scope has to be chosen per call by the
    caller, and n8n's MCP Client node takes its auth header from a static
    credential while its endpoint URL is an ordinary expression field. Putting
    it in the path is what lets one agent, with one tool list, be handed a
    different remit per invocation - no second copy of the workflow, no second
    set of tools to keep in step.

    Safe because the transport is stateless_http: every MCP call is its own HTTP
    request, so a scope can never outlive the call it came with. The ContextVar
    behind vault.write_scope() is what keeps concurrent requests from seeing
    each other's.

    Reads are untouched. An agent confined to one note still has to read the
    conventions and whatever the note refers to; it is writing outside its remit
    that must be impossible.
    """

    def __init__(self, inner) -> None:
        self.inner = inner

    async def __call__(self, scope, receive, send) -> None:
        path = scope.get("path", "") if scope["type"] == "http" else ""
        if not path.startswith(WRITE_SCOPE_PREFIX):
            return await self.inner(scope, receive, send)

        confined = urllib.parse.unquote(path[len(WRITE_SCOPE_PREFIX):]).strip("/")
        if not confined:
            return await self.inner(scope, receive, send)

        # Rewritten to the path the MCP app is actually mounted on, so the
        # transport never learns this happened. raw_path goes too, or Starlette
        # re-derives the original from it and routes to nothing.
        scope = dict(scope)
        scope["path"] = "/mcp"
        scope.pop("raw_path", None)

        log.info("write scope for this request: %s", confined)
        with vault.write_scope(confined):
            await self.inner(scope, receive, send)


app = mcp.streamable_http_app(
    streamable_http_path="/mcp",
    stateless_http=True,
    transport_security=TransportSecuritySettings(
        allowed_hosts=list(settings.allowed_hosts),
        allowed_origins=["*"],  # no browser origin - MCP clients only
    ),
)

app = VaultRoutes(app, rest_app)
app = ScopedWrites(app)


class BearerAuth:
    """Static shared secret on a private network.

    Non-HTTP scopes pass through untouched so the lifespan still runs and starts
    the session manager.
    """

    def __init__(self, inner, key: str) -> None:
        self.inner = inner
        self.expected = b"Bearer " + key.encode()

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            return await self.inner(scope, receive, send)
        supplied = dict(scope["headers"]).get(b"authorization", b"")
        if not hmac.compare_digest(supplied, self.expected):
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"www-authenticate", b'Bearer realm="vault-mcp"'),
                        (b"content-type", b"text/plain; charset=utf-8"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": b"Unauthorized"})
            return
        await self.inner(scope, receive, send)


app = BearerAuth(app, settings.api_key)


def main() -> None:
    log.info(
        "serving MCP on %s:%d/mcp and REST on %s:%d%s (allowed hosts: %s)",
        settings.host,
        settings.port,
        settings.host,
        settings.port,
        "{/vault/<path>,/frontmatter,/maintenance,/callouts}",
        ", ".join(settings.allowed_hosts),
    )
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info", access_log=False)


if __name__ == "__main__":
    main()
