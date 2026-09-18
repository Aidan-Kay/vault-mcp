"""The REST surface n8n calls, and the three things it could not do before.

These exist because of a migration: n8n reached the vault through
obsidian-local-rest-api, which enforced none of the containment, protection or
write-scoping this server does, and three call shapes stood in the way of
pointing it here instead. Each is covered below, along with the one behaviour
that deliberately differs from the plugin.

    the structured read   GET with an Accept header, answered as JSON
    frontmatter PATCH     Target-Type: frontmatter, the claim in claim-before-act
    /frontmatter          find notes by an exact field value, without the index

Removing a frontmatter field came later, and for a different reason: the refusal
a null value earns names `delete`, and REST had no spelling of it.

The claim is the correctness-critical one. `status: "approved"` written with its
quotes intact compares equal to nothing downstream, and a status that never
matches is a proposal that can never be resolved - so the assertions here are on
the bytes in the YAML, not on the call succeeding.

Requests go through the real `app`, so they cross BearerAuth and the prefix
routing on the way in. This runner writes, so it builds its own temp vault.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Before `from src import ...`, and an assignment rather than setdefault:
# tests/__init__ has already pointed this at the real vault, and this runner
# must not write there.
_VAULT = Path(tempfile.mkdtemp(prefix="vault-rest-"))
os.environ["VAULT_PATH"] = str(_VAULT)

import httpx  # noqa: E402

from src import callouts  # noqa: E402
from src import maintenance  # noqa: E402
from src.server import app  # noqa: E402

FAILURES: list[str] = []


def check(name: str, actual, expected) -> None:
    if actual == expected:
        return
    FAILURES.append(f"{name}\n    expected: {expected!r}\n    actual:   {actual!r}")


def check_in(name: str, needle: str, haystack: str) -> None:
    if needle in haystack:
        return
    FAILURES.append(f"{name}\n    expected to contain: {needle!r}\n    actual: {haystack!r}")


def report() -> int:
    for failure in FAILURES:
        print(f"FAIL {failure}")
    return len(FAILURES)


# --------------------------------------------------------------------------
# Fixture vault
#
# Workflows/Approvals is deliberate: it is in SEARCH_EXCLUDE_DIRS, so it is invisible
# to search, and it is where every note the frontmatter query exists to find
# actually lives. A query that quietly used the index would return nothing here
# and pass every other assertion in this file.
# --------------------------------------------------------------------------

PROPOSAL = """---
type: proposal
title: {title}
description: A proposal awaiting a decision.
tags: [approval, lyra]
status: {status}
thread_id: "{thread}"
rev: 1
timestamp: 2026-09-12T09:00:00Z
---

# {title}

## Summary

The body of the proposal.
"""

NOTE = """---
type: note
title: Alpha
description: An ordinary note.
tags:
  - lyra
  - ops
timestamp: 2026-09-12T09:00:00Z
---

# Alpha

## Summary

First section.

## Detail

Second section.
"""

BROKEN = """---
title: Broken
tags: [unclosed
  : : :
---

# Broken

A note whose YAML does not parse.
"""


def write_fixture() -> None:
    """A pristine vault, rebuilt before each test.

    Rebuilt rather than shared because the claim test changes the very statuses
    the query test counts. Coupling those would make one of them pass or fail on
    the order they happen to be called in, which is not a property either is
    trying to assert.
    """
    for child in _VAULT.iterdir():
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    (_VAULT / "Workflows" / "Approvals").mkdir(parents=True)
    (_VAULT / "Notes").mkdir(parents=True)
    for name, status, thread in (
        ("a1", "pending", "1547"),
        ("a2", "approved", "1548"),
        ("a3", "pending", "1549"),
    ):
        (_VAULT / "Workflows" / "Approvals" / f"{name}.md").write_text(
            PROPOSAL.format(title=name, status=status, thread=thread), encoding="utf-8"
        )
    (_VAULT / "Notes" / "Alpha.md").write_text(NOTE, encoding="utf-8")
    (_VAULT / "Notes" / "Broken.md").write_text(BROKEN, encoding="utf-8")


def read_note(rel: str) -> str:
    return (_VAULT / rel).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Driving the app
# --------------------------------------------------------------------------

JSON_ACCEPT = "application/json"
OLRAPI_ACCEPT = "application/vnd.olrapi.note+json"


async def _call(method: str, url: str, *, headers=None, content=None) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    sent = {"Authorization": "Bearer test"}
    sent.update(headers or {})
    async with httpx.AsyncClient(transport=transport, base_url="http://vault-mcp:8080") as c:
        return await c.request(method, url, headers=sent, content=content)


def call(method: str, url: str, *, headers=None, content=None) -> httpx.Response:
    return asyncio.run(_call(method, url, headers=headers, content=content))


# --------------------------------------------------------------------------
# 2.1  The structured read
# --------------------------------------------------------------------------


def test_structured_read() -> None:
    write_fixture()
    plain = call("GET", "/vault/Notes/Alpha.md")
    check("no Accept header still returns markdown", plain.status_code, 200)
    check("markdown body is the note verbatim", plain.text, NOTE)
    check_in("markdown content-type", "text/plain", plain.headers["content-type"])

    for label, accept in (("application/json", JSON_ACCEPT), ("the vendor type", OLRAPI_ACCEPT)):
        got = call("GET", "/vault/Notes/Alpha.md", headers={"Accept": accept})
        check(f"{label} returns 200", got.status_code, 200)
        # The deliberate difference from the plugin. n8n does not treat the
        # vendor type as JSON, so answering with it would hand the workflow a
        # string; answering application/json makes n8n parse it.
        check_in(f"{label} is answered as JSON", "application/json", got.headers["content-type"])
        body = got.json()
        check(f"{label}: path", body["path"], "Notes/Alpha.md")
        check(f"{label}: content is the file, byte for byte", body["content"], NOTE)
        # `body` exists so a caller wanting the prose does not carry its own
        # frontmatter regex. Three n8n nodes did, and one of them assembles
        # Lyra's whole system prompt.
        check(f"{label}: body drops the frontmatter", body["body"], NOTE.split("---\n", 2)[2].lstrip())
        check(f"{label}: body keeps the prose whole", body["body"].splitlines()[0], "# Alpha")
        check(f"{label}: title", body["frontmatter"]["title"], "Alpha")
        check(f"{label}: tags", body["frontmatter"]["tags"], ["lyra", "ops"])
        check(
            f"{label}: no stat or tags key invented",
            sorted(body),
            ["body", "content", "frontmatter", "path"],
        )

    scoped = call(
        "GET", "/vault/Notes/Alpha.md?section=Detail", headers={"Accept": JSON_ACCEPT}
    ).json()
    check("section= narrows content", scoped["content"].strip().splitlines()[0], "## Detail")
    check("section= leaves frontmatter whole", scoped["frontmatter"]["title"], "Alpha")
    # A section carries no frontmatter of its own, so there is nothing to strip.
    check("section= makes body the same text", scoped["body"], scoped["content"])

    # Malformed YAML must not make a note unreadable. `Get Triage Rules` wants
    # the content; a parse failure in a field it never mentions should not be
    # what stops it.
    broken = call("GET", "/vault/Notes/Broken.md", headers={"Accept": JSON_ACCEPT})
    check("malformed YAML still returns 200", broken.status_code, 200)
    check("malformed YAML yields empty frontmatter", broken.json()["frontmatter"], {})
    check_in("malformed YAML still returns content", "does not parse", broken.json()["content"])
    # The reason `content` is not the stripped field. metadata() answers {} for
    # a block it cannot parse, so if `content` had the block removed too, this
    # note would come back with no trace of its frontmatter anywhere - and four
    # notes in the real vault parse exactly this badly.
    check(
        "malformed YAML is still removed from body",
        "title: Broken" in broken.json()["body"],
        False,
    )
    check_in(
        "but the block itself survives in content",
        "title: Broken",
        broken.json()["content"],
    )


# --------------------------------------------------------------------------
# 2.2  The claim: PATCH with Target-Type: frontmatter
# --------------------------------------------------------------------------


def test_frontmatter_patch() -> None:
    write_fixture()
    got = call(
        "PATCH",
        "/vault/Workflows/Approvals/a1.md",
        headers={"Target": "status", "Target-Type": "frontmatter", "Operation": "replace"},
        content=json.dumps("approved"),
    )
    check("claiming a status returns 200", got.status_code, 200)
    # The whole point. A quoted value here compares equal to nothing downstream.
    check_in("status is written unquoted", "\nstatus: approved\n", read_note("Workflows/Approvals/a1.md"))
    check("the JSON string's quotes are not written", '"approved"' in read_note("Workflows/Approvals/a1.md"), False)

    call(
        "PATCH",
        "/vault/Workflows/Approvals/a1.md",
        headers={"Target": "rev", "Target-Type": "frontmatter"},
        content="2",
    )
    check_in("a number stays a number", "\nrev: 2\n", read_note("Workflows/Approvals/a1.md"))

    # A real snowflake, because the length is the point. Written bare this is an
    # integer, and one past JavaScript's MAX_SAFE_INTEGER - n8n's JSON.parse
    # would round it to ...038800 and the reply would go to a thread that does
    # not exist. The plugin quoted it; so must this.
    call(
        "PATCH",
        "/vault/Workflows/Approvals/a1.md",
        headers={"Target": "thread_id", "Target-Type": "frontmatter"},
        content=json.dumps("1548070648281038848"),
    )
    check_in(
        "a thread id is quoted, so it stays a string",
        'thread_id: "1548070648281038848"',
        read_note("Workflows/Approvals/a1.md"),
    )
    # The assertion that actually matters: what a caller reads back.
    got = call(
        "GET", "/vault/Workflows/Approvals/a1.md", headers={"Accept": "application/json"}
    )
    fm = got.json()["frontmatter"]
    check("and reads back as that exact string", fm["thread_id"], "1548070648281038848")
    check("status still compares equal downstream", fm["status"], "approved")
    check("rev is still a number, not a string", fm["rev"], 2)

    # Removal is an operation, not a null value. A null is refused outright, and
    # the refusal tells the caller to delete the field - which, until this,
    # named a spelling no REST caller had.
    removed = call(
        "PATCH",
        "/vault/Workflows/Approvals/a1.md",
        headers={"Target": "rev", "Target-Type": "frontmatter", "Operation": "delete"},
    )
    check("deleting a field returns 200", removed.status_code, 200)
    after = read_note("Workflows/Approvals/a1.md")
    check("the field is gone", "\nrev:" in after, False)
    check_in("the rest of the block is untouched", "\nstatus: approved\n", after)
    check("and the note was timestamped", "2026-09-12T09:00:00Z" in after, False)

    # Deleting a key that is not there is not an error: the caller asked for the
    # field to be absent, and afterwards it is.
    again = call(
        "PATCH",
        "/vault/Workflows/Approvals/a1.md",
        headers={"Target": "rev", "Target-Type": "frontmatter", "Operation": "delete"},
    )
    check("deleting an absent field is not an error", again.status_code, 200)

    nulled = call(
        "PATCH",
        "/vault/Workflows/Approvals/a3.md",
        headers={"Target": "status", "Target-Type": "frontmatter"},
        content="null",
    )
    check("a null value is still refused", nulled.status_code, 400)
    check_in("and names a spelling REST has", "Operation: delete", nulled.text)
    check_in(
        "status is untouched", "\nstatus: pending\n", read_note("Workflows/Approvals/a3.md")
    )

    # A bare word is not JSON. Rejecting it is what stops `approved` and
    # `"approved"` quietly becoming different values in the same field.
    before = read_note("Workflows/Approvals/a2.md")
    bad = call(
        "PATCH",
        "/vault/Workflows/Approvals/a2.md",
        headers={"Target": "status", "Target-Type": "frontmatter"},
        content="approved",
    )
    check("an unquoted body is refused", bad.status_code, 400)
    check_in("and says what was expected", "must be JSON", bad.text)
    check("and changes nothing", read_note("Workflows/Approvals/a2.md"), before)

    # set_frontmatter replaces outright, so an append would discard the rest of
    # a list rather than add to it.
    appended = call(
        "PATCH",
        "/vault/Workflows/Approvals/a2.md",
        headers={"Target": "tags", "Target-Type": "frontmatter", "Operation": "append"},
        content=json.dumps("extra"),
    )
    check("append on frontmatter is refused", appended.status_code, 400)
    check_in("and names the supported operation", "'replace'", appended.text)
    check("and changes nothing", read_note("Workflows/Approvals/a2.md"), before)

    unknown = call(
        "PATCH",
        "/vault/Workflows/Approvals/a2.md",
        headers={"Target": "status", "Target-Type": "elsewhere"},
        content=json.dumps("approved"),
    )
    check("an unknown Target-Type is refused", unknown.status_code, 400)
    check_in("and lists the ones that work", "'frontmatter'", unknown.text)
    check("and changes nothing", read_note("Workflows/Approvals/a2.md"), before)

    # The existing behaviour, which must survive the branch that was added
    # around it - both when Target-Type says heading and when it is absent.
    for headers in (
        {"Target": "Summary", "Target-Type": "heading"},
        {"Target": "Summary"},
    ):
        patched = call(
            "PATCH", "/vault/Notes/Alpha.md", headers=headers, content="Rewritten.\n"
        )
        check(f"heading patch still works ({headers})", patched.status_code, 200)
    check_in("the heading's section was replaced", "Rewritten.", read_note("Notes/Alpha.md"))
    check_in("and its neighbour was not", "Second section.", read_note("Notes/Alpha.md"))

    missing_target = call(
        "PATCH", "/vault/Notes/Alpha.md", headers={"Target-Type": "frontmatter"}, content='"x"'
    )
    check("PATCH without a Target is refused", missing_target.status_code, 400)

    # The reason for the migration, asserted on the new write path: a frontmatter
    # PATCH reaches the vault through the same guard every other write does.
    protected = call(
        "PATCH",
        "/vault/index.md",
        headers={"Target": "status", "Target-Type": "frontmatter"},
        content='"anything"',
    )
    check("index.md is protected from a frontmatter PATCH", protected.status_code, 400)
    check_in("and says why", "protected", protected.text)


# --------------------------------------------------------------------------
# 2.3  The frontmatter query
# --------------------------------------------------------------------------


def names(response: httpx.Response) -> list[str]:
    return sorted(hit["filename"] for hit in response.json())


def test_frontmatter_query() -> None:
    write_fixture()
    # Every note here is under Workflows/, which is in SEARCH_EXCLUDE_DIRS. A query
    # served from the semantic index would return an empty list.
    pending = call("GET", "/frontmatter?key=status&value=pending")
    check("pending proposals are found", pending.status_code, 200)
    check(
        "and only the pending ones",
        names(pending),
        ["Workflows/Approvals/a1.md", "Workflows/Approvals/a3.md"],
    )
    check_in("answered as JSON", "application/json", pending.headers["content-type"])
    check("each hit carries filename", sorted(pending.json()[0]), ["filename"])

    thread = call("GET", "/frontmatter?key=thread_id&value=1549")
    check("a quoted scalar matches unquoted", names(thread), ["Workflows/Approvals/a3.md"])

    check("nothing matching is an empty list", call("GET", "/frontmatter?key=status&value=nope").json(), [])
    check("an absent key is an empty list", call("GET", "/frontmatter?key=nosuch&value=x").json(), [])

    # Both list shapes this vault writes, since a key that silently matched
    # nothing would look exactly like a key with no matches.
    check(
        "inline list membership matches",
        names(call("GET", "/frontmatter?key=tags&value=approval")),
        ["Workflows/Approvals/a1.md", "Workflows/Approvals/a2.md", "Workflows/Approvals/a3.md"],
    )
    check(
        "block list membership matches",
        names(call("GET", "/frontmatter?key=tags&value=ops")),
        ["Notes/Alpha.md"],
    )
    check(
        "a value in both shapes finds both",
        names(call("GET", "/frontmatter?key=tags&value=lyra")),
        [
            "Notes/Alpha.md",
            "Workflows/Approvals/a1.md",
            "Workflows/Approvals/a2.md",
            "Workflows/Approvals/a3.md",
        ],
    )

    narrowed = call("GET", "/frontmatter?key=tags&value=lyra&dir=Notes")
    check("dir= narrows the walk", names(narrowed), ["Notes/Alpha.md"])

    check("key= is required", call("GET", "/frontmatter?value=pending").status_code, 400)
    check("value= is required", call("GET", "/frontmatter?key=status").status_code, 400)
    check_in(
        "and the error shows the shape",
        "key=status&value=pending",
        call("GET", "/frontmatter?key=status").text,
    )
    check("an empty value is a real query, not a missing one",
          call("GET", "/frontmatter?key=status&value=").status_code, 200)
    check("a dir that is not there is 404", call("GET", "/frontmatter?key=status&value=pending&dir=Nope").status_code, 404)
    check("a dir that is a note is 400", call("GET", "/frontmatter?key=status&value=pending&dir=Notes/Alpha.md").status_code, 400)


# --------------------------------------------------------------------------
# 4  Missing is 404, everything else is 400
# --------------------------------------------------------------------------


def test_status_codes() -> None:
    write_fixture()
    missing = call("GET", "/vault/Notes/Nope.md")
    check("a missing note is 404", missing.status_code, 404)
    check_in("and says so", "no such path", missing.text)
    check(
        "including on the structured read",
        call("GET", "/vault/Notes/Nope.md", headers={"Accept": JSON_ACCEPT}).status_code,
        404,
    )
    check("and on a write to one", call("DELETE", "/vault/Notes/Nope.md").status_code, 404)

    # The distinction is the point: a caller branching on "no such proposal"
    # must not also catch its own malformed target.
    bad_target = call(
        "PATCH", "/vault/Notes/Alpha.md", headers={"Target": "No Such Heading"}, content="x"
    )
    check("a bad target is 400, not 404", bad_target.status_code, 400)

    # The separators are encoded too, so this stays a single path segment. An
    # unencoded ../ is normalised away by the client, and even %2E%2E between
    # real slashes is normalised by the ASGI server - in both cases the path
    # stops starting with /vault and is refused by the router without ever
    # reaching the resolver. Safe, but a different mechanism, and asserting on
    # it here would leave containment itself untested. Verified against the
    # running server: this form arrives intact and safe_resolve refuses it.
    #
    # Containment is a refusal, not an absence: answering 404 would tell a
    # caller the path was merely missing.
    escaping = call("GET", "/vault/%2E%2E%2Fetc%2Fpasswd")
    check("an escaping path is 400, not 404", escaping.status_code, 400)
    check_in("and says it escaped rather than that it is missing", "escapes the vault", escaping.text)

    # Resolving through a parent back into the vault is not an escape, and must
    # not be reported as one - it is an ordinary missing note.
    inward = call("GET", "/vault/Notes/%2E%2E/Nope.md")
    check("a path that resolves back inside is a plain 404", inward.status_code, 404)
    check_in("and names the resolved path", "no such path", inward.text)
    check(
        "a non-.md write is 400, not 404",
        call("PUT", "/vault/Notes/Alpha.txt", content="x").status_code,
        400,
    )

    check("auth is still enforced", asyncio.run(_unauthenticated()), 401)



# --------------------------------------------------------------------------
# 2.4  The body write: PATCH with Target-Type: body
# --------------------------------------------------------------------------


def test_body_patch() -> None:
    """Replacing the prose must not take the frontmatter with it.

    `Open WebUI: Sync Lyra System Prompt` regenerates Aidan Summary.md every
    Saturday. With PUT as the only whole-note write, the note lost its
    frontmatter on the first run and never had any again.
    """
    write_fixture()

    done = call(
        "PATCH",
        "/vault/Notes/Alpha.md",
        headers={"Target-Type": "body"},
        content="# Alpha\n\nEntirely new prose.\n",
    )
    check("a body patch succeeds", done.status_code, 200)

    got = call("GET", "/vault/Notes/Alpha.md", headers={"Accept": JSON_ACCEPT}).json()
    check("the body is what was sent", got["body"], "# Alpha\n\nEntirely new prose.\n")
    check("the title survived", got["frontmatter"]["title"], "Alpha")
    check("so did the tags", got["frontmatter"]["tags"], ["lyra", "ops"])
    check_in("and the block is still in content", "type: note", got["content"])
    # The whole point: nothing of the old prose is left behind.
    check("the old prose is gone", "First section." in got["body"], False)
    # Every other write bumps it, and so must this one.
    check(
        "timestamp was bumped",
        got["frontmatter"]["timestamp"] != "2026-09-12T09:00:00Z",
        True,
    )

    # A block this server cannot parse is still carried across untouched. It is
    # sliced, not re-rendered, so there is nothing for a YAML dump to lose.
    call(
        "PATCH",
        "/vault/Notes/Broken.md",
        headers={"Target-Type": "body"},
        content="# Broken\n\nNew prose.\n",
    )
    broken = call("GET", "/vault/Notes/Broken.md", headers={"Accept": JSON_ACCEPT}).json()
    check_in("a malformed block survives a body patch", "tags: [unclosed", broken["content"])
    check("and the new prose is in place", broken["body"], "# Broken\n\nNew prose.\n")

    # No Target header is needed, and sending Operation: append is refused
    # rather than silently treated as a replace - POST already appends.
    appended = call(
        "PATCH",
        "/vault/Notes/Alpha.md",
        headers={"Target-Type": "body", "Operation": "append"},
        content="more",
    )
    check("append is refused on the body", appended.status_code, 400)
    check_in("and points at POST instead", "POST to the note", appended.text)


# --------------------------------------------------------------------------
# 2.6  /maintenance
#
# The fixture vault has no .scripts/, and that is the point: every checker
# reports itself missing, which exercises the shape of the answer and the
# not-found branch without running eight real walks over a temp directory.
#
# The assertion that matters is the status code. A vault full of findings must
# still be a 200 - the caller branches on "did the suite run", and a route that
# answered 500 for a broken link would make the workflow's error path fire on
# exactly the weeks the report is worth reading.
# --------------------------------------------------------------------------


def test_maintenance() -> None:
    got = call("GET", "/maintenance")
    check("a run with nothing to run is still 200", got.status_code, 200)

    body = got.json()
    check("one entry per checker", len(body["checks"]), len(maintenance.CHECKS))
    check("none of them exited zero", body["summary"]["exit_zero"], 0)
    check("all of them are reported as errored", body["summary"]["errored"], len(maintenance.CHECKS))

    # The markdown rendering is what the workflow actually puts under its
    # prompt, so a run that produced no findings must still produce a block.
    check_in("the rendering names every checker", maintenance.CHECKS[-1].title, body["markdown"])
    check_in("and says a check could not run", "did not run", body["markdown"])

    first = body["checks"][0]
    check("the order is the order in CHECKS", first["script"], maintenance.CHECKS[0].script)
    check_in("and a missing script says where it looked", ".scripts", first["error"])
    check("a missing script has no exit code to report", first["exit_code"], None)

    # POST is accepted so a caller that only sends POSTs needs no special case,
    # and no query parameter may become an argument - the argv is built from
    # the constants in src/maintenance.py and nothing else.
    posted = call("POST", "/maintenance?script=;id")
    check("POST is accepted too", posted.status_code, 200)
    check(
        "and a query parameter changes nothing about what ran",
        [c["script"] for c in posted.json()["checks"]],
        [c.script for c in maintenance.CHECKS],
    )


# --------------------------------------------------------------------------
# 2.7  /callouts
#
# The fixture vault has no .scripts/ either, so the extractor cannot run and
# the route must say so in the one way the workflow can branch on: a 500 with
# a message, not a 200 carrying an empty list. The two are not the same answer
# - an empty list means the vault has nothing open, and a run that never
# happened must never be read as a clean one.
# --------------------------------------------------------------------------


def test_callouts() -> None:
    got = call("GET", "/callouts")
    check("an extractor that is not there is a 500", got.status_code, 500)
    check_in("and the error says which script", callouts.SCRIPT, got.json()["error"])
    check_in("and where it looked", ".scripts", got.json()["error"])

    # POST is accepted for the same reason /maintenance accepts it, and the
    # script's own filters are not exposed - the argv is built from the
    # constants in src/callouts.py and nothing else.
    posted = call("POST", "/callouts?type=warning&path=;id")
    check("POST is accepted too", posted.status_code, 500)
    check_in("and a query parameter changes nothing", callouts.SCRIPT, posted.json()["error"])


async def _unauthenticated() -> int:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://vault-mcp:8080") as c:
        return (await c.get("/frontmatter?key=status&value=pending")).status_code


def main() -> int:
    test_structured_read()
    test_frontmatter_patch()
    test_frontmatter_query()
    test_status_codes()
    test_body_patch()
    test_maintenance()
    test_callouts()
    if report():
        return 1
    print("rest: all checks passed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        shutil.rmtree(_VAULT, ignore_errors=True)
