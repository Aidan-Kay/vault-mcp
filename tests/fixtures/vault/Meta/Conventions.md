---
type: reference
title: Conventions
description: How notes in this fixture vault are named, structured and filed.
tags: [meta]
timestamp: 2026-01-04T09:12:00Z
---

# Conventions

The rules every note here follows. This note exists so the fixture has one
deeply nested reference note, the shape most of the real vault's Meta/ folder has.

## Frontmatter

Every note opens with `type`, `title`, `description`, `tags` and `timestamp`, in
that order. A note without frontmatter is a scratch note and is expected to be
tidied or deleted.

### Timestamps

UTC, second precision, no offsets. The timestamp records when the note last said
something new, not when a typo was fixed.

### Descriptions

One sentence, no trailing full stop in the older notes and a full stop in the
newer ones, which is itself a thing the maintenance checkers complain about.

## File naming

Title case, spaces rather than hyphens, and no dates in filenames except for
one-off reports and filed source documents, which are immutable artefacts whose
date is their identity.

## Folder structure

One folder per domain, and a hub note named after its folder where a domain has
more than three notes in it.

### Attachments

Source documents live in a `Files/` subfolder beside the note that owns them, and
are linked from a `## Documents` section in that note.
