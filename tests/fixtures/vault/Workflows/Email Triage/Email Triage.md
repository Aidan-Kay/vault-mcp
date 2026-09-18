---
type: workflow
title: Email Triage
description: The n8n workflow that classifies incoming mail and files source documents.
tags: [workflow, automation]
timestamp: 2026-09-16T10:30:00Z
---

# Email Triage

Excluded from search like everything under `Workflows/`. It names the Kestrel Energy
account KE-8842071 and the MPAN 1900012345678 so that a lookup query for either
has a decoy it must not return.

## Steps

1. Gmail trigger on the household label
2. Classify with `qwen2.5:3b`
3. Where there is a PDF attachment, read it, choose the destination and upload
4. Patch the owning note's `## Documents` table
