---
type: reference
title: Ollama
description: The Ollama instance on the server, the models pulled and what uses them.
tags: [tech, ai]
timestamp: 2026-09-01T08:52:00Z
---

# Ollama

Runs on the server in Docker, CPU-only, and is shared by everything that needs
embeddings or a small local model.

## Models

- `nomic-embed-text` — 768 dimensions, used for every vault embedding
- `qwen2.5:3b` — used for classification in the email triage workflow

## Notes

The embedding model name is the reason the tokeniser splits on hyphens: a query
for `nomic` should match a note that only ever writes `nomic-embed-text`.
Keeping the identifier whole would make that query miss.
