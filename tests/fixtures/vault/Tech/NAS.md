---
type: reference
title: NAS
description: The Cobalt NAS used for media and as the second copy of the server backups.
tags: [tech, storage]
timestamp: 2026-07-19T14:20:00Z
---

# NAS

A Cobalt N920+ with four 12TB drives in SHR-1, used for media and as the
second copy of everything the server holds.

## Configuration

- Model: N920+, CobaltOS 7.2
- Static IP: 10.0.4.12
- Volume 1: 32TB usable

## Backups

This is the third `Backups` heading in the fixture, and the one that makes a bare
`Backups` target ambiguous. The NAS is both a backup target and a thing that
needs backing up: it receives the server's weekly `zfs send`, and its own
configuration is exported to the server monthly.
