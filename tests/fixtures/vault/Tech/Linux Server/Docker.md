---
type: reference
title: Docker
description: Docker and Compose conventions on the server, and how container data is backed up.
tags: [tech, docker]
timestamp: 2026-08-30T17:48:00Z
---

# Docker

Compose v2, one stack per service, every stack in its own directory under `/srv`
with a `.env` beside it. Images are pinned by tag and updated deliberately rather
than by Watchtower.

## Conventions

- Bind mounts rather than named volumes, so the data is visible on the host
- `restart: unless-stopped` on everything
- No container runs as root unless it cannot be avoided
- Ports bound to the LAN interface, never `0.0.0.0`, unless behind the proxy

## Backups

Container data is backed up by the host's ZFS snapshots rather than by anything
container-aware. Databases are the exception: Postgres and MariaDB are dumped to
disk before the nightly snapshot, because a snapshot of a live database is a
crash-consistent copy rather than a clean one.

## Networking

Each stack gets its own bridge network. The reverse proxy joins the networks of
the services it fronts rather than everything sharing one.
