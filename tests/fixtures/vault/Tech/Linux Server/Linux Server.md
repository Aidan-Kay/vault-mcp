---
type: reference
title: Linux Server
description: The home server — hardware, services, storage layout and its backup regime.
tags: [tech, server]
timestamp: 2026-09-14T22:05:00Z
---

# Linux Server

Debian 13 on a Ryzen 5600G with 64GB of ECC, running everything in Docker.

This note carries the shape the heading resolver exists for: `Backups` appears
twice, under two different parents, so a bare `Backups` target here must be
refused with an error naming the ancestors to prepend. `Docker.md` and `NAS.md`
each carry a third and fourth `Backups`, which is a different problem — those are
unambiguous within their own note.

## Hardware

- CPU: Ryzen 5 5600G, 6 cores
- RAM: 64GB ECC DDR4-3200
- Boot: 1TB NVMe, mirrored
- Bulk: four 8TB SATA in a ZFS raidz1

### Backups

The boot mirror is imaged monthly to the pool, because rebuilding Debian and the
compose stacks from scratch is a day's work and the image is 40GB.

## Storage

The pool is `tank`, one raidz1 vdev, 72% full as of September 2026. Datasets are
split per service so snapshots can be rolled back independently.

### Backups

Nightly ZFS snapshots kept for 14 days, weekly for 8 weeks, monthly for 12
months. Offsite is a weekly `zfs send` to a rotated external disk, and the
critical datasets also go to Stonevault S2 nightly.

## Services

Docker Compose, one stack per directory under `/srv`. Portainer is not installed
deliberately — the compose files are the interface.
