---
type: reference
title: Networking
description: VLANs, static addresses and the port allocations on the home network.
tags: [tech, network]
timestamp: 2026-06-11T11:35:00Z
---

# Networking

Flat 10.0.0.0/22 with three VLANs on top, all handled by the Halcyon SG-8.

## Addresses

| Host          | Address    | Notes                         |
| ------------- | ---------- | ----------------------------- |
| Gateway       | 10.0.0.1   | Halcyon SG-8                        |
| Linux server  | 10.0.4.10  | static lease                  |
| NAS           | 10.0.4.12  | static lease                  |
| Home Assistant| 10.0.4.14  | static lease                  |

## VLANs

- VLAN 1 — trusted, everything by default
- VLAN 20 — IoT, no route to trusted
- VLAN 30 — guest, internet only
