---
type: reference
title: Home Assistant
description: The Home Assistant instance, its integrations and the automations that matter.
tags: [tech, automation]
timestamp: 2026-05-27T19:15:00Z
---

No H1 wrapper, which is the shape the other 14% of the real vault's notes have, and
which the heading resolver needs represented: in an H1-wrapped note every heading
inherits the H1, so no bare leaf can ever be a heading's *whole* path. Here `Notes`
can be, and is, which is how a repeated leaf gets resolved by exact whole-path match
rather than refused as ambiguous.

Container on the server, 10.0.4.14, database on Postgres rather than SQLite because
the default recorder grew to 4GB in six months.

## Integrations

- Zigbee via a USB dongle on VLAN 20
- The electricity meter via the supplier's API
- The boiler via an OpenTherm gateway

### Notes

The Zigbee dongle needs a USB extension lead to get it away from the case, which
halved the number of dropped messages. The electricity integration polls rather than
streams, so its figures lag the meter by about an hour.

## Automations

Three that matter: the heating schedule, the front door notification, and the alert
when the freezer temperature rises above -15C for more than an hour.

## Notes

Upgrades are done by tag rather than `latest`, and the database is dumped before each
one. A restore has been needed once, after a breaking change to the recorder schema.
