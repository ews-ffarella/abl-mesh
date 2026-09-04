---
status: as-built
covers: what the lab book is in this repo, and the one rule that keeps it from becoming load-bearing history
---

# Lab book

What was done, when, and what came of it. One file per theme, chronological
within a file, **append-only**. Nothing links into an entry: a reader finds
a fact by subject in `docs/`, never by date here.

Estate rule, `[FF-0309]`: *"A log book when things are hot, a promotion loop
in ADRs / docs / skills / rules, and fails in gotchas."* An entry is written
verbose while the round is hot; when it settles, a dead end goes to
[`../gotchas/`](../gotchas/README.md) in short form, a choice with rejected
alternatives goes to an ADR, and the entry is condensed to a date, a line
and links out. Never delete or reorder an entry.

| File | Theme |
| --- | --- |
| [`archeology.md`](archeology.md) | the read of this repo against `ablmesh`: what was implemented, what stayed on the roadmap, what survived |

This repo takes no pipeline work, so the book is expected to grow rarely.
