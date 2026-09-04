# CLAUDE.md — abl-mesh

@AGENTS.md

Claude Code entrypoint. Shared instructions live in [`AGENTS.md`](AGENTS.md)
(imported above).

## Claude-local pointers

- **`.claude/` is generated — do not edit it.** Author under
  [`.cursor/`](.cursor/), then
  `uv run python scripts/sync_agent_config.py`. There is no `prek` hook in
  this repo; run the `--check` form by hand before a commit.
- Rules: [`.claude/rules/`](.claude/rules/) — generated from
  `.cursor/rules/*.mdc`. A rule carrying `paths:` frontmatter loads lazily,
  when a matching file is read; a rule without it loads every session. Each
  keeps an **Applies to** header for human readers.
- Memory written during a session lands in
  `~/.claude/projects/<slug>/memory/`, which is per-host and unversioned.
  Anything durable belongs in this repo or in `fabien-context`.
