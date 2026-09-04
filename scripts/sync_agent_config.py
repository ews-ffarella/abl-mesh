"""Generate the ``.claude/`` agent tree from ``.cursor/`` (the source of truth).

``.cursor/`` is authored by hand; ``.claude/`` is **generated** and should never be
edited directly. Run ``--check`` in CI / prek to fail on drift.

    uv run python scripts/sync_agent_config.py            # write .claude/
    uv run python scripts/sync_agent_config.py --check    # exit 1 if out of date

Why a generator instead of the old "mirror both trees by hand" rule: the two tools spell
the same scoping differently. Cursor rules carry ``globs`` / ``alwaysApply`` frontmatter
and load only when matching files are open. Claude Code reads ``paths:`` frontmatter: a
rule without it loads into every session, a rule with it loads lazily when a matching
file is read (documented at code.claude.com/docs/en/memory; ruled for this estate
2026-08-31 - fabien-context #57). Hand-mirroring drifted silently before the generator -
four rules that ``AGENTS.md`` calls always-on had no Claude counterpart at all.

The generator translates the Cursor ``globs`` into Claude ``paths:`` (always-on rules
stay frontmatter-free) and keeps a readable **Applies to** header for humans.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CURSOR_ROOT = REPO_ROOT / ".cursor"
CLAUDE_ROOT = REPO_ROOT / ".claude"

GENERATED_BANNER = (
    "<!-- GENERATED FROM .cursor/rules/{source} BY scripts/sync_agent_config.py.\n"
    "     Edit the .cursor source, then run: uv run python scripts/sync_agent_config.py -->"
)

SKILL_EXCLUDE_DIRS = {"__pycache__", ".ipynb_checkpoints"}


class Frontmatter:
    """Parsed Cursor ``.mdc`` frontmatter plus the markdown body."""

    def __init__(self, *, description: str, globs: str, always_apply: bool, body: str):
        self.description = description
        self.globs = globs
        self.always_apply = always_apply
        self.body = body

    @classmethod
    def parse(cls, text: str) -> Frontmatter:
        """Split a ``.mdc`` file into frontmatter fields and body.

        Cursor frontmatter is a flat ``key: value`` block; values are never nested, so a
        line scan is sufficient and keeps this script dependency-free (it runs as a prek
        hook, where pulling in a YAML parser would be extra setup).
        """
        if not text.startswith("---"):
            return cls(description="", globs="", always_apply=False, body=text.strip())

        _, _, rest = text.partition("---")
        raw_front, sep, body = rest.partition("---")
        if not sep:
            return cls(description="", globs="", always_apply=False, body=text.strip())

        fields: dict[str, str] = {}
        for line in raw_front.splitlines():
            key, colon, value = line.partition(":")
            if colon:
                fields[key.strip()] = value.strip()

        return cls(
            description=fields.get("description", ""),
            globs=fields.get("globs", "").strip("\"'"),
            always_apply=fields.get("alwaysApply", "").lower() == "true",
            body=body.strip(),
        )


def split_globs(globs: str) -> list[str]:
    """Split a Cursor ``globs`` value on commas, keeping brace groups whole.

    ``{.claude,.cursor}/**`` is one glob; a naive ``split(",")`` cuts it in half, which
    is exactly what the old header rendering did.
    """
    parts: list[str] = []
    buf, depth = "", 0
    for ch in globs:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            if buf.strip():
                parts.append(buf.strip())
            buf = ""
        else:
            buf += ch
    if buf.strip():
        parts.append(buf.strip())
    return parts


def render_claude_rule(source_name: str, front: Frontmatter) -> str:
    """Render a Cursor rule as a Claude rule, carrying its scope as ``paths:``.

    A rule with ``paths:`` frontmatter loads lazily - when Claude reads a matching
    file - instead of into every session, matching Cursor's ``globs`` behaviour.
    Frontmatter must be the first bytes of the file, before the banner comment.
    Always-on rules stay frontmatter-free so they keep loading at session start.
    """
    lines: list[str] = []
    if front.globs and not front.always_apply:
        lines.append("---")
        lines.append("paths:")
        lines.extend(f'  - "{g}"' for g in split_globs(front.globs))
        lines.append("---")

    lines.extend([GENERATED_BANNER.format(source=source_name), ""])

    if front.always_apply:
        lines.append("**Applies to:** always - this rule is in force for every change.")
    elif front.globs:
        globs = ", ".join(f"`{g}`" for g in split_globs(front.globs))
        lines.append(f"**Applies to:** {globs}")
        lines.append("")
        lines.append("Skip this rule if your change does not touch those paths.")
    else:
        lines.append("**Applies to:** on demand - consult when the topic below comes up.")

    lines.extend(["", front.body, ""])
    return "\n".join(lines)


def collect_rules() -> dict[Path, str]:
    """Map every target ``.claude/rules/*.md`` path to its generated content."""
    generated: dict[Path, str] = {}
    for source in sorted((CURSOR_ROOT / "rules").glob("*.mdc")):
        front = Frontmatter.parse(source.read_text(encoding="utf-8"))
        target = CLAUDE_ROOT / "rules" / f"{source.stem}.md"
        generated[target] = render_claude_rule(source.name, front)
    return generated


def iter_skill_files() -> list[tuple[Path, Path]]:
    """Yield ``(source, target)`` for every skill file that mirrors byte-for-byte."""
    pairs: list[tuple[Path, Path]] = []
    skills_root = CURSOR_ROOT / "skills"
    if not skills_root.is_dir():
        return pairs

    for source in sorted(skills_root.rglob("*")):
        if not source.is_file():
            continue
        if any(part in SKILL_EXCLUDE_DIRS for part in source.relative_to(skills_root).parts):
            continue
        target = CLAUDE_ROOT / "skills" / source.relative_to(skills_root)
        pairs.append((source, target))
    return pairs


def stale_targets(expected: set[Path], roots: list[Path], suffixes: set[str]) -> list[Path]:
    """Find generated files that no longer have a ``.cursor/`` source.

    Without this, renaming or consolidating a Cursor rule would leave the old Claude file
    behind and it would keep loading forever - exactly the failure this script exists to
    prevent.
    """
    stale: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            if any(part in SKILL_EXCLUDE_DIRS for part in path.relative_to(root).parts):
                continue
            if path not in expected:
                stale.append(path)
    return stale


def sync(*, check_only: bool) -> int:
    """Write (or verify) the generated ``.claude/`` tree. Returns a process exit code."""
    rules = collect_rules()
    skills = iter_skill_files()
    expected = set(rules) | {target for _, target in skills}

    drift: list[str] = []

    for target, content in rules.items():
        current = target.read_text(encoding="utf-8") if target.is_file() else None
        if current == content:
            continue
        drift.append(f"rule out of date: {target.relative_to(REPO_ROOT)}")
        if not check_only:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8", newline="\n")

    for source, target in skills:
        if target.is_file() and filecmp.cmp(source, target, shallow=False):
            continue
        drift.append(f"skill out of date: {target.relative_to(REPO_ROOT)}")
        if not check_only:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    for path in stale_targets(
        expected,
        [CLAUDE_ROOT / "rules", CLAUDE_ROOT / "skills"],
        {".md", ".py", ".txt", ".json", ".yaml", ".yml"},
    ):
        drift.append(f"orphaned (no .cursor source): {path.relative_to(REPO_ROOT)}")
        if not check_only:
            path.unlink()

    if not drift:
        print(f"agent config in sync ({len(rules)} rules, {len(skills)} skill files)")
        return 0

    for line in drift:
        print(line)

    if check_only:
        print("\n.claude/ is generated from .cursor/. Regenerate with:")
        print("    uv run python scripts/sync_agent_config.py")
        return 1

    print(f"\nregenerated .claude/ ({len(drift)} change(s))")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify only; exit 1 if .claude/ differs from what .cursor/ would generate",
    )
    args = parser.parse_args()
    return sync(check_only=args.check)


if __name__ == "__main__":
    sys.exit(main())
