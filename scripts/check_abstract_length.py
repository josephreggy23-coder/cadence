"""Check the repository abstract against the AAN's 300-word abstract limit.

This is a whitespace-delimited, conservative repository guard rather than an
official AAN word-counting tool. It deliberately excludes the Markdown title
and the repository-status notice, which are not part of the abstract body.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_LIMIT = 300


def abstract_body(markdown: str) -> str:
    """Return non-heading, non-blockquote content from an abstract Markdown file."""
    lines = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith(">"):
            continue
        lines.append(stripped)
    return " ".join(lines)


def word_count(markdown: str) -> int:
    """Count whitespace-delimited body words consistently across platforms."""
    return len(abstract_body(markdown).split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=Path("ABSTRACT.md"))
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be positive")
    if not args.path.is_file():
        parser.error(f"abstract not found: {args.path}")

    count = word_count(args.path.read_text(encoding="utf-8"))
    print(f"{args.path}: {count}/{args.limit} abstract-body words")
    if count > args.limit:
        raise SystemExit(
            f"abstract exceeds the configured {args.limit}-word limit by "
            f"{count - args.limit} words"
        )


if __name__ == "__main__":
    main()
