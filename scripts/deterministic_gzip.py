"""Create reproducible gzip files (timestamp-free header)."""

from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path


def compress(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_stream, destination.open("wb") as raw_output:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_output, mtime=0, compresslevel=9
        ) as gzip_output:
            shutil.copyfileobj(input_stream, gzip_output, length=1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    compress(args.source, args.destination)


if __name__ == "__main__":
    main()
