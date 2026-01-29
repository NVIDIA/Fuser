#!/usr/bin/env python3

import os
import sys
from typing import Optional
from pathlib import Path


def get_env(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value:
        value = value.strip()
    return value or None


def ensure_dir(path) -> None:
    """
    Create directory if it doesn't exist. Accepts str or pathlib.Path.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def write_to_path(output_path: Path | str, filename: str, content: str) -> None:
    """
    Write content to a file inside the given directory path, creating the directory if needed.
    """
    print(content, file=sys.stderr)
    base = Path(output_path)
    base.mkdir(parents=True, exist_ok=True)
    filepath = base / filename
    with open(filepath, "w") as f:
        f.write(content)


