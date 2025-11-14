#!/usr/bin/env python3

import subprocess
from typing import Optional


def run_git_command(arguments: list[str]) -> str:
    completed = subprocess.run(
        ["git"] + arguments,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def resolve_base_sha_from_ref(base_ref: str) -> Optional[str]:
    if not base_ref:
        return None
    candidate_ref = f"origin/{base_ref}"
    try:
        return run_git_command(["rev-parse", candidate_ref])
    except subprocess.CalledProcessError:
        return None


def compute_merge_base_sha(base_sha: str, head_sha: str) -> Optional[str]:
    try:
        return run_git_command(["merge-base", base_sha, head_sha])
    except subprocess.CalledProcessError:
        return None


