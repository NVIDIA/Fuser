# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# "python benchmark_thunder.py -h" for help.

import argparse
import os
import subprocess
from typing import Iterable


# Switches to the given branch, syncs to head, and returns the short hash of
# the head.
def switch_to(branch: str, sync: bool) -> str:
    subprocess.check_call(f"git fetch origin {branch}", shell=True)

    # `advice.detachedHead=false` silences the detached HEAD warning.
    subprocess.check_call(
        f"git -c advice.detachedHead=false checkout {branch}", shell=True
    )

    if sync:
        subprocess.check_call(f"git reset --hard origin/{branch}", shell=True)

    return subprocess.check_output(
        "git rev-parse --short HEAD", shell=True, text=True
    ).strip()


# Sanitize the branch name so it can be used in a filename.
def sanitize_branch_name(branch: str) -> str:
    return branch.replace("/", "_")


class BenchmarkRunner:
    def __init__(self, storage: str, benchmark_filter: str, sync: bool):
        self._storage = storage
        self._benchmark_filter = benchmark_filter
        self._sync = sync

    def run_setting(self, setting: str) -> None:
        thunder_branch, nvfuser_branch = setting.split(":")
        print(
            f"Running Thunder benchmarks for nvFuser branch '{nvfuser_branch}' and Thunder branch '{thunder_branch}'..."
        )

        os.chdir("/opt/pytorch/nvfuser")
        nvfuser_commit = switch_to(nvfuser_branch, self._sync)
        subprocess.check_call("git submodule update --init --recursive", shell=True)
        subprocess.check_call("_bn", shell=True)

        os.chdir("/opt/pytorch/lightning-thunder")
        thunder_commit = switch_to(thunder_branch, self._sync)
        subprocess.check_call("pip install -r requirements/devel.txt", shell=True)

        out_stem = f"thunder_{sanitize_branch_name(thunder_branch)}_{thunder_commit}_nvfuser_{sanitize_branch_name(nvfuser_branch)}_{nvfuser_commit}"
        # Benchmarks fail occasionally for various reasons, e.g. OOM. Use `run`
        # instead of `check_call` to continue with other settings and report only
        # benchmarks that succeeded.
        command = f"pytest thunder/benchmarks/targets.py --color=no -k '{self._benchmark_filter}' --benchmark-save={out_stem}"
        if self._storage:
            command += f" --benchmark-storage={self._storage}"
        subprocess.run(command, shell=True)

    def run_settings(self, settings: Iterable[str]) -> None:
        for setting in settings:
            self.run_setting(setting)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs Thunder benchmarks with multiple settings. "
        "It stores benchmark results to the specified storage path, which "
        "can be compared by running `pytest-benchmark --storage <storage_path> "
        "compare <run IDs> --group-by name`."
    )
    parser.add_argument(
        "settings",
        type=str,
        nargs="+",
        help="a list of settings to benchmark. Each setting is in format of <thunder_branch>:<nvfuser_branch>",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="",
        help="the path to give to `pytest --benchmark-storage`. If not specified, `pytest` will use its default storage path. This flag is useful to save benchmark results out of a transient Docker container.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="thunder]",
        help="a benchmark filter that will be passed to `pytest -k <filter>`. By default, the filter is 'thunder]'.",
    )
    parser.add_argument(
        "--sync", action="store_true", help="whether to `git reset origin`"
    )
    args = parser.parse_args()

    runner = BenchmarkRunner(args.storage, args.filter, args.sync)
    runner.run_settings(args.settings)
