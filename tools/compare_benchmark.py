# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# "compare_benchmark.py -h" for help.

import argparse
from dataclasses import dataclass
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
from typing import Iterable


# If `arg` uses a forbidden option, returns that forbidden option; otherwise,
# returns an empty string.
def uses_forbidden_option(arg: str) -> str:
    for forbidden_option in (
        "--benchmark_out",
        "--benchmark_out_format",
        "--benchmark_format",
    ):
        # Depending on which shell, the name of a long option and the value can
        # be split by a space or an =.
        if arg == forbidden_option or arg.startswith(forbidden_option + "="):
            return forbidden_option
    return ""


def sanitize_benchmark_args(args: list[str]) -> list[str]:
    # Skip the leading "--". It's sometimes written before the benchmark args
    # as a convention.
    if args and args[0] == "--":
        args = args[1:]

    for arg in args:
        if forbidden_option := uses_forbidden_option(arg):
            raise ValueError(
                f"{forbidden_option} should be specified by run_benchmark not the user."
            )

    return args


def check_out(branch_or_commit: str) -> None:
    # `advice.detachedHead=false` silences the detached HEAD warning.
    subprocess.check_call(
        f"git -c advice.detachedHead=false checkout {branch_or_commit}", shell=True
    )
    subprocess.check_call("git submodule update --init --recursive", shell=True)


# Runs nvfuser_bench with `benchmark_args` on the given branch or commit. Dumps
# outputs to `out_dir`. Returns `out_dir`/`branch_or_commit`.json that captures
# the benchmark result. If the output already exists, skips benchmarking and
# uses that output. This is useful, for example, when comparing multiple
# contenders to the same base.
def run_benchmark(
    branch_or_commit: str, benchmark_args: list[str], out_dir: str
) -> str:
    benchmark_out = os.path.join(out_dir, branch_or_commit + ".json")
    if os.path.exists(benchmark_out):
        print(f"{benchmark_out} already exists. Skip benchmarking {branch_or_commit}.")
        return benchmark_out

    check_out(branch_or_commit)

    subprocess.check_call("pip install -e .", shell=True)

    benchmark_command = " ".join(
        ["bin/nvfuser_bench"]
        + benchmark_args
        + [f"--benchmark_out={benchmark_out}", "--benchmark_format=json"]
    )
    stdout_path = os.path.join(out_dir, branch_or_commit + ".stdout")
    stderr_path = os.path.join(out_dir, branch_or_commit + ".stderr")
    print("Running benchmark command: " + benchmark_command)
    print(f"Stdout and stderr are redirected to {stdout_path} and {stderr_path}.")
    with open(stdout_path, "w") as stdout, open(stderr_path, "w") as stderr:
        subprocess.check_call(
            benchmark_command, stdout=stdout, stderr=stderr, shell=True
        )
    print(f"The benchmark output is stored in {benchmark_out}.")

    return benchmark_out


@dataclass
class Comparison:
    name: str
    baseline_time: float
    contender_time: float
    # contender_time divided by baseline_time. Smaller is better.
    change: float
    time_unit: str

    def __str__(self):
        return f"Benchmark {self.name} changed from {self.baseline_time}{self.time_unit} to {self.contender_time}{self.time_unit} ({self.change:.2f}x)"


# Compares the two given benchmark results and produces a .json file containing
# the comparison.
def compare(baseline_out: str, contender_out: str, out_dir: str) -> str:
    baseline, _ = os.path.splitext(os.path.basename(baseline_out))
    contender, _ = os.path.splitext(os.path.basename(contender_out))
    comparison_out = os.path.join(out_dir, f"{baseline}_vs_{contender}.json")
    subprocess.check_call(
        f"third_party/benchmark/tools/compare.py -d {comparison_out} benchmarks {baseline_out} {contender_out}",
        shell=True,
    )
    return comparison_out


def load_comparison(comparison_out: str) -> list[Comparison]:
    comparisons: list[Comparison] = []
    with open(comparison_out) as f:
        data = json.loads(f.read())

    for row in data:
        for measurement in row["measurements"]:
            if row["run_type"] != "iteration":
                continue
            comparisons.append(
                Comparison(
                    name=row["name"],
                    baseline_time=measurement["real_time"],
                    contender_time=measurement["real_time_other"],
                    # measurement["time"] means (contender-baseline)/baseline.
                    # That plus 1 is the ratio that we want.
                    change=measurement["time"] + 1,
                    time_unit=row["time_unit"],
                )
            )

    return comparisons


def summarize_comparison(comparisons: Iterable[Comparison], out_dir: str) -> None:
    sorted_comparisons = sorted(comparisons, key=lambda x: x.change)

    # Print top improvements and regressions.
    num_tops = 5
    print(f"Top {num_tops} improvements:")
    for i in range(min(num_tops, len(comparisons))):
        comparison = comparisons[i]
        if comparison.change >= 1:
            break
        print(f"  {comparison}")
    print()
    print(f"Top {num_tops} regressions:")
    for i in range(min(num_tops, len(comparisons))):
        comparison = comparisons[-(i + 1)]
        if comparison.change <= 1:
            break
        print(f"  {comparison}")

    # Generate and save the histogram of time changes.
    plt.xlabel("Time change (contender/baseline)")
    plt.ylabel("Number of benchmarks")
    n, bins, patches = plt.hist([comparison.change for comparison in comparisons])
    bin_centers = np.diff(bins) / 2 + bins[:-1]
    for bar, count, x in zip(patches, n, bin_centers):
        plt.text(x, count + 0.5, str(count), ha="center", va="bottom")

    histogram_out = os.path.join(out_dir, "histogram.png")
    plt.savefig(histogram_out)
    print()
    print(f"Saved the histogram of time changes to {histogram_out}.")


def get_head_branch_or_commit() -> str:
    # Return the branch name if possible.
    head_branch_or_commit = subprocess.check_output(
        "git rev-parse --abbrev-ref HEAD", text=True, shell=True
    ).strip()

    if head_branch_or_commit != "HEAD":
        return head_branch_or_commit
    # Head is detached. Return the commit instead.
    return subprocess.check_output("git rev-parse HEAD", text=True, shell=True).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs nvfuser_bench on two commits and compares their results. See https://github.com/NVIDIA/Fuser/wiki/Developer-guide#benchmark-nvfuser for usage."
    )

    parser.add_argument("baseline", type=str, help="The baseline branch or commit")
    parser.add_argument("contender", type=str, help="The contender branch or commit")
    parser.add_argument(
        "out_dir",
        type=str,
        help="The output folder that will contain benchmark results and comparison. It will be created if doesn't exist.",
    )
    parser.add_argument(
        "benchmark_args",
        type=str,
        nargs=argparse.REMAINDER,
        help="Arguments passed to nvfuser_bench, e.g., --benchmark_filter=NvFuserScheduler",
    )
    args = parser.parse_args()

    benchmark_args = sanitize_benchmark_args(args.benchmark_args)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    original_branch_or_commit = get_head_branch_or_commit()
    try:
        baseline_out = run_benchmark(args.baseline, benchmark_args, args.out_dir)
        contender_out = run_benchmark(args.contender, benchmark_args, args.out_dir)
    finally:
        # Check out the original branch even when benchmarking failed.
        check_out(original_branch_or_commit)

    comparison_out = compare(baseline_out, contender_out, args.out_dir)
    comparisons = load_comparison(comparison_out)
    summarize_comparison(comparisons, args.out_dir)
