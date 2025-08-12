# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Find corresponding .cu files for matching tests, even when new tests are
introduced between two commits. Diffs are displayed and the return value is the
number of mismatched corresponding tests.

Tests are skipped if they produce different numbers of .cu files, or if they
exist in only one of the given runs.

Example usage:
    python tools/diff_codegen_nvfuser_tests.py \
            codegen_comparison/{$commit1,$commit2}/binary_tests
"""

import os

from codediff import TestRun, TestDifferences

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        epilog="This command must be run from within a git checkout of the NVFuser repo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dir1", help="Directory containing 'stdout' and 'cuda/'")
    parser.add_argument("dir2", help="Directory containing 'stdout' and 'cuda/'")
    parser.add_argument(
        "--hide-env",
        action="store_true",
        help="Hide environment variables and nvcc versions in output?",
    )
    parser.add_argument("--html", action="store_true", help="Write HTML file?")
    parser.add_argument(
        "--hide-diffs",
        "--no-print-diff",
        action="store_true",
        help="Print diffs to STDOUT?",
    )
    parser.add_argument(
        "--kernel-inclusion-criterion",
        "-i",
        choices=("mismatched_cuda_or_ptx", "mismatched_ptx", "all"),
        default="mismatched_cuda_or_ptx",
        help="Which kernels should we include?",
    )
    parser.add_argument(
        "--html-max-diffs",
        default=200,
        type=int,
        help="Limit number of included kernel diffs in HTML output to this many (does not affect exit code).",
    )
    parser.add_argument(
        "--html-omit-preamble",
        action="store_true",
        help="Omit the preamble in HTML output?",
    )
    parser.add_argument(
        "-o", "--output-file", help="Location of HTML file output if -h is given."
    )
    parser.add_argument(
        "--json",
        help="Location to write JSON output, if given",
    )
    args = parser.parse_args()

    td = TestDifferences(
        TestRun(args.dir1),
        TestRun(args.dir2),
        show_diffs=not args.hide_diffs,
        inclusion_criterion=args.kernel_inclusion_criterion,
    )

    if args.hide_env:
        td.hide_env()

    if args.html:
        output_file = args.output_file
        if output_file is None:
            # determine default output file
            def get_abbrev(d):
                return os.path.basename(os.path.dirname(os.path.abspath(d)))

            abbrev1 = get_abbrev(args.dir1)
            abbrev2 = get_abbrev(args.dir2)
            run_name = os.path.basename(os.path.abspath(args.dir1))
            output_file = f"codediff_{abbrev1}_{abbrev2}_{run_name}.html"
        with open(output_file, "w") as f:
            f.write(
                td.generate_html(
                    omit_preamble=args.html_omit_preamble, max_diffs=args.html_max_diffs
                )
            )

    if args.json is not None:
        import json

        d = asdict(td)
        # clean up the dict a bit by removing temporary data structures
        del d["run1"]["kernel_map"]
        del d["run2"]["kernel_map"]
        json.dump(d, open(args.json, "w"), indent=2)

    if len(td.test_diffs) == 0:
        print("No differences found in overlapping tests!")
    else:
        print(
            td.total_num_diffs,
            "kernel differences from",
            len(td.test_diffs),
            "tests found",
        )
    if len(td.new_tests) > 0:
        print(len(td.new_tests), "new tests found")
    if len(td.removed_tests) > 0:
        print(len(td.removed_tests), "removed tests found")

    # Return 1 if preamble or any kernels are changed, else 0
    exit(1 if len(td.test_diffs) > 0 or len(td.preamble_diff) > 0 else 0)
