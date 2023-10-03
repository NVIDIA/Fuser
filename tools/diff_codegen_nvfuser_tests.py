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
import re
import subprocess
import sys

# precompile an RE we'll apply over and over


def get_test_map(directory: str) -> dict[str, list[str]]:
    """
    Get a map from test name to list of cuda filenames
    """
    # first find the stdout log file
    logfile = None
    for fname in os.listdir(directory):
        if fname.find("stdout") != -1:
            if logfile is not None:
                raise RuntimeError(
                    f"Input directory {directory} contains multiple "
                    'possible logs (filenames containing "stdout")'
                )
            logfile = os.path.join(directory, fname)
    if logfile is None:
        raise RuntimeError(
            f"Input directory {directory} contains no log (filenames "
            'containing "stdout")'
        )

    # regex for stripping ANSI color codes
    ansi_re = re.compile(r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]")
    kernel_map = {}
    current_test = None
    current_files = []
    for line in open(logfile, "r").readlines():
        line = ansi_re.sub("", line.strip())
        if line[:13] == "[ RUN      ] ":
            current_test = line[13:]
        elif line[:13] == "[       OK ] ":
            # Finalize test
            assert current_test is not None
            kernel_map[current_test] = current_files
            current_test = None
            current_files = []
        elif line[:10] == "PRINTING: ":
            if line[-3:] == ".cu":
                # This avoids comparing the .ptx files that are created then
                # removed by the MemoryTest.LoadCache tests
                current_files.append(line[10:])

    return kernel_map


def diff_nvfuser_tests_dirs(dir1: str, dir2: str):
    """
    Given directories for two
    """
    # check that commands are equal
    command1 = open(os.path.join(dir1, "command"), "r").read()
    command2 = open(os.path.join(dir2, "command"), "r").read()

    if command1 != command2:
        print("WARNING: commands differ between runs", file=sys.stderr)
        print(f"  {dir1}: {command1}", file=sys.stderr)
        print(f"  {dir2}: {command2}", file=sys.stderr)

    # check that command includes "nvfuser_tests"
    if command1.find("nvfuser_tests") == -1:
        print(
            "ERROR: Command does not appear to be nvfuser_tests. Aborting.",
            file=sys.stderr,
        )
        sys.exit(1)

    # check that exit codes are equal
    exitcode1 = open(os.path.join(dir1, "exitcode"), "r").read()
    exitcode2 = open(os.path.join(dir2, "exitcode"), "r").read()
    if exitcode1 != exitcode2:
        print(
            f"WARNING: Exit codes {exitcode1} and {exitcode2} do not match.",
            file=sys.stderr,
        )

    # get a map from test name to list of .cu files for each directory
    map1 = get_test_map(dir1)
    map2 = get_test_map(dir2)

    differing_tests = set()
    for testname, kernels1 in map1.items():
        if testname not in map2:
            print(
                f"WARNING: Test {testname} present in {dir1} but not in {dir2}",
                file=sys.stderr,
            )
            continue

        kernels2 = map2[testname]

        if len(kernels1) != len(kernels2):
            print(
                f"WARNING: Test {testname} has different number of kernels "
                f"in {dir1} than in {dir2}. Not showing diffs.",
                file=sys.stderr,
            )
            differing_tests.add(testname)

        for k1, k2 in zip(kernels1, kernels2):
            f1 = os.path.join(dir1, "cuda", k1)
            f2 = os.path.join(dir2, "cuda", k2)
            # -U50 gives us plenty of context
            # -I "void kernel" ignores mismatches in kernel signature line
            #    The intention is to avoid false positives from differently
            #    numbered kernels, but this can also hide true differences if
            #    the kernel signature changes.
            args = ["diff", "-U50", "-I", "void kernel", f1, f2]
            ret = subprocess.run(args, capture_output=True)
            if ret.returncode != 0:
                print(testname, ret.args)
                print(ret.stdout.decode("utf-8"))
                differing_tests.add(testname)

    for testname, kernels2 in map2.items():
        if testname not in map1:
            print(
                f"WARNING: Test {testname} present in {dir2} but not in {dir1}",
                file=sys.stderr,
            )

    return differing_tests


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dir1", help="Directory containing stdout-*.log and cuda/")
    parser.add_argument("dir2", help="Directory containing stdout-*.log and cuda/")
    args = parser.parse_args()

    differing_tests = diff_nvfuser_tests_dirs(args.dir1, args.dir2)

    if len(differing_tests) == 0:
        print("No differences found in overlapping tests!")
    else:
        print("Differences found in the following tests:")
        for t in differing_tests:
            print(f"  {t}")

    exit(len(differing_tests))
