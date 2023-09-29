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

from collections import OrderedDict
from dataclasses import dataclass, field
import difflib
import os
import re
import subprocess
import sys
from datetime import datetime
from typing import Optional, Set


@dataclass
class GitBranch:
    name: str
    # TODO: get the name of tracking branch
    # tracking_branch

    def __post_init__(self):
        # TODO: find tracking branch for this branch
        pass


@dataclass
class GitRev:
    abbrev: str
    full_hash: str = None
    in_branches: list[GitBranch] = field(default_factory=list)
    author_name: str = None
    author_email: str = None
    author_time: datetime.time = None
    commit_time: datetime.time = None

    def __post_init__(self):
        self.full_hash = (
            subprocess.run(["git", "rev-parse", self.abbrev], capture_output=True)
            .stdout.strip()
            .decode("utf-8")
        )
        for line in (
            subprocess.run(
                ["git", "branch", "--quiet", "--color=never", self.full_hash],
                capture_output=True,
            )
            .stdout.strip()
            .splitlines()
        ):
            # Possible output:
            #
            #     main
            #     * scalar_seg_edges
            #
            # In this case, we have checked out the HEAD of the
            # scalar_seg_edges branch. Here we just strip the *.
            if line[0] == "*":
                line = line[2:]
                in_branches.append(line)

        date_fmt = "%Y/%m/%d %H:%M:%S %z"
        git_show = (
            lambda fmt: subprocess.run(
                [
                    "git",
                    "show",
                    "--no-patch",
                    f"--format={fmt}",
                    f"--date=format:{date_fmt}",
                    self.full_hash,
                ],
                capture_output=True,
            )
            .stdout.strip()
            .decode("utf-8")
        )
        self.author_name = git_show("%an")
        self.author_email = git_show("%ae")

        # Get date and time for this commit in datetime format
        get_datetime = lambda time_str: datetime.strptime(time_str, date_fmt)
        self.author_time = get_datetime(git_show("%ad"))
        self.commit_time = get_datetime(git_show("%cd"))


@dataclass
class TestRun:
    directory: str
    git_rev: GitRev = None
    run_name: str = None
    command: str = None
    exit_code: int = None
    # map from name of test to list of kernel base filenames
    kernel_map: dict[str, list[str]] = field(default_factory=dict)
    # collecting the preamble lets us skip it when diffing, and lets us compare
    # only the preamble between runs
    preamble: str = None
    # lets us seek past preamble
    preamble_size_bytes: int = None

    def __post_init__(self):
        # get description of this git rev
        abbrev = os.path.basename(os.path.dirname(os.path.abspath(self.directory)))
        self.git_rev = GitRev(abbrev)

        self.command = open(os.path.join(self.directory, "command"), "r").read()

        # check that command includes "nvfuser_tests"
        if self.command.find("nvfuser_tests") == -1:
            print(
                "ERROR: Command does not appear to be nvfuser_tests. Aborting.",
                file=sys.stderr,
            )
            sys.exit(1)

        self.exit_code = int(open(os.path.join(self.directory, "exitcode"), "r").read())

        self.compute_kernel_map()

        self.find_preamble()

        print("End of TestRun post_init")

    def compute_kernel_map(self):
        """
        Compute a map from test name to list of cuda filenames
        """
        # first find the stdout log file
        logfile = None
        for fname in os.listdir(self.directory):
            if fname.find("stdout") != -1:
                if logfile is not None:
                    raise RuntimeError(
                        f"Input directory {self.directory} contains multiple "
                        'possible logs (filenames containing "stdout")'
                    )
                logfile = os.path.join(self.directory, fname)
        if logfile is None:
            raise RuntimeError(
                f"Input directory {self.directory} contains no log (filenames "
                'containing "stdout")'
            )

        # regex for stripping ANSI color codes
        ansi_re = re.compile(r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]")
        current_test = None
        current_files = []
        for line in open(logfile, "r").readlines():
            line = ansi_re.sub("", line.strip())
            if line[:13] == "[ RUN      ] ":
                current_test = line[13:]
            elif line[:13] == "[       OK ] ":
                # Finalize test
                assert current_test is not None
                self.kernel_map[current_test] = current_files
                current_test = None
                current_files = []
            elif line[:10] == "PRINTING: ":
                if line[-3:] == ".cu":
                    # This avoids comparing the .ptx files that are created then
                    # removed by the MemoryTest.LoadCache tests
                    current_files.append(line[10:])

    def find_preamble(self):
        """Look for common preamble in collected kernels"""
        preamble_lines = []
        first = True
        for cufile in os.listdir(os.path.join(self.directory, "cuda")):
            cufile_full = os.path.join(self.directory, "cuda", cufile)
            with open(cufile_full, "r") as f:
                for i, line in enumerate(f.readlines()):
                    line = line.rstrip()
                    # we set nvfuser_index_t in the preamble. We ignore that change for the purposes of this diff
                    if line[:8] == "typedef " and line[-17:] == " nvfuser_index_t;":
                        line = "typedef int nvfuser_index_t; // NOTE: hardcoded to int for easier diffing"
                    if first:
                        preamble_lines.append(line)
                    elif i >= len(preamble_lines) or preamble_lines[i] != line:
                        break
                    self.preamble_size_bytes = f.tell()
                preamble_lines = preamble_lines[:i]
            if self.preamble_size_bytes == 0:
                # early return if preamble is determined to be empty
                break
            first = False
        self.preamble = "\n".join(preamble_lines)

    def get_kernel(self, test_name, kernel_number, strip_preamble=True) -> str:
        """Get a string of the kernel, optionally stripping the preamble"""
        basename = self.kernel_map[test_name][kernel_number]
        fullname = os.path.join(self.directory, "cuda", basename)
        with open(fullname, "r") as f:
            if strip_preamble:
                f.seek(self.preamble_size_bytes)
            return f.read().strip()


@dataclass
class KernelDiff:
    testname: str
    kernel_num: int
    code1: str
    code2: str
    diff: str


# Lets us maintain test order
class LastUpdatedOrderedDict(OrderedDict):
    """Just an ordered dict with insertion at the end"""

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)


@dataclass
class TestDifferences:
    run1: TestRun
    run2: TestRun
    differing_tests: LastUpdatedOrderedDict[str, list[KernelDiff]] = field(
        default_factory=list
    )
    new_tests: list[str] = field(default_factory=list)
    removed_tests: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.run1.command != self.run2.command:
            print("WARNING: commands differ between runs", file=sys.stderr)
            print(f"  {self.run1.directory}: {self.run1.command}", file=sys.stderr)
            print(f"  {self.run2.directory}: {self.run2.command}", file=sys.stderr)

        if self.run1.exit_code != self.run1.exit_code:
            print(
                f"WARNING: Exit codes {self.run1.exit_code} and {self.run2.exit_code} do not match.",
                file=sys.stderr,
            )

        if self.run1.preamble != self.run2.preamble:
            print("Preambles differ between runs indicating changes to runtime files")

        differing_tests_set = set()
        for testname, kernels1 in self.run1.kernel_map.items():
            if testname not in self.run2.kernel_map:
                removed_tests.append(testname)
                continue

            kernels2 = self.run2.kernel_map[testname]

            if len(kernels1) != len(kernels2):
                print(
                    f"WARNING: Test {testname} has different number of kernels "
                    f"in {dir1} than in {dir2}. Not showing diffs.",
                    file=sys.stderr,
                )
                if testname not in differing_tests_set:
                    differing_tests.append(testname)
                    differing_tests_set.add(testname)

            for kernel_num in range(len(kernels1)):
                code1 = self.run1.get_kernel(testname, kernel_num, strip_preamble=True)
                code2 = self.run2.get_kernel(testname, kernel_num, strip_preamble=True)

                lines1 = code1.splitlines()
                lines2 = code2.splitlines()

                diff_str = "\n".join(difflib.unified_diff(
                    lines1,
                    lines2,
                    fromfile=self.run1.git_rev.abbrev,
                    tofile=self.run2.git_rev.abbrev,
                    n=5,
                ))
                if len(diff_str) > 0:
                    print(testname, kernel_num, diff_str)
                    diff_obj = KernelDiff(testname, kernel_num, code1, code2, diff_str)
                    if testname in self.differing_tests:
                        self.differing_tests[testname].append(diff_obj)
                    else:
                        self.differing_tests[testname] = [diff_obj]

        for testname, kernels2 in self.run2.kernel_map.items():
            if testname not in self.run1.kernel_map:
                new_tests.append(testname)

    def __len__(self):
        return len(self.differing_tests)

    def to_dict(self):
        """Convert to hierarchical dict format for use with jinja"""
        d = {}
        d["git1"] = self.run1.git_rev.to_dict()
        d["git2"] = self.run2.git_rev.to_dict()

    def generate_html(self, output_file: str) -> str:
        """Return a self-contained HTML string summarizing the codegen comparison"""
        import jinja2
        import pygments
        from pygments.lexers import CppLexer, DiffLexer
        from pygments.formatters import HtmlFormatter

        env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath="."))
        template = env.get_template(
            os.path.join(os.path.dirname(__file__), "templates", "codediff.html")
        )
        context = self.to_dict()
        context["pygments_style_defs"] = HtmlFormatter().get_style_defs(".highlight")

        return template.render(template_vars)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        epilog="This command must be run from within a git checkout of the NVFuser repo."
    )
    parser.add_argument("dir1", help="Directory containing stdout-*.log and cuda/")
    parser.add_argument("dir2", help="Directory containing stdout-*.log and cuda/")
    parser.add_argument("--html", action="store_true", help="Write HTML file?")
    parser.add_argument(
        "-o", "--output-file", help="Location of HTML file output if -h is given."
    )
    args = parser.parse_args()

    test_diffs = TestDifferences(TestRun(args.dir1), TestRun(args.dir2))

    if args.html:
        output_file = args.output_file
        if output_file is None:
            # determine default output file
            get_abbrev = lambda d: os.path.basename(os.path.dirname(os.path.abspath(d)))
            abbrev1 = get_abbrev(args.dir1)
            abbrev2 = get_abbrev(args.dir2)
            run_name = os.path.basename(os.path.abspath(args.dir1))
            output_file = f"codediff_{abbrev1}_{abbrev2}_{run_name}.html"
        with open(output_file, "w") as f:
            f.write(differing_tests.generate_html())

    if len(differing_tests) == 0:
        print("No differences found in overlapping tests!")
    else:
        print("Differences found in the following tests:")
        for t in differing_tests:
            print(f"  {t}")

    exit(len(differing_tests))
