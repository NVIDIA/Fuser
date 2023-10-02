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
from dataclasses import dataclass, field, InitVar
import difflib
import os
import re
import subprocess
import sys


@dataclass
class GitRev:
    abbrev: str
    title: str = field(init=False)
    full_hash: str = field(init=False)
    author_name: str = field(init=False)
    author_email: str = field(init=False)
    author_time: str = field(init=False)
    commit_time: str = field(init=False)

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

        def git_show(fmt) -> str:
            return (
                subprocess.run(
                    [
                        "git",
                        "show",
                        "--no-patch",
                        f"--format={fmt}",
                        self.full_hash,
                    ],
                    capture_output=True,
                )
                .stdout.strip()
                .decode("utf-8")
            )

        self.title = git_show("%s")
        self.author_name = git_show("%an")
        self.author_email = git_show("%ae")
        self.author_time = git_show("%ad")
        self.commit_time = git_show("%cd")

    def to_dict(self):
        return {
            "abbrev": self.abbrev,
            "full_hash": self.full_hash,
            "author_name": self.author_name,
            "author_email": self.author_email,
            "author_time": str(self.author_time),
            "commit_time": str(self.commit_time),
            "title": self.title,
        }


@dataclass
class CompiledKernel:
    filename: str
    ptxas_info: str | None = None
    gmem_bytes: int | None = None
    smem_bytes: int | None = None
    # maps from constant memory bank to bytes
    cmem_bank_to_bytes: dict[int, int] | None = None
    registers: int | None = None
    target_arch: str | None = None
    stack_frame_bytes: int | None = None
    spill_store_bytes: int | None = None
    spill_load_bytes: int | None = None

    def __post_init__(self):
        self.parse_ptxas()

    def parse_ptxas(self):
        # Example input:
        #
        #   ptxas info    : 307 bytes gmem
        #   ptxas info    : Compiling entry function '_ZN11CudaCodeGen7kernel1ENS_6TensorIfLi2ELi2EEES1_S1_' for 'sm_86'
        #   ptxas info    : Function properties for _ZN11CudaCodeGen7kernel1ENS_6TensorIfLi2ELi2EEES1_S1_
        #   ptxas         .     0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
        #   ptxas info    : Used 203 registers, 16 bytes smem, 472 bytes cmem[0], 8 bytes cmem[2]
        #
        # Here we parse this into the fields presented, and we replace the
        # mangled kernel name since it includes the kernel number and is
        # useless for the purposes of diffing since the kernel signature is
        # already included.
        if self.ptxas_info is None:
            return

        self.ptxas_info = re.sub(r"\b_Z.*\b", "[mangled kernel name]", self.ptxas_info)

        def find_unique_int(pattern) -> int | None:
            g = re.search(r"(\d+) bytes gmem").groups()
            return None if len(g) == 0 else int(g[0])

        self.stack_frame_bytes = find_unique_int(r"(\d+) bytes stack frame")
        self.spill_store_bytes = find_unique_int(r"(\d+) bytes spill stores")
        self.spill_load_bytes = find_unique_int(r"(\d+) bytes spill loads")
        self.registers = find_unique_int(r"(\d+) registers")
        self.gmem_bytes = find_unique_int(r"(\d+) bytes gmem")
        self.smem_bytes = find_unique_int(r"(\d+) bytes smem")

        cmem = {}
        for m in re.finditer(r"(\d+) bytes cmem\[(\d+)\]", self.ptxas_info):
            nbytes, bank = m.groups()
            cmem[bank] = nbytes
        if len(cmem) != 0:
            self.cmem_bank_to_bytes = cmem


@dataclass
class TestRun:
    directory: str
    git_rev: GitRev = field(init=False)
    run_name: str = field(init=False)
    command: str = field(init=False)
    exit_code: int = field(init=False)
    # map from name of test to list of kernel base filenames
    kernel_map: dict[str, list[str]] = field(init=False)
    # collecting the preamble lets us skip it when diffing, and lets us compare
    # only the preamble between runs
    preamble: str = field(init=False)
    # The following lets us skip preamble when loading kernels. Note that the
    # preamble can change length due to differing index types, so we can't rely
    # on f.seek()
    preamble_size_lines: int = field(init=False)

    def __post_init__(self):
        self.run_name = os.path.basename(self.directory)

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
        self.kernel_map = {}
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
                    current_files.append(CompiledKernel(line[10:]))
            elif line[:6] == "ptxas ":
                # NVFUSER_DUMP=ptxas_verbose corresponds to nvcc --ptxas-options=-v or --resources-usage
                # This always prints after printing the cuda filename
                if len(current_files) == 0:
                    print("WARNING: Cannot associate ptxas info with CUDA kernel")
                    continue
                if current_files[-1].ptxas_info is None:
                    current_files[-1].ptxas_info = line
                else:
                    current_files[-1].ptxas_info += line + "\n"

    def find_preamble(self):
        """Look for common preamble in collected kernels"""
        preamble_lines = []
        first = True
        files_processed = 0  # limit how many files to check
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
                preamble_lines = preamble_lines[:i]
            if len(preamble_lines) == 0:
                # early return if preamble is determined to be empty
                break
            first = False
            files_processed += 1
            if files_processed >= 50:
                break
        self.preamble_size_lines = len(preamble_lines)
        self.preamble = "\n".join(preamble_lines)

    def to_dict(self):
        d = {}
        d["name"] = self.run_name
        d["command"] = self.command
        d["exitcode"] = self.exit_code
        d["git"] = self.git_rev.to_dict()
        # NOTE: including preamble can add 5-6MB to the file size.
        # TODO: Optionally skip including the preamble in the report in order
        # to reduce file size
        d["preamble"] = self.preamble
        return d

    def get_kernel(self, test_name, kernel_number, strip_preamble=True) -> str:
        """Get a string of the kernel, optionally stripping the preamble"""
        kern = self.kernel_map[test_name][kernel_number]
        basename = kern.filename
        fullname = os.path.join(self.directory, "cuda", basename)
        code = ""
        with open(fullname, "r") as f:
            for i, line in enumerate(f.readlines()):
                if not strip_preamble or i >= self.preamble_size_lines:
                    # replace kernel934 with kernel1 to facilitate diffing
                    code += re.sub(r"\bkernel\d+\b", "kernelN", line)
        code = code.rstrip()
        if strip_preamble and code[-1] == "}":
            # trailing curly brace is close of namespace. This will clean it up so that we have just the kernel
            code = code[:-1].rstrip()
        return code


@dataclass
class KernelDiff:
    testname: str
    kernel_num: int
    code1: str
    code2: str
    diff: str

    def to_dict(self):
        return {
            "number": self.kernel_num,
            "code1": self.code1,
            "code2": self.code2,
            "diff": self.diff,
        }


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
    # either a list of diffs, or different numbers of kernels present
    differing_tests: LastUpdatedOrderedDict[
        str, tuple[int, int] | list[KernelDiff]
    ] = field(init=False)
    new_tests: list[str] = field(init=False)
    removed_tests: list[str] = field(init=False)
    total_num_diffs: int = field(init=False)
    show_diffs: InitVar[bool] = False

    def __post_init__(self, show_diffs: bool):
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

        self.differing_tests = {}
        self.new_tests = []
        self.removed_tests = []
        self.total_num_diffs = 0
        for testname, kernels1 in self.run1.kernel_map.items():
            if testname not in self.run2.kernel_map:
                self.removed_tests.append(testname)
                continue

            kernels2 = self.run2.kernel_map[testname]

            if len(kernels1) != len(kernels2):
                print(
                    f"WARNING: Test {testname} has different number of kernels "
                    f"in {dir1} than in {dir2}. Not showing diffs for this test.",
                    file=sys.stderr,
                )
                self.differing_tests[testname] = (len(kernels1), len(kernels2))

            for kernel_num in range(len(kernels1)):
                code1 = self.run1.get_kernel(testname, kernel_num, strip_preamble=True)
                code2 = self.run2.get_kernel(testname, kernel_num, strip_preamble=True)

                lines1 = code1.splitlines()
                lines2 = code2.splitlines()

                diff_str = "\n".join(
                    difflib.unified_diff(
                        lines1,
                        lines2,
                        fromfile=self.run1.git_rev.abbrev,
                        tofile=self.run2.git_rev.abbrev,
                        n=5,
                    )
                )
                if len(diff_str) > 0:
                    if show_diffs:
                        print(testname, kernel_num, diff_str)
                    self.total_num_diffs += 1
                    diff_obj = KernelDiff(testname, kernel_num, code1, code2, diff_str)
                    if testname in self.differing_tests:
                        self.differing_tests[testname].append(diff_obj)
                    else:
                        self.differing_tests[testname] = [diff_obj]

        for testname, kernels2 in self.run2.kernel_map.items():
            if testname not in self.run1.kernel_map:
                self.new_tests.append(testname)

    def to_dict(self):
        """Convert to hierarchical dict format for use with jinja"""
        d = {}
        d["run1"] = self.run1.to_dict()
        d["run2"] = self.run2.to_dict()

        d["preamble_diff"] = "\n".join(
            difflib.unified_diff(
                self.run1.preamble.splitlines(),
                self.run2.preamble.splitlines(),
                fromfile=self.run1.git_rev.abbrev,
                tofile=self.run2.git_rev.abbrev,
                n=5,
            )
        )

        d["test_diffs"] = []
        for testname, diffs in self.differing_tests.items():
            if isinstance(diffs, tuple):
                # differing numbers of kernels produced by this test
                d["test_diffs"].append(diffs)
            else:
                d["test_diffs"].append(
                    {
                        "name": testname,
                        "kernels": [di.to_dict() for di in diffs],
                    }
                )

        d["new_tests"] = []
        for testname in self.new_tests:
            kernels_code = []
            for i in range(len(self.run2.kernel_map[testname])):
                kernels_code.append(
                    self.run2.get_kernel(testname, i, strip_preamble=True)
                )
            d["new_tests"].append(
                {
                    "name": testname,
                    "code": kernels_code,
                }
            )

        d["removed_tests"] = []
        for testname in self.removed_tests:
            kernels_code = []
            for i in range(len(self.run1.kernel_map[testname])):
                kernels_code.append(
                    self.run1.get_kernel(testname, i, strip_preamble=True)
                )
            d["removed_tests"].append(
                {
                    "name": testname,
                    "code": kernels_code,
                }
            )

        d["total_num_diffs"] = self.total_num_diffs

        return d

    def generate_html(self, omit_preamble: bool, max_diffs: bool) -> str:
        """Return a self-contained HTML string summarizing the codegen comparison"""
        import jinja2

        tools_dir = os.path.dirname(__file__)
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=tools_dir))
        template = env.get_template("templates/codediff.html")
        context = self.to_dict()
        context["omit_preamble"] = omit_preamble
        context["max_diffs"] = max_diffs

        return template.render(context)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        epilog="This command must be run from within a git checkout of the NVFuser repo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dir1", help="Directory containing stdout-*.log and cuda/")
    parser.add_argument("dir2", help="Directory containing stdout-*.log and cuda/")
    parser.add_argument("--html", action="store_true", help="Write HTML file?")
    parser.add_argument(
        "--show-diffs", action="store_true", help="Print diffs to STDOUT?"
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
    args = parser.parse_args()

    test_diffs = TestDifferences(TestRun(args.dir1), TestRun(args.dir2))

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
                test_diffs.generate_html(
                    omit_preamble=args.html_omit_preamble, max_diffs=args.html_max_diffs
                )
            )

    num_differing_kernels = 0
    for k, v in test_diffs.differing_tests.items():
        if isinstance(v, list):
            num_differing_kernels += len(v)

    if len(test_diffs.differing_tests) == 0:
        print("No differences found in overlapping tests!")
    else:
        print(
            num_differing_kernels,
            "from",
            len(test_diffs.differing_tests),
            "tests found",
        )
    if len(test_diffs.new_tests) > 0:
        print(len(test_diffs.new_tests), "new tests found")
    if len(test_diffs.removed_tests) > 0:
        print(len(test_diffs.removed_tests), "removed tests found")

    exit(len(test_diffs.differing_tests))
