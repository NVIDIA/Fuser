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

from dataclasses import asdict, dataclass, field, InitVar
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


@dataclass
class CompiledKernel:
    filename: str
    code: str | None = None
    ptxas_info: str | None = None
    gmem_bytes: int = 0
    smem_bytes: int = 0
    cmem_bank_bytes: list[int] | None = None
    registers: int | None = None
    stack_frame_bytes: int = 0
    spill_store_bytes: int = 0
    spill_load_bytes: int = 0
    mangled_name: str | None = None
    arch: str | None = None
    index_type: str | None = None

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

        m = re.search(r"Compiling entry function '(.*)' for '(.*)'", self.ptxas_info)
        if m is not None:
            self.mangled_name, self.arch = m.groups()

        def find_unique_int(pattern) -> int | None:
            m = re.search(pattern, self.ptxas_info)
            return 0 if m is None else int(m.groups()[0])

        self.stack_frame_bytes = find_unique_int(r"(\d+) bytes stack frame")
        self.spill_store_bytes = find_unique_int(r"(\d+) bytes spill stores")
        self.spill_load_bytes = find_unique_int(r"(\d+) bytes spill loads")
        self.registers = find_unique_int(r"(\d+) registers")
        self.gmem_bytes = find_unique_int(r"(\d+) bytes gmem")
        self.smem_bytes = find_unique_int(r"(\d+) bytes smem")

        self.cmem_bank_bytes = []
        cmem_banks = 0
        for m in re.finditer(r"(\d+) bytes cmem\[(\d+)\]", self.ptxas_info):
            nbytes_str, bank_str = m.groups()
            bank = int(bank_str)
            if len(self.cmem_bank_bytes) <= bank:
                self.cmem_bank_bytes += [0] * (bank + 1 - len(self.cmem_bank_bytes))
            self.cmem_bank_bytes[bank] = int(nbytes_str)
            cmem_banks += 1


@dataclass
class CompiledTest:
    name: str
    kernels: list[CompiledKernel] | None = None


@dataclass
class TestRun:
    directory: str
    git: GitRev = field(init=False)
    run_name: str = field(init=False)
    command: str = field(init=False)
    exit_code: int = field(init=False)
    env: str = field(init=False)
    gpu_names: str = field(init=False)
    nvcc_version: str = field(init=False)
    # map from name of test to list of kernel base filenames
    kernel_map: dict[str, CompiledTest] = field(default_factory=dict)
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
        self.git = GitRev(abbrev)

        self.command = open(os.path.join(self.directory, "command"), "r").read()

        # check that command includes "nvfuser_tests"
        if self.command.find("nvfuser_tests") == -1:
            print(
                "ERROR: Command does not appear to be nvfuser_tests. Aborting.",
                file=sys.stderr,
            )
            sys.exit(1)

        try:
            self.env = open(os.path.join(self.directory, "env"), "r").read()
        except FileNotFoundError:
            self.env = None

        try:
            self.nvcc_version = open(
                os.path.join(self.directory, "nvcc_version"), "r"
            ).read()
        except FileNotFoundError:
            self.nvcc_version = None

        try:
            self.gpu_names = list(
                open(os.path.join(self.directory, "gpu_names"), "r").readlines()
            )
        except FileNotFoundError:
            self.gpu_names = None

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
        current_file = None
        ptxas_info = ""
        kernels = []

        def finalize_kernel():
            nonlocal ptxas_info
            nonlocal current_file
            if current_file is not None:
                kernels.append(CompiledKernel(current_file, ptxas_info=ptxas_info))
            ptxas_info = ""
            current_file = None

        def finalize_test():
            nonlocal current_test
            nonlocal kernels
            assert current_test is not None
            finalize_kernel()
            self.kernel_map[current_test] = CompiledTest(current_test, kernels)
            current_test = None
            kernels = []

        for line in open(logfile, "r").readlines():
            line = ansi_re.sub("", line.strip())
            if line[:13] == "[ RUN      ] ":
                current_test = line[13:]
            elif line[:13] == "[       OK ] ":
                finalize_test()
            elif line[:10] == "PRINTING: ":
                if line[-3:] == ".cu":
                    finalize_kernel()
                    # This avoids comparing the .ptx files that are created then
                    # removed by the MemoryTest.LoadCache tests
                    current_file = line[10:]
            elif line[:6] == "ptxas ":
                # NVFUSER_DUMP=ptxas_verbose corresponds to nvcc --ptxas-options=-v or --resources-usage
                # This always prints after printing the cuda filename
                if current_file is None:
                    print("WARNING: Cannot associate ptxas info with CUDA kernel")
                    continue
                ptxas_info += line + "\n"

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
                        line = "typedef int nvfuser_index_t; // NOTE: index type hard-coded as int for display only"
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

    def get_kernel(
        self, test_name, kernel_number, strip_preamble=True
    ) -> CompiledKernel:
        """Get a string of the kernel, optionally stripping the preamble"""
        kern = self.kernel_map[test_name].kernels[kernel_number]
        basename = kern.filename
        fullname = os.path.join(self.directory, "cuda", basename)
        kern.code = ""
        with open(fullname, "r") as f:
            for i, line in enumerate(f.readlines()):
                if kern.index_type is None:
                    m = re.search(r"typedef\s+(\S*)\s+nvfuser_index_t;", line)
                    if m is not None:
                        kern.index_type = m.groups()[0]
                if not strip_preamble or i >= self.preamble_size_lines:
                    # replace kernel934 with kernel1 to facilitate diffing
                    kern.code += re.sub(r"\bkernel\d+\b", "kernelN", line)
        kern.code = kern.code.rstrip()
        if strip_preamble and kern.code[-1] == "}":
            # trailing curly brace is close of namespace. This will clean it up so that we have just the kernel
            kern.code = kern.code[:-1].rstrip()
        return kern


@dataclass
class KernelDiff:
    testname: str
    kernel_num: int
    kernel1: CompiledKernel
    kernel2: CompiledKernel
    diff_lines: InitVar[list[str]]
    diff: str = field(init=False)
    new_lines: int = 0
    removed_lines: int = 0

    def __post_init__(self, diff_lines: list[str]):
        self.diff = "\n".join(diff_lines)

        for line in diff_lines:
            if line[:2] == "+ ":
                self.new_lines += 1
            elif line[:2] == "- ":
                self.removed_lines += 1


@dataclass
class TestDiff:
    testname: str
    kernel_diffs: list[KernelDiff] | None = None
    kernel_number_mismatch: tuple[int, int] | None = None


@dataclass
class TestDifferences:
    run1: TestRun
    run2: TestRun
    # either a list of diffs, or different numbers of kernels present
    test_diffs: list[TestDiff] = field(default_factory=list)
    new_tests: list[CompiledTest] = field(default_factory=list)
    removed_tests: list[CompiledTest] = field(default_factory=list)
    total_num_diffs: int = 0
    show_diffs: InitVar[bool] = False
    preamble_diff: str = field(init=False)

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

        self.preamble_diff = "\n".join(
            difflib.unified_diff(
                self.run1.preamble.splitlines(),
                self.run2.preamble.splitlines(),
                fromfile=self.run1.git.abbrev,
                tofile=self.run2.git.abbrev,
                n=5,
            )
        )
        if len(self.preamble_diff) > 0:
            print("Preambles differ between runs indicating changes to runtime files")

        for testname, compiled_test1 in self.run1.kernel_map.items():
            if testname not in self.run2.kernel_map:
                compiled_test1.kernels = [
                    self.run1.get_kernel(testname, i)
                    for i in range(len(compiled_test1.kernels))
                ]
                self.removed_tests.append(compiled_test1)
                continue

            compiled_test2 = self.run2.kernel_map[testname]

            if len(compiled_test1.kernels) != len(compiled_test2.kernels):
                print(
                    f"WARNING: Test {testname} has different number of kernels "
                    f"in {dir1} than in {dir2}. Not showing diffs for this test.",
                    file=sys.stderr,
                )
                self.test_diffs.append(
                    TestDiff(
                        testname,
                        None,
                        len(compiled_test1.kernels),
                        len(compiled_test2.kernels),
                    )
                )

            kernel_diffs = []
            for kernel_num in range(len(compiled_test1.kernels)):
                kern1 = self.run1.get_kernel(testname, kernel_num, strip_preamble=True)
                kern2 = self.run2.get_kernel(testname, kernel_num, strip_preamble=True)

                diff_lines = list(
                    difflib.unified_diff(
                        kern1.code.splitlines(),
                        kern2.code.splitlines(),
                        fromfile=self.run1.git.abbrev,
                        tofile=self.run2.git.abbrev,
                        n=5,
                    )
                )
                if len(diff_lines) > 0:
                    kd = KernelDiff(testname, kernel_num, kern1, kern2, diff_lines)
                    if show_diffs:
                        print(testname, kernel_num, kd.diff)
                    self.total_num_diffs += 1
                    kernel_diffs.append(kd)

            if len(kernel_diffs) > 0:
                self.test_diffs.append(TestDiff(testname, kernel_diffs))

        for testname, compiled_test2 in self.run2.kernel_map.items():
            if testname not in self.run1.kernel_map:
                compiled_test2.kernels = [
                    self.run2.get_kernel(testname, i)
                    for i in range(len(compiled_test2.kernels))
                ]
                self.new_tests.append(compiled_test2)

    def hide_env(self):
        """Remove private information like env vars and lib versions"""
        self.run1.env = None
        self.run2.env = None
        self.run1.nvcc_version = None
        self.run2.nvcc_version = None

    def generate_html(self, omit_preamble: bool, max_diffs: bool) -> str:
        """Return a self-contained HTML string summarizing the codegen comparison"""
        import jinja2

        tools_dir = os.path.dirname(__file__)
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=tools_dir))
        template = env.get_template("templates/codediff.html")
        context = asdict(self)
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
    parser.add_argument(
        "--hide-env",
        action="store_true",
        help="Hide environment variables and nvcc versions in output?",
    )
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
    parser.add_argument(
        "--json",
        help="Location to write JSON output, if given",
    )
    args = parser.parse_args()

    td = TestDifferences(TestRun(args.dir1), TestRun(args.dir2))

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
