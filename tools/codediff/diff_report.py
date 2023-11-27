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

from dataclasses import asdict, dataclass, field, InitVar
import difflib
from enum import auto, Enum
import os
import re
import subprocess
import sys


@dataclass
class GitRev:
    full_hash: str
    diff: str | None = None
    abbrev: str = field(init=False)
    title: str = field(init=False)
    author_name: str = field(init=False)
    author_email: str = field(init=False)
    author_time: str = field(init=False)
    commit_time: str = field(init=False)

    def __post_init__(self):
        self.abbrev = (
            subprocess.run(
                ["git", "rev-parse", "--short", self.full_hash], capture_output=True
            )
            .stdout.strip()
            .decode("utf-8")
        )
        for line in (
            subprocess.run(
                ["git", "branch", "--quiet", "--color=never", self.full_hash],
                capture_output=True,
            )
            .stdout.strip()
            .decode("utf-8")
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
class LaunchParams:
    blockDim: tuple[int]
    gridDim: tuple[int]
    dynamic_smem_bytes: int


@dataclass
class CompiledKernel:
    filename: str
    code: str | None = None
    ptx: str | None = None
    ptxas_info: str | None = None
    launch_params_str: str | None = None
    launch_params: LaunchParams | None = None
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
        self.parse_launch_params()

    def parse_ptxas(self):
        # Example input:
        #
        #   ptxas info    : 307 bytes gmem
        #   ptxas info    : Compiling entry function
        #   '_ZN76_GLOBAL__N__00000000_37___tmp_kernel_pointwise_f0_c1_r0_g0_cu_8995cef2_3255329nvfuser_pointwise_f0_c1_r0_g0ENS_6TensorIfLi2ELi2EEES1_S1_'
        #   for 'sm_86'
        #   ptxas info    : Function properties for
        #   _ZN76_GLOBAL__N__00000000_37___tmp_kernel_pointwise_f0_c1_r0_g0_cu_8995cef2_3255329nvfuser_pointwise_f0_c1_r0_g0ENS_6TensorIfLi2ELi2EEES1_S1_
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
            assert self.ptxas_info is not None
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

    def parse_launch_params(self):
        # If NVFUSER_DUMP=launch_param is given we will get a line like this for every launch:
        #   Launch Parameters: BlockDim.x = 32, BlockDim.y = 2, BlockDim.z = 2, GridDim.x = 8, GridDim.y = 8, GridDim.z = -1, Smem Size = 49152
        # This is not done by default since we might have hundreds of thousands of these lines.
        # Still, if we recognize it, we will parse this info. If there are
        # multiple lines, we just check that they are all equal and if not then
        # we keep the first version and print a warning.
        if self.launch_params_str is None:
            return

        for line in self.launch_params_str.splitlines():
            m = re.search(
                r"Launch Parameters: BlockDim.x = (.*), BlockDim.y = (.*), BlockDim.z = (.*), "
                r"GridDim.x = (.*), GridDim.y = (.*), GridDim.z = (.*), Smem Size = (.*)$",
                line,
            )
            bx, by, bz, gx, gy, gz, s = m.groups()
            lp = LaunchParams((bx, by, bz), (gx, gy, gz), s)
            if self.launch_params is None:
                self.launch_params = lp
            else:
                if lp != self.launch_params:
                    # Found multiple mismatched launch params for one kernel. Only using first
                    return


@dataclass
class BenchmarkResult:
    gpu_time: float
    gpu_time_unit: str
    cpu_time: float
    cpu_time_unit: float
    iterations: int | None = None


@dataclass
class CompiledTest:
    """One grouping of kernels. A run holds multiple of these"""

    name: str
    kernels: list[CompiledKernel]
    passed: bool = True
    benchmark_result: BenchmarkResult | None = None


class CommandType(Enum):
    """Denotes what type of command was run"""

    UNKNOWN = auto()
    GOOGLETEST = auto()
    GOOGLEBENCH = auto()
    PYTEST = auto()

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, type_str: str):
        l = type_str.lower()
        if l[:3] == "unk":
            # Specified unknown. Don't print warning
            return cls.UNKNOWN
        elif l == "gtest" or l == "googletest":
            return cls.GOOGLETEST
        elif l == "gbench" or l == "googlebench":
            return cls.GOOGLEBENCH
        elif l == "pytest":
            return cls.PYTEST
        else:
            print(
                f"WARNING: Unrecognized command type '{type_str}'. Parsing as UNKNOWN.",
                file=sys.stderr,
            )
            return cls.UNKNOWN


class LogParser:
    """General parser for STDOUT of NVFuser commands

    This parser does not group into individual tests, but rather places all
    kernels into a single CompiledTest whose name is "Ungrouped Kernels".
    """

    def __init__(self, log_file: str):
        self.compile_regex()

        self.kernel_map: dict[str, CompiledTest] = {}

        self.reset_test_state()

        self.parse(log_file)

    def compile_regex(self):
        # regex for stripping ANSI color codes
        self.ansi_re = re.compile(r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]")

    def reset_kernel_state(self):
        self.current_file = None
        self.ptxas_info = ""
        self.launch_params_str = ""

    def reset_test_state(self):
        """Initialize temporary variables used during parsing pass"""
        self.reset_kernel_state()
        self.current_test = None
        self.kernels = []

    def parse(self, log_file: str):
        for line in open(log_file, "r").readlines():
            line = self.ansi_re.sub("", line.rstrip())
            self.parse_line(line)
        self.finalize()

    def finalize_kernel(self):
        if self.current_file is not None:
            k = CompiledKernel(
                self.current_file,
                ptxas_info=self.ptxas_info,
                launch_params_str=self.launch_params_str,
            )
            self.kernels.append(k)
        self.reset_kernel_state()

    def finalize_test(self, passed: bool):
        assert self.current_test is not None
        self.finalize_kernel()
        new_test = CompiledTest(self.current_test, self.kernels, passed)
        self.kernel_map[self.current_test] = new_test
        self.reset_test_state()
        return new_test

    def finalize(self):
        if len(self.kernels) > 0:
            group_name = "Ungrouped Kernels"
            self.kernel_map[group_name] = CompiledTest(group_name, self.kernels)

    def parse_line(self, line):
        """Parse a line of log. Return True if consumed"""
        if line[:10] == "PRINTING: ":
            if line[-3:] == ".cu":
                self.finalize_kernel()
                # This avoids comparing the .ptx files that are created then
                # removed by the MemoryTest.LoadCache tests
                self.current_file = line[10:]
        elif line[:6] == "ptxas ":
            # NVFUSER_DUMP=ptxas_verbose corresponds to nvcc --ptxas-options=-v
            # or --resources-usage. This always prints after printing the cuda
            # filename
            if self.current_file is None:
                print("WARNING: Cannot associate ptxas info with CUDA kernel")
                return False
            self.ptxas_info += line + "\n"
        elif line[:19] == "Launch Parameters: ":
            if self.current_file is None:
                print("WARNING: Cannot associate launch params with CUDA kernel")
                return False
            self.launch_params_str += line + "\n"
        else:
            return False
        return True


class LogParserGTest(LogParser):
    """Parse output of googletest binaries like nvfuser_tests"""

    def parse_line(self, line):
        if super().parse_line(line):
            return True

        if line[:13] == "[ RUN      ] ":
            self.current_test = line[13:]
        elif line[:13] == "[       OK ] ":
            self.finalize_test(True)
        elif line[:13] == "[  FAILED  ] ":
            if self.current_test is not None and self.current_file is not None:
                # Avoid the summary of failed tests, such as
                #   [  FAILED  ] 1 test, listed below:
                #   [  FAILED  ] NVFuserTest.FusionTuringMatmulSplitK_CUDA
                self.finalize_test(False)
        else:
            return False
        return True


class LogParserGBench(LogParser):
    """Parse output of google benchmark binaries like nvfuser_bench"""

    def compile_regex(self):
        super().compile_regex()

        # Example line:
        #   benchmark_name   34.0 us      1.53 ms   2007  /Launch_Parameters[block(2/2/32)/grid(32/2/2)/49664]
        # This is the only kind of line we match for benchmarks. Note that this is printed at the end of each benchmark
        self.result_re = re.compile(
            r"^(\S+)\s+([-+\.\d]+)\s+(\S+)\s+([-+\.\d]+)\s+(\S+)\s+(\d+).*$"
        )

    def parse_line(self, line):
        if super().parse_line(line):
            return True

        m = re.match(self.result_re, line)
        if m is not None and self.current_file is not None:
            self.current_test, time, time_unit, cpu, cpu_unit, iterations = m.groups()
            # Skip metadata which for nvfuser_bench sometimes includes LaunchParams
            # meta = m.groups()[6]
            new_test = self.finalize_test(True)
            new_test.benchmark_result = BenchmarkResult(
                time, time_unit, cpu, cpu_unit, iterations
            )
            return True
        return False


class LogParserPyTest(LogParser):
    """Parse output of pytest tests.

    Note that the tests must be run with both the -v and -s options
    """

    def compile_regex(self):
        super().compile_regex()

        self.all_test_names: list[str] | None = None

    def parse_line(self, line):
        if self.all_test_names is None:
            m = re.match(r"Running \d+ items in this shard: (.*)$", line)
            if m is not None:
                # grab the test list
                self.all_test_names = m.groups()[0].split(", ")
                return True

        if self.all_test_names is not None:
            # Try to match a line like this:
            #
            #   python_tests/test_python_frontend.py::TestNvFuserFrontend::test_pad_expanded_empty PRINTING: __tmp_kernel5.cu
            #
            # The first column is the test name, which should not have spaces.
            # After that is an ordinary line of STDOUT. In these cases we should
            # mark the beginning of a new test, and process the remainder of the
            # line as a separate line.
            testrest = line.split(maxsplit=1)
            if len(testrest) > 0 and testrest[0] in self.all_test_names:
                self.current_test = testrest[0]
                if len(testrest) > 1:
                    line = testrest[1]
                else:
                    return True

        if line == "PASSED":
            self.finalize_test(True)
        elif line == "FAILED" and self.current_test is not None:
            self.finalize_test(False)

        if super().parse_line(line):
            return True

        return False


@dataclass
class TestRun:
    """A single process that might contain many kernels, grouped into tests"""

    directory: str
    git: GitRev = field(init=False)
    name: str = field(init=False)
    command: str = field(init=False)
    command_type: CommandType = field(init=False)
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
        if not os.path.isdir(self.directory):
            print(f"ERROR: {self.directory} does not name a directory")
            sys.exit(1)

        try:
            self.name = (
                open(os.path.join(self.directory, "run_name"), "r").read().rstrip()
            )
        except FileNotFoundError:
            self.name = os.path.basename(self.directory)

        # get description of this git rev
        gitdiff = None
        try:
            gitdiff = open(os.path.join(self.directory, "git_diff"), "r").read()
        except FileNotFoundError:
            pass
        git_hash = open(os.path.join(self.directory, "git_hash"), "r").read().rstrip()
        self.git = GitRev(git_hash, diff=gitdiff)

        self.command = open(os.path.join(self.directory, "command"), "r").read()

        try:
            self.command_type = CommandType.from_string(
                open(os.path.join(self.directory, "command_type"), "r").read().rstrip()
            )
        except FileNotFoundError:
            print(
                f"WARNING: Could not find {os.path.join(self.directory, 'command_type')}. "
                "Parsing as UNKNOWN command type means kernels will be ungrouped.",
                file=sys.stderr,
            )
            self.command_type = CommandType.UNKNOWN

        try:
            self.env = ""
            for line in open(os.path.join(self.directory, "env"), "r").readlines():
                # remove $testdir which is set by compare_codegen.sh
                # NOTE: compare_codegen.sh should have already removed these lines
                if re.search(r"^testdir=", line) is None:
                    self.env += line
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
        logfile = os.path.join(self.directory, "stdout")
        if not os.path.isfile(logfile):
            raise RuntimeError(
                f"Input directory {self.directory} contains no file named 'stdout'"
            )

        if self.command_type == CommandType.GOOGLETEST:
            parser = LogParserGTest(logfile)
        elif self.command_type == CommandType.GOOGLEBENCH:
            parser = LogParserGBench(logfile)
        elif self.command_type == CommandType.PYTEST:
            parser = LogParserPyTest(logfile)
        else:
            # The base class provides a parser that groups everything into a
            # single "test" called "Ungrouped Kernels"
            parser = LogParser(logfile)

        self.kernel_map = parser.kernel_map

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
                    if re.search(r"void (nvfuser|kernel)_?\d+\b", line) is not None:
                        # we arrived at the kernel definition
                        break
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
                    # also match kernel_43 to handle new-style naming with static fusion count
                    kern.code += re.sub(r"\bnvfuser_\d+\b", "nvfuser_N", line)
        kern.code = kern.code.rstrip()
        if strip_preamble and kern.code[-1] == "}":
            # trailing curly brace is close of namespace. This will clean it up so that we have just the kernel
            kern.code = kern.code[:-1].rstrip()
        # find ptx file if it exists
        ptx_basename = os.path.splitext(basename)[0] + ".ptx"
        ptx_fullname = os.path.join(self.directory, "ptx", ptx_basename)
        try:
            kern.ptx = open(ptx_fullname, "r").read().rstrip()
        except FileNotFoundError:
            pass
        return kern


@dataclass
class KernelDiff:
    testname: str
    kernel_num: int
    kernel1: CompiledKernel
    kernel2: CompiledKernel
    diff_lines: InitVar[list[str]]
    ptx_diff_lines: InitVar[list[str] | None]
    diff: str = field(init=False)
    new_lines: int = 0
    removed_lines: int = 0
    ptx_diff: str | None = None
    new_ptx_lines: int = 0
    removed_ptx_lines: int = 0

    def __post_init__(self, diff_lines: list[str], ptx_diff_lines: list[str] | None):
        self.diff = "\n".join(diff_lines)

        for line in diff_lines:
            if line[:2] == "+ ":
                self.new_lines += 1
            elif line[:2] == "- ":
                self.removed_lines += 1

        if ptx_diff_lines is not None:
            self.ptx_diff = "\n".join(ptx_diff_lines)

            for line in ptx_diff_lines:
                if line[:2] == "+ ":
                    self.new_ptx_lines += 1
                elif line[:2] == "- ":
                    self.removed_ptx_lines += 1


@dataclass
class TestDiff:
    testname: str
    test1: CompiledTest
    test2: CompiledTest
    kernel_diffs: list[KernelDiff] | None = None


def sanitize_ptx_lines(lines: list[str]) -> list[str]:
    """Remove comments and remove kernel id"""
    sanitary_lines = []
    for l in lines:
        # Replace mangled kernel names like
        #   _ZN76_GLOBAL__N__00000000_37___tmp_kernel_pointwise_f0_c1_r0_g0_cu_8995cef2_3255329nvfuser_pointwise_f0_c1_r0_g0ENS_6TensorIfLi2ELi2EEES1_S1_
        # or
        #   _ZN76_GLOBAL__N__00000000_37___tmp_kernel_4_cu_8995cef2_3255329nvfuser_4ENS_6TensorIfLi2ELi2EEES1_S1_
        # with
        #   _ZN76_GLOBAL__N__00000000_37__kernel_cu_8995cef2_3255329kernelENS_6TensorIfLi2ELi2EEES1_S1_
        l = re.sub(
            r"(_tmp_kernel|nvfuser)(_[a-z]+)?_(\d+|f\d+_c\d*_r\d+_g\d+)", "kernel", l
        )
        # This part standardizes the _ZN76_ part which can change when more
        # digits are needed for later parts of the mangled name
        l = re.sub(r"_ZN\d+_", "_ZN76_", l)
        # This part removes the hash and timestamp of the cu file
        l = re.sub(r"kernel_cu_[0-9a-z]{8}_\d{5}", "kernel_cu_", l)

        # Remove comments. This fixes mismatches in PTX "callseq" comments, which appear to be non-repeatable.
        l = re.sub(r"//.*$", "", l)
        sanitary_lines.append(l)
    return sanitary_lines


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
    inclusion_criterion: InitVar[str] = "mismatched_cuda_or_ptx"
    preamble_diff: str = field(init=False)
    env_diff: str = field(init=False)

    def __post_init__(self, show_diffs: bool, kernel_inclusion_criterion: str):
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
                fromfile=self.run1.name,
                tofile=self.run2.name,
                n=5,
            )
        )
        if len(self.preamble_diff) > 0:
            print("Preambles differ between runs indicating changes to runtime files")

        self.env_diff = "\n".join(
            difflib.unified_diff(
                self.run1.env.splitlines(),
                self.run2.env.splitlines(),
                fromfile=self.run1.name,
                tofile=self.run2.name,
                n=5,
            )
        )

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
                    f"in {self.run1.directory} than in {self.run2.directory}. "
                    "Not showing diffs for this test.",
                    file=sys.stderr,
                )
                self.test_diffs.append(
                    TestDiff(
                        testname,
                        compiled_test1,
                        compiled_test2,
                        None,
                    )
                )

            kernel_diffs = []
            for kernel_num in range(len(compiled_test1.kernels)):
                kern1 = self.run1.get_kernel(testname, kernel_num, strip_preamble=True)
                kern2 = self.run2.get_kernel(testname, kernel_num, strip_preamble=True)
                assert kern1.code is not None
                assert kern2.code is not None

                ptx_diff_lines = None
                if kern1.ptx is not None and kern2.ptx is not None:
                    ptx_diff_lines = list(
                        difflib.unified_diff(
                            sanitize_ptx_lines(kern1.ptx.splitlines()),
                            sanitize_ptx_lines(kern2.ptx.splitlines()),
                            fromfile=self.run1.name,
                            tofile=self.run2.name,
                            n=5,
                        )
                    )

                diff_lines = list(
                    difflib.unified_diff(
                        kern1.code.splitlines(),
                        kern2.code.splitlines(),
                        fromfile=self.run1.name,
                        tofile=self.run2.name,
                        n=5,
                    )
                )
                if (
                    kernel_inclusion_criterion == "all"
                    or (
                        kernel_inclusion_criterion == "mismatched_cuda_or_ptx"
                        and diff_lines is not None
                        and len(diff_lines) > 0
                    )
                    or (
                        kernel_inclusion_criterion
                        in ["mismatched_cuda_or_ptx", "mismatched_ptx"]
                        and ptx_diff_lines is not None
                        and len(ptx_diff_lines) > 0
                    )
                ):
                    kd = KernelDiff(
                        testname,
                        kernel_num + 1,
                        kern1,
                        kern2,
                        diff_lines,
                        ptx_diff_lines=ptx_diff_lines,
                    )
                    if show_diffs:
                        print(testname, kernel_num, kd.diff)
                    self.total_num_diffs += 1
                    kernel_diffs.append(kd)

            if len(kernel_diffs) > 0:
                self.test_diffs.append(
                    TestDiff(
                        testname,
                        compiled_test1,
                        compiled_test2,
                        kernel_diffs,
                    )
                )

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
        template_dir = os.path.join(tools_dir, "templates")
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(searchpath=template_dir)
        )
        template = env.get_template("codediff.html")
        # dict_factory lets us provide custom serializations for classes like Enums
        # https://stackoverflow.com/questions/61338539/how-to-use-enum-value-in-asdict-function-from-dataclasses-module
        context = asdict(
            self,
            dict_factory=lambda data: {
                # Serialize CommandType as string so that jinja can recognize it
                field: value.name if isinstance(value, CommandType) else value
                for field, value in data
            },
        )
        context["omit_preamble"] = omit_preamble
        context["max_diffs"] = max_diffs
        head_hash = (
            subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True)
            .stdout.strip()
            .decode("utf-8")
        )
        context["tool_git"] = GitRev(head_hash)

        return template.render(context)


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
