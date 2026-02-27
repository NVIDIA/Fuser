import argparse
import concurrent.futures
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import time
from enum import Enum
from pathlib import Path
from typing import Any, List, NamedTuple, Optional, Pattern

# Nvfuser directory root
result = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"],
    stdout=subprocess.PIPE,
    check=True,
)
NVFUSER_ROOT = result.stdout.decode("utf-8").strip()
IS_WINDOWS: bool = os.name == "nt"


# Returns '/usr/local/include/python<version number>'
def get_python_include_dir() -> str:
    return sysconfig.get_paths()["include"]


def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, flush=True, **kwargs)


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    path: Optional[str]
    line: Optional[int]
    char: Optional[int]
    code: str
    severity: LintSeverity
    name: str
    original: Optional[str]
    replacement: Optional[str]
    description: Optional[str]


def as_posix(name: str) -> str:
    return name.replace("\\", "/") if IS_WINDOWS else name


# c10/core/DispatchKey.cpp:281:26: error: 'k' used after it was moved [bugprone-use-after-move]
RESULTS_RE: Pattern[str] = re.compile(
    r"""(?mx)
    ^
    (?P<file>.*?):
    (?P<line>\d+):
    (?:(?P<column>-?\d+):)?
    \s(?P<severity>\S+?):?
    \s(?P<message>.*)
    \s(?P<code>\[.*\])
    $
    """
)


def run_command(
    args: List[str],
) -> subprocess.CompletedProcess[str]:
    logging.debug("$ %s", " ".join(args))
    start_time = time.monotonic()
    try:
        return subprocess.run(
            args,
            capture_output=True,
            check=True,
            text=True,
        )
    finally:
        end_time = time.monotonic()
        logging.debug("took %dms", (end_time - start_time) * 1000)


# Severity is either "error" or "note":
# https://github.com/python/mypy/blob/8b47a032e1317fb8e3f9a818005a6b63e9bf0311/mypy/errors.py#L46-L47
severities = {
    "error": LintSeverity.ERROR,
    "warning": LintSeverity.WARNING,
}


def clang_search_dirs() -> List[str]:
    # Compilers are ordered based on fallback preference
    # We pick the first one that is available on the system
    compilers = ["clang", "gcc", "cpp", "cc"]
    compilers = [c for c in compilers if shutil.which(c) is not None]
    if len(compilers) == 0:
        raise RuntimeError(f"None of {compilers} were found")
    compiler = compilers[0]

    result = subprocess.run(
        [compiler, "-E", "-x", "c++", "-", "-v"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
    )
    stderr = result.stderr.decode().strip().split("\n")
    search_start = r"#include.*search starts here:"
    search_end = r"End of search list."

    append_path = False
    search_paths = []
    for line in stderr:
        if re.match(search_start, line):
            if append_path:
                continue
            else:
                append_path = True
        elif re.match(search_end, line):
            break
        elif append_path:
            search_paths.append(line.strip())

    return search_paths


# NVFUSER_USE_PCH=ON somehow requires the CUDA include dir e.g.
# "/usr/local/cuda-13.0/targets/x86_64-linux/include". I'll try to add that in
# a future PR.
include_dir = [
    get_python_include_dir(),
    os.path.join(NVFUSER_ROOT, "third_party/pybind11/include"),
] + clang_search_dirs()

include_args = []
for dir in include_dir:
    include_args += ["--extra-arg", f"-I{dir}"]


def check_file(
    filename: str,
    binary: str,
    build_dir: Path,
) -> List[LintMessage]:
    try:
        proc = run_command(
            [binary, f"-p={build_dir}", *include_args, filename],
        )
    except subprocess.CalledProcessError as err:
        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code="CLANGTIDY",
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=(
                    f"Failed due to {err}\n\nstdout:\n{err.stdout}\nstderr:\n{err.stderr}"
                ),
            )
        ]
    lint_messages = []
    try:
        # Change the current working directory to the build directory, since
        # clang-tidy will report files relative to the build directory.
        saved_cwd = os.getcwd()
        os.chdir(build_dir)

        for match in RESULTS_RE.finditer(proc.stdout):
            # Convert the reported path to an absolute path.
            abs_path = str(Path(match["file"]).resolve())
            message = LintMessage(
                path=abs_path,
                name=match["code"],
                description=match["message"],
                line=int(match["line"]),
                char=int(match["column"])
                if match["column"] is not None and not match["column"].startswith("-")
                else None,
                code="CLANGTIDY",
                severity=severities.get(match["severity"], LintSeverity.ERROR),
                original=None,
                replacement=None,
            )
            lint_messages.append(message)
    finally:
        os.chdir(saved_cwd)

    return lint_messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="clang-tidy wrapper linter.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--build-dir",
        "--build_dir",
        required=True,
        help=(
            "Where the compile_commands.json file is located. "
            "Gets passed to clang-tidy -p"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET
        if args.verbose
        else logging.DEBUG
        if len(args.filenames) < 1000
        else logging.INFO,
        stream=sys.stderr,
    )

    try:
        llvm_bindir = subprocess.check_output(
            ["llvm-config", "--bindir"], text=True
        ).strip()
    except FileNotFoundError:
        err_msg = LintMessage(
            path="<none>",
            line=None,
            char=None,
            code="CLANGTIDY",
            severity=LintSeverity.ERROR,
            name="command-failed",
            original=None,
            replacement=None,
            description="llvm-config doesn't exist (is LLVM installed?).",
        )
        print(json.dumps(err_msg._asdict()), flush=True)
        exit(0)

    binary_path = os.path.join(llvm_bindir, "clang-tidy")
    if not os.path.exists(binary_path):
        err_msg = LintMessage(
            path="<none>",
            line=None,
            char=None,
            code="CLANGTIDY",
            severity=LintSeverity.ERROR,
            name="command-failed",
            original=None,
            replacement=None,
            description=f"Could not find clang-tidy binary at {binary_path} (is LLVM installed?)",
        )
        print(json.dumps(err_msg._asdict()), flush=True)
        exit(0)

    abs_build_dir = Path(args.build_dir).resolve()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=os.cpu_count(),
        thread_name_prefix="Thread",
    ) as executor:
        futures = {
            executor.submit(
                check_file,
                filename,
                binary_path,
                abs_build_dir,
            ): filename
            for filename in args.filenames
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                for lint_message in future.result():
                    print(json.dumps(lint_message._asdict()), flush=True)
            except Exception:
                logging.critical('Failed at "%s".', futures[future])
                raise


if __name__ == "__main__":
    main()
