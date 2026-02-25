#!/usr/bin/env python3
"""
Scan all TransformerEngine source files for usages of sm_100a-specific guards,
find the enclosing function/kernel for each usage, and write a CSV with
GitHub permalinks.

Guards searched:
  Macro-level (in any file):
    ARCH_BLACKWELL_FAMILY
    ARCH_HAS_STOCHASTIC_ROUNDING
    NVTE_CUDA_ARCH_MATCHES   (with FamilySpecific/ArchSpecific)
    __CUDA_ARCH_HAS_FEATURE__(SM100_ALL)
    _ENABLE_MXFMA
    is_sm_100f (local variable alias)
  Variable-level (local aliases set from the macros above):
    is_blackwell
    is_blackwell_arch
    has_fp4
    has_rs

Enclosing-function detection uses libclang (real C++ AST) with a regex
fallback for lines where the AST lookup returns nothing.
"""

import re
import csv
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# libclang setup
# ---------------------------------------------------------------------------
try:
    import clang.cindex as _cx
except ImportError:
    raise SystemExit(
        "ERROR: libclang Python bindings are required but not installed.\n"
        "Install with: pip install libclang"
    )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_SOURCE_DIR = "/opt/pytorch/TransformerEngine"
DEFAULT_OUTPUT_CSV = "/opt/pytorch/nvfuser/te_sm100a_guard_usages.csv"
GITHUB_REPO = "https://github.com/NVIDIA/TransformerEngine"
SOURCE_EXTENSIONS = {".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc"}
SKIP_DIRS = {"build", ".git", "3rdparty"}

# ---------------------------------------------------------------------------
# Guard patterns: (guard_label, compiled_regex)
# We match the *definition* or *use* of variables/macros derived from the
# three fundamental sm_100a guards.
# ---------------------------------------------------------------------------
GUARD_PATTERNS = [
    # Primary macro definitions / direct uses
    ("ARCH_BLACKWELL_FAMILY", re.compile(r"\bARCH_BLACKWELL_FAMILY\b")),
    ("ARCH_HAS_STOCHASTIC_ROUNDING", re.compile(r"\bARCH_HAS_STOCHASTIC_ROUNDING\b")),
    (
        "NVTE_CUDA_ARCH_MATCHES(FamilySpecific/ArchSpecific<100>)",
        re.compile(
            r"NVTE_CUDA_ARCH_MATCHES\s*\([^)]*(?:FamilySpecific|ArchSpecific)\s*<\s*10[013]"
        ),
    ),
    (
        "__CUDA_ARCH_HAS_FEATURE__(SM100_ALL/SM101_ALL)",
        re.compile(r"__CUDA_ARCH_HAS_FEATURE__\s*\(\s*SM10[01]"),
    ),
    ("_ENABLE_MXFMA", re.compile(r"\b_ENABLE_MXFMA\b")),
    # Local variable aliases
    (
        "is_blackwell (local alias for ARCH_BLACKWELL_FAMILY)",
        re.compile(r"\bis_blackwell\b"),
    ),
    (
        "is_blackwell_arch (local alias for ARCH_BLACKWELL_FAMILY)",
        re.compile(r"\bis_blackwell_arch\b"),
    ),
    ("has_fp4 (local alias for ARCH_BLACKWELL_FAMILY)", re.compile(r"\bhas_fp4\b")),
    (
        "has_rs (local alias for ARCH_HAS_STOCHASTIC_ROUNDING)",
        re.compile(r"\bhas_rs\b"),
    ),
    ("is_sm_100f (local alias for FamilySpecific<100>)", re.compile(r"\bis_sm_100f\b")),
]

# ---------------------------------------------------------------------------
# libclang-based enclosing-function detection
# ---------------------------------------------------------------------------
_CLANG_INDEX = None  # created once, reused across all files
_CLANG_TU_CACHE: dict = {}  # filepath -> TranslationUnit

# ---------------------------------------------------------------------------
# Virtual stub header: satisfies CUDA built-in declarations that clang
# (in C++ mode) doesn't know about, without pulling in the full CUDA RT.
# Intentionally omits types that TE source files define themselves
# (e.g. bf16x2, fp16x2 which ptx.cuh defines as FPx2<> aliases).
# ---------------------------------------------------------------------------
_CUDA_STUB_PATH = "/tmp/__te_guard_scan_cuda_stubs__.h"
_CUDA_STUB_SRC = """\
// Auto-generated stub for libclang CUDA analysis (TransformerEngine guard scan)
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
typedef __nv_bfloat16 bf16;
typedef __half fp16;
struct __cuda_dim3 { unsigned x, y, z; };
extern __cuda_dim3 threadIdx, blockIdx, blockDim, gridDim;
inline void __syncthreads() {}
inline void __syncwarp(unsigned = 0xffffffff) {}
template<typename T> inline T __shfl_sync(unsigned, T, int, int=32) { return T{}; }
template<typename T> inline T __shfl_xor_sync(unsigned, T, int, int=32) { return T{}; }
template<typename T> inline T __shfl_down_sync(unsigned, T, int, int=32) { return T{}; }
inline int __clz(unsigned) { return 0; }
inline int __clzll(unsigned long long) { return 0; }
inline int __popc(unsigned) { return 0; }
inline unsigned __ballot_sync(unsigned, int) { return 0; }
inline unsigned __activemask() { return 0; }
inline float fmaxf(float a, float b) { return a > b ? a : b; }
inline float fminf(float a, float b) { return a < b ? a : b; }
inline unsigned long long __cvta_generic_to_shared(const void*) { return 0; }
inline int   __float_as_int(float f)  { int   r; __builtin_memcpy(&r, &f, 4); return r; }
inline float __int_as_float(int   i)  { float r; __builtin_memcpy(&r, &i, 4); return r; }
inline float __frcp_rn(float x) { return 1.0f / x; }
template<typename T> inline T atomicMax(T* p, T v) { if (*p < v) *p = v; return *p; }
template<typename T> inline T atomicMin(T* p, T v) { if (*p > v) *p = v; return *p; }
// FP4 types (Blackwell CUDA) - not in older CUDA header installs
struct __nv_fp4_e2m1  { unsigned char  x; };
struct __nv_fp4x2_e2m1 { unsigned char  x; };  // sizeof = 1
struct __nv_fp4x4_e2m1 { unsigned short x; };  // sizeof = 2
#include <cstdio>
#include <cmath>
using std::isinf; using std::isnan;
// CUTLASS function-attribute macros.  The cutlass submodule may not be
// initialised on this machine, so libclang never sees cutlass.h and leaves
// these macros undefined.  Without definitions libclang mis-parses
// CUTLASS_DEVICE-decorated functions as variable declarations and therefore
// cannot determine the enclosing function for guard uses inside them.
// We define them as empty / plain `inline` so libclang sees standard C++
// function signatures (the CUDA-specific __device__ / __host__ qualifiers
// are not understood in -x c++ mode and cause the same mis-parse).
#define CUTLASS_DEVICE
#define CUTLASS_HOST_DEVICE   inline
#define CUTLASS_HOST
#define CUTLASS_GLOBAL
// Minimal CUTLASS type stubs so libclang can parse template-based return
// types such as cutlass::Array<cutlass::float_e2m1_t, 8>.  Without these,
// the `<` in the return type is parsed as a less-than operator, the
// function body `{` becomes a namespace body, and the enclosing-function
// lookup returns nothing for guard uses inside those functions.
namespace cutlass {
  using float_e2m1_t  = unsigned char;
  using float_e4m3_t  = unsigned char;
  using float_e5m2_t  = unsigned char;
  using bfloat16_t    = unsigned short;
  using half_t        = unsigned short;
  template<typename Element, int N> struct Array { Element data[N]; };
  template<typename T> struct NumericConverter { };
  template<typename T> struct NumericArrayConverter { };
} // namespace cutlass
"""

# ---------------------------------------------------------------------------
# Inline-asm stripper
# ---------------------------------------------------------------------------
_ASM_RE = re.compile(r"\b(?:asm|__asm__|__asm)\b\s*(?:volatile|__volatile__)?\s*\(")


def _strip_asm_blocks(source: str) -> str:
    """Replace asm(...); with (void)0; to avoid inline PTX asm parse errors.

    Uses bracket counting so it handles multi-line / nested parens correctly.
    The line *count* is preserved (newlines inside asm blocks are kept) so
    libclang line numbers remain accurate.
    """
    result: list[str] = []
    i = 0
    n = len(source)
    while i < n:
        m = _ASM_RE.search(source, i)
        if not m:
            result.append(source[i:])
            break
        result.append(source[i : m.start()])
        # Walk from the opening '(' to the matching ')'
        j = m.end() - 1  # index of opening '('
        depth = 1
        j += 1
        while j < n and depth > 0:
            c = source[j]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            j += 1
        # Skip optional trailing ';'
        k = j
        while k < n and source[k] in " \t":
            k += 1
        if k < n and source[k] == ";":
            k += 1
        # Preserve all newlines inside the asm block so line numbers stay correct
        newlines = source[m.start() : k].count("\n") * "\n"
        result.append("(void)0;" + newlines)
        i = k
    return "".join(result)


_CLANG_PARSE_ARGS = [
    "-x",
    "c++",
    "-std=c++17",
    "-ferror-limit=0",
    "-fno-spell-checking",
    "-I/usr/lib/llvm-18/lib/clang/18/include",
    "-I/usr/include/c++/13",
    "-I/usr/include/aarch64-linux-gnu/c++/13",
    "-I/usr/include",
    "-I/usr/local/cuda/include",
    "-I/opt/pytorch/TransformerEngine",
    "-I/opt/pytorch/TransformerEngine/transformer_engine",
    "-I/opt/pytorch/TransformerEngine/3rdparty/cutlass/include",
    # Stub out CUDA compile-time arch macros.
    # _SPECIFIC_ values must equal __CUDA_ARCH__ to satisfy the
    # static_assert inside ptx::ArchSpecific<N>::compatible().
    "-D__CUDA_ARCH__=1000",
    "-D__CUDA_ARCH_SPECIFIC__=1000",
    "-D__CUDA_ARCH_FAMILY_SPECIFIC__=1000",
    # Enable FP4 type block in ptx.cuh (L537-918) — macro is never defined in
    # the CUDA headers on this machine, so we define it here to force libclang
    # to see all functions in that range.
    "-DFP4_TYPE_SUPPORTED=1",
    # Force-include the CUDA builtins stub before any file is parsed
    f"-include{_CUDA_STUB_PATH}",
]

_CLANG_FUNC_KINDS = frozenset(
    [
        _cx.CursorKind.FUNCTION_DECL,
        _cx.CursorKind.CXX_METHOD,
        _cx.CursorKind.FUNCTION_TEMPLATE,
        _cx.CursorKind.CONSTRUCTOR,
        _cx.CursorKind.DESTRUCTOR,
    ]
)


def _get_tu(filepath: str):
    """Parse *filepath* with libclang, caching the TU for reuse.

    Two-pass strategy to eliminate inline asm parse errors:
    1. Parse with only the target file asm-stripped.  Collect all files that
       still produce errors (e.g. included CUDA headers with '=h' constraints).
    2. Re-parse with every error-producing file asm-stripped as well.

    Line numbers are preserved (newlines inside asm blocks are kept) so
    libclang source locations remain accurate.
    """
    global _CLANG_INDEX
    if filepath in _CLANG_TU_CACHE:
        return _CLANG_TU_CACHE[filepath]
    if _CLANG_INDEX is None:
        _CLANG_INDEX = _cx.Index.create()

    def _unsaved(paths: set[str]) -> list[tuple[str, str]]:
        result = [(_CUDA_STUB_PATH, _CUDA_STUB_SRC)]  # always include stub
        for p in paths:
            try:
                src = Path(p).read_text(errors="replace")
            except OSError:
                continue
            result.append((p, _strip_asm_blocks(src)))
        return result

    # Pass 1 — strip the target file
    uf = _unsaved({filepath})
    tu = _CLANG_INDEX.parse(
        filepath,
        unsaved_files=uf,
        args=_CLANG_PARSE_ARGS,
        options=_cx.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
    )

    # Find any other files that still have errors after pass 1
    error_files = {
        d.location.file.name
        for d in tu.diagnostics
        if d.severity >= _cx.Diagnostic.Error and d.location.file
    }
    extra = error_files - {filepath}

    if extra:
        # Pass 2 — strip the target + all discovered error files
        uf2 = _unsaved({filepath} | extra)
        tu = _CLANG_INDEX.parse(
            filepath,
            unsaved_files=uf2,
            args=_CLANG_PARSE_ARGS,
            options=_cx.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
        )

    _CLANG_TU_CACHE[filepath] = tu
    return tu


# Per-file cache: filepath → {lineno: func_name}
# Built lazily on first query for a file; subsequent lookups are O(1).
_LINE_MAP_CACHE: dict[str, dict[int, str]] = {}


def _build_line_func_map(filepath: str) -> dict[int, str]:
    """Parse *filepath* once with libclang and build a complete lineno→function
    mapping for every line that falls inside a function/method body.

    Innermost enclosing function wins (last written wins in a single DFS pass
    that visits children after recording the parent, so deeper nodes overwrite
    shallower ones for the lines they cover).
    """
    if filepath in _LINE_MAP_CACHE:
        return _LINE_MAP_CACHE[filepath]

    tu = _get_tu(filepath)
    result: dict[int, str] = {}

    # Iterative DFS over the AST.
    # Key optimisation: skip entire subtrees whose root node originates from a
    # *different* file (e.g. included headers).  The vast majority of AST nodes
    # in a CUDA/CUTLASS TU belong to headers — pruning them early cuts the walk
    # from O(100k nodes) to O(file-local nodes), making this fast enough to run
    # on every source file in find_callers without a timeout.
    # Innermost function wins: process parents before children so children can
    # overwrite the parent name for the lines they own.
    global_kernels: set[str] = set()
    stack = [tu.cursor]
    while stack:
        node = stack.pop()
        ext = node.extent
        # Only descend into nodes that touch the target file.
        if ext.start.file and ext.start.file.name != filepath:
            continue
        if node.kind in _CLANG_FUNC_KINDS:
            if ext.start.file and node.spelling:
                for ln in range(ext.start.line, ext.end.line + 1):
                    result[ln] = node.spelling
                # Side-effect: detect __global__ kernels while we're here.
                if (
                    node.kind == _cx.CursorKind.FUNCTION_DECL
                    and filepath not in _GLOBAL_FUNCS_CACHE
                ):
                    for tok in node.get_tokens():
                        if tok.spelling == "__global__":
                            global_kernels.add(node.spelling)
                            break
        stack.extend(node.get_children())

    # Only write _GLOBAL_FUNCS_CACHE if it wasn't already set by
    # _collect_global_kernel_names for this file.
    if filepath not in _GLOBAL_FUNCS_CACHE:
        _GLOBAL_FUNCS_CACHE[filepath] = global_kernels

    _LINE_MAP_CACHE[filepath] = result
    return result


def enclosing_function(filepath: str, target_lineno: int) -> str:
    """Return the name of the function/kernel that contains *target_lineno*.

    Uses a per-file lineno→function map that is built once via a single
    libclang AST walk and then cached, making repeated lookups O(1).
    """
    return _build_line_func_map(filepath).get(target_lineno, "")


def should_skip(path: Path) -> bool:
    for part in path.parts:
        if part in SKIP_DIRS:
            return True
    return False


def make_github_link(repo: str, commit: str, rel_path: str, lineno: int) -> str:
    return f"{repo}/blob/{commit}/{rel_path}#L{lineno}"


def get_commit(source_dir: str) -> str:
    import subprocess

    try:
        r = subprocess.run(
            ["git", "-C", source_dir, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return r.stdout.strip()
    except Exception:
        return "HEAD"


import os


# ---------------------------------------------------------------------------
# Shared helper: pre-compute #define continuation lines for a file's text
# ---------------------------------------------------------------------------


def _define_lines(lines: list[str]) -> set[int]:
    """Return 1-based line numbers that are part of any #define block."""
    result: set[int] = set()
    in_define = False
    for i, ln in enumerate(lines):
        if re.match(r"\s*#\s*define\b", ln):
            result.add(i + 1)
            in_define = ln.rstrip("\r").endswith("\\")
        elif in_define:
            result.add(i + 1)
            in_define = ln.rstrip("\r").endswith("\\")
        else:
            in_define = False
    return result


def scan(source_dir: str, commit: str) -> list[dict]:
    root = Path(source_dir)
    results = []
    seen = set()  # deduplicate (file, lineno, guard)

    for fpath in sorted(root.rglob("*")):
        if fpath.suffix not in SOURCE_EXTENSIONS:
            continue
        if should_skip(fpath):
            continue
        try:
            text = fpath.read_text(errors="replace")
        except OSError:
            continue

        lines = text.splitlines()
        rel = str(fpath.relative_to(root))

        # Pre-compute lines that are part of #define blocks (including
        # backslash-continuation lines).  Guard uses on these lines are macro
        # *definitions*, not real code inside a function, and must be skipped.
        dl = _define_lines(lines)

        for guard_label, pattern in GUARD_PATTERNS:
            for lineno, line in enumerate(lines, 1):
                if not pattern.search(line):
                    continue
                # Skip preprocessor directive lines — these have no enclosing
                # C++ function: #define, #if, #elif, #else, #endif, etc.
                # Also skip continuation lines of multi-line #define blocks.
                if re.match(r"\s*#\s*(?:define|if|elif|else|endif)\b", line):
                    continue
                if lineno in dl:
                    continue
                key = (rel, lineno, guard_label)
                if key in seen:
                    continue
                seen.add(key)

                func = enclosing_function(str(fpath), lineno)
                link = make_github_link(GITHUB_REPO, commit, rel, lineno)
                results.append(
                    {
                        "guard": guard_label,
                        "function": func,
                        "github_link": link,
                    }
                )

    # Sort by guard then github_link
    results.sort(key=lambda r: (r["guard"], r["github_link"]))
    return results


# ---------------------------------------------------------------------------
# Fast regex-based enclosing-function finder (used for caller discovery)
# ---------------------------------------------------------------------------
# Matches common C/CUDA function definition openings, e.g.:
#   void foo(
#   __global__ static void foo(
#   __device__ __forceinline__ RetType foo(
#   template<...> RetType foo(
# We scan *backwards* from the call site to find the nearest such line
# whose brace-depth is 0 at that point (i.e. it's a top-level definition).
_FUNC_DEF_RE = re.compile(
    r"^\s*"
    r"(?:template\s*<[^>]*>\s*)?"  # optional template<...>
    r"(?:(?:__device__|__host__|__global__|__forceinline__|"
    r"static|inline|virtual|explicit|constexpr|"
    r"CUTLASS_DEVICE|CUTLASS_HOST_DEVICE|CUTLASS_GLOBAL)\s+)*"
    r"(?:[\w:<>,*& ]+?\s+)?"  # return type (greedy but lazy)
    r"(\w+)"  # ← function name (group 1)
    r"\s*(?:<[^>]*>)?\s*\(",  # optional <tmpl> then (
)


def _enclosing_function_regex(lines: list[str], lineno: int) -> str:
    """Scan backwards from *lineno* (1-based) to find the nearest function name."""
    # Track brace depth going backwards; the function header is the last line
    # before we reach depth 0 that looks like a function definition.
    depth = 0
    for i in range(lineno - 1, -1, -1):
        ln = lines[i]
        depth -= ln.count("{") - ln.count("}")
        if depth <= 0:
            m = _FUNC_DEF_RE.match(ln)
            if m:
                return m.group(1)
            # Also check the previous line (CUTLASS_DEVICE on its own line)
            if i > 0:
                prev = lines[i - 1].strip()
                if prev in (
                    "CUTLASS_DEVICE",
                    "CUTLASS_HOST_DEVICE",
                    "CUTLASS_GLOBAL",
                    "__device__ __forceinline__",
                    "__device__",
                    "__host__",
                ):
                    m2 = _FUNC_DEF_RE.match(ln)
                    if m2:
                        return m2.group(1)
    return ""


def find_callers(
    callee_names: set[str],
    source_dir: str,
    commit: str,
    seen_pairs: set[tuple[str, str]],
) -> list[dict]:
    """Search *source_dir* for call sites of any function in *callee_names*.

    Returns a list of dicts with keys: caller, callee, github_link.
    *seen_pairs* is the set of (caller, callee) pairs already recorded across
    all previous levels; newly found pairs are added to it in-place.
    """
    root = Path(source_dir)
    results = []
    seen_keys: set[tuple] = set()  # (rel, lineno, callee) dedup within this call

    # Combined pattern — word-boundary match for any of the callee names
    combined = re.compile(
        r"\b(" + "|".join(re.escape(n) for n in sorted(callee_names)) + r")\b"
    )

    for fpath in sorted(root.rglob("*")):
        if fpath.suffix not in SOURCE_EXTENSIONS:
            continue
        if should_skip(fpath):
            continue
        try:
            text = fpath.read_text(errors="replace")
        except OSError:
            continue

        lines = text.splitlines()
        rel = str(fpath.relative_to(root))
        dl = _define_lines(lines)

        for lineno, line in enumerate(lines, 1):
            m = combined.search(line)
            if not m:
                continue
            # Skip preprocessor lines
            if re.match(r"\s*#\s*(?:define|if|elif|else|endif)\b", line):
                continue
            if lineno in dl:
                continue

            callee = m.group(1)
            key = (rel, lineno, callee)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            caller = enclosing_function(str(fpath), lineno)
            # Skip: no enclosing function, or the line is inside the callee's
            # own definition / a recursive call
            if not caller or caller == callee:
                continue

            pair = (caller, callee)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            link = make_github_link(GITHUB_REPO, commit, rel, lineno)
            results.append(
                {
                    "caller": caller,
                    "callee": callee,
                    "github_link": link,
                }
            )

    results.sort(key=lambda r: (r["caller"], r["callee"]))
    return results


# ---------------------------------------------------------------------------
# BFS call-chain builder
# ---------------------------------------------------------------------------

# Per-file cache: filepath → set of __global__ kernel names in that file.
# Populated lazily by _build_line_func_map as a side-effect of its AST walk.
# Never pre-populated by a separate scan — checked only AFTER find_callers
# has already parsed (and thus populated the cache for) the relevant files.
_GLOBAL_FUNCS_CACHE: dict[str, set[str]] = {}


def _is_global_kernel(name: str) -> bool:
    """Return True if *name* has been identified as a __global__ kernel.

    Uses the lazily-built _GLOBAL_FUNCS_CACHE that is populated as a side
    effect of _build_line_func_map (called by enclosing_function, which is
    called by find_callers).  This means by the time we check whether a newly
    discovered caller is a GPU entry point, its file has already been parsed
    and its kernels recorded — so no additional parsing is needed.
    """
    return any(name in kernels for kernels in _GLOBAL_FUNCS_CACHE.values())


def build_call_chain(
    l0_results: list[dict],
    source_dir: str,
    commit: str,
    max_depth: int = 6,
) -> list[list[dict]]:
    """BFS outward from the L0 guard-using functions.

    Returns *levels* where levels[0] is l0_results and levels[i] (i≥1) is the
    list of {caller, callee, github_link} dicts found at depth i.

    Termination conditions (in addition to max_depth):
    - A caller that is a __global__ kernel is a GPU entry point; no host code
      calls it directly, so we don't search further above it.
    - Cycle detection: once a function has appeared anywhere in the chain it is
      never added to the search frontier again, preventing infinite loops.
    """
    # _GLOBAL_FUNCS_CACHE is lazily populated by _build_line_func_map inside
    # find_callers, so we do NOT pre-scan for __global__ kernels here.
    # Instead we check _is_global_kernel() AFTER each find_callers call, when
    # the relevant files have already been parsed and cached.

    levels: list[list[dict]] = [l0_results]

    # all_names: every function seen at any level — used for cycle detection.
    # If a caller is already in this set, it has been (or will be) explored,
    # so we never add it to the frontier again.
    all_names: set[str] = {r["function"] for r in l0_results if r["function"]}
    all_pairs: set[tuple[str, str]] = set()

    current_names = set(all_names)

    for depth in range(1, max_depth + 1):
        if not current_names:
            break
        print(f"  [L{depth}] searching callers of {len(current_names)} function(s)…")
        level_results = find_callers(current_names, source_dir, commit, all_pairs)
        if not level_results:
            print(f"  [L{depth}] no new callers found — stopping.")
            break

        levels.append(level_results)
        print(f"  [L{depth}] found {len(level_results)} call edge(s).")

        new_names = {r["caller"] for r in level_results} - all_names
        all_names |= new_names

        # Don't search above __global__ kernels — they are GPU entry points.
        # _is_global_kernel() queries _GLOBAL_FUNCS_CACHE, which was populated
        # by _build_line_func_map for every file parsed during find_callers above.
        kernel_hits = {n for n in new_names if _is_global_kernel(n)}
        if kernel_hits:
            print(
                f"  [L{depth}] stopping at __global__ kernel(s): {sorted(kernel_hits)}"
            )

        current_names = new_names - kernel_hits

        if not current_names:
            break

    return levels


# ---------------------------------------------------------------------------
# CSV output — one file per level
# ---------------------------------------------------------------------------


def write_level_csvs(
    levels: list[list[dict]],
    output_dir: str,
    base: str,
) -> list[str]:
    """Write per-level CSV files; returns list of file paths written."""
    paths: list[str] = []

    # L0 — guard usages
    l0_path = os.path.join(output_dir, f"{base}_L0.csv")
    with open(l0_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["function", "github_link", "guard"])
        writer.writeheader()
        writer.writerows(levels[0])
    paths.append(l0_path)
    print(f"  L0  → {l0_path}  ({len(levels[0])} rows)")

    # L1+ — caller edges (deduplicated to unique pairs)
    for i, level_data in enumerate(levels[1:], 1):
        seen_pairs: dict[tuple, dict] = {}
        for r in level_data:
            k = (r["caller"], r["callee"])
            if k not in seen_pairs:
                seen_pairs[k] = r
        rows = sorted(seen_pairs.values(), key=lambda x: (x["caller"], x["callee"]))
        lpath = os.path.join(output_dir, f"{base}_L{i}.csv")
        with open(lpath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["caller", "callee", "github_link"])
            writer.writeheader()
            writer.writerows(rows)
        paths.append(lpath)
        print(f"  L{i}  → {lpath}  ({len(rows)} rows)")

    return paths


# ---------------------------------------------------------------------------
# Markdown / Mermaid call-graph output
# ---------------------------------------------------------------------------


def _mermaid_id(name: str) -> str:
    """Convert a function name to a valid Mermaid node identifier."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def write_callgraph_md(
    levels: list[list[dict]],
    output_path: str,
) -> None:
    """Write a Markdown file with one Mermaid call-graph section per guard."""

    # ------------------------------------------------------------------ #
    # Build the complete directed edge set: caller → callee               #
    # And collect per-guard L0 functions                                  #
    # ------------------------------------------------------------------ #
    # guard_short → set of L0 function names
    guard_funcs: dict[str, set[str]] = {}
    # (caller, callee) edges from L1+
    all_edges: list[tuple[str, str]] = []
    # function → representative github link (first seen wins)
    func_link: dict[str, str] = {}

    for r in levels[0]:
        fn = r["function"]
        if not fn:
            continue
        guard_short = r["guard"].split(" ")[0]  # e.g. "ARCH_HAS_STOCHASTIC_ROUNDING"
        guard_funcs.setdefault(guard_short, set()).add(fn)
        func_link.setdefault(fn, r["github_link"])

    for level_data in levels[1:]:
        for r in level_data:
            caller, callee = r["caller"], r["callee"]
            if caller and callee:
                edge = (caller, callee)
                if edge not in all_edges:
                    all_edges.append(edge)
                func_link.setdefault(caller, r["github_link"])

    # Reverse adjacency: callee → set of callers
    callee_to_callers: dict[str, set[str]] = {}
    for caller, callee in all_edges:
        callee_to_callers.setdefault(callee, set()).add(caller)

    # ------------------------------------------------------------------ #
    # Helper: BFS from a set of seed nodes upward (via callee→callers)   #
    # ------------------------------------------------------------------ #
    def upstream_nodes(seeds: set[str]) -> set[str]:
        visited: set[str] = set()
        frontier = set(seeds)
        while frontier:
            visited |= frontier
            next_f: set[str] = set()
            for node in frontier:
                next_f |= callee_to_callers.get(node, set())
            frontier = next_f - visited
        return visited

    # ------------------------------------------------------------------ #
    # Produce Markdown                                                    #
    # ------------------------------------------------------------------ #
    md: list[str] = []
    md.append("# SM100a Guard Call Graphs\n")
    md.append(
        "Each section shows one guard macro.  "
        "Arrows point **caller → callee**; "
        "the guard macro itself is the leaf node.\n"
    )
    md.append(
        f"Depth searched: L0–L{len(levels) - 1}  |  "
        f"Total call edges: {len(all_edges)}\n"
    )
    md.append("---\n")

    for guard_short, l0_fns in sorted(guard_funcs.items()):
        md.append(f"## `{guard_short}`\n")

        # Collect all nodes reachable upstream from L0 functions of this guard
        reachable = upstream_nodes(l0_fns)

        # The subgraph edges: only edges where both endpoints are reachable
        sub_edges = [(c, e) for c, e in all_edges if c in reachable and e in reachable]

        # Summary table  (L0 functions)
        md.append("### Functions directly using this guard\n")
        md.append("| Function | Source |")
        md.append("|---|---|")
        for fn in sorted(l0_fns):
            link = func_link.get(fn, "")
            md.append(f"| `{fn}` | {link} |")
        md.append("")

        if not sub_edges and not reachable - l0_fns:
            md.append("_No callers found in the codebase._\n")
            continue

        # Mermaid diagram
        md.append("### Call graph\n")
        md.append("```mermaid")
        md.append("flowchart LR")

        # Node declarations with link annotations where available
        all_nodes = {n for edge in sub_edges for n in edge} | l0_fns
        # Guard leaf node
        guard_node_id = _mermaid_id(guard_short)
        md.append(f'    {guard_node_id}(("{guard_short}"))')

        for node in sorted(all_nodes):
            nid = _mermaid_id(node)
            link = func_link.get(node, "")
            if link:
                md.append(f'    {nid}["`{node}`"]\n    click {nid} "{link}" _blank')
            else:
                md.append(f'    {nid}["`{node}`"]')

        # Edges from L1+ callers
        for caller, callee in sub_edges:
            md.append(f"    {_mermaid_id(caller)} --> {_mermaid_id(callee)}")

        # L0 functions → guard leaf
        for fn in sorted(l0_fns):
            md.append(f"    {_mermaid_id(fn)} --> {guard_node_id}")

        md.append("```\n")

    with open(output_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"  MD   → {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Find all sm_100a guard usages in TransformerEngine source, "
            "then walk the call chain outwards and write per-level CSVs "
            "and a Mermaid call-graph Markdown file."
        )
    )
    parser.add_argument(
        "--source-dir",
        default=DEFAULT_SOURCE_DIR,
        help=f"Path to TransformerEngine source (default: {DEFAULT_SOURCE_DIR})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_CSV,
        help=f"Legacy single-CSV output path (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--commit",
        default=None,
        help="Git commit hash for permalink generation (auto-detected if omitted)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum call-chain depth to explore (default: 6)",
    )
    args = parser.parse_args()

    output_dir = os.path.dirname(os.path.abspath(args.output))
    base = os.path.splitext(os.path.basename(args.output))[
        0
    ]  # e.g. "te_sm100a_guard_usages"

    commit = args.commit or get_commit(args.source_dir)
    print(f"Source dir : {args.source_dir}")
    print(f"Commit     : {commit[:12]}")

    # ------------------------------------------------------------------ #
    # L0 — scan for direct guard usages                                   #
    # ------------------------------------------------------------------ #
    print("\n[L0] Scanning for guard usages…")
    l0 = scan(args.source_dir, commit)
    print(f"  Found {len(l0)} guard usages.")

    from collections import Counter

    counts = Counter(r["guard"] for r in l0)
    for guard, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {count:4d}  {guard}")

    # Also write the legacy single CSV for backward compatibility
    fieldnames = ["function", "github_link", "guard"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(l0)
    print(f"\n  (legacy) guard usages CSV → {args.output}")

    # ------------------------------------------------------------------ #
    # L1+ — walk the call chain outward                                   #
    # ------------------------------------------------------------------ #
    print(f"\nWalking call chain (max depth {args.max_depth})…")
    levels = build_call_chain(l0, args.source_dir, commit, max_depth=args.max_depth)
    print(f"\nCall chain depth reached: L{len(levels) - 1}")

    # ------------------------------------------------------------------ #
    # Write per-level CSVs                                                #
    # ------------------------------------------------------------------ #
    print("\nWriting per-level CSVs…")
    write_level_csvs(levels, output_dir, base)

    # ------------------------------------------------------------------ #
    # Write Mermaid Markdown                                              #
    # ------------------------------------------------------------------ #
    md_path = os.path.join(output_dir, f"{base}_callgraph.md")
    print("\nWriting call-graph Markdown…")
    write_callgraph_md(levels, md_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
