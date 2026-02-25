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

Enclosing-function detection uses libclang (real C++ AST).
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
DEFAULT_OUTPUT_CSV = "/opt/pytorch/nvfuser/te_sm100a_guard_usages_L0.csv"
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

# Per-file cache: filepath → set of __global__ kernel names in that file.
_GLOBAL_FUNCS_CACHE: dict[str, set[str]] = {}


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
    # on every source file without a timeout.
    # Innermost function wins: process parents before children so children can
    # overwrite the parent name for the lines they own.
    # Read raw source lines for __global__ detection (more reliable than
    # token-walking, which is fragile when __global__ is redefined by CUDA headers).
    try:
        src_lines = Path(filepath).read_text(errors="replace").splitlines()
    except OSError:
        src_lines = []

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
                # Detect __global__: scan raw text of the function header.
                #
                # For regular functions (FUNCTION_DECL / CXX_METHOD),
                # ext.start.line points at the return-type line, so __global__
                # is at or up to 4 lines *before* the extent start.
                #
                # For template functions (FUNCTION_TEMPLATE), ext.start.line
                # points at the 'template' keyword, and __global__ appears
                # *after* the (possibly multi-line) template parameter list,
                # e.g.:
                #   line N+0: template <typename T,
                #   line N+1:           std::enable_if_t<…> = 0>
                #   line N+2: __global__ void my_kernel(…) {
                #
                # Strategy: scan from (ext.start.line - 4) forward until we
                # hit the opening '{' of the function body (which marks the
                # end of the header) or reach a 20-line cap.  This covers
                # both cases without accidentally reading into the body.
                header_start = max(0, ext.start.line - 4)
                header_end = min(len(src_lines), ext.start.line + 20)
                for raw_ln in src_lines[header_start:header_end]:
                    if "__global__" in raw_ln:
                        global_kernels.add(node.spelling)
                        break
                    # Stop at the opening brace — we've passed the header.
                    if "{" in raw_ln:
                        break
        stack.extend(node.get_children())

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
                # is_kernel is resolved after _build_line_func_map has run
                # (which populates _GLOBAL_FUNCS_CACHE as a side effect).
                is_kernel = (
                    any(func in ks for ks in _GLOBAL_FUNCS_CACHE.values())
                    if func
                    else False
                )
                results.append(
                    {
                        "guard": guard_label,
                        "function": func,
                        "is_kernel": is_kernel,
                        "github_link": link,
                    }
                )

    # Sort by guard then github_link
    results.sort(key=lambda r: (r["guard"], r["github_link"]))

    # Deduplicate: one row per unique function name (first occurrence wins).
    # For hits with no enclosing function, fall back to the github_link as key.
    seen_funcs: set[str] = set()
    deduped = []
    for r in results:
        fn = r["function"]
        key = fn if fn else r["github_link"]
        if key in seen_funcs:
            continue
        seen_funcs.add(key)
        deduped.append(r)
    return deduped


# ---------------------------------------------------------------------------
# Caller discovery (one level up from L0 non-kernel functions)
# ---------------------------------------------------------------------------


def _call_sites_in_file(filepath: str, callee_names: set[str]) -> list[tuple[int, str]]:
    """Walk the libclang AST of *filepath* and return (lineno, callee_name) for
    every call site whose callee is in *callee_names*.

    Primary pass: AST CALL_EXPR nodes — precise, handles regular and
    fully-resolved template calls.

    Supplemental pass: line-level regex for explicit-template-argument calls
    of the form ``name<...>(`` or ``name<...> (`` that live inside uninstantiated
    template bodies.  Clang emits UNEXPOSED_EXPR (not CALL_EXPR) for dependent
    calls in unspecialised templates, so they are invisible to the AST walk.
    The regex supplement catches exactly those missed sites.

    Uses the cached TU (and triggers _build_line_func_map as a side-effect so
    that _GLOBAL_FUNCS_CACHE is populated for *filepath*).
    """
    # Ensure the line→function map (and GLOBAL_FUNCS_CACHE) is built first.
    _build_line_func_map(filepath)
    tu = _get_tu(filepath)

    seen: set[tuple[int, str]] = set()
    hits: list[tuple[int, str]] = []

    # --- Primary AST pass ---------------------------------------------------
    stack = [tu.cursor]
    while stack:
        node = stack.pop()
        ext = node.extent
        # Only visit nodes whose call site is in *filepath* (prune headers).
        if ext.start.file and ext.start.file.name != filepath:
            continue
        if node.kind == _cx.CursorKind.CALL_EXPR:
            name = node.spelling
            if name in callee_names:
                key = (ext.start.line, name)
                if key not in seen:
                    seen.add(key)
                    hits.append(key)
        stack.extend(node.get_children())

    # --- Supplemental regex pass for dependent template calls ---------------
    # Pattern: word-boundary callee name followed (possibly with whitespace) by
    # an explicit template-argument list '<…>' then '('.  These are invisible to
    # the AST walk when they appear in an uninstantiated template body.
    try:
        lines = Path(filepath).read_text(errors="replace").splitlines()
    except OSError:
        return hits

    # Build one combined pattern covering all callee names.
    alt = "|".join(re.escape(n) for n in sorted(callee_names))
    tmpl_call_re = re.compile(r"\b(" + alt + r")\s*<[^;{]*>\s*\(")
    for lineno, line in enumerate(lines, 1):
        m = tmpl_call_re.search(line)
        if m:
            name = m.group(1)
            key = (lineno, name)
            if key not in seen:
                seen.add(key)
                hits.append(key)

    return hits


def find_callers(
    callee_names: set[str],
    source_dir: str,
    commit: str,
    kernel_names: set[str] | None = None,
) -> list[dict]:
    """Search *source_dir* for call sites of any function in *callee_names*
    using the libclang AST (CALL_EXPR nodes).

    Rules applied to *callee_names* (F — the functions being searched):
    1. F must not be a ``__global__`` kernel. Pass the complete set of known
       kernel names via *kernel_names* so they can be excluded upfront.
       Callers (CF) are *not* filtered — a CF that is ``__global__`` is
       perfectly valid and will appear in the results.
    2. Direct recursion is skipped: call sites where CF == F are ignored.

    A fast regex pre-filter first eliminates files that cannot contain any
    call site (text doesn't mention any callee name at all), so the expensive
    libclang parse is only triggered for files that are likely to have hits.

    Deduplicates on (caller, callee) pairs; first occurrence wins.
    Returns a list of dicts: caller, is_caller_kernel, callee, github_link.
    """
    # ------------------------------------------------------------------ #
    # Guard 1: skip __global__ F — derived from explicitly-passed set,   #
    # NOT from _GLOBAL_FUNCS_CACHE (which is populated lazily and may    #
    # be incomplete at call time).  CF (__global__ callers) are allowed. #
    # ------------------------------------------------------------------ #
    known_kernels: set[str] = kernel_names or set()
    search_names = callee_names - known_kernels
    if not search_names:
        return []
    if search_names != callee_names:
        skipped = callee_names - search_names
        print(
            f"    Skipping {len(skipped)} __global__ kernel(s) (F) from caller search: "
            f"{sorted(skipped)}"
        )

    root = Path(source_dir)
    seen_pairs: set[tuple[str, str]] = set()
    results: list[dict] = []

    # Quick text pre-filter: word-boundary check any callee name in file text.
    prefilter = re.compile(
        r"\b(" + "|".join(re.escape(n) for n in sorted(search_names)) + r")\b"
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

        # Skip files with no textual mention of any callee — avoids libclang parse.
        if not prefilter.search(text):
            continue

        rel = str(fpath.relative_to(root))
        fp = str(fpath)

        # AST-based call-site discovery for this file.
        for lineno, callee in _call_sites_in_file(fp, search_names):
            caller = enclosing_function(fp, lineno)

            # Guard 2: skip calls with no enclosing function, or direct
            # recursion (function calling itself).
            if not caller or caller == callee:
                continue

            pair = (caller, callee)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            is_caller_kernel = any(caller in ks for ks in _GLOBAL_FUNCS_CACHE.values())
            link = make_github_link(GITHUB_REPO, commit, rel, lineno)
            results.append(
                {
                    "caller": caller,
                    "is_caller_kernel": is_caller_kernel,
                    "callee": callee,
                    "github_link": link,
                }
            )

    results.sort(key=lambda r: (r["caller"], r["callee"]))
    return results


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Find all sm_100a guard usages in TransformerEngine source and "
            "write a CSV with the enclosing function name and GitHub permalink."
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
        help=f"Output CSV path for L0 guard usages (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--commit",
        default=None,
        help="Git commit hash for permalink generation (auto-detected if omitted)",
    )
    args = parser.parse_args()

    commit = args.commit or get_commit(args.source_dir)
    print(f"Source dir : {args.source_dir}")
    print(f"Commit     : {commit[:12]}")

    print("\nScanning for guard usages…")
    results = scan(args.source_dir, commit)
    print(f"  Found {len(results)} guard usages.")

    from collections import Counter

    counts = Counter(r["guard"] for r in results)
    for guard, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {count:4d}  {guard}")

    kernels = sum(1 for r in results if r["is_kernel"])
    print(
        f"  {kernels} __global__ kernel(s), {len(results) - kernels} device/host function(s)"
    )

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["function", "is_kernel", "github_link", "guard"]
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\nL0 output  → {args.output}")

    # ------------------------------------------------------------------ #
    # Caller discovery: find callers of L0 functions.                     #
    # find_callers internally skips __global__ kernels (no C++ callers)   #
    # and recursive self-calls.                                            #
    # ------------------------------------------------------------------ #
    callee_names = {r["function"] for r in results if r["function"]}
    # Derive __global__ kernel names from L0 scan results (is_kernel=True).
    # These are excluded as F (callee) in the search; __global__ CF are fine.
    l0_kernel_names: set[str] = {r["function"] for r in results if r["is_kernel"]}
    print(
        f"\nSearching callers of {len(callee_names)} function(s) "
        f"({len(l0_kernel_names)} __global__ skipped as F)…"
    )
    callers = find_callers(
        callee_names, args.source_dir, commit, kernel_names=l0_kernel_names
    )
    print(f"  Found {len(callers)} unique (caller, callee) pair(s).")
    caller_kernels = sum(1 for r in callers if r["is_caller_kernel"])
    print(f"  {caller_kernels} caller(s) are __global__ kernel(s) (CF).")

    callers_l1_output = args.output.replace(".csv", "_callers_L1.csv")
    with open(callers_l1_output, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["caller", "is_caller_kernel", "callee", "github_link"]
        )
        writer.writeheader()
        writer.writerows(callers)
    print(f"\nL1 callers output → {callers_l1_output}")

    # ------------------------------------------------------------------ #
    # L2 caller discovery: find callers of non-kernel L1 callers.        #
    # Rules: skip __global__ F (derived from is_caller_kernel); CF can   #
    # be __global__. Also skip functions already covered at L0.          #
    # ------------------------------------------------------------------ #
    # L1 callers that are __global__ kernels become the excluded-F set
    # for L2 — they are GPU entry points with no meaningful C++ callers.
    l1_kernel_names: set[str] = {r["caller"] for r in callers if r["is_caller_kernel"]}
    # L2 callees = all unique L1 callers that are NOT __global__.
    l2_callee_names = {
        r["caller"] for r in callers if r["caller"] and not r["is_caller_kernel"]
    }
    # Also exclude functions already searched at L0 to avoid re-scanning.
    l0_names = {r["function"] for r in results if r["function"]}
    l2_callee_names -= l0_names

    print(
        f"\nSearching L2 callers of {len(l2_callee_names)} non-kernel L1 function(s) "
        f"({len(l1_kernel_names)} __global__ skipped as F)…"
    )
    callers_l2 = find_callers(
        l2_callee_names, args.source_dir, commit, kernel_names=l1_kernel_names
    )
    print(f"  Found {len(callers_l2)} unique (caller, callee) pair(s).")
    caller_kernels_l2 = sum(1 for r in callers_l2 if r["is_caller_kernel"])
    print(f"  {caller_kernels_l2} caller(s) are __global__ kernel(s).")

    callers_l2_output = args.output.replace(".csv", "_callers_L2.csv")
    with open(callers_l2_output, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["caller", "is_caller_kernel", "callee", "github_link"]
        )
        writer.writeheader()
        writer.writerows(callers_l2)
    print(f"\nL2 callers output → {callers_l2_output}")


if __name__ == "__main__":
    main()
