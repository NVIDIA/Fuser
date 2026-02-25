# `te_guard_uses.py` — SM100a Guard Usage Scanner

Scans the [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) C++/CUDA source tree for usages of **Blackwell (SM100a) architecture-specific guards**, identifies the enclosing function or kernel for each usage via a real C++ AST, and writes CSV reports with GitHub permalinks. It also walks the call graph upward (L1, L2) to find which higher-level functions and kernels ultimately depend on Blackwell-specific code.

Scanned commit: `c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8`

---

## Outputs

| File | Description |
|------|-------------|
| `te_sm100a_guard_usages_L0.csv` | **L0** — one row per unique function/kernel that directly contains an arch guard |
| `te_sm100a_guard_usages_L0_callers_L1.csv` | **L1** — callers of the L0 functions |
| `te_sm100a_guard_usages_L0_callers_L2.csv` | **L2** — callers of the non-kernel L1 callers |

### L0 CSV columns

| Column | Description |
|--------|-------------|
| `function` | Name of the enclosing C++ function or kernel |
| `is_kernel` | `True` if the function has a `__global__` qualifier |
| `github_link` | Permalink to the guard usage in the TE repository |
| `guard` | Which guard pattern triggered the match (see below) |

### L1 / L2 CSV columns

| Column | Description |
|--------|-------------|
| `caller` | Name of the function that calls the callee |
| `is_caller_kernel` | `True` if the caller has a `__global__` qualifier |
| `callee` | Name of the function being called (from the previous level) |
| `github_link` | Permalink to the call site |

---

## Current Results

### L0 — 16 unique functions

| Function | `is_kernel` | Guard |
|----------|-------------|-------|
| `modify_base_tensor_map` | False | `ARCH_BLACKWELL_FAMILY` |
| `group_row_col_rht_gemm_device_graph_safe` | **True** | `ARCH_BLACKWELL_FAMILY` |
| `group_row_col_rht_gemm_device` | **True** | `ARCH_BLACKWELL_FAMILY` |
| `cvt_fp32_to_fp4_4x_with_rn` | False | `ARCH_BLACKWELL_FAMILY` |
| `try_cancel_cta` | False | `ARCH_BLACKWELL_FAMILY` |
| `get_cancelled_cta_id_2D` | False | `ARCH_BLACKWELL_FAMILY` |
| `float_to_e8m0` | False | `ARCH_BLACKWELL_FAMILY` |
| `mul_cvt_bf16_to_fp4_4x_with_rn` | False | `ARCH_BLACKWELL_FAMILY` |
| `mul_cvt_fp32_to_fp4_4x_with_rn` | False | `ARCH_BLACKWELL_FAMILY` |
| `mul_cvt_bf16_to_fp4_8x_round_to_nearest` | False | `ARCH_BLACKWELL_FAMILY` |
| `StochasticNumericConverterBase` | False | `ARCH_HAS_STOCHASTIC_ROUNDING` |
| `cvt_fp32_to_fp4_4x_with_stochastic_rounding` | False | `ARCH_HAS_STOCHASTIC_ROUNDING` |
| `mul_cvt_bf16_to_fp4_4x_with_stochastic_rounding` | False | `ARCH_HAS_STOCHASTIC_ROUNDING` |
| `mul_cvt_fp32_to_fp4_4x_with_stochastic_rounding` | False | `ARCH_HAS_STOCHASTIC_ROUNDING` |
| `mul_cvt_bf16_to_fp4_8x_stochastic_rounding` | False | `ARCH_HAS_STOCHASTIC_ROUNDING` |
| `reduce_sync_max_abs_f32` | False | `NVTE_CUDA_ARCH_MATCHES(FamilySpecific/ArchSpecific<100>)` |

2 `__global__` kernels, 14 device/host functions.

### L1 — 14 unique (caller, callee) pairs

The 2 `__global__` L0 kernels (`group_row_col_rht_gemm_device*`) are excluded as F. 3 callers are themselves `__global__` kernels.

| Caller | `is_caller_kernel` | Callee |
|--------|--------------------|--------|
| `colwise_scaling` | False | `mul_cvt_bf16_to_fp4_8x_round_to_nearest` |
| `colwise_scaling` | False | `mul_cvt_bf16_to_fp4_8x_stochastic_rounding` |
| `cvt_fp32_to_fp4_4x` | False | `cvt_fp32_to_fp4_4x_with_rn` |
| `cvt_fp32_to_fp4_4x` | False | `cvt_fp32_to_fp4_4x_with_stochastic_rounding` |
| `mul_cvt_bf16_to_fp4_4x` | False | `mul_cvt_bf16_to_fp4_4x_with_rn` |
| `mul_cvt_bf16_to_fp4_4x` | False | `mul_cvt_bf16_to_fp4_4x_with_stochastic_rounding` |
| `mul_cvt_fp32_to_fp4_4x` | False | `mul_cvt_fp32_to_fp4_4x_with_rn` |
| `mul_cvt_fp32_to_fp4_4x` | False | `mul_cvt_fp32_to_fp4_4x_with_stochastic_rounding` |
| `quantize_mxfp8_kernel_cast_only` | **True** | `reduce_sync_max_abs_f32` |
| `quantize_transpose_nvfp4_tuned_1D_kernel` | **True** | `get_cancelled_cta_id_2D` |
| `quantize_transpose_nvfp4_tuned_1D_kernel` | **True** | `try_cancel_cta` |
| `rowwise_scaling` | False | `mul_cvt_bf16_to_fp4_8x_round_to_nearest` |
| `rowwise_scaling` | False | `mul_cvt_bf16_to_fp4_8x_stochastic_rounding` |
| `to_e8m0` | False | `float_to_e8m0` |

### L2 — 10 unique (caller, callee) pairs

The 3 `__global__` L1 callers (`quantize_mxfp8_kernel_cast_only`, `quantize_transpose_nvfp4_tuned_1D_kernel`) are excluded as F. All 10 L2 callers are `__global__` kernels — the call graph is exhausted at this level.

| Caller | `is_caller_kernel` | Callee |
|--------|--------------------|--------|
| `block_scaled_1d_cast_transpose_kernel` | **True** | `cvt_fp32_to_fp4_4x` |
| `group_quantize_transpose_nvfp4_kernel` | **True** | `mul_cvt_bf16_to_fp4_4x` |
| `group_quantize_transpose_nvfp4_kernel` | **True** | `mul_cvt_fp32_to_fp4_4x` |
| `quantize_mxfp8_kernel_cast_only` | **True** | `to_e8m0` |
| `quantize_transpose_nvfp4_2D_kernel` | **True** | `mul_cvt_bf16_to_fp4_4x` |
| `quantize_transpose_nvfp4_2D_kernel` | **True** | `mul_cvt_fp32_to_fp4_4x` |
| `quantize_transpose_nvfp4_kernel` | **True** | `mul_cvt_bf16_to_fp4_4x` |
| `quantize_transpose_nvfp4_kernel` | **True** | `mul_cvt_fp32_to_fp4_4x` |
| `quantize_transpose_nvfp4_tuned_1D_kernel` | **True** | `colwise_scaling` |
| `quantize_transpose_nvfp4_tuned_1D_kernel` | **True** | `rowwise_scaling` |

---

## Guard Patterns

Ten patterns are searched across all `.cu`, `.cuh`, `.h`, `.hpp`, `.cpp`, `.cc` files:

| Label | What it matches |
|-------|----------------|
| `ARCH_BLACKWELL_FAMILY` | Primary macro guarding Blackwell-only code paths |
| `ARCH_HAS_STOCHASTIC_ROUNDING` | Macro for SM100a stochastic rounding support |
| `NVTE_CUDA_ARCH_MATCHES(FamilySpecific/ArchSpecific<100>)` | Template-based arch dispatch for SM100/SM101/SM103 |
| `__CUDA_ARCH_HAS_FEATURE__(SM100_ALL/SM101_ALL)` | CUDA built-in feature query (SM100/SM101) |
| `_ENABLE_MXFMA` | Enables mixed FP4/FP8 MXFMA path |
| `is_blackwell` | Local boolean alias for `ARCH_BLACKWELL_FAMILY` |
| `is_blackwell_arch` | Local boolean alias for `ARCH_BLACKWELL_FAMILY` |
| `has_fp4` | Local alias: Blackwell has hardware FP4 support |
| `has_rs` | Local alias for `ARCH_HAS_STOCHASTIC_ROUNDING` |
| `is_sm_100f` | Local alias for `FamilySpecific<100>` dispatch |

Preprocessor lines (`#define`, `#if`, `#elif`, `#endif`) and lines inside multi-line `#define` blocks are skipped — these are macro *definitions*, not real code paths inside functions.

Rows are **deduplicated by function name**: if the same function contains multiple guard usages (different lines or different guards), only the first occurrence is kept.

---

## Architecture

### 1. libclang-based enclosing-function detection

Each source file is parsed once with **libclang** in `-x c++` mode using a two-pass strategy:

1. **First pass** — parse the raw source. Collect all files that produce parse errors (e.g. included CUDA headers with inline PTX `asm` constraints).
2. **Second pass** — strip all `asm(…)` blocks in the target file and any discovered error files (replacing them with `(void)0;`, preserving line counts), then re-parse. Line numbers remain accurate.

The resulting AST is walked once to build a `{line_number → function_name}` map for the file (cached in `_LINE_MAP_CACHE`). Enclosing-function lookup for any line is then an O(1) dictionary lookup.

#### CUDA stub header

libclang runs in plain C++ mode (not CUDA mode), so it doesn't know about CUDA intrinsics. A virtual stub header (`/tmp/__te_guard_scan_cuda_stubs__.h`) is force-included before every file. It provides:

- CUDA built-ins: `threadIdx`, `blockIdx`, `__syncthreads`, `__shfl_sync`, etc.
- FP4 types: `__nv_fp4_e2m1`, `__nv_fp4x4_e2m1` (not present in older CUDA installs)
- CUTLASS macros and minimal type stubs (`cutlass::Array`, `float_e2m1_t`, etc.) so template return types parse correctly
- Key `-D` defines: `-D__CUDA_ARCH__=1000`, `-DFP4_TYPE_SUPPORTED=1`

### 2. `__global__` kernel detection

Because `__global__` is a CUDA-specific qualifier unknown to libclang in C++ mode, it is detected via a **raw source text scan** around each function's AST extent. Results are cached per file in `_GLOBAL_FUNCS_CACHE`.

For regular functions (`FUNCTION_DECL`/`CXX_METHOD`), libclang's `ext.start.line` points at the return-type/qualifier line, so `__global__` appears at or up to 4 lines before the extent start.

For template functions (`FUNCTION_TEMPLATE`), `ext.start.line` points at the `template` keyword, and `__global__` appears *after* the (possibly multi-line) template parameter list:

```cpp
template <typename CastTraits,                     // ← ext.start.line
          std::enable_if_t<…> = 0>
__global__ void quantize_mxfp8_kernel_cast_only(…) {
```

The scan reads forward from `ext.start.line - 4` up to 20 lines ahead, stopping early on the first `{` (opening of the function body). This covers both cases.

### 3. Caller discovery (`find_callers`)

Caller search follows two rules about the **callee** function F being searched:

1. **F must not be `__global__`** — GPU entry points have no C++ caller. The set of known kernel names is passed *explicitly* from scan results (`is_kernel` / `is_caller_kernel` fields) rather than from the lazy `_GLOBAL_FUNCS_CACHE`, which may be incomplete when called. Callers (CF) that are `__global__` are **not** filtered and appear in the output.
2. **Direct recursion is skipped** — call sites where `caller == callee` are ignored.

#### Performance: regex pre-filter

Parsing every TE file with libclang would be expensive. Before invoking the AST, each file is checked with a fast regex:

```
\b(name1|name2|…)\b
```

Only files with a textual hit are parsed. This eliminates the vast majority of files.

#### Two-pass call-site detection

libclang emits `CALL_EXPR` nodes for most function calls, but **dependent template calls** — e.g. `ptx::mul_cvt_bf16_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(…)` inside an uninstantiated template body — are emitted as `UNEXPOSED_EXPR` and are invisible to the AST walk.

`_call_sites_in_file` therefore runs two passes:

1. **AST pass** — walk `CALL_EXPR` nodes; precise, no false positives.
2. **Regex supplement** — scan for `name<…>(` patterns (explicit template-argument calls) to catch the dependent-template cases the AST missed.

Results are deduplicated by `(line, callee_name)`.

### 4. Multi-level caller walk (L1 → L2)

After L0, the script runs two further levels:

- **L1**: search callers of all 16 L0 functions, skipping the 2 `__global__` L0 kernels as F. Kernel set = `{r["function"] for r in L0 if r["is_kernel"]}`.
- **L2**: search callers of the non-kernel L1 callers only, skipping any L1 caller with `is_caller_kernel=True`. Kernel set = `{r["caller"] for r in L1 if r["is_caller_kernel"]}`. L0 function names are also excluded to avoid re-scanning already-covered names.

At L2 all 10 found callers are `__global__` kernels — the call graph is exhausted.

---

## Usage

```bash
# Run with defaults (TE at /opt/pytorch/TransformerEngine)
python3 te_guard_uses.py

# Custom source dir and output path
python3 te_guard_uses.py \
    --source-dir /path/to/TransformerEngine \
    --output /path/to/output_L0.csv \
    --commit abc1234
```

The `--output` path is used as the L0 CSV filename. L1 and L2 filenames are derived from it by inserting `_callers_L1` / `_callers_L2` before `.csv`.

### Dependencies

```bash
pip install libclang
```

---

## Known Limitations

- **`__global__` detection** relies on raw text scanning of a bounded window around the function extent. Unusual formatting where `__global__` appears more than 4 lines before `ext.start.line` (for non-template functions) could cause a miss.
- **Template function names** returned by the AST are base names only (e.g. `mul_cvt_bf16_to_fp4_4x`, not the full specialisation). This is intentional for deduplication but means different specialisations of the same template are merged into one row.
- **Function pointer dispatch** (e.g. `auto kernel = specialized::quantize_mxfp8_kernel_cast_only<traits>; kernel<<<…>>>()`) is not detected by either AST or regex, because no call token is present at the assignment site.
- The scan is **read-only** and does not build or compile TransformerEngine.
