# SM100a Guard Usage — Call Chains

Full call chains from entry-point kernels down to the Blackwell-guarded L0 functions.
Commit: `c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8`

Legend:
- **[kernel]** — `__global__` GPU entry point (`is_kernel` / `is_caller_kernel` = True)
- **[device]** — `__device__` / `__forceinline__` helper function
- Guard label shown on each L0 leaf

---

## Chains with three levels (L2 → L1 → L0)

### `block_scaled_1d_cast_transpose_kernel` [kernel]

- [block_scaled_1d_cast_transpose_kernel](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/transpose/quantize_transpose_vector_blockwise_fp4.cu#L544) \[L2, kernel]
  - [cvt_fp32_to_fp4_4x](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/transpose/quantize_transpose_vector_blockwise_fp4.cu#L314) \[L1, device]
    - [cvt_fp32_to_fp4_4x_with_rn](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/transpose/quantize_transpose_vector_blockwise_fp4.cu#L284) \[L0] `ARCH_BLACKWELL_FAMILY`
    - [cvt_fp32_to_fp4_4x_with_stochastic_rounding](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/transpose/quantize_transpose_vector_blockwise_fp4.cu#L262) \[L0] `ARCH_HAS_STOCHASTIC_ROUNDING`

---

### `group_quantize_transpose_nvfp4_kernel` [kernel]

- [group_quantize_transpose_nvfp4_kernel](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/cast/nvfp4/group_quantize_transpose_nvfp4.cuh#L456) \[L2, kernel]
  - [mul_cvt_bf16_to_fp4_4x](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/util/ptx.cuh#L654) \[L1, device]
    - [mul_cvt_bf16_to_fp4_4x_with_rn](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/util/ptx.cuh#L604) \[L0] `ARCH_BLACKWELL_FAMILY`
    - [mul_cvt_bf16_to_fp4_4x_with_stochastic_rounding](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/util/ptx.cuh#L564) \[L0] `ARCH_HAS_STOCHASTIC_ROUNDING`
  - [mul_cvt_fp32_to_fp4_4x](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/util/ptx.cuh#L742) \[L1, device]
    - [mul_cvt_fp32_to_fp4_4x_with_rn](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/util/ptx.cuh#L697) \[L0] `ARCH_BLACKWELL_FAMILY`
    - [mul_cvt_fp32_to_fp4_4x_with_stochastic_rounding](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/util/ptx.cuh#L661) \[L0] `ARCH_HAS_STOCHASTIC_ROUNDING`

---

### `quantize_mxfp8_kernel_cast_only` [kernel]

This kernel appears at both L1 (calling `reduce_sync_max_abs_f32` directly) and L2 (calling `to_e8m0` via the `to_e8m0` device function).

- [quantize_mxfp8_kernel_cast_only](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/cast/mxfp8/specialized/quantize_mxfp8.cuh#L280) \[L2, kernel]
  - [to_e8m0](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/cast/mxfp8/specialized/quantize_mxfp8.cuh#L85) \[L1, device]
    - [float_to_e8m0](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/util/ptx.cuh#L345) \[L0] `ARCH_BLACKWELL_FAMILY`

- [quantize_mxfp8_kernel_cast_only](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/cast/mxfp8/specialized/quantize_mxfp8.cuh#L1389) \[L1, kernel]
  - [reduce_sync_max_abs_f32](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/util/ptx.cuh#L1129) \[L0] `NVTE_CUDA_ARCH_MATCHES(FamilySpecific/ArchSpecific<100>)`

---

### `quantize_transpose_nvfp4_2D_kernel` [kernel]

- [quantize_transpose_nvfp4_2D_kernel](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh#L940) \[L2, kernel]
  - [mul_cvt_bf16_to_fp4_4x](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/util/ptx.cuh#L654) \[L1, device]
    - [mul_cvt_bf16_to_fp4_4x_with_rn](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/util/ptx.cuh#L604) \[L0] `ARCH_BLACKWELL_FAMILY`
    - [mul_cvt_bf16_to_fp4_4x_with_stochastic_rounding](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/util/ptx.cuh#L564) \[L0] `ARCH_HAS_STOCHASTIC_ROUNDING`
  - [mul_cvt_fp32_to_fp4_4x](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/util/ptx.cuh#L742) \[L1, device]
    - [mul_cvt_fp32_to_fp4_4x_with_rn](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/util/ptx.cuh#L697) \[L0] `ARCH_BLACKWELL_FAMILY`
    - [mul_cvt_fp32_to_fp4_4x_with_stochastic_rounding](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/util/ptx.cuh#L661) \[L0] `ARCH_HAS_STOCHASTIC_ROUNDING`

---

### `quantize_transpose_nvfp4_kernel` [kernel]

- [quantize_transpose_nvfp4_kernel](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh#L361) \[L2, kernel]
  - [mul_cvt_bf16_to_fp4_4x](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/util/ptx.cuh#L654) \[L1, device]
    - [mul_cvt_bf16_to_fp4_4x_with_rn](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/util/ptx.cuh#L604) \[L0] `ARCH_BLACKWELL_FAMILY`
    - [mul_cvt_bf16_to_fp4_4x_with_stochastic_rounding](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/util/ptx.cuh#L564) \[L0] `ARCH_HAS_STOCHASTIC_ROUNDING`
  - [mul_cvt_fp32_to_fp4_4x](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/util/ptx.cuh#L742) \[L1, device]
    - [mul_cvt_fp32_to_fp4_4x_with_rn](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/util/ptx.cuh#L697) \[L0] `ARCH_BLACKWELL_FAMILY`
    - [mul_cvt_fp32_to_fp4_4x_with_stochastic_rounding](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/util/ptx.cuh#L661) \[L0] `ARCH_HAS_STOCHASTIC_ROUNDING`

---

### `quantize_transpose_nvfp4_tuned_1D_kernel` [kernel]

This kernel appears at both L1 (calling `try_cancel_cta` / `get_cancelled_cta_id_2D` directly) and L2 (calling `colwise_scaling` / `rowwise_scaling`).

- [quantize_transpose_nvfp4_tuned_1D_kernel](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/cast/nvfp4/specialized/quantize_transpose_nvfp4_tuned_1D.cuh#L578) \[L2, kernel]
  - [colwise_scaling](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/cast/nvfp4/specialized/quantize_transpose_nvfp4_tuned_1D.cuh#L253) \[L1, device]
    - [mul_cvt_bf16_to_fp4_8x_round_to_nearest](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/util/ptx.cuh#L750) \[L0] `ARCH_BLACKWELL_FAMILY`
    - [mul_cvt_bf16_to_fp4_8x_stochastic_rounding](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/util/ptx.cuh#L842) \[L0] `ARCH_HAS_STOCHASTIC_ROUNDING`
  - [rowwise_scaling](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/cast/nvfp4/specialized/quantize_transpose_nvfp4_tuned_1D.cuh#L341) \[L1, device]
    - [mul_cvt_bf16_to_fp4_8x_round_to_nearest](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/util/ptx.cuh#L750) \[L0] `ARCH_BLACKWELL_FAMILY`
    - [mul_cvt_bf16_to_fp4_8x_stochastic_rounding](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/util/ptx.cuh#L842) \[L0] `ARCH_HAS_STOCHASTIC_ROUNDING`

- [quantize_transpose_nvfp4_tuned_1D_kernel](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/cast/nvfp4/specialized/quantize_transpose_nvfp4_tuned_1D.cuh#L520) \[L1, kernel]
  - [get_cancelled_cta_id_2D](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/util/ptx.cuh#L299) \[L0] `ARCH_BLACKWELL_FAMILY`
  - [try_cancel_cta](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/util/ptx.cuh#L281) \[L0] `ARCH_BLACKWELL_FAMILY`

---

## Standalone L0 kernels (directly guarded, no callers in source tree)

These `__global__` kernels contain the Blackwell guard directly in their own body. They are GPU entry points launched from the host — the scan found no C++ call sites for them within the TE source tree.

### `group_row_col_rht_gemm_device` [kernel]

- [group_row_col_rht_gemm_device](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/hadamard_transform/group_row_cast_col_hadamard_transform_cast_fusion.cu#L186) \[L0, kernel] `ARCH_BLACKWELL_FAMILY`

---

### `group_row_col_rht_gemm_device_graph_safe` [kernel]

- [group_row_col_rht_gemm_device_graph_safe](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/hadamard_transform/graph_safe_group_row_cast_col_hadamard_transform_cast_fusion.cu#L194) \[L0, kernel] `ARCH_BLACKWELL_FAMILY`

---

## L0 device functions with no detected callers

`modify_base_tensor_map` and `StochasticNumericConverterBase` contain Blackwell guards but no call sites were found for them within the scanned source tree.

| Function | `is_kernel` | Guard | Source |
|----------|-------------|-------|--------|
| [modify_base_tensor_map](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/cast/mxfp8/group_quantize_mxfp8.cuh#L143) | False | `ARCH_BLACKWELL_FAMILY` | `cast/mxfp8/group_quantize_mxfp8.cuh#L143` |
| [StochasticNumericConverterBase](https://github.com/NVIDIA/TransformerEngine/blob/c68ec3101d0dc16fe6eb40294a5fed3a9370b6a8/transformer_engine/common/hadamard_transform/graph_safe_group_row_cast_col_hadamard_transform_cast_fusion.cu#L98) | False | `ARCH_HAS_STOCHASTIC_ROUNDING` | `hadamard_transform/graph_safe_group_row_cast_col_hadamard_transform_cast_fusion.cu#L98` |
