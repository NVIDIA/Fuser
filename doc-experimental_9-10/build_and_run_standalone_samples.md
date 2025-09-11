## Building and Running Standalone nvFuser Samples

This guide provides reliable ways to build and run small nvFuser C++ samples, using the helper scripts in `../doc-bot/experimenting/` and equivalent manual commands. It also includes troubleshooting for common runtime issues.

### Prerequisites

- CUDA toolkit and compatible NVIDIA driver
- PyTorch (ATen) installed (for linking and runtime tensor inputs)
- nvFuser repository (this project)

### Option A: Use helper script (recommended)

Script: `../doc-bot/experimenting/how_to_build`

```bash
# Source the script, then build a sample
source ../doc-bot/experimenting/how_to_build /opt/pytorch/nvfuser/doc-bot/experimenting/8-22-2025/tv_add_2d_SAMPLE_2.cpp

# Resulting binary is placed next to the .cpp (same basename)
```

Run setup for libraries and execute:

```bash
source ../doc-bot/experimenting/setup_libs_to_run
/opt/pytorch/nvfuser/doc-bot/experimenting/8-22-2025/tv_add_2d_SAMPLE_2
```

Notes:
- The build script resolves Torch include/lib paths and nvFuser `.so` locations automatically.
- It links required libraries and adds rpaths so the binary can find them at runtime.

### Option B: Manual build command

Use this when you need explicit control or to integrate into other build systems.

```bash
TORCH_LIB=$(python3 -c 'import torch, os, sys; sys.stdout.write(os.path.join(os.path.dirname(torch.__file__), "lib"))')
TORCH_INCLUDE=$(dirname "$TORCH_LIB")/include
NVFUSER_LIB_DIR=$(dirname $(find /opt/pytorch/nvfuser/python/nvfuser_common/lib -name libnvfuser.so -o -name libnvfuser_codegen.so | head -n1))

g++ -std=c++20 -O2 \
  -I/opt/pytorch/nvfuser/csrc \
  -I/opt/pytorch/nvfuser/lib/dynamic_type/src \
  -I/opt/pytorch/nvfuser/third_party/flatbuffers/include \
  -I"$TORCH_INCLUDE" \
  -I"$TORCH_INCLUDE/torch/csrc/api/include" \
  -I/usr/local/cuda/include \
  /opt/pytorch/nvfuser/doc-bot/experimenting/8-22-2025/tv_add_2d_SAMPLE_2.cpp \
  -L"$NVFUSER_LIB_DIR" \
  -L"$TORCH_LIB" \
  -L/usr/local/cuda/lib64 \
  -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -lcudart -lcupti -lnvrtc -ldl -pthread \
  -Wl,-rpath,"$NVFUSER_LIB_DIR" \
  -Wl,-rpath,"$TORCH_LIB" \
  -Wl,-rpath,/usr/local/cuda/lib64 \
  -lnvfuser -lnvfuser_codegen -lnvfuser_direct \
  -o /opt/pytorch/nvfuser/doc-bot/experimenting/8-22-2025/tv_add_2d_SAMPLE_2
```

Run:

```bash
LD_LIBRARY_PATH=$TORCH_LIB:/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
  /opt/pytorch/nvfuser/doc-bot/experimenting/8-22-2025/tv_add_2d_SAMPLE_2
```

### Troubleshooting and environment workarounds

See `../doc-bot/experimenting/9-3-2025/workarounds` for detailed notes. Highlights:

- Missing shared libraries (e.g., `libc10_cuda.so`):
  - Prepend runtime path:
    ```bash
    LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:\
    /opt/pytorch/nvfuser/python/nvfuser_common/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    <path-to-sample>
    ```

- Driver mismatch:
  - If your driver is too old/new for the CUDA toolkit, defer CUDA execution or use an environment that matches the driver.
  - For inspection without CUDA, use IR-only mode if supported by the sample:
    ```bash
    NVFUSER_IR_ONLY=1 <path-to-sample>
    ```

### Related references

- `../doc-bot/experimenting/how_to_build`
- `../doc-bot/experimenting/how_to_run`
- `../doc-bot/experimenting/setup_libs_to_run`
- `../doc-bot/experimenting/9-3-2025/workarounds`


