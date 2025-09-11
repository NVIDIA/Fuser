# Build and Run Standalone nvFuser Samples (Heavily Commented)

## Prerequisites
- C++17 compiler (g++/clang++) and CUDA toolkit (for execution)
- PyTorch/ATen installed (wheel or from source); libs visible at runtime
- Repo path assumed: `/opt/pytorch/nvfuser`

## What you’ll learn
- Practical ways to compile and run small C++ nvFuser programs outside PyTorch proper
- How to satisfy include paths and link ATen/Torch + CUDA
- How to set `LD_LIBRARY_PATH` (or rpath) so binaries can find shared libraries
- How to run in IR-only mode when CUDA execution isn’t available

Primary references:
- Sample program: `../doc-bot/experimenting/8-22-2025/tv_add_2d_SAMPLE_2.cpp`
- Environment notes/workarounds: `../doc-bot/experimenting/9-3-2025/workarounds`

---

## Option A: Use the provided helper scripts (recommended)

If available in your tree, prefer the helper scripts; they encapsulate include paths, link flags, and rpath.

- Build helper: `../doc-bot/how_to_build`
- Run helper: `../doc-bot/how_to_run`

Example (heavily commented):

```bash
# 1) Build a sample into a colocated binary
#    The helper takes a C++ source path and produces a binary alongside it.
#    Internally, it sets include paths, Torch/nvfuser link flags, and rpath.
source ../doc-bot/how_to_build ../doc-bot/experimenting/8-22-2025/tv_add_2d_SAMPLE_2.cpp

# 2) Run the produced binary
#    The run helper ensures the necessary LD_LIBRARY_PATH (or uses baked-in rpath)
#    so libtorch/libc10/libc10_cuda and nvFuser libs are discoverable.
source ../doc-bot/how_to_run ../doc-bot/experimenting/8-22-2025/tv_add_2d_SAMPLE_2
```

Quick verify:
```bash
# You should see a printed 2D tensor (C) and IR dumps if the sample prints them
```

---

## Option B: Manual build (portable fallback)

Below is a template showing the moving parts. Adjust paths to match your environment.

```bash
# 0) Choose a compiler (g++ or clang++) with C++17 and CUDA-capable toolchain
CXX=${CXX:-g++}

# 1) Locate Torch/ATen headers and libs. If you installed torch via pip,
#    headers and libs typically live under site-packages/torch/include and torch/lib.
PY_TORCH_LIB=/usr/local/lib/python3.12/dist-packages/torch/lib
PY_TORCH_INC=/usr/local/lib/python3.12/dist-packages/torch/include

# 2) nvFuser headers (repo checkout). The sample includes <fusion.h>, <ops/...> etc.
#    Point an include to the nvfuser source tree so headers resolve.
NVFUSER_SRC=/opt/pytorch/nvfuser

# 3) CUDA SDK path (for libcudart and device headers), if needed at link/run time.
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

# 4) Build flags (C++17, PIC, warnings). Torch needs -D_GLIBCXX_USE_CXX11_ABI=1 or 0 depending on your build;
#    most modern wheels use the new ABI (1). If you see link errors, toggle this.
CXXFLAGS="-std=c++17 -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=1"

# 5) Include paths: nvFuser source includes + Torch headers (two-level include for torch/extension)
INCLUDES="-I${NVFUSER_SRC} -I${NVFUSER_SRC}/csrc -I${PY_TORCH_INC} -I${PY_TORCH_INC}/torch/csrc/api/include"

# 6) Libraries to link: Torch, C10, CUDA. Order matters on some systems.
LDFLAGS="-L${PY_TORCH_LIB} -L${CUDA_HOME}/lib64 \
  -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -lcudart"

# 7) rpath so the binary finds libs without setting LD_LIBRARY_PATH
RPATH="-Wl,-rpath,${PY_TORCH_LIB} -Wl,-rpath,${CUDA_HOME}/lib64"

# 8) Compile the sample
${CXX} ${CXXFLAGS} ${INCLUDES} \
  ../doc-bot/experimenting/8-22-2025/tv_add_2d_SAMPLE_2.cpp \
  -o ./tv_add_2d_SAMPLE_2 ${LDFLAGS} ${RPATH}
```

Notes:
- If `-ltorch_cuda` is not present in your wheel, link `-ltorch` and `-lc10` and rely on `-lcuda`/`-lcudart` from the CUDA SDK. Check `${PY_TORCH_LIB}` for actual library filenames.
- If the compiler cannot find `<fusion.h>`, double-check the `-I` paths to nvFuser headers (`${NVFUSER_SRC}` and `${NVFUSER_SRC}/csrc`).

---

## Environment setup for running

If the binary cannot locate libraries at runtime (common error: `error while loading shared libraries: libc10_cuda.so`), set `LD_LIBRARY_PATH` to include Torch, nvFuser common, and CUDA libraries. These values are from our notes:

```bash
# Export library search paths for a one-off run
export LD_LIBRARY_PATH=\
/usr/local/lib/python3.12/dist-packages/torch/lib:\
/opt/pytorch/nvfuser/python/nvfuser_common/lib:\
/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Then run your binary
./tv_add_2d_SAMPLE_2
```

Reference: `../doc-bot/experimenting/9-3-2025/workarounds`

Driver note: If your NVIDIA driver is too old for the installed CUDA or Torch, execution will fail. In that case, use IR-only mode (below) to still inspect graphs.

---

## IR-only mode (no CUDA execution needed)

When CUDA execution isn’t possible (e.g., driver mismatch), you can still build and run samples purely to print Fusion IR and transforms.

```bash
# Prevent device execution; still allows printing IR and transforms
export NVFUSER_IR_ONLY=1
./tv_add_2d_SAMPLE_2  # program prints IR without launching GPU kernels
```

This mode is useful for documentation, debugging IR shapes, and validating schedules without needing a working CUDA runtime.

---

## Troubleshooting checklist

- Includes not found: verify `-I${NVFUSER_SRC}` and `-I${NVFUSER_SRC}/csrc` point to your nvFuser headers
- Link errors (undefined refs): check Torch libs in `${PY_TORCH_LIB}` and adjust `-D_GLIBCXX_USE_CXX11_ABI` if needed
- Runtime shared library errors: set `LD_LIBRARY_PATH` (or use `-Wl,-rpath,...`) as shown above
- Driver/CUDA mismatch: try `NVFUSER_IR_ONLY=1` to run in IR-only mode

---

## Appendix: Inline sanity check

A tiny IR-only test (no ATen) to confirm headers and IR print work:

```cpp
// file: sanity_ir_only.cpp
#include <fusion.h>
#include <fusion_guard.h>
#include <ops/arith.h>
#include <ops/alias.h>
#include <ir/iostream.h>
using namespace nvfuser;
int main(){
  Fusion f; FusionGuard g(&f);
  auto a = iota(IrBuilder::create<Val>(16, DataType::Index),
                IrBuilder::create<Val>(0.0, DataType::Float),
                IrBuilder::create<Val>(1.0, DataType::Float), DataType::Float);
  auto b = set(a); auto c = add(a,b); f.addOutput(c);
  f.print(std::cout, true); return 0; }
```

Build (adjust paths as in the manual build section; no ATen or CUDA link needed):

```bash
${CXX} -std=c++17 -I${NVFUSER_SRC} -I${NVFUSER_SRC}/csrc \
  sanity_ir_only.cpp -o sanity_ir_only
./sanity_ir_only | head -n 50 | cat
```
