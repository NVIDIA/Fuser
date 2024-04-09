# Build

```
mkdir -p build
cmake -B build
ninja -C build
```

# Test

```
NVFUSER_DISABLE=matmul_expr_eval NVFUSER_MATMUL_HEURISTIC_PLUGIN=build/libmatmul_heuristic_plugin.so ../../build/test_matmul --gtest_filter='*EpilogueBias*'
```
