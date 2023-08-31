# Build

```
mkdir build
cd build
CMAKE_PREFIX_PATH=$(python -c 'import nvfuser.utils; import torch.utils; print(nvfuser.utils.cmake_prefix_path, torch.utils.cmake_prefix_path, sep=";")')
cmake -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH -G Ninja ..
cmake --build .
```

# Test

```
./sinh_example
```
