# Build

```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import nvfuser.utils; import torch.utils; print(nvfuser.utils.cmake_prefix_path, torch.utils.cmake_prefix_path, sep=";")')" ..
make -j
```

# Test

```
./sinh_example
```
