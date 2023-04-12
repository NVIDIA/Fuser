# Build

```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import nvfuser; import torch; print(nvfuser.cmake_prefix_path, torch.utils.cmake_prefix_path, sep=";")')" ..
make -j
```

# Test

```
./sinh_example
```
