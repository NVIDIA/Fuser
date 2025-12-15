# How to run multidevice tests?

1. Install NvFuser with distributed supported enabled. Set `export NVFUSER_BUILD_WITHOUT_DISTRIBUTED=0`.
2. Check that `NVFUSER_DISABLE` environment variable does not contain `multidevice`. Do not `export NVFUSER_DISABLE=multidevice`.
3. Install test specific dependencies with `pip install pytest-mpi mpi4py`
4. Run command: `mpirun -np [num_devices] pytest tests/python/multidevice/[test_name].py --only-mpi -s`
