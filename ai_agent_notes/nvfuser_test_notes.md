# NVFuser Test Process

## Three Required Steps

1. Find container name:
```bash
docker ps
# Look for container running nvfuser-dev:csarofeen image
```

2. Build (if needed):
```bash
docker exec nvfuser-dev:csarofeen /bin/bash -c "cd /opt/pytorch/Fuser && pip install . -v"
```

3. Run test with output capture:
```bash
docker exec nvfuser-dev:csarofeen /bin/bash -c "/opt/pytorch/Fuser/bin/test_nvfuser --gtest_filter='TEST_FILTER' ; echo '=== TEST COMPLETE ==='" > local_test_log.txt 2>&1
```

## Building and Running the sinh_libtorch Example

1. **Find the Docker Container**:
   ```bash
   docker ps
   # Look for container running nvfuser-dev:csarofeen image
   ```

2. **Navigate to the Example Directory**:
   ```bash
   docker exec <container_name> /bin/bash -c "cd /opt/pytorch/Fuser/examples/sinh_libtorch"
   ```

3. **Build the Example**:
   ```bash
   docker exec <container_name> /bin/bash -c "cd /opt/pytorch/Fuser/examples/sinh_libtorch && make"
   ```

4. **Run the Example**:
   ```bash
   docker exec <container_name> /bin/bash -c "cd /opt/pytorch/Fuser/examples/sinh_libtorch && ./sinh_example"
   ```

5. **Verify Output**:
   - Ensure the output matches the expected results as shown in the console.

## Notes
- Replace `TEST_FILTER` with the specific test name (e.g. `NVFuserTest.FusionMagicSchedulerInstanceNormalization_CUDA`)
- Output will be saved to `local_test_log.txt`
- Ensure the correct installation paths for Torch and nvFuser are set in the environment. This may involve checking the `CMAKE_PREFIX_PATH` or `Torch_DIR` and `Nvfuser_DIR` variables in the CMake configuration. 