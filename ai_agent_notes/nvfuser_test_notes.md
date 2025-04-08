# NVFuser Build and Test Instructions

## Environment
- Container: `bold_gould` (image: nvfuser:csarofeen)
- Build directory: `/opt/pytorch/Fuser/`

## Docker Command Format
Always use `/bin/bash -c` format for docker exec commands to ensure proper command execution:
```bash
docker exec bold_gould /bin/bash -c "command"
```

## Build Command
```bash
docker exec bold_gould /bin/bash -c "cd /opt/pytorch/Fuser && pip install ."
```

## Test Command
Add completion marker to ensure command termination is detected. Replace TEST_FILTER with desired test name:
```bash
docker exec bold_gould /bin/bash -c "/opt/pytorch/Fuser/bin/test_nvfuser --gtest_filter='TEST_FILTER' ; echo '=== TEST COMPLETE ==='"
```

### Example Test Filters
- Single test: `*FusionMagicSchedulerInstanceNormalization_CUDA`
- All CUDA tests: `*_CUDA`
- All tests in a category: `FusionMagicScheduler*`
- Specific test case: `FusionMagicScheduler.LayerNorm_CUDA`

## Example Interaction
```
> Run build command
"Build command completed successfully"
> Run test command with filter "*FusionMagicSchedulerInstanceNormalization_CUDA"
"Test command completed (failed)"
```
or
```
> Run build command
"Build command completed successfully"
> Run test command with filter "FusionMagicScheduler*"
"Test command completed (succeeded)"
``` 