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

## Notes
- Replace `TEST_FILTER` with the specific test name (e.g. `NVFuserTest.FusionMagicSchedulerInstanceNormalization_CUDA`)
- Output will be saved to `local_test_log.txt` 