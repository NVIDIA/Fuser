# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import pytest


def pytest_runtest_call(item: "Item") -> None:
    """Called to run the test for test item (the call phase).

    The default implementation calls ``item.runtest()``.
    """
    retry = False
    try:
        print("first runtest")
        item.runtest()
    except torch.OutOfMemoryError:
        retry = True

    if not retry:
        return

    # We have hit an OOM error, so clear the cache and retry
    gc.collect()
    torch.cuda.empty_cache()

    print("collected garbage")
    try:
        item.runtest()
    except torch.OutOfMemoryError as e:
        # If we hit an OOM this time, then skip the test
        import pytest

        pytest.skip(f"Test failed due to OutOfMemoryError: {e}")

