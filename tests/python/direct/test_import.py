# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]


def test_import_correct():
    try:
        import nvfuser_direct  # noqa: F401
    except Exception as e:
        raise RuntimeError("Failed to import nvfuser_direct.")


def test_import_conflict_direct_then_nvfuser():
    try:
        import nvfuser_direct  # noqa: F401
        import nvfuser  # noqa: F401
    except AssertionError as e:
        expected_msg = (
            "Cannot import nvfuser if nvfuser_direct module is already imported."
        )
        assert expected_msg in str(e)
        return
    raise AssertionError("Expected AssertionError from imports.")
