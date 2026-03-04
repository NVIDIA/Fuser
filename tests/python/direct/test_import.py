# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]


def test_import_correct():
    try:
        import nvfuser_direct  # noqa: F401
    except Exception as e:
        raise RuntimeError("Failed to import nvfuser_direct.")
