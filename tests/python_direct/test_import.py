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
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        import nvfuser_direct  # noqa: F401
        import nvfuser  # noqa: F401

        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert (
            "Be careful! You've imported nvfuser when the nvfuser_direct module is already imported."
            in str(w[-1].message)
        )
