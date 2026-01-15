# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os


def pytest_configure(config):
    """
    Hook called after command line options have been parsed and all plugins
    and initial conftest files have been loaded.
    """
    # Append to NVFUSER_ENABLE environment variable for all tests in this directory
    existing = os.environ.get("NVFUSER_ENABLE", "")
    new_options = "id_model,id_model_extra_validation"

    if existing:
        os.environ["NVFUSER_ENABLE"] = f"{existing},{new_options}"
    else:
        os.environ["NVFUSER_ENABLE"] = new_options


def pytest_make_parametrize_id(config, val, argname):
    return f"{argname}={val}"
