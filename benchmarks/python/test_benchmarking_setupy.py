# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os


def test_env_vars_are_set():
    for var in [
        "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING",
        "TORCHINDUCTOR_COORDINATE_DESCENT_CHECK_ALL_DIRECTIONS",
    ]:
        assert os.environ[var] == "1"
