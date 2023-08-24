# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


def get_available_memory_gb():
    """Returns the available memory in GB."""
    try:
        import psutil

        return psutil.virtual_memory().available / 1024 / 1024 / 1024
    except:  # noqa: E722
        pass

    try:
        with open("/proc/meminfo", "r") as f:
            while True:
                line = f.readline()
                if line.startswith("MemAvailable:"):
                    mem = line.split()[1]
                    assert line.split()[2] == "kB"
                    return int(mem) / 1024 / 1024
                if not line:
                    break
    except:  # noqa: E722
        pass

    return 0
