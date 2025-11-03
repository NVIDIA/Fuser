# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from cupti import cupti
import cxxfilt
import pytest
from ._C_DIRECT import get_fusion_profile


# Base class for all timers used by pytest-benchmark.
class Timer:
    def __init__(self):
        self.current_time = 0.0

    def _increment_global_time(self, elapsed_time: float) -> None:
        self.current_time += elapsed_time

    def __call__(self):
        raise NotImplementedError("Subclass must implement this method")

    def cleanup(self):
        pass


def demangle_kernel_name(mangled_name):
    try:
        return cxxfilt.demangle(mangled_name)
    except Exception:
        return mangled_name  # Return original if demangling fails


def cupti_call_safe(func, *args):
    """Wrapper for CUPTI calls. Failing CUPTI calls will exit the program."""
    try:
        return func(*args)
    except Exception as e:
        print(f"CUPTI call {func.__name__} failed: {e}")
        pytest.exit(1)


class CuptiProfiler:
    # List of activities to be recorded by CUPTI.
    activity_kinds: list[cupti.ActivityKind] = [
        cupti.ActivityKind.CONCURRENT_KERNEL,
    ]

    # Private class variable to store the subscriber handle.
    __subscriber_handle = None

    def _error_if_not_valid(self) -> None:
        if not self.is_valid:
            raise RuntimeError(
                "CuptiProfiler is not valid. " "This instance has been torn down."
            )

    def _func_buffer_requested(self) -> tuple[int, int]:
        # 8MB buffer size as recommended by CUPTI samples.
        # max_num_records=0 indicates the buffer is filled with as many records as possible.
        buffer_size = 8 * 1024 * 1024
        max_num_records = 0
        return buffer_size, max_num_records

    def _func_buffer_completed(self, activities: list[cupti.ActivityAPI]) -> None:
        for activity in activities:
            # Activity.end and Activity.start are in nanoseconds.
            duration = (activity.end - activity.start) / 1e9
            self.profiler_output.append((demangle_kernel_name(activity.name), duration))

    def __init__(self):
        if CuptiProfiler.__subscriber_handle is not None:
            raise RuntimeError(
                "Only one instance of CuptiProfiler can be created. "
                "CUPTI only supports one subscriber at a time."
            )

        self.profiler_output: list[tuple[str, float]] = []

        # Subscribe to CUPTI and register activity callbacks.
        CuptiProfiler.__subscriber_handle = cupti_call_safe(cupti.subscribe, None, None)
        cupti_call_safe(
            cupti.activity_register_callbacks,
            self._func_buffer_requested,
            self._func_buffer_completed,
        )
        self.is_valid = True

    def start(self) -> None:
        self._error_if_not_valid()
        cupti_call_safe(cupti.activity_flush_all, 1)
        self.profiler_output = []
        for activity_kind in CuptiProfiler.activity_kinds:
            cupti_call_safe(cupti.activity_enable, activity_kind)

    def stop(self) -> list[tuple[str, float]]:
        self._error_if_not_valid()
        for activity_kind in CuptiProfiler.activity_kinds:
            cupti_call_safe(cupti.activity_disable, activity_kind)
        cupti_call_safe(cupti.activity_flush_all, 0)
        return self.profiler_output

    def teardown_cupti(self) -> None:
        self._error_if_not_valid()
        if CuptiProfiler.__subscriber_handle is None:
            return
        cupti_call_safe(cupti.unsubscribe, CuptiProfiler.__subscriber_handle)
        cupti_call_safe(cupti.finalize)
        CuptiProfiler.__subscriber_handle = None
        # Invalidate the profiler so it cannot be used again.
        self.is_valid = False


class CuptiTimer(Timer):
    def __init__(self):
        super().__init__()
        self.cupti_profiler = CuptiProfiler()
        self.is_running = False

    def __call__(self):
        torch.cuda.synchronize()

        if not self.is_running:
            self.cupti_profiler.start()
            self.is_running = True
            return self.current_time

        profiler_output = self.cupti_profiler.stop()
        self.is_running = False

        # Check if any activities were recorded
        if len(profiler_output) == 0:
            self.cleanup()
            raise RuntimeError("No activities were recorded.")

        self._increment_global_time(sum(duration for _, duration in profiler_output))
        return self.current_time

    def cleanup(self):
        self.is_running = False
        self.cupti_profiler.teardown_cupti()


class FusionProfileTimer(Timer):
    def __init__(self):
        super().__init__()
        self.fd = None
        # Specifies if the timer in host measurement is called at the start/finish of execution.
        # Timings are measured at the end of execution.
        self.execution_start = True

    def set_fd(self, fd):
        self.fd = fd

    def __call__(self):
        if not self.execution_start:
            profile = get_fusion_profile()
            elapsed_host_time = profile.host_time_ms / 1e3
            self._increment_global_time(elapsed_host_time)
        self.execution_start = not self.execution_start
        return self.current_time
