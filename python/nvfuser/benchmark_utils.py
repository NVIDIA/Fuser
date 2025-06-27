# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from torch.autograd import DeviceType
from torch.profiler import profile, ProfilerActivity
import torch


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


class TorchProfileTimer(Timer):
    def __init__(self):
        super().__init__()
        self.prof = profile(activities=[ProfilerActivity.CUDA])

    def _get_kernel_time(
        self, prof_averages: torch.autograd.profiler_util.EventList
    ) -> float:
        """
        Arguments:
            prof_averages: Output of self.prof.key_averages()
        Returns:
            time_value: Elapsed CUDA time in seconds.
        """
        elapsed_cuda_time = 0
        has_cuda_event = False
        for event in prof_averages:
            if event.device_type != DeviceType.CUDA:
                continue
            has_cuda_event = True
            # Re: torch profiler API changes in https://github.com/pytorch/pytorch/pull/123247
            elapsed_cuda_time = (
                elapsed_cuda_time + event.self_device_time_total
                if hasattr(event, "self_device_time_total")
                else event.self_cuda_time_total
            )
        assert has_cuda_event, "No CUDA events found"
        return elapsed_cuda_time / 1e6

    def __call__(self):
        """
        Custom torchprofiler-based timer used by pytest-benchmark.
        At every timer call, the profiler is stopped to compute the elapsed CUDA time
        and the global clock is incremented. The profiler is restarted before returning to continue tracing.

        Returns:
            self.current_time: Global monotonic clock variable
        """
        try:
            self.prof.stop()
        except AssertionError:
            self.prof.start()
            return self.current_time

        prof_averages = self.prof.key_averages()
        elapsed_cuda_time = self._get_kernel_time(prof_averages)
        self._increment_global_time(elapsed_cuda_time)
        # Clear the internal profiler object to avoid accumulating function events and then restart the profiler
        # See PR: https://github.com/pytorch/pytorch/pull/125510
        self.prof.profiler = None

        return self.current_time

    def cleanup(self):
        """
        Stops a running torchprofiler instance if found.
        """
        try:
            self.prof.stop()
        except AssertionError:
            pass


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
            profile = self.fd.profile()
            elapsed_host_time = profile.host_time_ms / 1e3
            self._increment_global_time(elapsed_host_time)
        self.execution_start = not self.execution_start
        return self.current_time
