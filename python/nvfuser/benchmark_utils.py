# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import atexit
import sys

try:
    from cupti import cupti
except ImportError:
    print("CUPTI not installed. Installing cupti-python...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cupti-python"])
    from cupti import cupti


# try:
#     import cxxfilt
#     def demangle_kernel_name(mangled_name):
#         try:
#             return cxxfilt.demangle(mangled_name)
#         except:
#             return mangled_name  # Return original if demangling fails
# except ImportError:
#     def demangle_kernel_name(mangled_name):
#         return mangled_name  # Return original if cxxfilt not available

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

class CuptiProfiler:
    # List of activities to be recorded by CUPTI.
    activity_kinds = [
        cupti.ActivityKind.CONCURRENT_KERNEL,
    ]
    _subscriber_handle = None
    
    # Failing CUPTI calls will exit the program.
    @staticmethod
    def with_error_handling(func, *args):
        try:
            return func(*args)
        except Exception as e:
            print (f"CUPTI call {func.__name__} failed: {e}")
            sys.exit(1)
    
    @staticmethod
    def enable_cupti_activities():
        for activity_kind in CuptiProfiler.activity_kinds:
            CuptiProfiler.with_error_handling(cupti.activity_enable, activity_kind)
    
    @staticmethod
    def disable_cupti_activities():
        for activity_kind in CuptiProfiler.activity_kinds:
            CuptiProfiler.with_error_handling(cupti.activity_disable, activity_kind)
    
    @staticmethod
    def func_buffer_requested():
        buffer_size = 8 * 1024 * 1024
        max_num_records = 0
        return buffer_size, max_num_records
    
    def func_buffer_completed(self, activities: list[cupti.ActivityAPI]):
        for activity in activities:
            duration = (activity.end - activity.start) / 1e9
            self.profiler_output[activity.name] = duration
    
    def __init__(self):
        if CuptiProfiler._subscriber_handle is not None:
          raise RuntimeError(
            "Only one instance of CuptiProfiler can be created. "
            "CUPTI only supports one subscriber at a time."
          )
        
        atexit.register(self.teardown_cupti)
        self.profiler_output = dict()
            
        # Subscribe to CUPTI and register activity callbacks.
        CuptiProfiler._subscriber_handle = CuptiProfiler.with_error_handling(cupti.subscribe, None, None)
        CuptiProfiler.with_error_handling(cupti.activity_register_callbacks, self.func_buffer_requested, self.func_buffer_completed)
    
    def start(self):
        CuptiProfiler.with_error_handling(cupti.activity_flush_all, 1)
        self.profiler_output.clear()
        self.enable_cupti_activities()
    
    def stop(self):
        self.disable_cupti_activities()
        CuptiProfiler.with_error_handling(cupti.activity_flush_all, 0)
        return self.profiler_output
    
    @classmethod
    def teardown_cupti(cls):
        if cls._subscriber_handle is None:
            return

        CuptiProfiler.with_error_handling(cupti.unsubscribe, cls._subscriber_handle)
        CuptiProfiler.with_error_handling(cupti.finalize)
        cls._subscriber_handle = None
      

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
        if not profiler_output:
            self.cleanup()
            raise RuntimeError("No activities were recorded.")
            
        self._increment_global_time(sum(profiler_output.values()))
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
            profile = self.fd.profile()
            elapsed_host_time = profile.host_time_ms / 1e3
            self._increment_global_time(elapsed_host_time)
        self.execution_start = not self.execution_start
        return self.current_time
