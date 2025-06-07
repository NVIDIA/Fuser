# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from cupti import cupti
import atexit
from cuda import cuda

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

class CuptiManager:
    # List of activities to be recorded by CUPTI.
    activity_kinds = [
        cupti.ActivityKind.CONCURRENT_KERNEL,
    ]
    
    _cupti_manager = None
    _profiler_output = dict()
    _subscriber_handle = None
    
    @staticmethod
    def _cudaGetErrorEnum(error):
        if isinstance(error, cupti.CUresult):
            err, name = cupti.cuGetErrorName(error)
            return name if err == cupti.CUresult.CUDA_SUCCESS else "<unknown>"
        elif isinstance(error, cupti.cudaError_t):
            return cupti.cudaGetErrorName(error)[1]
        elif isinstance(error, cupti.nvrtcResult):
            return cupti.nvrtcGetErrorString(error)[1]
        else:
            raise RuntimeError("Unknown error type: {}".format(error))
   
    @staticmethod
    def checkCudaErrors(result):
        if result[0].value:
            raise RuntimeError(
                "CUDA error code={}({})".format(
                    result[0].value, CuptiManager._cudaGetErrorEnum(result[0])
                )
            )
        if len(result) == 1:
            return None
        elif len(result) == 2:
            return result[1]
        else:
            return result[1:]
    
    @staticmethod
    def enable_cupti_activities():
        for activity_kind in CuptiManager.activity_kinds:
            cupti.activity_enable(activity_kind)
    
    @staticmethod
    def disable_cupti_activities():
        for activity_kind in CuptiManager.activity_kinds:
            cupti.activity_disable(activity_kind)
    
    @staticmethod
    def func_buffer_requested():
        buffer_size = 8 * 1024 * 1024
        max_num_records = 0
        return buffer_size, max_num_records
    
    @classmethod
    def func_buffer_completed(activities: list[cupti.Activity]):
        for activity in activities:
            duration = (activity.end - activity.start) / 1e3
            CuptiManager._profiler_output[activity.name] = duration
    
    @staticmethod
    def on_profiler_start():
        CuptiManager.enable_cupti_activities()
    
    @staticmethod
    def on_profiler_stop():
        cupti.activity_flush_all(0)
        CuptiManager.disable_cupti_activities()
        
    @staticmethod
    def callback(userdata, domain, callback_id, callback_data):
        if callback_id == cupti.driver_api_trace_cbid.cuProfilerStart:
            if callback_data.callback_site == cupti.ApiCallbackSite.API_EXIT:
                CuptiManager.on_profiler_start()
        if callback_id == cupti.driver_api_trace_cbid.cuProfilerStop:
            if callback_data.callback_site == cupti.ApiCallbackSite.API_ENTER:
                CuptiManager.on_profiler_stop()
    
    def __init__(self):
        assert CuptiManager._subscriber_handle is None, "CUPTI subscriber already initialized."
        
        # Subscribe to CUPTI and register callbacks for cuProfilerStart and cuProfilerStop
        try:
            CuptiManager._subscriber_handle = cupti.subscribe(self.callback, None)
            cupti.enable_callback(
                1,
                CuptiManager._subscriber_handle,
                cupti.CallbackDomain.DRIVER_API,
                cupti.driver_api_trace_cbid.cuProfilerStart,
            )
            cupti.enable_callback(
                1,
                CuptiManager._subscriber_handle,
                cupti.CallbackDomain.DRIVER_API,
                cupti.driver_api_trace_cbid.cuProfilerStop,
            )
            cupti.activity_register_callbacks(CuptiManager.func_buffer_requested, CuptiManager.func_buffer_completed)

        except Exception as e:
            raise RuntimeError(f"Error initializing CUPTI manager: {e}")

    @classmethod
    def get_cupti_manager(cls):
        if not cls._cupti_manager:
            cls._cupti_manager = CuptiManager()
        return cls._cupti_manager
      
    @classmethod
    def get_profiler_output(cls):
        return CuptiManager._profiler_output
    
    @classmethod
    def teardown_cupti(cls):
        CuptiManager.disable_cupti_activities()
        cupti.activity_flush_all(1)
        cupti.unsubscribe(CuptiManager._subscriber_handle)
        cupti.finalize()
        CuptiManager._profiler_output.clear()
        CuptiManager._subscriber_handle = None
        CuptiManager._cupti_manager = None

    @classmethod
    def start_profiling(cls) -> None:
        cls.checkCudaErrors(cuda.cuProfilerStart())

    @classmethod
    def stop_profiling(cls) -> dict:
        CuptiManager.checkCudaErrors(cuda.cuProfilerStop())
        return CuptiManager._profiler_output

class CuptiTimer(Timer):
    def __init__(self):
        super().__init__()
        self.cupti_manager = CuptiManager.get_cupti_manager()
        self.is_running = False

    def __call__(self):
        torch.cuda.synchronize()
        
        if not self.cupti_manager.is_running:
            self.cupti_manager.start_profiling()
            return self.current_time

        timings = self.cupti_manager.stop_profiling()
        
        # Check if any activities were recorded
        try:
            assert timings, "No activities were recorded"
        except AssertionError as e:
            self.cleanup()
            raise e
            
        self._increment_global_time(sum(timings.values()))
        return self.current_time
    
    def cleanup(self):
        self.cupti_manager.cleanup()

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
