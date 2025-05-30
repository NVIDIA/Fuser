# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from torch.autograd import DeviceType
from torch.profiler import profile, ProfilerActivity
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

class CuptiTimer(Timer):
    def _cudaGetErrorEnum(self, error):
      if isinstance(error, cupti.CUresult):
          err, name = cupti.cuGetErrorName(error)
          return name if err == cupti.CUresult.CUDA_SUCCESS else "<unknown>"
      elif isinstance(error, cupti.cudaError_t):
          return cupti.cudaGetErrorName(error)[1]
      elif isinstance(error, cupti.nvrtcResult):
          return cupti.nvrtcGetErrorString(error)[1]
      else:
        raise RuntimeError("Unknown error type: {}".format(error))


    def checkCudaErrors(self, result):
        if result[0].value:
            raise RuntimeError(
                "CUDA error code={}({})".format(
                    result[0].value, _cudaGetErrorEnum(result[0])
                )
            )
        if len(result) == 1:
            return None
        elif len(result) == 2:
            return result[1]
        else:
            return result[1:]
      
    def func_buffer_requested(self):
      buffer_size = 8 * 1024 * 1024
      max_num_records = 0
      return buffer_size, max_num_records
    
    def func_buffer_completed(self, activities):
      for activity in activities:
        duration = (activity.end - activity.start) / 1e3
        self.timings[activity.name] = duration
        print(f"Activity: {activity.name}, Duration: {duration} ms")

    def cupti_initialize(self):
      try:
        for activity in self.activity:
            cupti.activity_enable(activity)

      except cupti.cuptiError as e:
          print(f"Error while enabling Activity Kind {activity.name} : {e}")
          sys.exit(3) 
      cupti.activity_register_callbacks(self.func_buffer_requested, self.func_buffer_completed)
      
    def on_profiler_start(self):
        self.cupti_initialize()
        
    def on_profiler_stop(self):
        cupti.activity_flush_all(0)
        for activity in self.activity:
            cupti.activity_disable(activity)
    
    def at_exit_handler(self):
        cupti.activity_flush_all(1)
    
    def callback(self, userdata, domain, callback_id, callback_data):
        if callback_id == cupti.driver_api_trace_cbid.cuProfilerStart:
            if callback_data.callback_site == cupti.ApiCallbackSite.API_EXIT:
                self.on_profiler_start()
        if callback_id == cupti.driver_api_trace_cbid.cuProfilerStop:
            if callback_data.callback_site == cupti.ApiCallbackSite.API_ENTER:
                self.on_profiler_stop()
    
    def __init__(self):
      super().__init__()
      self.is_running = False
      self.timings = dict()
      self.activity = [cupti.ActivityKind.CONCURRENT_KERNEL]
        
      # Try to clean up any existing CUPTI state
      try:
          # First try to flush any existing activities
          cupti.activity_flush_all(1)
          
          # Try to disable any existing activities
          for activity in self.activity:
              try:
                  cupti.activity_disable(activity)
              except Exception:
                  pass  # Ignore errors if activity wasn't enabled
                  
          # Try to finalize CUPTI
          try:
              cupti.finalize()
          except Exception:
              pass  # Ignore errors if CUPTI wasn't initialized
              
      except Exception as e:
          print(f"Warning: Error cleaning up existing CUPTI state: {e}")
      
      atexit.register(self.at_exit_handler)
      
      # Now initialize our subscriber
      try:
          self.subscriber_handle = cupti.subscribe(self.callback, None)
          cupti.enable_callback(
              1,
              self.subscriber_handle,
              cupti.CallbackDomain.DRIVER_API,
              cupti.driver_api_trace_cbid.cuProfilerStart,
          )
          cupti.enable_callback(
              1,
              self.subscriber_handle,
              cupti.CallbackDomain.DRIVER_API,
              cupti.driver_api_trace_cbid.cuProfilerStop,
          )
      except Exception as e:
          print(f"Error initializing CUPTI subscriber: {e}")
          raise
    
    def __call__(self):
        torch.cuda.synchronize()
        if not self.is_running:
            self.timings.clear()
            self.is_running = True
            self.checkCudaErrors(cuda.cuProfilerStart())
            return self.current_time

        self.checkCudaErrors(cuda.cuProfilerStop())
        self.is_running = False
        
        # Check if any activities were recorded
        try:
          assert self.timings, "No activities were recorded"
        except AssertionError as e:
          self.cleanup()
          raise e
          
        self._increment_global_time(sum(self.timings.values()))
        self.timings.clear()
        return self.current_time
    
    def cleanup(self):
        for activity in self.activity:
            cupti.activity_disable(activity)
        cupti.unsubscribe(self.subscriber_handle)

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
