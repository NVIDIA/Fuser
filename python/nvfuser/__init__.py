# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import sys
import warnings

if "nvfuser_direct" in sys.modules:
    warnings.warn(
        "Be careful! You've imported nvfuser when the nvfuser_direct module is already imported.",
        UserWarning,
    )

import logging
import os
import re
from typing import Callable
import warnings

import torch

# This is needed when libnvfuser.so is patched and doesn't have the pytorch library location available.
pytorch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
if pytorch_lib_dir not in sys.path:
    sys.path.append(pytorch_lib_dir)

# we need to import _C here to avoid confusing error message generated from failure in this python script ended up with
# complaining on `_C` not defined for `_C._FusionDefinition`
from . import _C
from ._C import *  # noqa: F401,F403

from . import contrib  # noqa: F401


logger = logging.getLogger("nvfuser")


# Register automatic serialization of Nvfuser cache hierarchy and cuda kernels.
def enable_automatic_serialization():
    import atexit

    atexit.register(_C.serialize)

    # A separate process is created for each device in a distributed setting.
    # Each FusionCache becomes associated with a single device.
    # Automatic serialization saves a separate cache for each device.
    # Set the FusionCache id to the ddp local rank.
    env_var_ddp_local_rank = os.environ.get("LOCAL_RANK", None)
    if env_var_ddp_local_rank is not None:
        env_var_ddp_local_rank = int(env_var_ddp_local_rank)
    _C.FusionCache.get(max_fusions := 8192, env_var_ddp_local_rank)


# Unregister automatic serialization of Nvfuser cache hierarchy and cuda kernels.
def disable_automatic_serialization():
    import atexit

    atexit.unregister(_C.serialize)


class FusionDefinition(_C._FusionDefinition):
    def __init__(
        self,
        id=None,
        max_length=9999,
        use_multidevice_executor=False,
        backend_type=CommunicatorBackend.nccl,
    ):
        super(FusionDefinition, self).__init__(
            id, max_length, use_multidevice_executor, backend_type
        )
        self.profiled = False

    def segment(self, inputs):
        """
        Decompose this FusionDefinition into a sequence of segment
        FusionDefinitions.

        This function runs the nvfuser segmentation algorithm and translates the
        segments into their corresponding FusionDefinitions.

        Args:
            inputs (List[Union[Tensor, Scalar]]): A list of inputs to fusion.

        Returns:
            List[FusionDefinition]: The FusionDefinitions corresponding to the
            sub-fusion segments of this FusionDefinition.
        """
        num_segments = self._setup_segmentation(inputs)
        if num_segments == 1:
            self._finalize_segmentation()
            return []

        # Track all segments for this FusionDefinition
        self.segments = []

        # Track map_segment_fid_to_original_fid for each segment
        self.segment_index_space_maps = {}

        # Track the last segment a value is used as an input
        self.map_value_to_last_used_segment = {}

        for idx in range(num_segments):
            new_fd = FusionDefinition()
            map_segment_fid_to_original_fid = self._build_segment(new_fd, idx)

            for segment_input in new_fd.inputs():
                original_input = map_segment_fid_to_original_fid[segment_input]
                self.map_value_to_last_used_segment[original_input] = idx

            self.segment_index_space_maps[new_fd] = map_segment_fid_to_original_fid
            self.segments.append(new_fd)
        self._finalize_segmentation()
        return self.segments

    def __enter__(self):
        return self._setup_definition()

    def __exit__(self, type, value, traceback):
        try:
            self._finalize_definition()
        except Exception as err:
            logger.exception(self._repro_error_str("defining"))
            raise

    def definition(self):
        raise NotImplementedError("definition() should be implemented by child class!")

    def _execute_segments(self, input_arguments, *, device=None, profile=False):
        """
        Run the sequence of FusionDefinition segments to generate the results
        of this FusionDefinition.

        This FusionDefinition acts an argument manager. It gathers input
        arguments for the segments and stores their output results. After
        running a segment, any redundant intermediate values, which are
        unnecessary for any other segments, are deleted to save memory.

        Args:
            inputs (List[Union[Tensor, Scalar]]): A list of inputs to fusion.

        Kwargs:
            device (Optional[Union[int, str, torch.device]]): This is a hint to run
                the Fusion on the given CUDA device. This is not typically
                necessary, as the device is usually inferred from the locations
                of input tensors. However, for some fusion definitions, no
                tensors will be input (for example when all tensors are
                generated with `full` or `uniform` ops). In these cases, we
                must either tell NVFuser where to run the resulting kernel, or
                let it default to 0. Note that passing this option providing
                and input tensors that lie on another device is an error.
            profile (bool): Captures a CUPTI based profile of a fusion.


        Returns:
            List[Tensor]: The output results for this FusionDefinition.
        """
        assert len(self.segments) > 0
        assert len(self.segments) == len(self.segment_index_space_maps)

        input_arguments_with_extents = [*input_arguments]
        for a in input_arguments:
            if type(a) is torch.Tensor:
                input_arguments_with_extents.extend(a.size())

        # Map inputs arguments to original fid
        map_original_fid_to_value = {
            fd_state: argument
            for fd_state, argument in zip(
                self.inputs() + self.extents(), input_arguments_with_extents
            )
        }

        # Run all segments in correct order
        for idx, segment in enumerate(self.segments):
            segment_to_original_map = self.segment_index_space_maps[segment]

            # Gather segment input arguments
            segment_arguments = [
                map_original_fid_to_value[segment_to_original_map[fd_state]]
                for fd_state in segment.inputs()
            ]

            # Run segment
            segment_outputs = segment.execute(
                segment_arguments, device=device, profile=profile
            )

            # Update original fusion definition indices to outputs
            for fd_state, output in zip(segment.outputs(), segment_outputs):
                map_original_fid_to_value[segment_to_original_map[fd_state]] = output

            # Destroy any arguments that are not used by future segments
            for segment_input in segment.inputs():
                original_input = segment_to_original_map[segment_input]
                if (
                    original_input not in self.outputs()
                    and self.map_value_to_last_used_segment[original_input] == idx
                ):
                    del map_original_fid_to_value[original_input]

        # Map output fid to actual results
        return [map_original_fid_to_value[fd_state] for fd_state in self.outputs()]

    def execute(
        self,
        inputs,
        *,
        device=None,
        override_user_schedule=False,
        capture_debug_output=False,
        print_repro=False,
        profile=False,
        save_repro_inputs=False,
        _enable_options: list[str] = [],
        _disable_options: list[str] = [],
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], list[Sharding]]:
        """
        Executes an nvFuser set of kernels for a given Fusion

        The FusionDefinition will be executed on a single CUDA device.
        Typically, which device to run on is determined by the devices where
        the input tensors reside. However, if the Fusion is defined such that
        none of the inputs are tensors, we are not able to infer a device from
        the inputs. For example, the following FusionDefinition will be unable
        to unambiguously infer the device of its output:

            with FusionDefinition() as fd:
                tv1 = fd.ops.full([5])
                fd.add_output(tv1)

        In that case, we default to selecting the first CUDA
        device, i.e. `torch.device("cuda:0")`. This method enables selecting an
        alternative preferred device.

        Args:
            inputs (List[Union[Tensor, Scalar]]): A list of inputs to fusion.

        Kwargs:
            device (Optional[Union[int, str, torch.device]]): This is a hint to run
                the Fusion on the given CUDA device. This is not typically
                necessary, as the device is usually inferred from the locations
                of input tensors. However, for some fusion definitions, no
                tensors will be input (for example when all tensors are
                generated with `full` or `uniform` ops). In these cases, we
                must either tell NVFuser where to run the resulting kernel, or
                let it default to 0. Note that passing this option providing
                and input tensors that lie on another device is an error.
            override_user_schedule (bool): For a user defined schedule,
                override with auto-generated schedule (default: False)
            capture_debug_output (bool): Whether to capture any printed
                debugging information as a string. If True, the string can be
                retrieved after execution using :meth:`get_debug_output`. If False,
                then that method will return None when called.
            print_repro (bool): Prints a reproduction script to stdout.
            profile (bool): Captures a CUPTI based profile of a fusion.
            save_repro_inputs (bool): Saves the inputs for last_repro_script() to
                provide a provide a reproduction script.
            _enable_options/_disable_options (list): NVFUSER_ENABLE/DISABLE options to use.
                This is an alternative to environment variables.
                Note: Currently, we do not cache/store these options in the FusionCache which makes it
                    plausible to reuse kernels when executing the same fusion definition with different sets of options.
                    Reset the FusionCache manually to avoid inadvertent kernel reuse when between different sets of options.

        Returns:
            A list of output tensors and, if multidevice_schedule is defined, a
            list of output shardings. The latter is important to pack the outputs
            into DTensors for framework integration.
        """
        self.profiled = profile

        if not isinstance(device, int) and device is not None:
            if not isinstance(device, torch.device):
                device = torch.device(device)
            assert (
                device.type == "cuda"
            ), "If device argument is passed it must be a CUDA device"
            device = device.index

        # if definition is not defined by a context manager, try a child class
        defined_multidevice_schedule = hasattr(self, "multidevice_schedule")
        if self.id() is None:
            self._setup_definition()
            self.definition()
            self._finalize_definition()

            defined_schedule = hasattr(self, "schedule") and isinstance(
                self.schedule, Callable
            )
            assert not (
                defined_multidevice_schedule and defined_schedule
            ), "I haven't tested what if both are defined. We don't plan to support this use case although it may just work."

            if defined_multidevice_schedule:
                # Unlike `schedule`, `multidevice_schedule` is designed for inter-device
                # scheduling, The scheduling is done before concretization and therefore
                # before pre-segmentation. `schedule` however assumes the FusionDefinition
                # has been concretized and pre-segmented, and therefore requires
                # `_setup_schedule` and `_finalize_schedule` to be called before and after.
                #
                # Note: there's a plan to embed multidevice schedules into FusionDefinition
                # as annotating nodes. This may eventually replace `multidevice_schedule`.
                self._setup_multidevice_schedule()
                self.multidevice_schedule()
                self._finalize_multidevice_schedule()

            # If schedule is defined by child class and schedule is not defined for
            # inputs, make a schedule.
            if defined_schedule:
                # Schedule fusion if it does not exist yet or profiling fusion
                if profile or not self._exist_schedule(inputs):
                    self._setup_schedule(inputs, overwrite_existing_schedule=profile)
                    self.schedule()
                    self._finalize_schedule(inputs)

        if save_repro_inputs:
            from torch._subclasses.fake_tensor import FakeTensorMode

            fake_mode = FakeTensorMode()
            self.fake_inputs = [fake_mode.from_tensor(inp) for inp in inputs]

        if hasattr(self, "segments") and len(self.segments) > 0:
            return self._execute_segments(inputs, device=device, profile=profile)

        try:
            if print_repro:
                print(self.repro_script_for(inputs))
            if len(_enable_options) or len(_disable_options):
                warnings.warn(
                    "Reset the FusionCache manually to avoid reusing kernels when re-executing the fusion definition with different options."
                )

            out_tensors, out_shardings = self._execute(
                inputs,
                device=device,
                override_user_schedule=override_user_schedule,
                capture_debug_output=capture_debug_output,
                profile=profile,
                _enable_options=_enable_options,
                _disable_options=_disable_options,
            )

            if defined_multidevice_schedule:
                return out_tensors, out_shardings

            assert len(out_shardings) == 0
            return out_tensors

        except Exception as err:
            logger.exception(self._repro_error_str("executing", inputs))
            raise

    def debug_output(self):
        """
        Retrieve string of captured debug information from the previous execution.

        Note that `capture_debug_output=True` must be passed to `execute()` in
        order to enable capturing this output. Otherwise, this method will
        return `None`.

        Returns:
            Optional[String] : the captured debug output for the previous call
            to execute(). If the `capture_debug_output` argument to that call
            was False, returns None. Otherwise, returns the output as a string.
        """
        return self._debug_output()

    def from_pytorch(self, tensor, static_sizes=False):
        """
        Defines an nvfuser input tensor from a pytorch tensor and defaults
        to definining a symbolic tensor for dynamic shape usage.

        Args:
            tensor (torch.Tensor): Input tensor to nvFuser
            static_sizes (bool)  : Interprets sizes as static rather than
                                   as symbolic for dynamic shape usage

        Returns:
            nvfuser.Tensor
        """
        try:
            from .pytorch_utils import torch_dtype_to_nvfuser_dtype
        except ImportError:
            raise ImportError("Unable to import pytorch_utils!")

        supported_tensor = tensor.is_cuda or (tensor.is_cpu and len(tensor.size()) == 0)
        if not supported_tensor:
            raise ValueError(
                f"Found unsupported device {tensor.device}, only scalar CPU or CUDA tensors are supported"
            )

        return self.define_tensor(
            sizes=tensor.size(),
            strides=tensor.stride(),
            dtype=torch_dtype_to_nvfuser_dtype(tensor.dtype),
            static_sizes=static_sizes,
            is_cpu=tensor.is_cpu,
        )

    def fusion_ir(self):
        """
        Returns the uscheduled Fusion IR for the given definition that corresponds to all scheduled inputs.

        Returns:
            String
        """
        return self._fusion_ir()

    def last_cuda_code(self, intrinsic_code=False, **kwargs):
        """
        Returns the Cuda Code for the last executed set of inputs

        Args:
            intrinsic_code (Bool): Include all the additional code required to run kernel(s). (default: False)

        Kwargs:
            override_user_schedule (Bool): For a user defined schedule, override with auto-generated schedule (default: False)

        Returns:
            String
        """
        override_user_schedule = kwargs.pop("override_user_schedule", False)
        return self._last_cuda_code(intrinsic_code, override_user_schedule)

    def cuda_code_for(self, inputs, intrinsic_code=False, **kwargs):
        """
        Returns the Cuda Code for the given inputs

        Args:
            inputs (List[Union[Tensor, Scalar]]): A list of inputs to fusion.
            intrinsic_code (Bool): Include all the additional code required to run kernel(s). (default: False)

        Kwargs:
            override_user_schedule (Bool): For a user defined schedule, override with auto-generated schedule (default: False)

        Returns:
            String
        """
        override_user_schedule = kwargs.pop("override_user_schedule", False)
        return self._cuda_code_for(inputs, intrinsic_code, override_user_schedule)

    def last_scheduled_fusion_ir(self, tensor_transforms=False, **kwargs):
        """
        Returns the Scheduled Fusion IR for the last executed set of inputs

        Args:
            tensor_transforms (Bool): Include tensor transforms that were applied through scheduling. (default: False)

        Kwargs:
            override_user_schedule (Bool): For a user defined schedule, override with auto-generated schedule (default: False)

        Returns:
            String
        """
        override_user_schedule = kwargs.pop("override_user_schedule", False)
        return self._last_scheduled_fusion_ir(tensor_transforms, override_user_schedule)

    def scheduled_fusion_ir_for(self, inputs, tensor_transforms=False, **kwargs):
        """
        Returns the Scheduled Fusion IR for the last executed set of inputs

        Args:
            inputs (List[Union[Tensor, Scalar]]): A list of inputs to fusion.
            tensor_transforms (Bool): Include tensor transforms that were applied through scheduling. (default: False)

        Kwargs:
            override_user_schedule (Bool): For a user defined schedule, override with auto-generated schedule (default: False)

        Returns:
            String
        """
        override_user_schedule = kwargs.pop("override_user_schedule", False)
        return self._scheduled_fusion_ir_for(
            inputs, tensor_transforms, override_user_schedule
        )

    def profile(self):
        """
        Returns the FusionProfile object from the CUPTI based FusionProfiler

        Returns:
            FusionProfile
        """
        if not self.profiled:
            raise ValueError(
                "The execute() method was not previously called with profiling enabled!"
            )

        fp = self._profile()

        if fp.fusion_id < 0:
            raise ValueError(
                "Something went wrong with Fusion Profiling as an illegal fusion_id was returned! "
                + str(fp.fusion_id)
            )
        if fp.segments < 1:
            raise ValueError(
                "Something went wrong with Fusion Profiling as no kernel segments were profiled!"
                + str(fp.segments)
            )

        return fp

    def last_repro_script(self) -> str:
        assert (
            self.fake_inputs is not None
        ), "fd.last_repro_script() cannot provide a repro because fd.execute(inputs, save_repro_state=True) was not executed!"
        script = self.repro_script_for(self.fake_inputs)
        return script

    def repro_script_for(self, inputs: list | None = None) -> str:
        msg = "# CUDA devices:\n"
        for i in range(torch.cuda.device_count()):
            msg += f"#  {i}: {torch.cuda.get_device_name(i)}\n"
        fusion_func_name = (
            "nvfuser_incomplete_fusion"
            if self.id() is None
            else f"nvfuser_fusion_id{self.id()}"
        )
        msg += (
            f"# torch version: {torch.__version__}\n"
            f"# cuda version: {torch.version.cuda}\n"
            f"# nvfuser version: {version()}\n"
            "import torch\n"
            "from nvfuser import FusionDefinition, DataType\n"
            f"{self}"
            "with FusionDefinition() as fd:\n"
            f"    {fusion_func_name}(fd)\n"
        )
        if inputs is not None:
            msg += "\ninputs = [\n"
            for i in inputs:
                if isinstance(i, torch.Tensor):
                    if i.is_contiguous():
                        msg += f"    torch.testing.make_tensor({tuple(i.size())}, dtype={i.dtype}, device='{i.device}'),\n"
                    else:
                        # max linear index determines number of elements to generate
                        sz = 1
                        for szi, stri in zip(i.size(), i.stride()):
                            if szi == 0:
                                sz = 0
                                break
                            sz += (szi - 1) * stri
                        if i.dtype.is_floating_point:
                            msg += (
                                f"    torch.randn({sz}, dtype={i.dtype}, device='{i.device}')"
                                f".as_strided({tuple(i.size())}, {tuple(i.stride())}),\n"
                            )
                        else:
                            upper_bound = 2 if i.dtype == torch.bool else 10
                            msg += (
                                f"    torch.randint(0, {upper_bound}, ({sz},), dtype={i.dtype}, device='{i.device}')"
                                f".as_strided({tuple(i.size())}, {tuple(i.stride())}),\n"
                            )
                else:
                    input_as_string = str(i)
                    # `nan` and `inf` are stringified as is, which are not
                    # defined in Python. So we replace them with `float("nan")`
                    # and `float("inf")`. `-inf` is replaced with
                    # `-float("inf")`, which equals `float("-inf")`.
                    input_as_string = re.sub(
                        r"\binf\b", 'float("inf")', input_as_string
                    )
                    input_as_string = re.sub(
                        r"\bnan\b", 'float("nan")', input_as_string
                    )
                    msg += f"    {input_as_string},\n"
            msg += "]"
            msg += "\nfd.execute(inputs)\n"

        return msg

    def _repro_error_str(self, section: str, inputs: list | None = None):
        msg = (
            f"An error occurred while {section} nvFuser FusionDefinition {self.id()}.\n"
            "If you believe this is a bug or need assistance, please file an issue at "
            "https://github.com/NVIDIA/Fuser/issues/new\n"
            f"Here's a script to reproduce the error:\n"
            "```python\n"
        )
        msg += self.repro_script_for(inputs)
        msg += "```\n"
        return msg

    def validate(
        self,
        inputs: list[torch.Tensor],
        reference_outputs: list[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Validates the fusion outputs against the provided reference outputs, using variable tolerances determined based on datatype and reduction size.

        Inputs:
            inputs: A list of inputs expected by the fusion definition
            reference_outputs: A list of reference outputs to validate against
        """
        fusion_outputs = self.execute(inputs, **kwargs)

        if reference_outputs is None:
            return self.validate_with_auto_inferred_outputs(fusion_outputs, inputs)

        assert len(fusion_outputs) == len(
            reference_outputs
        ), f"Expected {len(fusion_outputs)} reference outputs for validation."

        tolerance_values = self.getValTolerances(inputs)
        assert len(tolerance_values) == len(
            fusion_outputs
        ), f"Missing tolerance values, expected {len(fusion_outputs)}, got {len(tolerance_values)}"

        for inx, fusion_output in enumerate(fusion_outputs):
            atol, rtol = tolerance_values[inx]
            reference_output = reference_outputs[inx]

            assert (
                reference_output.shape == fusion_output.shape
            ), "Mismatch in reference and fusion output dimensions"
            if torch.is_floating_point(fusion_output) or torch.is_complex(
                fusion_output
            ):
                assert torch.allclose(
                    fusion_output, reference_output, atol=atol, rtol=rtol
                ), f"Max error: {torch.abs(torch.max(fusion_output - reference_output))}, \
                    Absolute tolerance: {atol}, Relative tolerance: {rtol}"

            else:
                assert torch.equal(
                    fusion_output, reference_output
                ), "Mismatch in reference and fusion output values, datatype is not float/complex."


from .nvfuser_version import __version__


def version():
    r"""returns nvfuser version in format of a string 'm.n.p+git[7d-sha]'.

    We strip the git[7d-sha] and convert the string to
    `nvfuser_version.Version` for comparison. e.g. you can use it as:
        import nvfuser
        print(nvfuser.version())              # 0.0.1+git21df524
        nvfuser.version() == '0.0.1`          # True
        nvfuser.version() > '0.0.0`           # True

        from nvfuser_version import Version
        nvfuser.version() < Version('1.0.0')  # True
    """
    return __version__
