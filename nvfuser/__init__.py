# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import os
import sys
from typing import Optional, Union  # noqa: F401

import torch

# This is needed when libnvfuser.so is patched and doesn't have the pytorch library location available.
pytorch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
if pytorch_lib_dir not in sys.path:
    sys.path.append(pytorch_lib_dir)

# we need to import _C here to avoid confusing error message generated from failure in this python script ended up with
# complaining on `_C` not defined for `_C._FusionDefinition`
try:
    from . import _C
except ImportError as err:
    logging.getLogger("nvfuser").error(
        """==== importing nvfuser failed ====
             try run `patch-nvfuser` if https://github.com/NVIDIA/Fuser is installed via pip package"""
    )
    raise err
from ._C import *  # noqa: F401,F403

from . import contrib  # noqa: F401


logger = logging.getLogger("nvfuser")


class FusionDefinition(_C._FusionDefinition):
    def __enter__(self):
        return self._setup_definition()

    def __exit__(self, type, value, traceback):
        self._finalize_definition()

    def definition(self):
        raise NotImplementedError("definition() should be implemented by child class!")

    def schedule(self):
        raise NotImplementedError("schedule() should be implemented by child class!")

    def execute(self, inputs, *, device=None, **kwargs):
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
            override_user_schedule (bool): For a user defined schedule, override with auto-generated schedule (default: False)
            device (Optional[Union[int, str, torch.device]]): This is a hint to run
            the Fusion on the given CUDA device. This is not typically
            necessary, as the device is usually inferred from the locations of
            input tensors. However, for some fusion definitions, no tensors
            will be input (for example when all tensors are generated with
            `full` or `uniform` ops). In these cases, we must either tell
            NVFuser where to run the resulting kernel, or let it default to 0.
            Note that passing this option providing and input tensors that lie
            on another device is an error.

        Returns:
            List[Tensor]
        """
        override_user_schedule = kwargs.pop("override_user_schedule", False)
        func_based_def = False

        if device is not None:
            if not isinstance(device, torch.device):
                device = torch.device(device)
            assert (
                device.type == "cuda"
            ), "If device argument is passed it must be a CUDA device"
            device = device.index

        # if definition is not defined by a context manager, try a child class
        if self.id() is None:
            self._setup_definition()
            self.definition()
            self._finalize_definition()
            func_based_def = True

        # If schedule is defined by child class, make a schedule for inputs
        if func_based_def and (super(type(self), self).schedule != self.schedule):
            self._setup_schedule(inputs)
            self.schedule()
            self._finalize_schedule(inputs)

        result = None
        try:
            result = self._execute(inputs, override_user_schedule, device=device)
        except Exception as err:
            msg = (
                f"An error occurred while executing nvFuser FusionDefinition {self.id()}.\n"
                "If you believe this is a bug or need assistance, please file an issue at "
                "https://github.com/NVIDIA/Fuser/issues/new\n"
            )
            msg += (
                f"Here's a script to reproduce the error:\n"
                "```\n"
                "import torch\n"
                "from nvfuser import FusionDefinition, DataType\n"
                f"{self}"
                "with FusionDefinition() as fd:\n"
                f"    nvfuser_fusion_id{self.id()}(fd)\n"
                "\n"
                "inputs = [\n"
            )
            for i in inputs:
                if isinstance(i, torch.Tensor):
                    if i.dtype.is_floating_point:
                        msg += (
                            f"    torch.randn({tuple(i.size())}, dtype={i.dtype}, device='{i.device}')"
                            f".as_strided({tuple(i.size())}, {tuple(i.stride())}),\n"
                        )
                    else:
                        msg += (
                            f"    torch.randint(0, 10, {tuple(i.size())}, dtype={i.dtype}, device='{i.device}')"
                            f".as_strided({tuple(i.size())}, {tuple(i.stride())}),\n"
                        )
                else:
                    msg += f"    {i},\n"
            msg += "]"
            msg += "\nfd.execute(inputs)\n"
            msg += "```\n"
            logger.exception(msg)
            raise

        return result

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

        if not tensor.is_cuda:
            raise ValueError("Tensor should be on a cuda device!")

        return self.define_tensor(
            sizes=tensor.size(),
            strides=tensor.stride(),
            dtype=torch_dtype_to_nvfuser_dtype(tensor.dtype),
            static_sizes=static_sizes,
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
