# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys

import torch

# This is needed when libnvfuser.so is patched and doesn't have the pytorch library location available.
sys.path.append(os.path.join(os.path.dirname(torch.__file__), "lib"))

# we need to import _C here to avoid confusing error message generated from failure in this python script ended up with
# complaining on `_C` not defined for `_C._FusionDefinition`
from . import _C
from ._C import *  # noqa: F401,F403


class FusionDefinition(_C._FusionDefinition):
    def __enter__(self):
        return self._setup_definition()

    def __exit__(self, type, value, traceback):
        self._finalize_definition()

    def definition(self):
        raise NotImplementedError("definition() should be implemented by child class!")

    def schedule(self):
        raise NotImplementedError("schedule() should be implemented by child class!")

    def execute(self, inputs, **kwargs):
        """
        Executes an nvFuser set of kernels for a given Fusion

        Args:
            inputs (List[Union[Tensor, Scalar]]): A list of inputs to fusion.

        Kwargs:
            override_user_schedule (bool): For a user defined schedule, override with auto-generated schedule (default: False)

        Returns:
            List[Tensor]
        """
        override_user_schedule = kwargs.pop("override_user_schedule", False)
        func_based_def = False

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
            result = self._execute(inputs, override_user_schedule)
        except Exception as err:
            print("\nError executing nvFuser FusionDefinition:")
            print(self)
            raise RuntimeError(err)

        return result

    def from_pytorch(self, tensor):
        """
        Defines an nvfuser input tensor from a pytorch tensor

        Args:
            tensor (torch.Tensor): Input tensor to nvFuser

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
        override_user_schedule = kwargs.pop('override_user_schedule', False)
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
        override_user_schedule = kwargs.pop('override_user_schedule', False)
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
        override_user_schedule = kwargs.pop('override_user_schedule', False)
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
        override_user_schedule = kwargs.pop('override_user_schedule', False)
        return self._scheduled_fusion_ir_for(inputs, tensor_transforms, override_user_schedule)

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
