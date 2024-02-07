from __future__ import annotations
import logging as logging
from nvfuser._C import DataType
from nvfuser._C import FusionCache
from nvfuser._C import Scalar
from nvfuser._C import Tensor
from nvfuser._C import Vector
from nvfuser._C import compute_contiguity
from nvfuser._C import compute_tensor_descriptor
from nvfuser._C import serialize
import os as os
import re as re
import sys as sys
import torch as torch
from . import _C
from . import contrib
from . import nvfuser_version
from . import pytorch_utils
__all__ = ['DataType', 'FusionCache', 'FusionDefinition', 'Scalar', 'Tensor', 'Vector', 'compute_contiguity', 'compute_tensor_descriptor', 'contrib', 'disable_automatic_serialization', 'enable_automatic_serialization', 'logger', 'logging', 'nvfuser_version', 'os', 'pytorch_lib_dir', 'pytorch_utils', 're', 'serialize', 'sys', 'torch', 'version']
class FusionDefinition(_C._FusionDefinition):
    def __enter__(self):
        ...
    def __exit__(self, type, value, traceback):
        ...
    def cuda_code_for(self, inputs, intrinsic_code = False, **kwargs):
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
    def definition(self):
        ...
    def execute(self, inputs, *, device = None, override_user_schedule = False, capture_debug_output = False):
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
                    override_user_schedule (bool): For a user defined schedule,
                        override with auto-generated schedule (default: False)
                    device (Optional[Union[int, str, torch.device]]): This is a hint to run
                        the Fusion on the given CUDA device. This is not typically
                        necessary, as the device is usually inferred from the locations
                        of input tensors. However, for some fusion definitions, no
                        tensors will be input (for example when all tensors are
                        generated with `full` or `uniform` ops). In these cases, we
                        must either tell NVFuser where to run the resulting kernel, or
                        let it default to 0. Note that passing this option providing
                        and input tensors that lie on another device is an error.
                    capture_debug_output (bool): Whether to capture any printed
                        debugging information as a string. If True, the string can be
                        retrieved after execution using :meth:`get_debug_output`. If False,
                        then that method will return None when called.
        
                Returns:
                    List[Tensor]
                
        """
    def from_pytorch(self, tensor, static_sizes = False):
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
    def fusion_ir(self):
        """
        
                Returns the uscheduled Fusion IR for the given definition that corresponds to all scheduled inputs.
        
                Returns:
                    String
                
        """
    def last_cuda_code(self, intrinsic_code = False, **kwargs):
        """
        
                Returns the Cuda Code for the last executed set of inputs
        
                Args:
                    intrinsic_code (Bool): Include all the additional code required to run kernel(s). (default: False)
        
                Kwargs:
                    override_user_schedule (Bool): For a user defined schedule, override with auto-generated schedule (default: False)
        
                Returns:
                    String
                
        """
    def last_scheduled_fusion_ir(self, tensor_transforms = False, **kwargs):
        """
        
                Returns the Scheduled Fusion IR for the last executed set of inputs
        
                Args:
                    tensor_transforms (Bool): Include tensor transforms that were applied through scheduling. (default: False)
        
                Kwargs:
                    override_user_schedule (Bool): For a user defined schedule, override with auto-generated schedule (default: False)
        
                Returns:
                    String
                
        """
    def schedule(self):
        ...
    def scheduled_fusion_ir_for(self, inputs, tensor_transforms = False, **kwargs):
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
    def validate(self, inputs: typing.List[torch.Tensor], reference_outputs: typing.List[torch.Tensor], kwargs = None):
        """
        
                Validates the fusion outputs against the provided reference outputs, using variable tolerances determined based on datatype and reduction size.
        
                Inputs:
                    inputs: A list of inputs expected by the fusion definition
                    reference_outputs: A list of reference outputs to validate against
                
        """
def disable_automatic_serialization():
    ...
def enable_automatic_serialization():
    ...
def version():
    """
    returns nvfuser version in format of a string 'm.n.p+git[7d-sha]'.
    
        We strip the git[7d-sha] and convert the string to
        `nvfuser_version.Version` for comparison. e.g. you can use it as:
            import nvfuser
            print(nvfuser.version())              # 0.0.1+git21df524
            nvfuser.version() == '0.0.1`          # True
            nvfuser.version() > '0.0.0`           # True
    
            from nvfuser_version import Version
            nvfuser.version() < Version('1.0.0')  # True
        
    """
__version__: nvfuser_version.NvfuserVersion  # value = '0.1.5+git6830ed2'
logger: logging.Logger  # value = <Logger nvfuser (INFO)>
pytorch_lib_dir: str = '/usr/local/lib/python3.10/dist-packages/torch/lib'
