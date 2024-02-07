from __future__ import annotations
import enum as enum
import nvfuser as nvfuser
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
import torch as torch
import typing
__all__: list = ['InstanceNorm1dNVFuser', 'InstanceNorm2dNVFuser', 'InstanceNorm3dNVFuser']
class InstanceNorm1dNVFuser(_InstanceNormNVFuser):
    def _check_input_dim(self, input: torch.Tensor) -> None:
        ...
class InstanceNorm2dNVFuser(_InstanceNormNVFuser):
    def _check_input_dim(self, input: torch.Tensor) -> None:
        ...
class InstanceNorm3dNVFuser(_InstanceNormNVFuser):
    def _check_input_dim(self, input: torch.Tensor) -> None:
        ...
class NamedAxis(enum.Enum):
    """
    An enumeration.
    """
    BATCH: typing.ClassVar[NamedAxis]  # value = <NamedAxis.BATCH: 1>
    CHANNEL: typing.ClassVar[NamedAxis]  # value = <NamedAxis.CHANNEL: 2>
class NormNVFuserFunction(torch.autograd.function.Function):
    _backward_cls = torch.autograd.function.NormNVFuserFunctionBackward
    @staticmethod
    def backward(ctx: typing.Any, grad_output: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, NoneType, NoneType, NoneType, NoneType, NoneType, NoneType, NoneType]:
        """
        Instance norm backward using NVFuser
        """
    @staticmethod
    def forward(ctx: typing.Any, x: torch.Tensor, weight: typing.Optional[torch.Tensor], bias: typing.Optional[torch.Tensor], running_mean: typing.Optional[torch.Tensor], running_var: typing.Optional[torch.Tensor], use_input_stats: bool, momentum: float, eps: float, unbiased: bool, stat_axes: typing.List[nvfuser.contrib.nn.normalization.NamedAxis]) -> torch.Tensor:
        ...
class _BatchNormNVFuser(_NormNVFuserBase):
    stat_axes: typing.ClassVar[list]  # value = [<NamedAxis.CHANNEL: 2>]
class _InstanceNormNVFuser(_NormNVFuserBase):
    stat_axes: typing.ClassVar[list]  # value = [<NamedAxis.BATCH: 1>, <NamedAxis.CHANNEL: 2>]
class _LayerNormNVFuser(_NormNVFuserBase):
    stat_axes: typing.ClassVar[list]  # value = [<NamedAxis.BATCH: 1>]
class _NormNVFuserBase(torch.nn.modules.batchnorm._NormBase):
    stat_axes = None
    def __init__(self, num_features: int, eps: float = 1e-05, momentum: float = 0.1, affine: bool = False, track_running_stats: bool = False, device: torch.device = None, dtype: torch.dtype = None) -> None:
        ...
    def _check_input_dim(self, input: torch.Tensor) -> None:
        ...
    def _load_from_state_dict(self, state_dict: typing.Dict[str, typing.Any], prefix: str, local_metadata: typing.Any, strict: bool, missing_keys: typing.List[str], unexpected_keys: typing.List[str], error_msgs: typing.List[str]) -> None:
        ...
    def forward(self, input: nvfuser._C.Tensor) -> nvfuser._C.Tensor:
        ...
def norm_fusion_backward(fd: nvfuser.FusionDefinition, inputs: typing.List[torch.Tensor], x: nvfuser.Tensor, grad_output: nvfuser.Tensor, mean: typing.Optional[torch.Tensor], invstd: torch.Tensor, weight: typing.Optional[ForwardRef('nvfuser.Tensor')], bias: typing.Optional[ForwardRef('nvfuser.Tensor')], running_mean: typing.Optional[ForwardRef('nvfuser.Tensor')], running_var: typing.Optional[ForwardRef('nvfuser.Tensor')], use_input_stats: bool, channels_last: bool, x_datatype: nvfuser.DataType, *, stat_axes: typing.List[nvfuser.contrib.nn.normalization.NamedAxis]) -> typing.Tuple[ForwardRef('nvfuser.Tensor'), ForwardRef('nvfuser.Tensor'), ForwardRef('nvfuser.Tensor')]:
    """
    
        Modify FusionDefinition to add a generic normalization layer (backward).
    
        Args:
            fd: An initialized FusionDefinition.
            inputs: A list of :class:'torch.Tensor' inputs to the
                `FusionDefinition` `fd`.
            x: The input NVFuser tensor.
            grad_output: NVFuser tensor representing gradient of loss with respect
                to downstream activation (typical input to backward()).
            mean: The mean used in the forward normalization.
            invstd: The reciprocal of standard deviation used in the forward normalization.
            weight: If given, multiply normed output by this `Tensor`. It should be
                one-dimensional if `NamedAxis.CHANNEL` is in `stat_axes`, and
                zero-dimensional otherwise. It will be broadcast along all other
                dimensions.
            bias: If given, add this `Tensor` to normed output. It should be
                one-dimensional if `NamedAxis.CHANNEL` is in `stat_axes`, and
                zero-dimensional otherwise. It will be broadcast along all other
                dimensions.
            running_mean: If given, a running mean estimate that will be modified
                in place.
            running_var: If given, a running variance estimate that will be
                modified in place.
            use_input_stats: Whether to compute the stats of this batch or to
                _only_ use the provided running_mean and running_var.
            channels_last: Whether channels are in position -1 (`True`) or 1
                (`False`).
            x_datatype: :class:'DataType' of input :class:'Tensor' `x`
            stat_axes: A list of `NamedAxis` objects indicating a combination of
                axes with which to index the computed statistics. This can be used
                to implement multiple types of normalization layers, since most of
                those differ only in which axes are reduced over.
        Returns:
            The normalized output, as well as mean and 1/std. Note that
            `fd.add_output` is _not_ called by this function.
        
    """
def norm_fusion_forward(fd: nvfuser.FusionDefinition, inputs: typing.List[torch.Tensor], x: nvfuser.Tensor, weight: typing.Optional[ForwardRef('nvfuser.Tensor')], bias: typing.Optional[ForwardRef('nvfuser.Tensor')], running_mean: typing.Optional[ForwardRef('nvfuser.Tensor')], running_var: typing.Optional[ForwardRef('nvfuser.Tensor')], eps: nvfuser.Scalar, use_input_stats: bool, momentum: nvfuser.Scalar, channels_last: bool, x_datatype: nvfuser.DataType, unbiased: bool = False, *, stat_axes: typing.List[nvfuser.contrib.nn.normalization.NamedAxis]) -> typing.Tuple[ForwardRef('nvfuser.Tensor'), ForwardRef('nvfuser.Tensor'), ForwardRef('nvfuser.Tensor')]:
    """
    Modify FusionDefinition to add a generic normalization layer (forward).
    
        This can be used to construct a BatchNorm, GroupNorm, InstanceNorm, or
        LayerNorm network by indicating different sets of axes to preserve.
    
        BatchNorm: `stat_axes = [NamedAxis.CHANNEL]`
        LayerNorm: `stat_axes = [NamedAxis.BATCH]`
        InstanceNorm: `stat_axes = [NamedAxis.BATCH, NamedAxis.CHANNEL]`
    
        Args:
            fd: An initialized FusionDefinition.
            inputs: A list of :class:'torch.Tensor' inputs to the
                `FusionDefinition` `fd`.
            x: An input NVFuser tensor.
            weight: If given, multiply normed output by this `Tensor`. It should be
                one-dimensional if `NamedAxis.CHANNEL` is in `stat_axes`, and
                zero-dimensional otherwise. It will be broadcast along all other
                dimensions.
            bias: If given, add this `Tensor` to normed output. It should be
                one-dimensional if `NamedAxis.CHANNEL` is in `stat_axes`, and
                zero-dimensional otherwise. It will be broadcast along all other
                dimensions.
            running_mean: If given, a running mean estimate that will be modified
                in place.
            running_var: If given, a running variance estimate that will be
                modified in place.
            eps: Amount to regularize the square root needed to convert variance to
                standard deviation.
            use_input_stats: Whether to compute the stats of this batch or to
                _only_ use the provided running_mean and running_var.
            momentum: Momentum for exponentially weighted moving average of running
                stats.
            channels_last: Whether channels are in position -1 (`True`) or 1
                (`False`).
            x_datatype: :class:'DataType' of input :class:'Tensor' `x`
            unbiased: Whether to use unbiased variance for computing current batch
                statistics. Note that unbiased estimates are always used for
                running variance updates, regardless of this argument's value.
            stat_axes: A list of `NamedAxis` objects indicating a combination of
                axes with which to index the computed statistics. This can be used
                to implement multiple types of normalization layers, since most of
                those differ only in which axes are reduced over.
        Returns:
            The normalized output, as well as mean and 1/std. Note that
            `fd.add_output` is _not_ called by this function.
        
    """
def partially_contig_tensor(fd: nvfuser.FusionDefinition, x: torch.Tensor) -> nvfuser.Tensor:
    ...
