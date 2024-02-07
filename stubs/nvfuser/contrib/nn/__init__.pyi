from __future__ import annotations
from nvfuser.contrib.nn.normalization import InstanceNorm1dNVFuser
from nvfuser.contrib.nn.normalization import InstanceNorm2dNVFuser
from nvfuser.contrib.nn.normalization import InstanceNorm3dNVFuser
from . import normalization
__all__: list = ['InstanceNorm1dNVFuser', 'InstanceNorm2dNVFuser', 'InstanceNorm3dNVFuser']
