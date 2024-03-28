# run it with
# NVFUSER_DUMP=fusion_ir mpirun --allow-run-as-root -n 2 python mpi_nvfuser.py

import torch
import nvfuser
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from torch import Tensor
from typing import Tuple

from nvfuser import FusionDefinition, DataType

import os

rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])

torch.cuda.set_device(rank)

### Inputs
inputs = [
    torch.randn(2, 4, device="cuda")[rank:rank+1, ...],
]

class MultiDeviceModel(FusionDefinition):

    def definition(self):
        # dynamic shape isn't working? at least I'm getting asserts
        #self.t0 = self.from_pytorch(inputs[0])

        self.t0 = self.define_tensor((2, 4), (False, False), dtype=DataType.Float)

        # looks like I cannot have scalar inputs to any expression, hitting an assert
        #self.s0 = self.define_constant(2.0)
        #self.t1 = self.ops.mul(self.t0, self.s0)

        # relu seems to also be complaining, I assume sharding operation has to be a set
        #self.t1 = self.ops.relu(self.t0)

        self.t1 = self.ops.set(self.t0)
        self.t2 = self.ops.add(self.t1, self.t1)
        self.add_output(self.t2)

    def schedule(self):
        self.mesh = self.sched.create_device_mesh((0, 1))
        self.sched.set_device_mesh(self.t0, self.mesh)
        self.sched.set_device_mesh(self.t1, self.mesh)
        self.sched.set_device_mesh(self.t2, self.mesh)
        self.sched.parallelize(self.t0, 0, nvfuser.ParallelType.DIDx)

fn = MultiDeviceModel()

### Repro code
o = fn.execute(inputs)

# multiple execution seems to be triggering a sync issue. at least the output doesn't look right
#torch.cuda.profiler.start()
#for i in range(3):
#    o = fn.execute(inputs)
#torch.cuda.profiler.stop()

print("input with rank\t", rank, "\t", inputs)
print("output with rank\t", rank, "\t", o)
