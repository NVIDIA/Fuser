# run it with
# mpirun -n 2 python mpi_test.py

import torch
import nvfuser
from nvfuser import FusionDefinition, DataType

import os

rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])

torch.cuda.set_device(rank)

# Inputs
inputs = [
    torch.randn(2, 4, device="cuda")[rank : rank + 1, ...],
]

# dynamic shape isn't supported;
# scalar inputs isn't supported;
class MultiDeviceModel(FusionDefinition):
    def definition(self):
        self.t0 = self.define_tensor((2, 4), (False, False), dtype=DataType.Float)
        self.t1 = self.ops.relu(self.t0)
        self.t2 = self.ops.add(self.t1, self.t1)
        self.add_output(self.t2)

    def schedule(self):
        self.mesh = self.sched._create_device_mesh((0, 1))
        self.sched._set_device_mesh(self.t0, self.mesh)
        self.sched._set_device_mesh(self.t1, self.mesh)
        self.sched._set_device_mesh(self.t2, self.mesh)
        self.sched._parallelize(self.t0, 0, nvfuser.ParallelType.mesh_x)


fn = MultiDeviceModel()

o = fn.execute(inputs)

for i in range(3):
    o = fn.execute(inputs)

assert (inputs[0] * 2).relu().allclose(o[0][rank])
