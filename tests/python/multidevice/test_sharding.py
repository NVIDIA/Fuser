# SPDX-FileCopyrightText: Copyright (c) 2026-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import torch

import nvfuser_direct as nvfuser


# Unit tests for multidevice_test.shard_tensor.
class TestShardTensor:
    @pytest.mark.mpi
    def test_inner_split(self, multidevice_test):
        d = multidevice_test.size

        with nvfuser.FusionDefinition() as fd:
            tv = fd.define_tensor([-1])
            fd.add_output(tv)

            mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))
            tv.set_device_mesh(mesh)
            tv.inner_split(0, d)
            tv.axis(1).parallelize(nvfuser.ParallelType.mesh_x)

        t_ref = torch.arange(3 * d, dtype=torch.float)
        t = multidevice_test.shard_tensor(t_ref, tv)
        torch.testing.assert_close(
            t.cpu(), t_ref.reshape(3, d)[:, multidevice_test.rank]
        )

    @pytest.mark.mpi
    def test_nonoutermost_split(self, multidevice_test):
        d = multidevice_test.size

        with nvfuser.FusionDefinition() as fd:
            tv = fd.define_tensor([-1])
            fd.add_output(tv)

            mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))
            tv.set_device_mesh(mesh)
            tv.outer_split(0, 2)
            tv.outer_split(1, d)
            tv.axis(1).parallelize(nvfuser.ParallelType.mesh_x)

        t_ref = torch.arange(2 * d * 3, dtype=torch.float)
        t = multidevice_test.shard_tensor(t_ref, tv)
        torch.testing.assert_close(
            t.cpu(), t_ref.reshape(2, d, 3)[:, multidevice_test.rank, :].flatten()
        )

    @pytest.mark.mpi
    def test_parallel_reduction(self, multidevice_test):
        d = multidevice_test.size
        with nvfuser.FusionDefinition() as fd:
            inp_tv = fd.define_tensor([-1, -1])
            out_tv = fd.ops.sum(inp_tv, [1])
            fd.add_output(out_tv)

            mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))
            for tv in [inp_tv, out_tv]:
                tv.set_device_mesh(mesh)
                tv.outer_split(1, d)
                tv.axis(1).parallelize(nvfuser.ParallelType.mesh_x)

        inp_ref = torch.arange(2 * d * 3, dtype=torch.float).reshape(2, d * 3)
        out_ref = inp_ref.sum([1])

        inp = multidevice_test.shard_tensor(inp_ref, inp_tv)
        out = multidevice_test.shard_tensor(out_ref, out_tv)
        print(inp)
        print(out)

    @pytest.mark.mpi
    def test_2d_sharded_matrix(self, multidevice_test):
        d = multidevice_test.size
        tp_size = 2
        if d % tp_size != 0:
            pytest.skip(
                f"Number of devices ({d}) must be divisible by tp_size ({tp_size})"
            )
        dp_size = d // tp_size

        with nvfuser.FusionDefinition() as fd:
            tv = fd.define_tensor([-1, -1])
            fd.add_output(tv)

            mesh = nvfuser.multidevice.DeviceMesh(
                torch.arange(d).reshape(dp_size, tp_size)
            )
            tv.set_device_mesh(mesh)
            tv.outer_split(0, dp_size)
            tv.axis(0).parallelize(nvfuser.ParallelType.mesh_y)
            tv.outer_split(2, tp_size)
            tv.axis(2).parallelize(nvfuser.ParallelType.mesh_x)

        rows_per_rank = 2
        cols_per_rank = 3
        t_ref = torch.arange(
            dp_size * rows_per_rank * tp_size * cols_per_rank, dtype=torch.float
        ).reshape(dp_size * rows_per_rank, tp_size * cols_per_rank)
        t = multidevice_test.shard_tensor(t_ref, tv)

        dp_rank = multidevice_test.rank // tp_size
        tp_rank = multidevice_test.rank % tp_size
        torch.testing.assert_close(
            t.cpu(),
            t_ref[
                dp_rank * rows_per_rank : (dp_rank + 1) * rows_per_rank,
                tp_rank * cols_per_rank : (tp_rank + 1) * cols_per_rank,
            ],
        )

    @pytest.mark.mpi
    def test_2d_sharded_vector(self, multidevice_test):
        d = multidevice_test.size
        tp_size = 2
        if d % tp_size != 0:
            pytest.skip(
                f"Number of devices ({d}) must be divisible by tp_size ({tp_size})"
            )
        dp_size = d // tp_size

        rows_per_rank = 2
        cols_per_rank = 3

        with nvfuser.FusionDefinition() as fd:
            tv = fd.define_tensor([-1])
            fd.add_output(tv)

            mesh = nvfuser.multidevice.DeviceMesh(
                torch.arange(d).reshape(dp_size, tp_size)
            )
            tv.set_device_mesh(mesh)
            tv.inner_split(0, tp_size * cols_per_rank)
            tv.outer_split(0, dp_size)
            tv.axis(0).parallelize(nvfuser.ParallelType.mesh_y)
            tv.outer_split(2, tp_size)
            tv.axis(2).parallelize(nvfuser.ParallelType.mesh_x)

        t_ref = torch.arange(
            dp_size * rows_per_rank * tp_size * cols_per_rank, dtype=torch.float
        )
        t = multidevice_test.shard_tensor(t_ref, tv)

        dp_rank = multidevice_test.rank // tp_size
        tp_rank = multidevice_test.rank % tp_size
        torch.testing.assert_close(
            t.cpu(),
            t_ref.reshape(dp_size, rows_per_rank, tp_size, cols_per_rank)[
                dp_rank, :, tp_rank, :
            ].flatten(),
        )

    @pytest.mark.mpi
    def test_expanded_broadcast(self, multidevice_test):
        d = multidevice_test.size
        with nvfuser.FusionDefinition() as fd:
            tv = fd.define_tensor([-1, -1], contiguity=[True, None])
            fd.add_output(tv)

            mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))
            tv.set_device_mesh(mesh)
            tv.outer_split(1, d)
            tv.axis(1).parallelize(nvfuser.ParallelType.mesh_x)

        # When d=2, t_ref =
        #   [[0, 0, 0, 0, 0, 0],
        #    [1, 1, 1, 1, 1, 1]]
        #
        # GPU 0 gets the left half, and GPU 1 gets the right half. So t =
        #   [[0, 0, 0],
        #    [1, 1, 1]]
        # regardless of which GPU.
        t_ref = torch.arange(2, dtype=torch.float).unsqueeze(-1).expand(-1, d * 3)
        t = multidevice_test.shard_tensor(t_ref, tv)
        torch.testing.assert_close(
            t.cpu(), torch.arange(2, dtype=torch.float).unsqueeze(-1).expand(-1, 3)
        )

    @pytest.mark.mpi
    def test_broadcast(self, multidevice_test):
        d = multidevice_test.size
        with nvfuser.FusionDefinition() as fd:
            tv = fd.define_tensor([1, -1], contiguity=[None, True])
            fd.add_output(tv)

            mesh = nvfuser.multidevice.DeviceMesh(torch.arange(d))
            tv.set_device_mesh(mesh)
            tv.outer_split(1, d)
            tv.axis(1).parallelize(nvfuser.ParallelType.mesh_x)

        # When d=2, t_ref = [[0, 1, 2, 3, 4, 5]]. GPU 0 gets [[0, 1, 2]], and GPU 1 gets [[3, 4, 5]].
        t_ref = torch.arange(d * 3, dtype=torch.float).unsqueeze(0)
        t = multidevice_test.shard_tensor(t_ref, tv)
        rank = multidevice_test.rank
        torch.testing.assert_close(t.cpu(), t_ref[:, rank * 3 : (rank + 1) * 3])
