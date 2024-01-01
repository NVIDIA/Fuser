import cudnn
import math
import torch


def test_sdpa_backward():
    b = 16
    h = 12
    s = 128
    d = 64
    qkv_shape = (b, h, s, d)

    stride_d = 1
    stride_h = d * stride_d
    stride_s = h * stride_h * 3
    stride_b = s * stride_s
    out_strides = (stride_b, stride_h, stride_s, stride_d)

    q_tensor = torch.randn(qkv_shape, dtype=torch.bfloat16).cuda()
    k_tensor = torch.randn(qkv_shape, dtype=torch.bfloat16).cuda()
    v_tensor = torch.randn(qkv_shape, dtype=torch.bfloat16).cuda()

    dqkv_tensor = torch.empty(math.prod(qkv_shape) * 3, dtype=torch.bfloat16).cuda()
    dq_tensor = torch.as_strided(dqkv_tensor, qkv_shape, out_strides, storage_offset=0)
    dk_tensor = torch.as_strided(
        dqkv_tensor, qkv_shape, out_strides, storage_offset=h * stride_h
    )
    dv_tensor = torch.as_strided(
        dqkv_tensor, qkv_shape, out_strides, storage_offset=h * stride_h * 2
    )
    o_tensor = torch.randn(qkv_shape, dtype=torch.bfloat16).cuda()
    do_tensor = torch.randn(qkv_shape, dtype=torch.bfloat16).cuda()
    stats_tensor = torch.randn(b, h, s, 1, dtype=torch.float32).cuda()

    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q = graph.tensor_like(q_tensor)
    k = graph.tensor_like(k_tensor)
    v = graph.tensor_like(v_tensor)
    o = graph.tensor_like(o_tensor)
    do = graph.tensor_like(do_tensor)
    stats = graph.tensor_like(stats_tensor)

    dq, dk, dv = graph.sdpa_backward(q, k, v, o, do, stats)
    dq.set_output(True).set_dim(dq_tensor.size()).set_stride(dq_tensor.stride())
    dk.set_output(True).set_dim(dk_tensor.size()).set_stride(dk_tensor.stride())
    dv.set_output(True).set_dim(dv_tensor.size()).set_stride(dv_tensor.stride())

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8).cuda()
    graph.execute(
        {
            q: q_tensor,
            k: k_tensor,
            v: v_tensor,
            o: o_tensor,
            do: do_tensor,
            stats: stats_tensor,
            dq: dq_tensor,
            dk: dk_tensor,
            dv: dv_tensor,
        },
        workspace,
    )
    torch.cuda.synchronize()
