import cudnn
import math
import torch
from dataclasses import dataclass


@dataclass
class Config:
    b: int
    h: int
    s: int
    d: int


def make_nanogpt_gpt2xl_config() -> Config:
    return Config(16, 25, 128, 64)


def test_torch_sdpa_backward():
    config = make_nanogpt_gpt2xl_config()

    qkv_shape = (config.b, config.h, config.s, config.d)
    q_tensor = torch.randn(qkv_shape, dtype=torch.bfloat16).cuda()
    k_tensor = torch.randn(qkv_shape, dtype=torch.bfloat16).cuda()
    v_tensor = torch.randn(qkv_shape, dtype=torch.bfloat16).cuda()
    o_tensor = torch.randn(qkv_shape, dtype=torch.bfloat16).cuda()
    do_tensor = torch.randn(qkv_shape, dtype=torch.bfloat16).cuda()
    # The logsumexp L returned by the forward pass. See https://arxiv.org/pdf/2307.08691.pdf
    stats_tensor = torch.randn(
        config.b, config.h, config.s, 1, dtype=torch.float32
    ).cuda()
    philox_seed = torch.randint(10, ())
    philox_offset = torch.randint(10, ())

    (
        dq_tensor,
        dk_tensor,
        dv_tensor,
        _,
    ) = torch.ops.aten._scaled_dot_product_efficient_attention_backward(
        do_tensor,
        q_tensor,
        k_tensor,
        v_tensor,
        None,
        o_tensor,
        stats_tensor,
        philox_seed,
        philox_offset,
        0.0,
        [False, False, False, False],
    )
    torch.cuda.synchronize()

    assert dq_tensor.dtype == torch.bfloat16
    assert dk_tensor.dtype == torch.bfloat16
    assert dv_tensor.dtype == torch.bfloat16
    assert dq_tensor.size() == qkv_shape
    assert dk_tensor.size() == qkv_shape
    assert dv_tensor.size() == qkv_shape


def test_cudnn_sdpa_backward():
    config = make_nanogpt_gpt2xl_config()

    qkv_shape = (config.b, config.h, config.s, config.d)
    stride_d = 1
    stride_h = config.d * stride_d
    stride_s = config.h * stride_h * 3
    stride_b = config.s * stride_s
    out_strides = (stride_b, stride_h, stride_s, stride_d)

    q_tensor = torch.randn(qkv_shape, dtype=torch.bfloat16).cuda()
    k_tensor = torch.randn(qkv_shape, dtype=torch.bfloat16).cuda()
    v_tensor = torch.randn(qkv_shape, dtype=torch.bfloat16).cuda()

    dqkv_tensor = torch.empty(math.prod(qkv_shape) * 3, dtype=torch.bfloat16).cuda()
    dq_tensor = torch.as_strided(dqkv_tensor, qkv_shape, out_strides, storage_offset=0)
    dk_tensor = torch.as_strided(
        dqkv_tensor, qkv_shape, out_strides, storage_offset=config.h * stride_h
    )
    dv_tensor = torch.as_strided(
        dqkv_tensor, qkv_shape, out_strides, storage_offset=config.h * stride_h * 2
    )
    o_tensor = torch.randn(qkv_shape, dtype=torch.bfloat16).cuda()
    do_tensor = torch.randn(qkv_shape, dtype=torch.bfloat16).cuda()
    stats_tensor = torch.randn(
        config.b, config.h, config.s, 1, dtype=torch.float32
    ).cuda()

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
