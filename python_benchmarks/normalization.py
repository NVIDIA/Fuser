from nvfuser import FusionDefinition, DataType
from .global_params import PROMOTE_DTYPES

def norm_fwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
    norm: str,
    num_dims: int,
    channels_last: bool,
    eps: float = 1e-5,
    momentum: float = 0.01,
) -> None:

    batch_dim = 0
    channel_dim = 1 if not channels_last else num_dims - 1
    bcast_mask = [True if i != channel_dim else False for i in range(num_dims)]
    channels_only_bcast_mask = [True if i != channel_dim else False for i in range(num_dims)]
    
    reduction_axes = [i for i in range(num_dims) if i != channel_dim]
    if norm == "instance_norm":
        reduction_axes.remove(batch_dim)
        bcast_mask[batch_dim] = False 

    input = fd.define_tensor(shape=[-1] * num_dims, contiguity=[True] * num_dims, dtype=dtype, is_cpu=False)
    weight = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)
    bias = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)

    running_mean= fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False)
    running_var = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False)
    
    if dtype in PROMOTE_DTYPES:
        input = fd.ops.cast(input, dtype=DataType.Float)
        weight = fd.ops.cast(weight, dtype=DataType.Float)
        bias = fd.ops.cast(bias, dtype=DataType.Float)
        
    var, mean = fd.ops.var_mean(input, axes=reduction_axes, correction=0, keepdim=False)
    
    var_bcast = fd.ops.broadcast(var, bcast_mask)
    mean_bcast = fd.ops.broadcast(mean, bcast_mask)

    eps = fd.define_scalar(eps, dtype=DataType.Double)
    x_sub_mean = fd.ops.sub(input, mean_bcast)
    var_eps = fd.ops.add(var_bcast, eps)
    invstd = fd.ops.rsqrt(var_eps)
    x_norm = fd.ops.mul(x_sub_mean, invstd)

    weight = fd.ops.broadcast(weight, channels_only_bcast_mask)
    x_scaled = fd.ops.mul(x_norm, weight)
    bias = fd.ops.broadcast(bias, channels_only_bcast_mask)
    output = fd.ops.add(x_scaled, bias)

    rev_momentum = fd.define_scalar(1-momentum, dtype=DataType.Double)
    momentum = fd.define_scalar(momentum, dtype=DataType.Double)
    
    running_mean = fd.ops.add(fd.ops.mul(momentum, mean), fd.ops.mul(rev_momentum, running_mean))
    running_var = fd.ops.add(fd.ops.mul(momentum, var), fd.ops.mul(rev_momentum, running_var))

    if dtype in PROMOTE_DTYPES:
        output = fd.ops.cast(output, dtype=dtype)
    
    fd.add_output(output)
    fd.add_output(mean)
    fd.add_output(invstd)

def norm_bwd_fusion(
    fd: FusionDefinition,
    dtype: DataType,
    norm: str,
    num_dims: int,
    channels_last: bool,
    eps: float = 1e-5,
) -> None:
    
    batch_dim = 0
    channel_dim = 1 if not channels_last else num_dims - 1
    bcast_mask = [True if i != channel_dim else False for i in range(num_dims)]
    reduction_axes = [i for i in range(num_dims) if i != channel_dim]
    if norm == "instance_norm":
        reduction_axes.remove(batch_dim)
        bcast_mask[batch_dim] = False 

    input = fd.define_tensor(shape=[-1] * num_dims, contiguity=[True] * num_dims, dtype=dtype, is_cpu=False)
    grad = fd.define_tensor(shape=[-1] * num_dims, contiguity=[True] * num_dims, dtype=dtype, is_cpu=False)
    weight = fd.define_tensor(shape=[-1], contiguity=[True], dtype=dtype, is_cpu=False)

    running_mean= fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False)
    running_var = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False)
    mean= fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False)
    invstd = fd.define_tensor(shape=[-1], contiguity=[True], dtype=DataType.Float, is_cpu=False)
    
    if dtype in PROMOTE_DTYPES:
        input = fd.ops.cast(input, dtype=DataType.Float)
        grad = fd.ops.cast(grad, dtype=DataType.Float)
        weight = fd.ops.cast(weight, dtype=DataType.Float)

    num_features = fd.define_scalar(1)
    for ax in reduction_axes:
        num_features *= input.size(ax)

    norm = fd.ops.reciprocal(num_features)

    mean = fd.ops.broadcast(mean, bcast_mask)
    invstd = fd.ops.broadcast(invstd, bcast_mask)
    
    grad_sum = fd.ops.sum(grad, axes=reduction_axes, keepdim = True)
    
    x_sub_mean = fd.ops.sub(input, mean)
    dot_p = fd.ops.sum(fd.ops.mul(grad, x_sub_mean), axes=reduction_axes, keepdim=True)
    
    grad_mean = fd.ops.mul(grad_sum, norm)
    proj_scale = fd.ops.mul(fd.ops.mul(dot_p, norm), fd.ops.mul(invstd, invstd))
    
    weight = fd.ops.broadcast(weight, bcast_mask)
    grad_scale = fd.ops.mul(weight, invstd)
    proj = fd.ops.mul(proj_scale, x_sub_mean)
    
    grad_input = fd.ops.mul(fd.ops.sub(fd.ops.sub(grad, proj), grad_mean), grad_scale)
    grad_weight = fd.ops.mul(dot_p, invstd)
    grad_bias = grad_sum

    if dtype in PROMOTE_DTYPES:
        grad_input = fd.ops.cast(grad_input, dtype=dtype)
        grad_weight = fd.ops.cast(grad_weight, dtype=dtype)
        grad_bias = fd.ops.cast(grad_bias, dtype=dtype)
    
    fd.add_output(grad_input)
    fd.add_output(grad_weight)
    fd.add_output(grad_bias)