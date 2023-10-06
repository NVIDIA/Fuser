import torch
import thunder
from thunder.core.transforms import inline, value_and_grad

torch.manual_seed(1234)

def func(a, b):
    result = a
    for _ in range(1000):
        result = result + b
    return result

x = torch.randn([128, 387], device="cuda")
y = torch.randn([128, 387], device="cuda")

forward_inference = thunder.compile(func, disable_preprocessing=True)
iout = forward_inference(x, y)
print(thunder.last_traces(forward_inference)[0])

'''
forward_grad = thunder.compile(inline(value_and_grad(func)), disable_preprocessing=True)
tout, tgrad = forward_grad(x, y)
print(thunder.last_traces(forward_grad)[0])
'''
