import torch
import thunder
from rope_ops import rope_setup
from torch.autograd import DeviceType
from torch.profiler import profile, ProfilerActivity
from thunder.executors.nvfuserex import nvfuserex
import sys
import random

variations = [
        "llama_2_7b_hf_rope",
        "llama_3_8B_rope",
        "hf_qwen2_rope",
        "hf_phi3_rope",
        "hf_mistral_nemo_rope"]

if len(sys.argv) > 1 and sys.argv[1] == "random-order": #random_order
  random.shuffle(variations)
  print (variations)
  
NUM_ROUNDS = 10

def clear_grads(inputs):
  for inp in inputs:
      if isinstance(inp, torch.Tensor):
          inp.grad = None

def process_profile(prof_averages):
    num_kernels = 0
    cuda_events = []
    for event in prof_averages:
        if event.device_type != DeviceType.CUDA:
            continue
        cuda_events.append(event)
        num_kernels += 1
    return cuda_events, num_kernels
  
for variation in variations:
  prof = profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU])
  model, gen_inputs, grad, _ = rope_setup[variation]()
  fwd_inputs = gen_inputs()

  def fwd_call(inp):
      return model(*inp)

  # execute the compiled fwd fn
  fwd_fn = thunder.jit(fwd_call, executors=[nvfuserex], nv_enable_matmul=True)
  outputs = fwd_fn(fwd_inputs)
  grads = grad()

  # accumulate all output, so we can feed a single grad and use the unary bwd function
  output = outputs[0]
  for i in range(1, len(outputs)):
      output += outputs[i]
  expected_num_kernels=None
  print (f"Profiling: {variation}")
  for i in range(NUM_ROUNDS):
    clear_grads([output, grads, *fwd_inputs])
    prof.start()
    output.backward(grads, retain_graph=True)
    prof.stop()
    prof_averages = prof.key_averages()
    cuda_events, num_kernels = process_profile(prof_averages)
    
    print(f"Round: {i}")
    # for event in cuda_events:
    #   print(event)
    # print()
    if expected_num_kernels is None:
      expected_num_kernels = num_kernels
    else:
        assert expected_num_kernels == num_kernels, f"Expected {expected_num_kernels}, got {num_kernels}"
    print (num_kernels)
    prof.profiler = None
    
