"""
Simulate intermediate variables in hierarchical split-K reduction
"""
import numpy as np

import math

def countLeadingZeros(x):
    if x == 0:
        return x.bit_length()
    # Keep shifting x by one until 
    # leftmost bit does not become 1.
    total_bits = x.bit_length()
    res = 0
    while ((x & (1 << (total_bits - 1))) == 0):
        x = (x << 1)
        res += 1
    return res

def mostSignificantBit(x):
    return x.bit_length() - countLeadingZeros(x)

def simulate(S):
    D = mostSignificantBit(S - 1)
    CL = math.ceil(math.log2(S))
    print(f"S={S}({bin(S)}) S-1={S-1}({bin(S-1)}) msb(S-1)={mostSignificantBit(S-1)} D={D} ceil(log2(S))={CL}")
    padded_size = 1 << D
    E = padded_size - S
    print(f" padded_size={padded_size} E={E}")
    cta_rank = np.arange(S)
    print(f" cta_rank={cta_rank}")
    work_buffer_pos = E + cta_rank
    for stage in range(D-1, -1, -1):
        print(f"  stage={stage}")
        print(f"   work_buffer_pos={work_buffer_pos}")
        stage_shift = 1 << stage
        print(f"   stage_shift={stage_shift}")
        active = work_buffer_pos >= 0
        print(f"   active={active}")
        write_sum = active & (work_buffer_pos < stage_shift)
        print(f"   write_sum={write_sum}")
        read_sum = active & (work_buffer_pos >= stage_shift) & (cta_rank >= stage_shift)
        print(f"   read_sum={read_sum}")
        work_buffer_pos -= stage_shift
        print(f"   shifted work_buffer_pos={work_buffer_pos}")

simulate(5)
"""
Sample output for simulate(5)
S=5(0b101) S-1=4(0b100) msb(S-1)=3 D=3 ceil(log2(S))=3
 padded_size=8 E=3
 cta_rank=[0 1 2 3 4]
  stage=2
   work_buffer_pos=[3 4 5 6 7]
   stage_shift=4
   active=[ True  True  True  True  True]
   write_sum=[ True False False False False]
   read_sum=[False False False False  True]
   shifted work_buffer_pos=[-1  0  1  2  3]
  stage=1
   work_buffer_pos=[-1  0  1  2  3]
   stage_shift=2
   active=[False  True  True  True  True]
   write_sum=[False  True  True False False]
   read_sum=[False False False  True  True]
   shifted work_buffer_pos=[-3 -2 -1  0  1]
  stage=0
   work_buffer_pos=[-3 -2 -1  0  1]
   stage_shift=1
   active=[False False False  True  True]
   write_sum=[False False False  True False]
   read_sum=[False False False False  True]
   shifted work_buffer_pos=[-4 -3 -2 -1  0]
"""
