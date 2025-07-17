

def ceilDiv(a, b):
    return (a + b - 1) // b

m = 8192
n = 8192
cta_m = 128
cta_n = 128
m_blocks = ceilDiv(m, cta_m)
n_blocks = ceilDiv(n, cta_n)
gridDim = 132
warpGroups = 2

print(m_blocks, n_blocks)

for blockIdx in range(10):
    print(f"==== {blockIdx}")
    for o in range(ceilDiv(ceilDiv(m_blocks * n_blocks, gridDim), warpGroups)):
        for i in range(warpGroups):
            bid = o * (gridDim * warpGroups) + i * gridDim + blockIdx
            m = (bid // warpGroups) % m_blocks
            n = (bid // warpGroups) // m_blocks * warpGroups + bid % warpGroups
            #m = bid % m_blocks;
            #n = bid // m_blocks;
            print(m, n)
