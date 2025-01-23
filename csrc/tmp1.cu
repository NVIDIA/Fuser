[ RUN      ] RNGTest.ValidateWithCURand
Assign
ROP: T2_l_float[iblockIdx.x11{( ceilDiv(( (nvfuser_index_t)(i0) ), 128) )}, iUS12{1}, ithreadIdx.x10{128}] ca_pos( 3 )
   = rng_uniform({i2}, float);

!is deterministic
kir::GetRNGSeedAndOffsetFromHost
OFFSET DEF: GetRNGSeedAndOffsetFromHost()
Assign
ROP: T2_l_float[iblockIdx.x11{( ceilDiv(( (nvfuser_index_t)(i0) ), 128) )}, iUS12{1}, ithreadIdx.x10{128}] ca_pos( 3 )
   = rng_uniform({i5}, float);

!is deterministic
kir::GetRNGSeedAndOffsetFromHost
OFFSET DEF: GetRNGSeedAndOffsetFromHost()
KernelExecutor::runFusion
kir::GetRNGSeedAndOffsetFromHost::evaluate
ELSE
KernelExecutor::runFusion END
KernelExecutor::runFusion
kir::GetRNGSeedAndOffsetFromHost::evaluate
ELSE
KernelExecutor::runFusion END
(nil), (nil), 0, 0
0, 0
1713891541, 3781805453, 3159862348, 2600524760,
(nil), (nil), 0, 4
0, 1
4175744164, 1555169499, 2980410603, 159317863,
 0.3990
 0.8805
 0.7357
 0.6055
 0.5167
 0.9397
 0.0590
 0.5155
 0.0249
 0.7129
 0.1489
 0.4658
 0.9401
 0.0678
 0.7456
 0.7204
[ CUDAFloatType{16} ]
 0.3990
 0.8805
 0.7357
 0.6055
 0.5167
 0.9397
 0.0590
 0.5155
 0.0249
 0.7129
 0.1489
 0.4658
 0.9401
 0.0678
 0.7456
 0.7204
[ CUDAFloatType{16} ]
 0.9722
 0.3621
 0.6939
 0.0371
 0.7910
 0.8507
 0.4938
 0.4273
 0.4690
 0.1352
 0.8455
 0.4252
 0.3300
 0.9178
 0.9638
 0.7722
[ CUDAFloatType{16} ]
 0.9722
 0.3621
 0.6939
 0.0371
 0.7910
 0.8507
 0.4938
 0.4273
 0.4690
 0.1352
 0.8455
 0.4252
 0.3300
 0.9178
 0.9638
 0.7722
[ CUDAFloatType{16} ]
[       OK ] RNGTest.ValidateWithCURand (585 ms)
[----------] 1 test from RNGTest (585 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (585 ms total)