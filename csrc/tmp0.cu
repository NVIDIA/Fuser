Assign
ROP: T2_l_float[iblockIdx.x11{( ceilDiv(( (nvfuser_index_t)(i0) ), 128) )}, iUS12{1}, ithreadIdx.x10{128}] ca_pos( 3 )
   = rng_uniform({i2}, float);

!is deterministic
kir::GetRNGSeedAndOffsetFromHost
OFFSET DEF: GetRNGSeedAndOffsetFromHost()
kir::GetRNGSeedAndOffsetFromHost::evaluate
ELSE
Assign
ROP: T2_l_float[iblockIdx.x11{( ceilDiv(( (nvfuser_index_t)(i0) ), 128) )}, iUS12{1}, ithreadIdx.x10{128}] ca_pos( 3 )
   = rng_uniform({i5}, float);

!is deterministic
kir::GetRNGSeedAndOffsetFromHost
OFFSET DEF: GetRNGSeedAndOffsetFromHost()
kir::GetRNGSeedAndOffsetFromHost::evaluate
ELSE
KernelExecutor::runFusion
kir::GetRNGSeedAndOffsetFromHost::evaluate
ELSE
KernelExecutor::runFusion END
KernelExecutor::runFusion
kir::GetRNGSeedAndOffsetFromHost::evaluate
ELSE
KernelExecutor::runFusion END
(nil), (nil), 0, 8
0, 2
83534633, 1372009126, 605361069, 1167144420,
(nil), (nil), 0, 12
0, 3
3381718825, 1782871206, 2596868943, 1832992260,
 0.0194
 0.3194
 0.1409
 0.2717
 0.7733
 0.7433
 0.2088
 0.6629
 0.8650
 0.8156
 0.9185
 0.5725
 0.8097
 0.0312
 0.7727
 0.0617
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
 0.7874
 0.4151
 0.6046
 0.4268
 0.1344
 0.6251
 0.3162
 0.2022
 0.2190
 0.9205
 0.0950
 0.6768
 0.8193
 0.4184
 0.4194
 0.9725
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