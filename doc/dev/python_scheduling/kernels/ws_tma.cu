__global__ void __launch_bounds__(
    /*maxThreadsPerBlock=*/384,
    /*minBlocksPerMultiprocessor=*/1)
    nvfuser_none_f0_c0_r0_g0(
        Tensor<__bfloat, 2, 2> T0,
        Tensor<__bfloat, 2, 2> T1,
        const __grid_constant__ TensorMap var0,
        const __grid_constant__ TensorMap var1,
        Tensor<__bfloat, 2, 2> T5) {
// mbarrier initialization
#pragma unroll
  for (int i14 = 0; i14 < 3; ++i14) {
    if ((Hopper::electSync(4294967295U) && b10)) {
      mbarrier::init(toSmem((&T11[i14])), 2U);
    }
  }
  __syncthreads();
  // producer
  if ((((int)threadIdx.x) >= 256)) {
    decreaseRegisters<40>();
    for (int i16 = 0; i16 < 27; ++i16) {
      if (((((((int)threadIdx.x) + -256) / 32ULL) == 0ULL) &&
           Hopper::electSync(4294967295U))) {
        mbarrier::waitParity(
            toSmem((&T11[((i16 % 3) + 3LL)])), __to_uint32(((i16 / 3) % 2)));
        mbarrier::arriveExpectTX(toSmem((&T11[(i16 % 3)])), 32768U);
        Hopper::cpAsyncBulkTensorTileG2S(
            (Hopper::CpAsyncBulkTensorTileG2SIndex<2>{
                ptr2,
                (Array<int, 2, 1>{
                    __to_int32((256 * (i17 % 32))),
                    __to_int32((64 * (i17 / 32)))}),
                toSmem((&T11[(i16 % 3)]))}),
            (i3 + i18));
      }
    }
  }
  // consumer
  else {
    increaseRegisters<232>();
#pragma unroll
    for (int i19 = 0; i19 < 3; ++i19) {
      mbarrier::arrive(toSmem((&T11[(i19 + 3LL)])));
    }
#pragma unroll
    for (int i20 = 0; i20 < 27; ++i20) {
      mbarrier::waitParity(
          toSmem((&T11[(i20 % 3)])), __to_uint32(((i20 / 3) % 2)));
      for (int i28 = 0; i28 < 2; ++i28) {
        for (int i31 = 0; i31 < 4; ++i31) {
          loadGeneric<__bfloat, 8>(
              &T8[(i30 + (8 * i31))], &T6[(i29 + (2048 * i31))]);
        }
      }
      mbarrier::arrive(toSmem((&T11[((i20 % 3) + 3LL)])));
#pragma unroll
      for (int i36 = 0; i36 < 2; ++i36) {
        for (int i40 = 0; i40 < 4; ++i40) {
          for (int i43 = 0; i43 < 8; ++i43) {
            // computations //
            if ((b26 && (i13 < (i39 - i41)))) {
              loadLocalToGlobal<
                  __bfloat,
                  /*vec_size=*/8,
                  /*is_volatile=*/false>(&T5[(i38 + (65536 * i40))], &T10[0]);
            }
          }
        }
      }
    }
  }
