__global__ void nvfuser_none_f0_c0_r0_g0(Tensor<float, 2, 2> T0, Tensor<float, 2, 2> T2) {
  alignas(16) extern __shared__ char array[];
  const unsigned smem_offset = 0;
  NVFUSER_DEFINE_MAGIC_ZERO;
  nvfuser_index_t i0;
  i0 = ceilDiv(T0.logical_size[1LL], 32LL);
  nvfuser_index_t i1;
  i1 = ((nvfuser_index_t)threadIdx.x) / 32LL;
  nvfuser_index_t i2;
  i2 = ((nvfuser_index_t)threadIdx.x) % 32LL;
  nvfuser_index_t i3;
  i3 = ((nvfuser_index_t)blockIdx.x) % i0;
  nvfuser_index_t i4;
  i4 = ((nvfuser_index_t)blockIdx.x) / i0;
  nvfuser_index_t i5;
  i5 = (((T0.alloc_stride[0LL] * i1) + (T0.alloc_stride[1LL] * i2)) + ((32LL * T0.alloc_stride[0LL]) * i4)) + ((32LL * T0.alloc_stride[1LL]) * i3);
  nvfuser_index_t i6;
  i6 = 8LL * T0.alloc_stride[0LL];
  nvfuser_index_t i7;
  i7 = 32LL * i1;
  nvfuser_index_t i8;
  i8 = 32LL * i2;
  nvfuser_index_t i9;
  i9 = 32LL * i4;
  nvfuser_index_t i10;
  i10 = ((i2 + (T0.logical_size[0LL] * i1)) + ((32LL * T0.logical_size[0LL]) * i3)) + i9;
  nvfuser_index_t i11;
  i11 = 8LL * T0.logical_size[0LL];
  bool b12;
  b12 = (i2 + i9) < T0.logical_size[0LL];
  nvfuser_index_t i13;
  i13 = ((-T0.logical_size[1LL]) + i1) + (32LL * i3);
  float* T1 = reinterpret_cast<float*>(array + smem_offset + 0LL);
  #pragma unroll
  for(nvfuser_index_t i14 = 0; i14 < 4LL; ++i14) {
    T1[((i7 + (256LL * i14)) + (i2 ^ (i1 + (8LL * i14))))]
       = T0[(i5 + (i6 * (i14 + nvfuser_zero)))];
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
  __syncthreads();
  #pragma unroll
  for(nvfuser_index_t i18 = 0; i18 < 4LL; ++i18) {
    nvfuser_index_t i19;
    i19 = i18 + nvfuser_zero;
    if ((b12 && (i13 < (-(8LL * i19))))) {
      T2[(i10 + (i11 * i19))]
         = T1[(i8 + (i2 ^ (i1 + (8LL * i18))))];
    }
  }
  NVFUSER_UPDATE_MAGIC_ZERO;
}
