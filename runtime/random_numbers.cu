// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
__device__ unsigned int mulhilo32(
    unsigned int a,
    unsigned int b,
    unsigned int* result_high) {
  *result_high = __umulhi(a, b);
  return a * b;
}

__device__ Array<uint32_t, 4> single_round(
    Array<uint32_t, 4> ctr,
    Array<uint32_t, 2> key) {
  constexpr unsigned long kPhiloxSA = 0xD2511F53;
  constexpr unsigned long kPhiloxSB = 0xCD9E8D57;
  unsigned int hi0;
  unsigned int hi1;
  unsigned int lo0 = mulhilo32(kPhiloxSA, ctr[0], &hi0);
  unsigned int lo1 = mulhilo32(kPhiloxSB, ctr[2], &hi1);
  Array<uint32_t, 4> ret = {
      hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0};
  return ret;
}

__device__ Array<uint32_t, 4> philox(
    unsigned long long seed,
    unsigned long long subsequence,
    unsigned long long offset) {
  constexpr unsigned long kPhilox10A = 0x9E3779B9;
  constexpr unsigned long kPhilox10B = 0xBB67AE85;
  Array<uint32_t, 2> key;
  key[0] = (unsigned int)seed;
  key[1] = (unsigned int)(seed >> 32);
  Array<uint32_t, 4> counter;
  counter[0] = (unsigned int)(offset);
  counter[1] = (unsigned int)(offset >> 32);
  counter[2] = (unsigned int)(subsequence);
  counter[3] = (unsigned int)(subsequence >> 32);

  Array<uint32_t, 4> output = {};
  Array<uint32_t, 2> key_ = key;
  Array<uint32_t, 4> counter_ = counter;
  for (int i = 0; i < 9; i++) {
    counter_ = single_round(counter_, key_);
    key_[0] += (kPhilox10A);
    key_[1] += (kPhilox10B);
  }
  output = single_round(counter_, key_);
  return output;
}

// This is a uniform double in the range (0, 1]
__device__ double raw_uniform_double(unsigned int x, unsigned int y) {
  constexpr double scale = 1.0 / (double)(1ll << 53);
  const unsigned long long z =
      (unsigned long long)x ^ ((unsigned long long)y << (53 - 32));
  return (double)z * scale + 0.5 * scale;
}

// This is a uniform float in the range (0, 1]
__device__ float raw_uniform_float(unsigned int x) {
  constexpr float scale = (float)(1.0 / (double)(1ll << 32));
  return (float)x * scale + 0.5f * scale;
}

__device__ __half uniform_half(unsigned int x) {
  __half result = __float2half(raw_uniform_float(x));
  return __heq(result, __float2half(1.0f)) ? __float2half(0.0f) : result;
}

__device__ __bfloat uniform_bfloat(unsigned int x) {
  __bfloat result = __float2bfloat(raw_uniform_float(x));
  return __heq(result, __float2bfloat(1.0f)) ? __float2bfloat(0.0f) : result;
}

__device__ float uniformf(unsigned int x) {
  float result = raw_uniform_float(x);
  return result == 1.0f ? 0.0f : result;
}

__device__ double uniform(unsigned int x, unsigned int y) {
  double result = raw_uniform_double(x, y);
  return result == 1.0 ? 0.0 : result;
}

__device__ double rng_uniform(
    const Array<uint32_t, 4>& rng_result,
    int rng_component) {
  return uniform(
      rng_result[rng_component * 2], rng_result[rng_component * 2 + 1]);
}

__device__ float rng_uniformf(
    const Array<uint32_t, 4>& rng_result,
    int rng_component) {
  return uniformf(rng_result[rng_component]);
}

__device__ __half
rng_uniform_half(const Array<uint32_t, 4>& rng_result, int rng_component) {
  return uniform_half(rng_result[rng_component]);
}

__device__ __bfloat
rng_uniform_bfloat(const Array<uint32_t, 4>& rng_result, int rng_component) {
  return uniform_bfloat(rng_result[rng_component]);
}

__device__ double rng_uniform_range(
    const Array<uint32_t, 4>& rng_result,
    int rng_component,
    double from,
    double to) {
  auto range = to - from;
  auto uniform01 = rng_uniform(rng_result, rng_component);
  return from + range * uniform01;
}

__device__ float rng_uniform_rangef(
    const Array<uint32_t, 4>& rng_result,
    int rng_component,
    float from,
    float to) {
  auto range = to - from;
  auto uniform01 = rng_uniformf(rng_result, rng_component);
  return from + range * uniform01;
}

__device__ __half rng_uniform_range_half(
    const Array<uint32_t, 4>& rng_result,
    int rng_component,
    float from,
    float to) {
  auto range = to - from;
  float uniform01 = raw_uniform_float(rng_result[rng_component]);
  __half result = __float2half(from + range * uniform01);
  return __heq(result, __float2half(to)) ? __float2half(from) : result;
}

__device__ __bfloat rng_uniform_range_bfloat(
    const Array<uint32_t, 4>& rng_result,
    int rng_component,
    float from,
    float to) {
  auto range = to - from;
  float uniform01 = raw_uniform_float(rng_result[rng_component]);
  __bfloat result = __float2bfloat(from + range * uniform01);
  return __heq(result, __float2bfloat(to)) ? __float2bfloat(from) : result;
}

__device__ float normalf(unsigned int x, unsigned int y, int rng_component) {
  float u = uniformf(x);
  float v = uniformf(y) * 6.2831855f;

  if (rng_component % 2 == 0) {
    return sqrtf(-2.0f * logf(u)) * sinf(v);
  } else {
    return sqrtf(-2.0f * logf(u)) * cosf(v);
  }
}

__device__ double normal(
    unsigned int x0,
    unsigned int x1,
    unsigned int y0,
    unsigned int y1,
    int rng_component) {
  double u = uniform(x0, x1);
  double v = uniform(y0, y1) * 6.2831853071795860;

  if (rng_component % 2 == 0) {
    return sqrt(-2.0 * log(u)) * sin(v);
  } else {
    return sqrt(-2.0 * log(u)) * cos(v);
  }
}

__device__ double rng_normal_standard(
    const Array<uint32_t, 4>& rng_result,
    int rng_component) {
  return normal(
      rng_result[0],
      rng_result[1],
      rng_result[2],
      rng_result[3],
      rng_component);
}

__device__ float rng_normal_standardf(
    const Array<uint32_t, 4>& rng_result,
    int rng_component) {
  return normalf(
      rng_result[rng_component / 2 * 2],
      rng_result[1 + rng_component / 2 * 2],
      rng_component);
}

__device__ __half rng_normal_standard_half(
    const Array<uint32_t, 4>& rng_result,
    int rng_component) {
  return __float2half(normalf(
      rng_result[rng_component / 2 * 2],
      rng_result[1 + rng_component / 2 * 2],
      rng_component));
}

__device__ __bfloat rng_normal_standard_bfloat(
    const Array<uint32_t, 4>& rng_result,
    int rng_component) {
  return __float2bfloat(normalf(
      rng_result[rng_component / 2 * 2],
      rng_result[1 + rng_component / 2 * 2],
      rng_component));
}

__device__ double rng_normal_general(
    const Array<uint32_t, 4>& rng_result,
    int rng_component,
    double mean,
    double std) {
  auto normal01 = rng_normal_standard(rng_result, rng_component);
  return normal01 * std + mean;
}

__device__ float rng_normal_generalf(
    const Array<uint32_t, 4>& rng_result,
    int rng_component,
    float mean,
    float std) {
  auto normal01 = rng_normal_standardf(rng_result, rng_component);
  return normal01 * std + mean;
}

__device__ __half rng_normal_general_half(
    const Array<uint32_t, 4>& rng_result,
    int rng_component,
    float mean,
    float std) {
  auto normal01 = normalf(
      rng_result[rng_component / 2 * 2],
      rng_result[1 + rng_component / 2 * 2],
      rng_component);
  return __float2half(normal01 * std + mean);
}

__device__ __bfloat rng_normal_general_bfloat(
    const Array<uint32_t, 4>& rng_result,
    int rng_component,
    float mean,
    float std) {
  auto normal01 = normalf(
      rng_result[rng_component / 2 * 2],
      rng_result[1 + rng_component / 2 * 2],
      rng_component);
  return __float2bfloat(normal01 * std + mean);
}
