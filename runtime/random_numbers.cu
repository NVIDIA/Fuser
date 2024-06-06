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

__device__ uint4 single_round(uint4 ctr, uint2 key) {
  constexpr unsigned long kPhiloxSA = 0xD2511F53;
  constexpr unsigned long kPhiloxSB = 0xCD9E8D57;
  unsigned int hi0;
  unsigned int hi1;
  unsigned int lo0 = mulhilo32(kPhiloxSA, ctr.x, &hi0);
  unsigned int lo1 = mulhilo32(kPhiloxSB, ctr.z, &hi1);
  uint4 ret = {hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0};
  return ret;
}

__device__ uint4 philox(
    unsigned long long seed,
    unsigned long long subsequence,
    unsigned long long offset) {
  constexpr unsigned long kPhilox10A = 0x9E3779B9;
  constexpr unsigned long kPhilox10B = 0xBB67AE85;
  uint2 key = {};
  key.x = (unsigned int)seed;
  key.y = (unsigned int)(seed >> 32);
  uint4 counter = make_uint4(0, 0, 0, 0);
  counter.x = (unsigned int)(offset);
  counter.y = (unsigned int)(offset >> 32);
  counter.z = (unsigned int)(subsequence);
  counter.w = (unsigned int)(subsequence >> 32);

  uint4 output = {};
  uint2 key_ = key;
  uint4 counter_ = counter;
  for (int i = 0; i < 9; i++) {
    counter_ = single_round(counter_, key_);
    key_.x += (kPhilox10A);
    key_.y += (kPhilox10B);
  }
  output = single_round(counter_, key_);
  return output;
}

template <unsigned int significand_bits>
__device__ float uniform_for_casting(unsigned int x) {
  // We scale the values to lie between 0 and 1 such that the highest generated
  // value does not round to 1.0 when converted to half. Scaling is done by the
  // transform x => scale * x + 0.5 * scale (see note below) so we choose scale
  // accordingly.
  //
  // For a floating point value with significand having B bits of precision,
  // nextafter(1.0, 0.0) == 1 - 1/(2^B). In round-to-nearest mode this
  // means the highest scaled generated value `scale * (N-1) + 0.5 * scale` must
  // be strictly less than 1 - 1/(2^(B+1)) (where N=2^32 for unsigned int input
  // x). Equality is achieved when scale = (1 - 1/(2^(B+1))) / (N - 0.5). Since
  // we need a scale strictly less than this, we do not subtract 0.5 from N.
  constexpr float scale =
      (float)((1.0 - 1.0 / (double)(1l << (significand_bits))) /
              (double)(1l << 32));

  // x is an int between 0 and N-1 (inclusive) for N=2^bits. After scaling this
  // becomes a float between 0 and (N-1)/N=1-1/N. We add 1/2N so that the mean
  // of the generated values equals 0.5.
  float result = (float)x * scale + (scale / 2.0f);
  return result == 1 ? 0.0f : result;
}

// Returns float since we might still need to scale to range
__device__ float uniform_half(unsigned int x) {
  // significand precision for float16 is 11 bits
  return uniform_for_casting<11>(x);
}

// Returns float since we might still need to scale to range
__device__ float uniform_bfloat(unsigned int x) {
  // significand precision for bfloat16 is 8 bits
  return uniform_for_casting<8>(x);
}

__device__ float uniformf(unsigned int x) {
  constexpr float kRanInvM32 = 2.3283064e-10f; // Inverse of 2^32.
  float result = x * kRanInvM32 + kRanInvM32 / 2.0f;
  return result == 1 ? 0.0f : result;
}

__device__ double uniform(unsigned int x, unsigned int y) {
  constexpr double kRan2Pow53Inv = 1.1102230246251565e-16;
  const unsigned long long z =
      (unsigned long long)x ^ ((unsigned long long)y << (53 - 32));
  double result = z * kRan2Pow53Inv + (kRan2Pow53Inv / 2.0);
  return result == 1 ? 0.0 : result;
}

__device__ double rng_uniform(const uint4& rng_result, int rng_component) {
  return uniform(
      (&rng_result.x)[rng_component * 2],
      (&rng_result.x)[rng_component * 2 + 1]);
}

__device__ float rng_uniformf(const uint4& rng_result, int rng_component) {
  return uniformf((&rng_result.x)[rng_component]);
}

__device__ __half rng_uniform_half(const uint4& rng_result, int rng_component) {
  return __float2half(uniform_half((&rng_result.x)[rng_component]));
}

__device__ __bfloat
rng_uniform_bfloat(const uint4& rng_result, int rng_component) {
  return __float2bfloat(uniform_bfloat((&rng_result.x)[rng_component]));
}

__device__ double rng_uniform_range(
    const uint4& rng_result,
    int rng_component,
    double from,
    double to) {
  auto range = to - from;
  auto uniform01 = rng_uniform(rng_result, rng_component);
  return from + range * uniform01;
}

__device__ float rng_uniform_rangef(
    const uint4& rng_result,
    int rng_component,
    float from,
    float to) {
  auto range = to - from;
  auto uniform01 = rng_uniformf(rng_result, rng_component);
  return from + range * uniform01;
}

__device__ __half rng_uniform_range_half(
    const uint4& rng_result,
    int rng_component,
    float from,
    float to) {
  auto range = to - from;
  auto uniform01 = uniform_half((&rng_result.x)[rng_component]);
  return __float2half(from + range * uniform01);
}

__device__ __bfloat rng_uniform_range_bfloat(
    const uint4& rng_result,
    int rng_component,
    float from,
    float to) {
  auto range = to - from;
  auto uniform01 = uniform_bfloat((&rng_result.x)[rng_component]);
  return __float2bfloat(from + range * uniform01);
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
    const uint4& rng_result,
    int rng_component) {
  return normal(
      rng_result.x, rng_result.y, rng_result.z, rng_result.w, rng_component);
}

__device__ float rng_normal_standardf(
    const uint4& rng_result,
    int rng_component) {
  return normalf(
      (&rng_result.x)[rng_component / 2 * 2],
      (&rng_result.y)[rng_component / 2 * 2],
      rng_component);
}

__device__ double rng_normal_general(
    const uint4& rng_result,
    int rng_component,
    double mean,
    double std) {
  auto normal01 = rng_normal_standard(rng_result, rng_component);
  return normal01 * std + mean;
}

__device__ float rng_normal_generalf(
    const uint4& rng_result,
    int rng_component,
    float mean,
    float std) {
  auto normal01 = rng_normal_standardf(rng_result, rng_component);
  return normal01 * std + mean;
}
