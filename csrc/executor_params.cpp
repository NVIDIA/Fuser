// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <debug.h>
#include <executor_params.h>

#include <ATen/cuda/CUDAContext.h>

namespace nvfuser {

void LaunchParams::assertValid() {
  NVF_ERROR(
      bdimx() * bdimy() * bdimz() > 0 &&
          bdimx() * bdimy() * bdimz() <=
              (int64_t)at::cuda::getCurrentDeviceProperties()
                  ->maxThreadsPerMultiProcessor,
      "Selected invalid number of threads for cuda: ",
      bdimx() * bdimy() * bdimz());
  NVF_ERROR(
      gdimx() > 0 && gdimx() < (std::int64_t(1) << 32) - 1,
      "Invalid number of blocks in x direction: ",
      gdimx());
  NVF_ERROR(
      gdimy() > 0 && gdimy() <= 65535,
      "Invalid number of blocks in y direction: ",
      gdimy());
  NVF_ERROR(
      gdimz() > 0 && gdimz() <= 65535,
      "Invalid number of blocks in z direction: ",
      gdimz());
}

void LaunchParams::bind(int64_t val, ParallelType p_type) {
  switch (p_type) {
    case ParallelType::TIDx:
      checkAndSet(val, bdimx_, "blockDim.x");
      break;
    case ParallelType::BIDx:
    case ParallelType::BIDxCluster:
      checkAndSet(val, gdimx_, "gridDim.x");
      break;
    case ParallelType::TIDy:
      checkAndSet(val, bdimy_, "blockDim.y");
      break;
    case ParallelType::BIDy:
      checkAndSet(val, gdimy_, "gridDim.y");
      break;
    case ParallelType::TIDz:
      checkAndSet(val, bdimz_, "blockdim.z");
      break;
    case ParallelType::BIDz:
      checkAndSet(val, gdimz_, "gridDim.z");
      break;
    default:
      NVF_ERROR(
          false,
          "Tried to bind invalid parallel type in launch config: ",
          p_type);
  }
  assertValid();
}

int64_t LaunchParams::getDim(ParallelType p_type) const {
  switch (p_type) {
    case ParallelType::TIDx:
      return bdimx();
    case ParallelType::BIDx:
    case ParallelType::BIDxCluster:
      return gdimx();
    case ParallelType::TIDy:
      return bdimy();
    case ParallelType::BIDy:
      return gdimy();
    case ParallelType::TIDz:
      return bdimz();
    case ParallelType::BIDz:
      return gdimz();
    default:
      NVF_ERROR(
          false,
          "Tried to get with invalid parallel type in launch config: ",
          p_type);
  }
}

bool LaunchParams::hasDim(ParallelType p_type) const {
  return getRawVal(p_type) != UNINITIALIZED_VAL;
}

const int64_t& LaunchParams::getRawVal(ParallelType p_type) const {
  switch (p_type) {
    case ParallelType::TIDx:
      return bdimx_;
    case ParallelType::BIDx:
    case ParallelType::BIDxCluster:
      return gdimx_;
    case ParallelType::TIDy:
      return bdimy_;
    case ParallelType::BIDy:
      return gdimy_;
    case ParallelType::TIDz:
      return bdimz_;
    case ParallelType::BIDz:
      return gdimz_;
    default:
      NVF_ERROR(
          false,
          "Tried to get with invalid parallel type in launch config: ",
          p_type);
  }
}

bool LaunchParams::operator==(const LaunchParams& other) const {
  return gdimx_ == other.gdimx_ && gdimy_ == other.gdimy_ &&
      bdimx_ == other.bdimx_ && bdimy_ == other.bdimy_ && smem_ == other.smem_;
}

void LaunchParams::print() const {
  debug() << toString();
}

std::string LaunchParams::toString() const {
  std::stringstream ss;
  ss << "Launch Parameters: "
     << "BlockDim.x = " << (bdimx_ == UNINITIALIZED_VAL ? -1 : bdimx_) << ", "
     << "BlockDim.y = " << (bdimy_ == UNINITIALIZED_VAL ? -1 : bdimy_) << ", "
     << "BlockDim.z = " << (bdimz_ == UNINITIALIZED_VAL ? -1 : bdimz_) << ", "
     << "GridDim.x = " << (gdimx_ == UNINITIALIZED_VAL ? -1 : gdimx_) << ", "
     << "GridDim.y = " << (gdimy_ == UNINITIALIZED_VAL ? -1 : gdimy_) << ", "
     << "GridDim.z = " << (gdimz_ == UNINITIALIZED_VAL ? -1 : gdimz_) << ", "
     << "Smem Size = " << smem() << "\n";
  return ss.str();
}

flatbuffers::Offset<serde::LaunchParams> LaunchParams::serialize(
    flatbuffers::FlatBufferBuilder& builder) const {
  // See table definition for LaunchParams in serde/fusion_cache.fbs
  using fb_tensor_shape = flatbuffers::Offset<serde::TensorShape>;
  std::vector<fb_tensor_shape> shapes_fb;
  shapes_fb.reserve(output_sizes.size());
  for (const auto& shape : output_sizes) {
    shapes_fb.push_back(serde::CreateTensorShapeDirect(builder, &shape));
  }
  return serde::CreateLaunchParamsDirect(
      builder,
      gdimx_,
      gdimy_,
      gdimz_,
      bdimx_,
      bdimy_,
      bdimz_,
      smem_,
      &shapes_fb);
}

void LaunchParams::deserialize(const serde::LaunchParams* buffer) {
  // See table definitions for LaunchParams and TensorShape in
  // serde/fusion_cache.fbs
  NVF_ERROR(buffer != nullptr, "serde::LaunchParams is nullptr.");

  gdimx_ = buffer->gdimx();
  gdimy_ = buffer->gdimy();
  gdimz_ = buffer->gdimz();
  bdimx_ = buffer->bdimx();
  bdimy_ = buffer->bdimy();
  bdimz_ = buffer->bdimz();
  smem_ = buffer->smem();

  for (auto fb_shape : *buffer->output_sizes()) {
    output_sizes.emplace_back(
        fb_shape->shape()->begin(), fb_shape->shape()->end());
  }
}

} // namespace nvfuser
