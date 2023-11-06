// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#pragma once

#include <multidevice/communicator.h>
#include <test/utils.h>

namespace nvfuser {

class MultiDeviceTest : public NVFuserTest {
 protected:
  static Communicator comm;
};

} // namespace nvfuser

#endif
