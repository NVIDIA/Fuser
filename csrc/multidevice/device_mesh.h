// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#pragma once
#include <multidevice/multidevice.h>

namespace nvfuser {

/*
    The class DevishMesh represents a set of devices on which a Pipeline Stage will be executed.
    This can be simply seen as a list (or n-array) of device indices (i.e. ints).
    Here, "dimensions" stands for the shape of the Mesh seen as an array.
    By default, we assume a flat Mesh, i.e., dimensions = (number_of_device_indices)
*/
class DeviceMesh final {
public:
    /* return the flatten vector of device indices */
    const auto& deviceIndices() const {
        return deviceIndices_;
    }

    /* return the shape of the mesh */
    const auto& dimensions() const {
        return dimensions_;
    }

    /* return the total length of the mesh */
    size_t size() const {
        size_t ret = 1;
            for (auto dimension: dimensions_) {
                ret *= dimension;
            }
            return ret;
    }

    /* set the attributes of the mesh */
    void set(std::vector<DeviceIdxType> deviceIndices, std::vector<DimensionType> dimensions)
    {
        deviceIndices_ = deviceIndices;
        dimensions_ = dimensions;

        TORCH_INTERNAL_ASSERT(validate(deviceIndices));
    }

    void set(std::vector<DeviceIdxType> deviceIndices)
    {
        DimensionType length = static_cast<DimensionType>(deviceIndices.size());
        set(deviceIndices, {length});
    }

private:
    /* returns whether the mesh is valid */
    bool validate(std::vector<DeviceIdxType> deviceIndices)
    {
        return (size() == deviceIndices.size() && size() > 0);
    }

    /* stores the device indices */
    std::vector<DeviceIdxType> deviceIndices_;
    /* stores the mesh shape */
    std::vector<DimensionType> dimensions_;
};

} // namespace nvfuser
