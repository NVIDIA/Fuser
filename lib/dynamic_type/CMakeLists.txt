# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

add_library(dynamic_type INTERFACE)
target_include_directories(dynamic_type INTERFACE src)

function(add_test_for_standard std_version)
    set(target test_dynamic_type_${std_version})
    add_executable(${target}
        test/test_dynamic_type.cpp
        test/ForAllTypes.cpp
        test/binary_ops.cpp
        test/examples.cpp
        test/move.cpp
        test/null.cpp
        test/opcheck.cpp
        test/typing.cpp
    )
    target_include_directories(${target} PUBLIC src)
    target_link_libraries(${target} PRIVATE gtest_main gmock_main)
    set_property(TARGET ${target} PROPERTY CXX_STANDARD ${std_version})
endfunction()

add_test_for_standard(17)

# add_test_for_standard(20)
# add_test_for_standard(23)
# add_test_for_standard(26)
