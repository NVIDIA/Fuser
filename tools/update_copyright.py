import os

ext_map = {
    "cpp": ("c", "cpp", "cu", "cc", "cuh", "h"),
    "py": ("py",),
    "txt": ("txt",),
}

licence_style_0 = """\
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
"""

licence_style_1 = """\
# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""

header_license = {
    "cpp": licence_style_0,
    "py": licence_style_1,
    "txt": licence_style_1,
}

exclude_list = (
    "./tools/update_copyright.py",
    "./examples/sinh_libtorch/main.cpp",
    "./version.txt",
)


def get_exclusions():
    parent_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/..")
    print(parent_path)
    return [os.path.abspath(os.path.join(parent_path, f)) for f in exclude_list]


def has_licence(file_handle, licence_str):
    header_content = file_handle.read(len(licence_str))
    file_handle.seek(0, 0)
    return header_content.startswith(licence_str)


def update_licence(file_handle, licence_str):
    if not has_licence(file_handle, licence_str):
        content = file_handle.read()
        file_handle.seek(0, 0)
        file_handle.write(licence_str + content)
        return True
    return False


def update_files(root_path):
    exclusions = get_exclusions()
    print(exclusions)
    for root, dirs, files in os.walk(root_path):
        for file_name in files:
            abs_file = os.path.abspath(os.path.join(root, file_name))
            print(abs_file)
            if file_name[0] == "." or abs_file in exclusions:
                continue
            file_ext = file_name.split(".")[-1]
            for k, v in ext_map.items():
                if file_ext in v:
                    licence_str = header_license[k]
                    with open(abs_file, "r+") as file_handle:
                        if update_licence(file_handle, licence_str):
                            print("attached licence header to ", abs_file)


if __name__ == "__main__":
    update_files(".")
