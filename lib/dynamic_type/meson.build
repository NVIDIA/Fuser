project('dynamic_type', 'cpp')

dynamic_type_dep = declare_dependency(
    include_directories: include_directories('src'),
    compile_args: '-std=c++17',
)

install_subdir('src/dynamic_type', install_dir : 'include')


# test
gtest_dep = dependency('gtest_main')
gmock_dep = dependency('gmock')

test_exe = executable('test_dynamic_type_17',
    [
        'test/ForAllTypes.cpp',
        'test/assignment.cpp',
        'test/binary_ops.cpp',
        'test/container.cpp',
        'test/examples.cpp',
        'test/hash.cpp',
        'test/member.cpp',
        'test/move.cpp',
        'test/null.cpp',
        'test/opcheck.cpp',
        'test/print.cpp',
        'test/typing.cpp',
        'test/unary_ops.cpp',
    ],
    dependencies: [
        dynamic_type_dep,
        gtest_dep,
        gmock_dep,
    ]
)

test('test_dynamic_type_17', test_exe)


# benchmarks
gbench_dep = dependency('benchmark')

bench_exe = executable('bench_dynamic_type_17',
    [
        'benchmark/main.cpp',
        'benchmark/knn.cpp',
        'benchmark/sort.cpp',
    ],
    dependencies: [
        dynamic_type_dep,
        gbench_dep,
    ],
)

benchmark('bench_dynamic_type_17', bench_exe, timeout: 0)
