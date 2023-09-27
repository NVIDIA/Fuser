project('dynamic_type', 'cpp')

dynamic_type = declare_dependency(
    include_directories: include_directories('src'),
    compile_args: '-std=c++17',
)

install_subdir('src/dynamic_type', install_dir : 'include')


# test
gtest_proj = subproject('gtest')
gtest_main_dep = gtest_proj.get_variable('gtest_main_dep')
gmock_dep = gtest_proj.get_variable('gmock_dep')

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
        dynamic_type,
        gtest_main_dep,
        gmock_dep,
    ]
)

test('test_dynamic_type_17', test_exe)


# benchmarks
gbench_proj = subproject('google-benchmark')
gbench_dep = gbench_proj.get_variable('google_benchmark_dep')

bench_exe = executable('bench_dynamic_type_17',
    [
        'benchmark/main.cpp',
        'benchmark/knn.cpp',
        'benchmark/sort.cpp',
    ],
    dependencies: [
        dynamic_type,
        gbench_dep,
    ],
)

test('bench_dynamic_type_17', bench_exe)
