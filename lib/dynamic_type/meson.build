project('dynamic_type', 'cpp')

dynamic_type_dep = declare_dependency(
    include_directories: include_directories('src')
)

install_subdir('src/dynamic_type', install_dir : 'include')


# test
gtest_dep = dependency('gtest_main')
gmock_dep = dependency('gmock')

foreach standard : ['17', '20', '23']
    name = 'test_dynamic_type_' + standard
    test_exe = executable(name,
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
        ],
        override_options: 'cpp_std=c++' + standard,
    )

    test(name, test_exe)
endforeach


# benchmarks
gbench_dep = dependency('benchmark')

foreach standard : ['17', '20', '23']
    name = 'bench_dynamic_type_' + standard
    bench_exe = executable(name,
        [
            'benchmark/main.cpp',
            'benchmark/knn.cpp',
            'benchmark/sort.cpp',
        ],
        dependencies: [
            dynamic_type_dep,
            gbench_dep,
        ],
        override_options: 'cpp_std=c++' + standard,
    )

    benchmark(name, bench_exe, timeout: 0)
endforeach
