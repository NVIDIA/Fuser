runtime_resource_targets = []

runtime_resources = [
    'array',
    'basic_type_traits',
    'bf16_support',
    'bit',
    'block_reduction',
    'block_sync_atomic',
    'block_sync_default',
    'block_welford_outer',
    'broadcast',
    'complex_number',
    'fp16_support',
    'fused_reduction',
    'fused_welford_helper',
    'fused_welford_impl',
    'fused_welford_impl_outer',
    'grid_broadcast',
    'grid_reduction',
    'grid_sync',
    'helpers',
    'index_utils',
    'memory',
    'random_numbers',
    'tensor',
    'tensorcore',
    'tuple',
    'type_traits',
    'warp',
    'welford',
]

foreach runtime : runtime_resources
    target = custom_target(
        'nvfuser_runtime_' + runtime,
        input : ['../tools/stringify_file.py', '../runtime/' + runtime + '.cu'],
        output : runtime + '.h',
        command : [
            python_installation,
            '@INPUT0@',
            '-i',
            '@INPUT1@',
            '-o',
            '@OUTPUT@',
        ]
    )
    runtime_resource_targets += target
endforeach
