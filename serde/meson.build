flatc = find_program('flatc')
flatbuffer_header = custom_target(
    'fusion_cache_generated.h',
    input : ['../csrc/serde/fusion_cache.fbs'],
    output : 'fusion_cache_generated.h',
    command : [
        flatc,
        '--gen-object-api',
        '-o',
        '@OUTPUT@',
        '-c',
        '-b',
        '@INPUT@',
    ]
)
