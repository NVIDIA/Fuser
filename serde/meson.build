flatc = find_program('flatc')
flatbuffer_header = custom_target(
    'flatbuffer_header',
    input : ['../csrc/serde/fusion_cache.fbs'],
    output : 'fusion_cache_generated.h',
    command : [
        flatc,
        '--gen-object-api',
        '-o',
        '@OUTDIR@',
        '-c',
        '-b',
        '@INPUT@',
    ]
)
