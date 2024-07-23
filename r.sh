NVFUSER_DUMP=lower_verbose,segmented_fusion NVFUSER_DISABLE=parallel_compile python bug.py 2>&1 |tee 1.log
NVFUSER_DUMP=segmented_fusion NVFUSER_DISABLE=parallel_compile python bug.py 2>&1 |tee 1.log
NVFUSER_DUMP=pre_segmenter_logging,segmented_fusion NVFUSER_DISABLE=parallel_compile python bug.py 2>&1 |tee 1.log
