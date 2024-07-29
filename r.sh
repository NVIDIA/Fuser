NVFUSER_DISABLE=parallel_compile NVFUSER_DUMP=segmented_fusion,scheduler_params,launch_param python 2702.py 2>&1 |tee 1.log
