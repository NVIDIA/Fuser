Welcome to nvFuser's documentation!
===================================

nvFuser is a PyTorch JIT compiler that fuses operations in PyTorch programs to provide better performance.

.. toctree::
   :maxdepth: 3
   :caption: Table of Contents

   Overview <installation.md>

.. toctree::
   :hidden:
   :caption: Python API

   api/general
   api/ops
   api/multidevice
   api/enum
   api/pod_class


.. toctree::
   :hidden:
   :caption: Developer References

   reading/divisibility-of-split.md
   reading/iterdomain.md
   reading/multigpu.md
   reading/tma-modeling-in-depth.md
   dev/debug.md
   dev/visibility.md
   dev/host_ir_jit.md
   dev/ldmatrix_stmatrix.md
   dev/tma.md
   dev/tmem.md
