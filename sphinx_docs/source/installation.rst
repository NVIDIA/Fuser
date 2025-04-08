Installation
============

Installing nvFuser
-----------------

You can install nvFuser using pip:

.. code-block:: bash

   pip install nvFuser

Building from Source
-------------------

To build nvFuser from source:

1. Clone the repository:
   
   .. code-block:: bash

      git clone https://github.com/NVIDIA/nvFuser.git
      cd nvFuser

2. Install dependencies:
   
   .. code-block:: bash

      pip install -r requirements.txt

3. Build the package:
   
   .. code-block:: bash

      python setup.py install

Requirements
-----------

* Python 3.8 or higher
* PyTorch 2.0 or higher
* CUDA toolkit (version compatible with your PyTorch installation) 
