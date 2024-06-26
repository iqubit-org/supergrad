.. SuperGrad documentation master file, created by
   sphinx-quickstart on Tue Jul 12 14:22:39 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SuperGrad's documentation!
====================================================================
Overview
--------------------------------------------------------------------
SuperGrad is a differentiable Hamiltonian simulator, primarily designed for superconducting quantum processors. However, it has the potential to be expanded to accommodate other physical platforms. Its key feature lies in its ability to compute gradients with respect to objective functions, such as gate fidelity or functions that approximate logical error rates. The motivation behind gradient computation is rooted in the fact that superconducting processors often consist of numerous qubits, resulting in a rapid increase in the total number of optimizable parameters. Consequently, when optimizing the parameters of a quantum system or fitting them to experimental data, leveraging gradients can significantly accelerate these tasks.

Documentations
--------------------------------------------------------------------
.. toctree::
   :maxdepth: 2

   Tutorials <tutorials/tutorials.rst>
   Examples <examples/examples.rst>

API References
--------------------------------------------------------------------
.. toctree::
   :maxdepth: 2

   API References <api/api_supergrad>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
