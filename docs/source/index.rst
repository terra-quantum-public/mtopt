tq_mtopt Documentation
============================

tq_mtopt is a lightweight Python library that implements two optimizers for black-box optimization on discrete grids:
Tensor Rank Cross (TRC) and Matrix Train (MT). Both methods build low-rank tensor representations of the objective
from a small number of function evaluations, then extract candidate optima directly from the representations.
The package targets use-cases where objective gradients are unavailable or unreliable,
and where each function call is expensive (simulation, hardware loop, or batched evaluation).
It offers:

* Tensor-rank-cross (TRC) and matrix-train (MT) optimizers
* Interfaces for common test functions (Ackley, Rastrigin, etc.)
* Simple runner scripts for benchmarking and plotting
* A modular design for adding your own optimizers and targets

.. note::

   This documentation is automatically generated using Sphinx and autodoc.
   Run ``./generate_docs.sh`` from the project root to regenerate both the
   reStructuredText API stubs and the HTML documentation.

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   usage
   benchmarks

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
