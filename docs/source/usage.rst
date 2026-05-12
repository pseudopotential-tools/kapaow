Usage
=====

The everyday entry point is the :doc:`command-line interface <cli>` —
``kapaow optimize rc`` for the bisection rc search, ``kapaow optimize
spread`` for the pareto front, ``kapaow plot periodic-table`` for the
sweep plots, and ``kapaow convert`` / ``kapaow confine`` for one-off
file manipulations.

The same building blocks are also available as a Python API.

Basis manipulation
------------------

.. automodule:: kapaow.basis
    :members:

.. automodule:: kapaow.extend
    :members:

Pseudoatomic solver
-------------------

.. automodule:: kapaow.solve
    :members:

Radius search and pareto front
------------------------------

.. automodule:: kapaow.rc_search
    :members:

.. automodule:: kapaow.pareto
    :members:
