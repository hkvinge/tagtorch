"""
Runtime hyperparameter utilities for persistent homology.

This module provides tools for estimating hyperperameters
(such as maximum dimension and epsilon thresholds in persistent
homology computations) that are likely to run within a specified
time limit.

Submodules:
    ph                  -- specialized 
    abstract            -- generic timeout/search helpers
    abstract_subprocess -- soft/hard timeout execution helpers
"""