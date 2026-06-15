"""
# Hyperparameter Utilities for Topological Data Analysis

This module provides tools for automatic selection of hyperparameters
used in persistent homology computations, such as maximum dimension and
epsilon thresholds.

## Background

Domain scientists are increasingly interested in use methods in TDA to identify complex nonlinear patterns to better understand their data.
However, to realize this goal they must select and tune TDA models which are appropriate for their specific scientific applications.
Proper selection currently requires both expert knowledge of TDA methods and labor-intensive analysis of user data.
This creates a gap between interest and ability for the vast majority of scientists, who lack both the expertise and the available labor for data analysis.
TAGTorch is building tools to help automate model selection and tuning, thus closing the gap.


## Example: Managing Compute Budgets

Researchers typically require accurate estimates of compute costs (in time, memory, and dollars) to plan a successful modeling workflow.
However, many topological models are resource-intensive and data-dependent.
The TDA component of a modeling workflow can thus be large and unpredictable.
To address this challenge, TAGTorch is building tools for automated exploration of trade-off curves, which can accurately forecast the maximum amount of topological information which can be extracted from a data set, given a fixed compute budget.
These tools leverage the shared experience of a community of TDA experts familiar with these trade-offs.
However, we are always looking to improve. 
If you have insights or suggestions, please reach out to us!
"""