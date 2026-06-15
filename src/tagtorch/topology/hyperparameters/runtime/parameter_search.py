"""
Tools to automatically select the largest argument which a function can
accept, subject to the requirement that the function runs within a
specified time limit.

**This module offers tools to end the execution of a function which is 
already running, using subprocesses.** However we are still debugging.
"""

import sys
import time
import reprlib
from typing import Literal

import numpy as np
from ripser import ripser

from tagtorch.topology.hyperparameters.runtime.timeout import (
    HardTimeoutError,
    SoftTimeoutError,
    run_with_timeout,
)


# =============================================================
# LINEAR SEARCH FOR MAX PARAMETER -- WITHOUT SUBPROCESS TIMEOUTS
# =============================================================

def max_argument_for_runtime_with_linear_search(
        function_to_time    =   None,
        max_runtime         =   5, 
        parameter_values    =   None
    ):
    """
    Returns the maximum parameter value for which function_to_time runs within max_runtime seconds.

    Uses a simple linear search through parameter_values, timing each one with time.time().
    """

    if function_to_time is None:
        raise ValueError("function_to_time must be provided.")
    if parameter_values is None or len(parameter_values) == 0:
        raise ValueError("parameter_values must be a non-empty list of parameter values to test.")
    good_param = None
    for param in parameter_values:
        start = time.time()
        function_to_time(param)
        duration = time.time() - start
        print(f"Parameter value {param} took {duration:.2f} seconds.")
        if duration > max_runtime:
            print(f"Parameter value {param} exceeded max runtime of {max_runtime} seconds.")
            return good_param
        else:
            good_param = param
    print(f"All parameter values tested within max runtime. Returning best value: {good_param}")
    return good_param



# =============================================================
# BINARY SEARCH FOR MAX PARAMETER -- WITH SUBPROCESS TIMEOUTS
# =============================================================


def max_argument_for_runtime_with_binary_search(
        function_to_time    =   None,
        max_runtime         =   5, 
        parameter_sets      =   None,
    mode: Literal["hard", "soft"] = "hard",
):
    """
    Uses binary search to find the largest argument set for which the function
    runs within the specified runtime.

    Each item in ``parameter_sets`` should be an ordered test case dictionary:
      - ``{"args": (...), "kwargs": {...}}``
      - optional ``"label"`` for logging

    The list must be sorted from easiest/fastest to hardest/slowest so that
    timeout success is monotonic and binary search is valid.

    Parameters:
    - function_to_time: The function to be timed.
    - max_runtime: The maximum allowed runtime in seconds.
    - parameter_sets: Ordered list of dictionaries containing ``args`` and ``kwargs``.
    - mode: "hard" to force-kill via subprocess, "soft" to use func_timeout (see run_with_timeout).
    Returns:
    - The best parameter-set dictionary, or ``None`` if all tested sets time out.
    """
    if function_to_time is None:
        raise ValueError("function_to_time must be provided.")

    if parameter_sets is None or len(parameter_sets) == 0:
        raise ValueError("parameter_sets must be a non-empty list of dictionaries with args/kwargs.")

    low, high = 0, len(parameter_sets) - 1
    best_spec = None

    while low <= high:
        mid = (low + high) // 2
        spec = parameter_sets[mid]
        args = tuple(spec.get("args", ()))
        kwargs = dict(spec.get("kwargs", {}))
        label = spec.get("label", reprlib.repr({"args": args, "kwargs": kwargs}))
        print(f"[max_argument_for_runtime_with_binary_search] Testing parameter set: {label}")
        try:
            run_with_timeout(function_to_time, timeout_sec=max_runtime, args=args, kwargs=kwargs, mode=mode)
            best_spec = spec
            low = mid + 1
        except (HardTimeoutError, SoftTimeoutError):
            high = mid - 1

    print(
        "[max_argument_for_runtime_with_binary_search] Best parameter set found: "
        f"{reprlib.repr(best_spec)}"
    )
    return best_spec


# =============================================================
# TESTS
# =============================================================

# ----------------
# tests
# ----------------

def test_max_argument_for_runtime_with_binary_search():
    "testing ripsers with hard and soft modes"
    thresh_values = np.linspace(0, 1, num=20)
    points = np.random.rand(5000, 2)
    parameter_sets = [
        {
            "args": (points,),
            "kwargs": {"maxdim": 1, "thresh": float(thresh)},
            "label": f"thresh={float(thresh):.4f}",
        }
        for thresh in thresh_values
    ]
    
    for mode in ("hard", "soft"):
        best_spec = max_argument_for_runtime_with_binary_search(
            ripser,
            max_runtime=1,
            parameter_sets=parameter_sets,
            mode=mode,
        )
        best_thresh = None if best_spec is None else best_spec["kwargs"]["thresh"]
        print(f"[{mode}] Maximum thresh that runs within 1 second: {best_thresh}")

def slow_func(x):
    import time
    print(f"  [slow_func] Sleeping for {x} seconds...")
    time.sleep(x)
    print(f"  [slow_func] Done sleeping for {x} seconds.")
    return x

def test_max_argument_for_runtime_with_binary_search_2():
    param_values = [0.1, 0.5, 1, 2, 8, 15]
    parameter_sets = [
        {"args": (value,), "kwargs": {}, "label": f"x={value}"}
        for value in param_values
    ]
    for mode in ("hard", "soft"):
        best_spec = max_argument_for_runtime_with_binary_search(
            slow_func, max_runtime=2, parameter_sets=parameter_sets, mode=mode
        )
        best_value = None if best_spec is None else best_spec["args"][0]
        print(f'[{mode}] Should find max param <= 2: {best_value}')

