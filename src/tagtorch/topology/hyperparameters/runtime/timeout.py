"""
Timeout helpers for executing arbitrary callables with either soft or hard timeout behavior.

**`soft` mode**
- pros
  - works well for ripser in practice
  - has low overhead (adds only a fraction of a second to the function call itself)
- cons
  - may fail to interrupt if the function calls into C/Cython code that doesn't check for
    Python signals
- dependencies
  - `func_timeout` and relies on asynchronous exception injection in a worker thread.

**`hard` mode**
- pros
  - forcibly terminates the function if it exceeds the timeout
- cons
  - higher overhead due to process creation
- dependencies
  - none, uses standard multiprocessing

Examples
--------

See tests/test_abstract_subprocesses_timeout.py for example usage
and timing diagnostics of both modes.
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
import importlib
from collections.abc import Callable
from queue import Empty
from typing import Any, Literal


class SoftTimeoutError(TimeoutError):
    """Raised when a soft timeout is exceeded."""


class HardTimeoutError(TimeoutError):
    """Raised when a hard timeout is exceeded."""


class SubprocessExecutionError(RuntimeError):
    """Raised when a subprocess execution fails with an exception."""


def _run_in_subprocess(
    result_queue: mp.Queue,
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    try:
        result = func(*args, **kwargs)
        result_queue.put(("ok", result))
    except BaseException as exc:
        result_queue.put(
            (
                "err",
                {
                    "exc": exc,
                    "exc_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                },
            )
        )


def run_with_timeout_soft(
    func: Callable[..., Any],
    timeout_sec: float,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Execute a function with a timeout using `func_timeout`.

    This mode is best-effort for code that spends long periods in C/Cython extensions.
    """
    try:
        func_timeout_module = importlib.import_module("func_timeout")
        func_timeout_function = getattr(func_timeout_module, "func_timeout")
        function_timed_out = getattr(func_timeout_module, "FunctionTimedOut")
    except ImportError as exc:
        raise ImportError(
            "run_with_timeout_soft requires func_timeout. Install it with: pip install func_timeout"
        ) from exc

    try:
        # func_timeout internally accesses func.__name__, so wrap callables that lack it
        # (e.g. functools.partial) in a thin named function before handing off.
        if not hasattr(func, '__name__'):
            def _wrapper(*a, **kw):
                return func(*a, **kw)
            _wrapper.__name__ = repr(func)
            _callable = _wrapper
        else:
            _callable = func
        return func_timeout_function(timeout_sec, _callable, args=args, kwargs=kwargs)
    except function_timed_out as exc:
        raise SoftTimeoutError(f"{getattr(func, '__name__', repr(func))} exceeded {timeout_sec} seconds") from exc


def run_with_timeout_hard(
    func: Callable[..., Any],
    timeout_sec: float,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Execute a function in a child process and force-terminate it if timeout is exceeded.

    `func`, `args`, and return value must be picklable for inter-process transport.
    """
    context = mp.get_context("spawn")
    result_queue: mp.Queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_run_in_subprocess,
        args=(result_queue, func, args, kwargs),
    )
    process.start()
    process.join(timeout_sec)

    if process.is_alive():
        process.terminate()
        process.join(1.0)
        if process.is_alive():
            process.kill()
            process.join(1.0)
        raise HardTimeoutError(f"{getattr(func, '__name__', repr(func))} exceeded {timeout_sec} seconds")

    if process.exitcode not in (0, None) and result_queue.empty():
        raise SubprocessExecutionError(
            f"Child process for {getattr(func, '__name__', repr(func))} exited with code {process.exitcode} and produced no result"
        )

    try:
        status, payload = result_queue.get_nowait()
    except Empty as exc:
        raise SubprocessExecutionError(
            f"Child process for {getattr(func, '__name__', repr(func))} finished without returning a result"
        ) from exc

    if status == "ok":
        return payload

    raise SubprocessExecutionError(
        f"{getattr(func, '__name__', repr(func))} raised {payload['exc_type']} in subprocess:\n{payload['traceback']}"
    ) from payload["exc"]


def run_with_timeout(
    func: Callable[..., Any],
    timeout_sec: float,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    mode: Literal["hard", "soft"] = "hard",
) -> Any:
    """
    Execute `func` with timeout control.

    Parameters
    ----------
    func:
        Callable to execute. This should be a top-level function or a picklable
        callable if using `hard` mode. Lambda functions are not picklable and 
        will cause `hard` mode to fail with a pickling error. To use a lambda or
        other non-picklable callable with `hard` mode, wrap it in a named function
        defined at the top level of a module. For example:
        ```
        def my_wrapper(x):
            return my_lambda(x)
        ```
    timeout_sec:
        Timeout in seconds.
    args:
        Positional arguments for `func`.
    kwargs:
        Keyword arguments for `func`.
    mode:
        `"hard"` to force-kill via subprocess, `"soft"` to use func_timeout.

    Choice of timeout mode:
    -----------------------    
    
    **`soft` mode**
    - pros
      - works well for ripser in practice
      - has low overhead (adds only a fraction of a second to the function call itself)
    - cons
      - may fail to interrupt if the function calls into C/Cython code that doesn't check for
        Python signals
    - dependencies
      - `func_timeout` and relies on asynchronous exception injection in a worker thread.

    **`hard` mode**
    - pros
      - forcibly terminates the function if it exceeds the timeout
    - cons
      - higher overhead due to process creation
    - dependencies
      - none, uses standard multiprocessing

    Examples
    --------

    See tests/test_abstract_subprocesses_timeout.py for example usage
    and timing diagnostics of both modes.      
    """
    if kwargs is None:
        kwargs = {}

    if mode == "hard":
        return run_with_timeout_hard(func, timeout_sec, *args, **kwargs)
    if mode == "soft":
        return run_with_timeout_soft(func, timeout_sec, *args, **kwargs)

    raise ValueError("mode must be either 'hard' or 'soft'")
