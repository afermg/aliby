"""Shared helpers for running per-time-series work in parallel."""

import concurrent.futures
import contextlib
import multiprocessing
import os
import re
import sys

# matches a real `if __name__ == "__main__":` guard line
main_guard_re = re.compile(r"""if\s+__name__\s*==\s*['"]__main__['"]\s*:""")
# cache the positive guard check so the driver is read at most once
main_guard_checked = False


def resolve_no_processors(no_processors):
    """
    Convert a no_processors argument to a worker count.

    Accept bool or int. ``True`` maps to one fewer than the number of
    available CPUs (so at least one core is free for the rest of the
    system); ``False`` and any integer ``<= 1`` map to ``1`` (serial).
    Use ``is`` to distinguish the boolean flags from the equal
    integers ``0`` and ``1``, since Python treats bool as a subclass
    of int.

    When the resolved count exceeds 1 and the caller is running
    inside IPython or Jupyter, downgrade to ``1`` and print a single
    line explaining the override. Forking a multiprocessing pool
    from an interactive interpreter can hang, notably on macOS, so
    the safe choice is to run in serial regardless of what the user
    asked for; a plain script gets the requested parallelism.

    Parameters
    ----------
    no_processors: bool or int
        Caller's requested worker count, or a flag selecting auto
        (``True``) or serial (``False``) mode.

    Returns
    -------
    int
        Safe worker count to pass through to the parallel code paths.
    """
    if no_processors is True:
        resolved = max(1, multiprocessing.cpu_count() - 1)
    elif no_processors is False:
        resolved = 1
    else:
        resolved = int(no_processors)
    if resolved > 1 and running_in_ipython():
        print(
            "Running in IPython or Jupyter; falling back to one worker"
            " because forking from an interactive interpreter can hang"
            " (notably on macOS). Run as a plain script to use the"
            " requested workers."
        )
        resolved = 1
    return resolved


def running_in_ipython():
    """
    Check if the code is running inside IPython or Jupyter.

    Forking multiprocessing workers from an IPython or Jupyter
    process can hang because the parent process is then unsafe to
    fork, particularly on macOS.

    Returns
    -------
    bool
        True if running inside IPython or Jupyter.
    """
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ImportError:
        return False


@contextlib.contextmanager
def silence_output():
    """
    Suppress text sent to stdout and stderr.

    gaussianprocessderivatives reports its numerical warnings with
    print, which corrupts the tqdm progress bar when several workers
    fit Gaussian processes in parallel.
    """
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            with contextlib.redirect_stderr(devnull):
                yield


def source_has_main_guard(source):
    """
    Return True if source contains a real main guard.

    Skip commented-out lines and inline comments so that a
    placeholder like ``# if __name__ == "__main__":`` does not count
    as a guard.

    Parameters
    ----------
    source: str
        Contents of the driver script.

    Returns
    -------
    bool
        True if an active `if __name__ == "__main__":` line is found.
    """
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # strip simple inline comments; safe for guard lines because
        # the guard expression contains no '#' characters
        active = stripped.split("#", 1)[0]
        if main_guard_re.search(active):
            return True
    return False


def require_main_guard():
    """
    Raise if the driver script lacks a main guard.

    The parallel pools use the spawn or forkserver start method, both
    of which re-import the driver's ``__main__`` in their helper
    process. A driver that runs its analysis at top level therefore
    re-executes it recursively in every worker spawn. Rather than
    letting the run proceed to a confusing flood of
    ``BrokenProcessPool`` messages, the first parallel call raises with
    a clear, actionable error.

    Stay silent in interactive contexts (Jupyter and IPython have no
    ``__main__.__file__``). Cache the positive result so the file is
    read at most once per session.
    """
    global main_guard_checked
    if main_guard_checked:
        return
    main_module = sys.modules.get("__main__")
    if main_module is None:
        return
    main_file = getattr(main_module, "__file__", None)
    if main_file is None:
        return
    try:
        with open(main_file) as fh:
            source = fh.read()
    except OSError:
        return
    if source_has_main_guard(source):
        main_guard_checked = True
        return
    raise RuntimeError(
        f"\nParallel processing requires {main_file}\n"
        'to use an `if __name__ == "__main__":` guard around its\n'
        "top-level analysis code. The pool is built with spawn or\n"
        "forkserver multiprocessing, which re-imports the main module\n"
        "in its helper process; without the guard your script would\n"
        "execute recursively in the workers and surface as a flood of\n"
        "`BrokenProcessPool` errors.\n\n"
        "Wrap the analysis (everything after the imports and\n"
        "module-level constants) in:\n\n"
        '    if __name__ == "__main__":\n'
        "        ...\n\n"
        "Module-level constants and imports can stay outside the"
        " guard.\n"
    )


def make_process_executor(max_workers, start_method="spawn"):
    """
    Build a ProcessPoolExecutor using a chosen start method.

    The start method only affects how workers are created, never their
    steady-state speed; pick it to suit the worker's native libraries.
    Plain ``fork`` is never safe on macOS here: Accelerate's BLAS
    dispatches through Grand Central Dispatch threads that do not
    survive ``fork()``, so a worker forked off the long-lived main
    process silently dies once the data is large enough to trigger
    multi-threaded BLAS (``BrokenProcessPool``, no traceback). This is
    why large datasets crash a forking pool. ``threadpoolctl`` cannot
    help: it does not support Accelerate, so capping BLAS threads is a
    no-op when numpy is built against Accelerate (the Apple Silicon
    default).

    Use ``start_method="spawn"`` for work that touches the GPU. Apple's
    Metal compiler service cannot be reached from any process produced
    by ``fork()``, and forkserver children are forked, so GPU work
    (e.g. torch on the MPS backend) only runs from a freshly ``exec``-ed
    spawn worker. ``start_method="forkserver"`` is a good choice for
    CPU- and BLAS-only work: it forks cheap copy-on-write children from
    a clean intermediate process, giving faster startup and lower
    memory than spawn while still sidestepping the fork-BLAS crash.

    Raise ``RuntimeError`` on the first call if the driver script lacks
    an ``if __name__ == "__main__":`` guard, since spawn and forkserver
    both re-import ``__main__`` and would otherwise recurse.

    Parameters
    ----------
    max_workers: int
        Number of worker processes.
    start_method: str
        Multiprocessing start method, ``"spawn"`` (default) or
        ``"forkserver"``.

    Returns
    -------
    concurrent.futures.ProcessPoolExecutor
        Executor using the requested multiprocessing context.
    """
    require_main_guard()
    mp_ctx = multiprocessing.get_context(start_method)
    return concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_ctx,
    )


def make_forkserver_executor(max_workers):
    """
    Build a forkserver ProcessPoolExecutor.

    Thin wrapper over ``make_process_executor`` for CPU- and BLAS-only
    callers (e.g. wela's Gaussian-process fits) that want forkserver's
    fast, low-memory worker startup.

    Parameters
    ----------
    max_workers: int
        Number of worker processes.

    Returns
    -------
    concurrent.futures.ProcessPoolExecutor
        Executor using a forkserver multiprocessing context.
    """
    return make_process_executor(max_workers, start_method="forkserver")
