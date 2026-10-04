"""Order-preserving process pool for per-genome work (sketching, k-mer passes, QC).

Every per-genome step of the Python stages (QC sketches, lineage sketches, the
k-mer passes of the unitig build, unitig queries, pairwise sketch distances) is
embarrassingly parallel. :func:`ordered_map` runs such a step in a pool of
worker processes and yields the results **in input order**, so a stage's output
is byte-identical for any ``threads`` value; only wall time changes.

Design choices
--------------
* Processes, not threads: the work is numpy-heavy but also parses FASTA in
  Python, and measured thread pools top out at ~2.5x. Processes scale with cores.
* The platform's default start method (``fork`` on Linux with Python 3.11,
  ``spawn`` on macOS). Workers only run numpy code on their inputs. Under
  ``spawn``/``forkserver`` a *script* that calls a stage function must guard its
  entry point with ``if __name__ == "__main__":`` (the ``python -m genome2mic``
  CLI already does).
* Bounded in-flight window (``max_pending``): a slow consumer never lets finished
  results (e.g. 40 MB k-mer arrays) pile up in memory.
* Small inputs (fewer than :data:`MIN_PARALLEL_ITEMS` items) and ``threads=1``
  run in the calling process: no start-up cost, and test monkeypatches apply.

Large read-only arrays (k-mer sets, sketch ranks) are handed to workers as
memory-mapped ``.npy`` files via :func:`share_arrays` / :func:`load_shared`, so N
workers share one copy through the page cache instead of N pickled copies.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import tempfile
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "MIN_PARALLEL_ITEMS",
    "resolve_threads",
    "worker_count",
    "ordered_map",
    "share_arrays",
    "load_shared",
]

T = TypeVar("T")
R = TypeVar("R")

MIN_PARALLEL_ITEMS: int = 32
"""Below this many items a step runs in the calling process (pool start-up costs ~1 s)."""

_START_METHOD: str | None = None
"""``None`` = the platform default (see module docstring)."""


def resolve_threads(threads: int | None) -> int:
    """Number of worker processes for ``threads``: ``None`` -> all usable cores.

    Raises:
        ValueError: if ``threads`` is given and smaller than 1.
    """
    if threads is None:
        try:
            available = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
        except AttributeError:  # macOS / Windows
            available = os.cpu_count() or 1
        return max(1, int(available))
    value = int(threads)
    if value < 1:
        raise ValueError(f"threads must be >= 1 (or None for all cores); got {threads!r}")
    return value


def worker_count(threads: int | None, n_items: int, *, min_items: int | None = None) -> int:
    """Processes to use for ``n_items`` tasks; ``1`` means "run in this process".

    Args:
        threads: Requested worker count (``None`` = all cores).
        n_items: Number of tasks.
        min_items: Inputs smaller than this run in-process (default
            :data:`MIN_PARALLEL_ITEMS`).
    """
    floor = MIN_PARALLEL_ITEMS if min_items is None else int(min_items)
    requested = resolve_threads(threads)
    if requested <= 1 or n_items < max(2, floor):
        return 1
    return max(1, min(requested, int(n_items)))


def _call_chunk(fn: Callable[[Any], Any], chunk: Sequence[Any]) -> list[Any]:
    """Worker entry point: apply ``fn`` to every item of one chunk."""
    return [fn(item) for item in chunk]


def _chunks(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def ordered_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    workers: int,
    initializer: Callable[..., None] | None = None,
    initargs: tuple[Any, ...] = (),
    chunksize: int = 1,
    max_pending: int | None = None,
) -> Iterator[R]:
    """Yield ``fn(item)`` for every item, in input order, using ``workers`` processes.

    Args:
        fn: Top-level (picklable) function of one argument.
        items: Task arguments (picklable). Materialised into a list.
        workers: From :func:`worker_count`; ``<= 1`` runs everything in this
            process (``initializer`` is then called here once).
        initializer, initargs: Run once per worker process before any task
            (typically :func:`load_shared` of memory-mapped arrays).
        chunksize: Items per submitted task (amortises IPC for tiny tasks).
        max_pending: Maximum chunks in flight (default ``2 x workers``). Bounds
            the memory held by finished-but-unconsumed results.

    If the worker processes cannot start (e.g. a script without the
    ``__main__`` guard under ``spawn``) or the pool breaks mid-run (a worker was
    killed, e.g. out of memory), the remaining items are computed in this
    process with a WARNING; the output is the same, only slower.

    Raises:
        Whatever ``fn`` raises, re-raised in the parent; pending work is cancelled.
    """
    if chunksize < 1:
        raise ValueError("chunksize must be >= 1")
    work = list(items)
    if workers <= 1 or len(work) <= 1:
        yield from _serial(fn, work, 0, initializer, initargs)
        return

    window = max(1, int(max_pending) if max_pending is not None else 2 * workers)
    logger.debug("ordered_map: %d item(s) on %d worker process(es), chunksize %d", len(work), workers, chunksize)
    executor: ProcessPoolExecutor | None = None
    n_yielded = 0
    try:
        try:
            executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp.get_context(_START_METHOD),
                initializer=initializer,
                initargs=initargs,
            )
            executor.submit(_probe).result()  # fail fast if workers cannot start
        except (BrokenProcessPool, OSError) as exc:
            _warn_fallback(exc, 0, len(work))
            yield from _serial(fn, work, 0, initializer, initargs)
            return

        pending: deque[Future[list[R]]] = deque()
        chunks = _chunks(work, chunksize)
        try:
            for chunk in chunks:
                pending.append(executor.submit(_call_chunk, fn, chunk))
                if len(pending) >= window:
                    break
            while pending:
                results = pending.popleft().result()
                nxt = next(chunks, None)
                if nxt is not None:
                    pending.append(executor.submit(_call_chunk, fn, nxt))
                for result in results:
                    yield result
                    n_yielded += 1
        except BrokenProcessPool as exc:
            executor.shutdown(wait=True, cancel_futures=True)
            executor = None
            _warn_fallback(exc, n_yielded, len(work))
            yield from _serial(fn, work, n_yielded, initializer, initargs)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)


def _probe() -> None:
    """No-op task: proves a worker process started (and ran the initializer)."""


def _serial(
    fn: Callable[[T], R],
    work: Sequence[T],
    start: int,
    initializer: Callable[..., None] | None,
    initargs: tuple[Any, ...],
) -> Iterator[R]:
    """``fn`` over ``work[start:]`` in this process (``initializer`` run here first)."""
    if initializer is not None:
        initializer(*initargs)
    for item in work[start:]:
        yield fn(item)


def _warn_fallback(exc: BaseException, done: int, total: int) -> None:
    logger.warning(
        "worker processes failed (%s: %s); computing the remaining %d of %d item(s) in this process. "
        "If this is a script, guard its entry point with `if __name__ == \"__main__\":`.",
        type(exc).__name__, exc, total - done, total,
    )


@contextmanager
def share_arrays(arrays: Mapping[str, np.ndarray], *, enabled: bool = True) -> Iterator[dict[str, Any]]:
    """Context manager: write ``arrays`` to temporary ``.npy`` files for worker processes.

    Yields a spec for :func:`load_shared`: ``{name: path}`` when ``enabled``, else
    the arrays themselves (in-process runs need no copy). The files are deleted on
    exit. The directory honours ``TMPDIR``; on a VM point it at the data disk when
    the k-mer set is large.
    """
    if not enabled:
        yield dict(arrays)
        return
    with tempfile.TemporaryDirectory(prefix="g2m_shared_") as tmp:
        spec: dict[str, Any] = {}
        for name, array in arrays.items():
            target = Path(tmp) / f"{name}.npy"
            np.save(target, np.ascontiguousarray(array), allow_pickle=False)
            spec[name] = str(target)
        yield spec


def load_shared(spec: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Resolve a :func:`share_arrays` spec: memory-map paths, pass arrays through."""
    out: dict[str, np.ndarray] = {}
    for name, value in spec.items():
        if isinstance(value, (str, os.PathLike)):
            out[name] = np.load(value, mmap_mode="r", allow_pickle=False)
        else:
            out[name] = value
    return out
