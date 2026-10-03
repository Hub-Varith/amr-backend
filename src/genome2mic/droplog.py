"""Row-drop accounting for every filter in the pipeline.

CLAUDE.md: *"Log dropped rows with reasons. Every filter emits a count. Silent drops
make dataset bugs invisible."* Each stage creates one :class:`DropLog`, calls
:meth:`DropLog.drop` (or :meth:`DropLog.keep_where`) for every filter -- including
filters that dropped nothing, so the log proves the filter ran -- and writes the
table to ``paths.drop_log(stage)``.

Example::

    log = DropLog("ingest")
    raw = log.keep_where(raw, raw["evidence"] == "Laboratory Method", "evidence != Laboratory Method")
    log.drop("unknown drug", n_unknown, detail="cefepime/tazo, ceftobiprole")
    log.write(paths.drop_log("ingest"))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

COLUMNS: tuple[str, ...] = ("stage", "reason", "n_dropped", "detail")


@dataclass(frozen=True)
class DropRecord:
    """One filter outcome."""

    stage: str
    reason: str
    n_dropped: int
    detail: str | None = None


class DropLog:
    """Collects (reason, count) pairs for one pipeline stage.

    Args:
        stage: Stage name, e.g. ``ingest``, ``qc``, ``splits``. Appears in every
            record and in the output file name.
    """

    def __init__(self, stage: str) -> None:
        if not stage or not str(stage).strip():
            raise ValueError("stage must be a non-empty string")
        self.stage = str(stage).strip()
        self._records: list[DropRecord] = []

    # ------------------------------------------------------------- recording
    def drop(self, reason: str, n: int, detail: str | None = None) -> None:
        """Record that ``n`` rows were dropped for ``reason`` and log it at INFO.

        ``n == 0`` is recorded too: it documents that the filter ran. Negative or
        non-integer counts raise ``ValueError``.
        """
        if not reason or not str(reason).strip():
            raise ValueError("reason must be a non-empty string")
        count = _as_count(n)
        text = None if detail is None else str(detail)
        self._records.append(DropRecord(self.stage, str(reason).strip(), count, text))
        if text:
            logger.info("[%s] dropped %d rows: %s (%s)", self.stage, count, reason, text)
        else:
            logger.info("[%s] dropped %d rows: %s", self.stage, count, reason)

    def keep_where(
        self,
        frame: pd.DataFrame,
        mask: pd.Series | np.ndarray,
        reason: str,
        detail: str | None = None,
    ) -> pd.DataFrame:
        """Return ``frame[mask]`` and record the number of rows that did not pass.

        ``mask`` must be boolean and aligned with ``frame`` (a Series with the same
        index or an array of the same length); nulls in the mask count as False.
        """
        keep = _as_bool_mask(frame, mask)
        dropped = int(len(frame) - int(keep.sum()))
        self.drop(reason, dropped, detail)
        return frame.loc[keep]

    def extend(self, other: "DropLog") -> None:
        """Append every record of another log (e.g. a sub-step) to this one."""
        self._records.extend(other._records)

    # ---------------------------------------------------------------- access
    @property
    def records(self) -> tuple[DropRecord, ...]:
        """All records in insertion order."""
        return tuple(self._records)

    def total(self) -> int:
        """Sum of all recorded drop counts."""
        return sum(r.n_dropped for r in self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"DropLog(stage={self.stage!r}, n_records={len(self)}, total_dropped={self.total()})"

    # ---------------------------------------------------------------- output
    def to_frame(self) -> pd.DataFrame:
        """Records as a table with columns ``stage, reason, n_dropped, detail``.

        ``detail`` is null (not ``""``) when absent. Dtypes are stable even when the
        log is empty: ``str``, ``str``, ``int64``, ``str``.
        """
        return pd.DataFrame(
            {
                "stage": pd.array([r.stage for r in self._records], dtype="str"),
                "reason": pd.array([r.reason for r in self._records], dtype="str"),
                "n_dropped": np.array([r.n_dropped for r in self._records], dtype="int64"),
                "detail": pd.array([r.detail for r in self._records], dtype="str"),
            },
            columns=list(COLUMNS),
        )

    def write(self, path: Path) -> None:
        """Write :meth:`to_frame` as CSV, creating parent directories."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(target, index=False)
        logger.info("[%s] wrote drop log with %d records (%d rows dropped) to %s",
                    self.stage, len(self), self.total(), target)


def _as_count(n: object) -> int:
    """Coerce a count to a non-negative Python int (accepts numpy integers)."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise ValueError(f"n must be an integer count, got {n!r}")
    count = int(n)
    if count < 0:
        raise ValueError(f"n must be >= 0, got {count}")
    return count


def _as_bool_mask(frame: pd.DataFrame, mask: pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(mask, pd.Series):
        aligned = mask.reindex(frame.index) if not mask.index.equals(frame.index) else mask
        values = aligned.to_numpy(dtype="object")
        keep = np.array([bool(v) if not pd.isna(v) else False for v in values], dtype=bool)
    else:
        arr = np.asarray(mask)
        if arr.dtype != bool:
            raise ValueError("mask must be boolean")
        keep = arr
    if len(keep) != len(frame):
        raise ValueError(f"mask length {len(keep)} != frame length {len(frame)}")
    return keep
