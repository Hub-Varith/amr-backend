"""Tests for genome2mic.droplog.DropLog."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genome2mic.droplog import COLUMNS, DropLog, DropRecord


class TestRecording:
    def test_drop_records_and_logs_at_info(self, caplog: pytest.LogCaptureFixture) -> None:
        log = DropLog("ingest")
        with caplog.at_level(logging.INFO, logger="genome2mic.droplog"):
            log.drop("evidence != Laboratory Method", 12)
            log.drop("unknown drug", 3, detail="ceftobiprole, cefiderocol")
        assert log.records == (
            DropRecord("ingest", "evidence != Laboratory Method", 12, None),
            DropRecord("ingest", "unknown drug", 3, "ceftobiprole, cefiderocol"),
        )
        assert len(log) == 2 and log.total() == 15
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert any("[ingest] dropped 12 rows: evidence != Laboratory Method" in m for m in messages)
        assert any("dropped 3 rows: unknown drug (ceftobiprole, cefiderocol)" in m for m in messages)

    def test_zero_drops_are_recorded_to_prove_the_filter_ran(self) -> None:
        log = DropLog("qc")
        log.drop("too fragmented (n_contigs > 500)", 0)
        assert log.to_frame()["n_dropped"].tolist() == [0]

    def test_numpy_integer_counts_accepted(self) -> None:
        log = DropLog("qc")
        log.drop("size", np.int64(4))
        log.drop("mash", int(np.sum(np.array([True, False, True]))))
        assert log.total() == 6
        assert all(isinstance(r.n_dropped, int) for r in log.records)

    @pytest.mark.parametrize("bad", [-1, 1.5, "3", None, True])
    def test_invalid_counts_raise(self, bad: object) -> None:
        log = DropLog("qc")
        with pytest.raises(ValueError):
            log.drop("reason", bad)  # type: ignore[arg-type]

    def test_empty_reason_or_stage_raise(self) -> None:
        with pytest.raises(ValueError):
            DropLog("")
        with pytest.raises(ValueError):
            DropLog("x").drop("  ", 1)

    def test_extend_merges_records(self) -> None:
        a, b = DropLog("ingest"), DropLog("ingest")
        a.drop("r1", 1)
        b.drop("r2", 2)
        a.extend(b)
        assert [r.reason for r in a.records] == ["r1", "r2"] and a.total() == 3


class TestKeepWhere:
    def test_filters_and_counts_dropped_rows(self) -> None:
        frame = pd.DataFrame({"evidence": ["Laboratory Method", "Computational Prediction", "Laboratory Method"]})
        log = DropLog("ingest")
        kept = log.keep_where(frame, frame["evidence"] == "Laboratory Method", "evidence != Laboratory Method")
        assert kept["evidence"].tolist() == ["Laboratory Method", "Laboratory Method"]
        assert kept.index.tolist() == [0, 2]
        assert log.records[0].n_dropped == 1
        # the input frame is untouched (copy-on-write)
        assert len(frame) == 3

    def test_null_in_mask_counts_as_dropped(self) -> None:
        frame = pd.DataFrame({"x": [1, 2, 3]})
        mask = pd.Series([True, None, False], dtype="boolean")
        log = DropLog("s")
        kept = log.keep_where(frame, mask, "nullable mask")
        assert kept["x"].tolist() == [1] and log.total() == 2

    def test_numpy_mask(self) -> None:
        frame = pd.DataFrame({"x": [1, 2, 3]})
        log = DropLog("s")
        kept = log.keep_where(frame, np.array([False, True, True]), "numpy mask")
        assert kept["x"].tolist() == [2, 3] and log.total() == 1

    def test_misaligned_or_non_boolean_mask_raises(self) -> None:
        frame = pd.DataFrame({"x": [1, 2, 3]})
        log = DropLog("s")
        with pytest.raises(ValueError):
            log.keep_where(frame, np.array([True, False]), "short")
        with pytest.raises(ValueError):
            log.keep_where(frame, np.array([1, 0, 1]), "ints")


class TestFrameAndWrite:
    def test_to_frame_columns_dtypes_and_nulls(self) -> None:
        log = DropLog("ingest")
        log.drop("a", 1)
        log.drop("b", 2, detail="why")
        frame = log.to_frame()
        assert list(frame.columns) == list(COLUMNS) == ["stage", "reason", "n_dropped", "detail"]
        assert frame["n_dropped"].dtype == np.int64
        assert str(frame["stage"].dtype) == "str" and str(frame["detail"].dtype) == "str"
        assert frame["stage"].tolist() == ["ingest", "ingest"]
        assert pd.isna(frame["detail"].iloc[0]) and frame["detail"].iloc[1] == "why"
        # missing detail is null, never an empty string
        assert "" not in frame["detail"].dropna().tolist()

    def test_empty_log_frame_has_stable_schema(self) -> None:
        frame = DropLog("qc").to_frame()
        assert list(frame.columns) == list(COLUMNS)
        assert len(frame) == 0
        assert frame["n_dropped"].dtype == np.int64

    def test_write_creates_parent_dirs_and_round_trips(self, tmp_path: Path) -> None:
        log = DropLog("ingest")
        log.drop("evidence", 12)
        log.drop("unknown drug", 0, detail="none seen")
        target = tmp_path / "data" / "processed" / "drop_log_ingest.csv"
        log.write(target)
        assert target.exists()
        back = pd.read_csv(target)
        assert list(back.columns) == list(COLUMNS)
        assert back["n_dropped"].tolist() == [12, 0]
        assert back["reason"].tolist() == ["evidence", "unknown drug"]
        assert pd.isna(back["detail"].iloc[0]) and back["detail"].iloc[1] == "none seen"

    def test_repr(self) -> None:
        log = DropLog("splits")
        log.drop("x", 5)
        assert repr(log) == "DropLog(stage='splits', n_records=1, total_dropped=5)"
