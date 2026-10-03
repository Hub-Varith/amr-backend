"""The interval rule from DATA_CONTRACT.md stage 2, plus the agreed snapping and `<` handling."""

import math

import pytest

from genome2mic.ingest.mic_interval import MicIntervalConverter


@pytest.mark.parametrize(
    ("sign", "value", "expected"),
    [
        ("=", 8.0, (4.0, 8.0, "interval")),
        ("<=", 0.25, (0.0, 0.25, "left")),
        ("<", 0.5, (0.0, 0.5, "left")),
        (">", 32.0, (32.0, math.inf, "right")),
        (">=", 16.0, (8.0, math.inf, "right")),
    ],
)
def test_from_mic_follows_the_contract_table(sign: str, value: float, expected: tuple) -> None:
    assert MicIntervalConverter.from_mic(sign, value) == expected


@pytest.mark.parametrize(
    ("sir", "expected"),
    [
        ("S", (0.0, 1.0, "left")),
        ("R", (2.0, math.inf, "right")),
        ("I", (1.0, 2.0, "interval")),
    ],
)
def test_from_sir_uses_s_and_r_breakpoints(sir: str, expected: tuple) -> None:
    assert MicIntervalConverter.from_sir(sir, s_breakpoint=1.0, r_breakpoint=2.0) == expected


def test_from_sir_rejects_intermediate_when_there_is_no_intermediate_zone() -> None:
    assert MicIntervalConverter.from_sir("I", s_breakpoint=2.0, r_breakpoint=2.0) is None


@pytest.mark.parametrize(
    ("reported", "step"),
    [
        (0.06, 0.0625),
        (0.12, 0.125),
        (0.03, 0.03125),
        (0.015, 0.015625),
        (0.25, 0.25),
        (8.0, 8.0),
        (256.0, 256.0),
    ],
)
def test_snap_maps_rounded_panel_values_to_doubling_steps(reported: float, step: float) -> None:
    assert MicIntervalConverter.snap_to_doubling_step(reported) == step


@pytest.mark.parametrize("off_grid", [0.094, 0.19, 0.38, 0.75, 1.5, 3.0, 6.0, 12.0, 24.0])
def test_snap_rejects_values_between_steps(off_grid: float) -> None:
    assert MicIntervalConverter.snap_to_doubling_step(off_grid) is None


@pytest.mark.parametrize("not_positive", [0.0, -1.0])
def test_snap_rejects_non_positive_values(not_positive: float) -> None:
    assert MicIntervalConverter.snap_to_doubling_step(not_positive) is None


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("8", 8.0),
        ("0.06", 0.06),
        ("8/4", 8.0),
        ("0.5/9.5", 0.5),
        (" 16 ", 16.0),
    ],
)
def test_parse_value_takes_the_first_number(text: str, value: float) -> None:
    assert MicIntervalConverter.parse_value(text) == value


@pytest.mark.parametrize("text", ["", "abc", "R", "nan"])
def test_parse_value_rejects_non_numbers(text: str) -> None:
    assert MicIntervalConverter.parse_value(text) is None


@pytest.mark.parametrize(
    ("raw", "sign"),
    [("", "="), ("==", "="), ("=", "="), ("≤", "<="), ("=<", "<="), ("<", "<"), (">=", ">="), ("≥", ">=")],
)
def test_normalize_sign(raw: str, sign: str) -> None:
    assert MicIntervalConverter.normalize_sign(raw) == sign


def test_normalize_sign_rejects_unknown() -> None:
    assert MicIntervalConverter.normalize_sign("~") is None
