"""Bloodstream breakpoints by species, drug, standard, and year. See configs/breakpoints/README.md."""

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

FILE_NAME = re.compile(r"^(clsi|eucast)_(\d{4})\.csv$")


class BreakpointTable:
    """Looks up the S and R breakpoints in force for a result's standard and year."""

    def __init__(self, table: pd.DataFrame) -> None:
        self.table = table

    @classmethod
    def from_directory(cls, directory: Path) -> "BreakpointTable":
        """Load every `<standard>_<effective_year>.csv` file in the directory."""
        frames = []
        for path in sorted(directory.glob("*.csv")):
            match = FILE_NAME.match(path.name)
            if match is None:
                logger.warning("Skipping breakpoint file with unexpected name: %s", path)
                continue
            frame = pd.read_csv(path)
            frame["standard"] = match.group(1).upper()
            frame["effective_year"] = int(match.group(2))
            frames.append(frame)
        table = pd.concat(frames, ignore_index=True)
        logger.info("Loaded breakpoints: files=%s rows=%s", len(frames), len(table))
        return cls(table)

    def lookup(self, species: str, drug: str, standard: str, year: int | None) -> tuple[float, float] | None:
        """Return (s_breakpoint, r_breakpoint) or None.

        Without a year, a value is returned only when every version agrees, so nothing is guessed.
        """
        rows = self.table[
            (self.table["species"] == species) & (self.table["drug"] == drug) & (self.table["standard"] == standard)
        ]
        if year is None:
            # Agreement only means something if the rows reach back to the oldest file of this
            # standard; otherwise an older, missing version could have had a different value.
            oldest_year = self.table.loc[self.table["standard"] == standard, "effective_year"].min()
            if rows.empty or rows["effective_year"].min() > oldest_year:
                return None
            distinct = rows[["s_breakpoint", "r_breakpoint"]].drop_duplicates()
            if len(distinct) != 1:
                return None
            return float(distinct.iloc[0]["s_breakpoint"]), float(distinct.iloc[0]["r_breakpoint"])
        in_force = rows[rows["effective_year"] <= year]
        if in_force.empty:
            return None
        latest = in_force.sort_values("effective_year").iloc[-1]
        return float(latest["s_breakpoint"]), float(latest["r_breakpoint"])
