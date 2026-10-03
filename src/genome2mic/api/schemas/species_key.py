"""Species keys in scope."""

from enum import StrEnum


class SpeciesKey(StrEnum):
    """The 5-letter species keys from DATA_CONTRACT.md."""

    ECOLI = "ECOLI"
    KPNEU = "KPNEU"
    SAUR = "SAUR"
    PAER = "PAER"
    ABAU = "ABAU"
