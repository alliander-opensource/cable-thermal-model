# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from enum import StrEnum


class BondingType(StrEnum):
    """Bonding type of a cable circuit."""

    OneSided = "one_sided"
    TwoSided = "two_sided"
    CrossBonding = "cross_bonding"
    NoBonding = "no_bonding"


class CircuitType(StrEnum):
    """Circuit type of a cable configuration."""

    Trefoil = "trefoil"
    Linear = "linear"
    Single = "single"
    LinearVertical = "linear_vertical"


class CircuitYReference(StrEnum):
    """Y-axis reference position for circuit placement."""

    Top = "top"
    Center = "center"
    Bottom = "bottom"
