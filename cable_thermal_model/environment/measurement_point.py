# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

import logging

from pydantic import BaseModel, Field, computed_field

from cable_thermal_model.cable.cable_circuit import CableKey

logger = logging.getLogger(__name__)

MeasurementPointKey = tuple[str, str, str]

MEASUREMENT_POINT_KEY_PREFIX = "measurement_point"


class MeasurementPoint(BaseModel):
    """Represents a single measurement point in the environment.

    Attributes:
        x: X-coordinate of the measurement point in meters.
        y: Y-coordinate of the measurement point in meters (must be below 0).
        distances_to_cables: Mapping from cable keys to distances in meters.
        distances_to_mirror_cables: Mapping from mirror cable keys to distances in meters.
        ndigits: Number of decimal places used when generating the key.
        key: Computed 3-level key used in result DataFrame columns.
    """

    x: float = Field(description="The x-coordinate of the measurement point in meters.")
    y: float = Field(description="The y-coordinate of the measurement point in meters.", lt=0.0)
    distances_to_cables: dict[CableKey, float] = Field(
        description=(
            "A dictionary mapping cable identifiers to their respective distances from the measurement point in meters."
        ),
        default_factory=dict,
    )
    distances_to_mirror_cables: dict[CableKey, float] = Field(
        description=(
            "A dictionary mapping mirror cable identifiers to their respective distances "
            "from the measurement point in meters."
        ),
        default_factory=dict,
    )
    ndigits: int = Field(
        default=3,
        description="Number of decimal places to round the coordinates for key generation.",
    )

    @computed_field  # type: ignore[misc]
    @property
    def key(self) -> MeasurementPointKey:
        """Create a stable output key for a measurement point.

        Returns:
            A 3-level tuple key in the format
            ``("measurement_point", "x=<value>m", "y=<value>m")``.
        """
        return (
            MEASUREMENT_POINT_KEY_PREFIX,
            f"x={self.x:.{self.ndigits}f}m",
            f"y={self.y:.{self.ndigits}f}m",
        )

    def __hash__(self) -> int:
        """Provide a stable hash so instances can be stored in sets.

        Hashing by the computed key keeps behavior aligned with how
        measurement points are deduplicated in the registry.
        """
        return hash(self.key)


class MeasurementPointRegistry:
    """Stores and deduplicates measurement points.

    Attributes:
        points: Set of unique measurement points.
    """

    def __init__(self) -> None:
        """Initialize an empty measurement-point registry."""
        self.points: set[MeasurementPoint] = set()

        return

    @property
    def measurement_point_keys(self) -> set[MeasurementPointKey]:
        """Return all registered measurement-point keys.

        Returns:
            The set of measurement-point keys currently in the registry.
        """
        return {mp.key for mp in self.points}

    def add_measurement_point(self, x: float, y: float, ndigits: int = 3) -> MeasurementPointKey:
        """Add a measurement point to the registry.

        Args:
            x: x-coordinate of the measurement point.
            y: y-coordinate of the measurement point.
            ndigits: Number of decimal places to round the coordinates for key generation.

        Returns:
            The key of the added or already-existing measurement point.
        """
        measurement_point = MeasurementPoint(x=x, y=y, ndigits=ndigits)
        if measurement_point.key in self.measurement_point_keys:
            _, x_str, y_str = measurement_point.key
            logger.warning(f"Measurement point at {x_str}, {y_str} already exists. Skipping addition.")
        else:
            self.points.add(measurement_point)

        return measurement_point.key
