# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

import logging

from pandera import Field
from pydantic import BaseModel

from cable_thermal_model.cable.cable_circuit import CableKey

logger = logging.getLogger(__name__)

MeasurementPointKey = tuple[str, str, str]

MEASUREMENT_POINT_KEY_PREFIX = "measurement_point"


class MeasurementPointInputSchema(BaseModel):
    """Input schema for constructing a measurement point.

    Attributes:
        x: Horizontal coordinate in meters.
        y: Vertical coordinate in meters. Must be below ground level.
        distances_to_cables: Distance from this point to each cable.
        distances_to_mirror_cables: Distance from this point to each mirror cable.
    """

    x: float = Field(description="The x-coordinate of the measurement point in meters.")
    y: float = Field(description="The y-coordinate of the measurement point in meters.", lt=0.0)
    distances_to_cables: dict[CableKey, float] = Field(
        description=(
            "A dictionary mapping cable identifiers to their respective distances from the measurement point in meters."
        ),
    )
    distances_to_mirror_cables: dict[CableKey, float] = Field(
        description=(
            "A dictionary mapping mirror cable identifiers to their respective distances "
            "from the measurement point in meters."
        )
    )


class MeasurementPoint:
    """Represents a single measurement point in the environment.

    Attributes:
        key: Unique 3-level key used in result DataFrame columns.
        distances_to_cables: Distance from this point to each cable.
        distances_to_mirror_cables: Distance from this point to each mirror cable.
    """

    def __init__(self, input_data: MeasurementPointInputSchema):
        """Initialize a measurement point from validated input data.

        Args:
            input_data: Validated measurement-point input values.
        """
        self.key: MeasurementPointKey = MeasurementPoint.create_key(x=input_data.x, y=input_data.y)
        self.distances_to_cables = input_data.distances_to_cables
        self.distances_to_mirror_cables = input_data.distances_to_mirror_cables

    @staticmethod
    def create_key(x: float, y: float) -> MeasurementPointKey:
        """Create a stable output key for a measurement point.

        Args:
            x: The x-coordinate of the measurement point in meters.
            y: The y-coordinate of the measurement point in meters.

        Returns:
            A 3-level tuple key in the format
            ``("measurement_point", "x=<value>m", "y=<value>m")``.
        """
        return (MEASUREMENT_POINT_KEY_PREFIX, f"x={x:.3f}m", f"y={y:.3f}m")


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

    def add_measurement_point(self, input_data: MeasurementPointInputSchema) -> MeasurementPointKey:
        """Add a measurement point to the registry.

        Args:
            input_data: Validated input data for the measurement point.

        Returns:
            The key of the added or already-existing measurement point.
        """
        measurement_point = MeasurementPoint(input_data=input_data)
        if measurement_point.key in self.measurement_point_keys:
            logger.warning(
                f"Measurement point at x={input_data.x:.3f}m, y={input_data.y:.3f}m already exists. Skipping addition."
            )
        else:
            self.points.add(measurement_point)

        return measurement_point.key
