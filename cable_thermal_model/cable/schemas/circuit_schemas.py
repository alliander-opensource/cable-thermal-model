# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, computed_field

from cable_thermal_model.cable.cable_builder import CableT
from cable_thermal_model.cable.enums.circuit_enums import BondingType, CircuitType, CircuitYReference
from cable_thermal_model.cable.schemas.cable_input_schemas import CableConstructionalInputSchema
from cable_thermal_model.cable.schemas.pipe_schemas import PipeInputSchema
from cable_thermal_model.model.cables.fd_cable import FDCable, FDCableInAir, FDCableTrefoilCircuitInSinglePipe


class BaseCircuitConfiguration(BaseModel):
    """Base schema for a cable configuration within a circuit."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    circuit_type: CircuitType = Field(description="Configuration of the cable(s)")
    dist: float | None = Field(default=None)
    length: float = Field(description="Length of the circuit in this configuration in meters", ge=0.0)


BaseCircuitConfigurationT = TypeVar("BaseCircuitConfigurationT", bound=BaseCircuitConfiguration)


class CircuitConfiguration(BaseCircuitConfiguration):
    """Circuit configuration that holds a pre-built FDCable instance."""

    cable: FDCable = Field(description="Cable object to use in this configuration.")


class CircuitConfigurationCableNotBuild(BaseCircuitConfiguration):
    """Circuit configuration where the cable is not pre-constructed."""

    pipe: PipeInputSchema | None = Field(default=None)

    @computed_field  # type: ignore[misc]
    @property
    def fd_cable_class(self) -> type[FDCable]:
        """Determine FDCable implementation for this configuration.

        Returns:
            type[FDCable]: `FDCableTrefoilCircuitInSinglePipe` when a trefoil
                single-pipe configuration is requested, otherwise `FDCable`.

        """
        if self.pipe is not None and self.pipe.trefoil_circuit_in_single_pipe:
            return FDCableTrefoilCircuitInSinglePipe

        return FDCable


class CircuitConfigurationFromCableId(CircuitConfigurationCableNotBuild):
    """Circuit configuration where the cable is identified by a string cable ID."""

    cable_id: str = Field(
        description="Cable id to use in this configuration. The cable id should be present in the cable source file."
    )


class CircuitConfigurationFromCableConstructionalInputSchema(CircuitConfigurationCableNotBuild):
    """Circuit configuration where the cable is built from a constructional input schema."""

    cable_constructional_information: CableConstructionalInputSchema = Field(
        description="Cable constructional input schema to use in this configuration."
    )


class BaseCircuitInputSchema(BaseModel, Generic[BaseCircuitConfigurationT]):
    """Base input schema shared by all circuit types."""

    # Identifier for the circuit
    circuit_name: str = Field(..., description="Name of the circuit")

    # Remaining parameters
    circuit_type: CircuitType | None = Field(default=None, description="Type of the circuit")
    dist: float | None = Field(default=None, description="Distance between the cables in the circuit in meters")
    pipe: PipeInputSchema | None = Field(default=None, description="Pipe information for the circuit")
    bonding_type: BondingType | None = Field(default=None, description="Bonding type of the circuit")
    multiple_configurations: list[BaseCircuitConfigurationT] = Field(
        default=[], description="Specifies different configurations in the connection"
    )


class Cable(BaseModel, Generic[CableT]):
    """Schema carrying a pre-built cable instance."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    cable: CableT = Field(..., description="FDCable instance to use in the circuit")


class CableId(BaseModel):
    """Schema identifying a cable by its string ID and source file path."""

    # Parameters that specify which cable to use and where to get the cable information from
    cable_id: str = Field(..., description="Identifier of the cable type to use in the circuit")
    cable_source_file_path: Path = Field(
        default=Path("data/example_cables.csv"), description="Source file to use for the cable in the circuit"
    )


class CircuitInSoilProperties(BaseModel):
    """Properties for a circuit buried in soil."""

    x: float = Field(..., description="Horizontal location of the center of the circuit in meters")
    y: float = Field(..., description="Depth of the center of the circuit in meters")
    y_ref: CircuitYReference = Field(
        default=CircuitYReference.Center, description="Reference of the circuit y position"
    )


class CircuitInAirProperties(BaseModel):
    """Properties for a circuit in air."""

    clipped_to_wall: bool = Field(default=False, description="Indicator if the circuit is clipped to a wall")


class CircuitFromCableInputSchema(BaseCircuitInputSchema[CircuitConfiguration], Cable[CableT], Generic[CableT]):
    """Input schema for the `add_circuit_from_cable` method of the StaticEnvironment class."""


class CircuitFromCableConstructionalInputSchema(
    BaseCircuitInputSchema[CircuitConfigurationFromCableConstructionalInputSchema]
):
    """Input schema used in the StaticEnvironment class.

    Input schema for the `add_circuit_from_cable_constructional_information` method.
    """

    cable_constructional_information: CableConstructionalInputSchema = Field(
        description="Cable constructional input schema to build the cable for this circuit."
    )


class CircuitFromCableIdInputSchema(BaseCircuitInputSchema[CircuitConfigurationFromCableId], CableId):
    """Input schema for the `add_circuit_from_cable_id` method of the StaticEnvironment class."""


class CircuitInSoilFromCableInputSchema(CircuitFromCableInputSchema[FDCable], CircuitInSoilProperties):
    """Input schema for the `add_circuit_from_cable` method of the StaticEnvironmentSoil class."""


class CircuitInSoilFromCableConstructionalInputSchema(
    CircuitFromCableConstructionalInputSchema, CircuitInSoilProperties
):
    """Input schema used in the StaticEnvironmentSoil class.

    Input schema for the `add_circuit_from_cable_constructional_information` method.
    """


class CircuitInSoilFromCableIdInputSchema(CircuitFromCableIdInputSchema, CircuitInSoilProperties):
    """Input schema for the `add_circuit_from_cable_id` method of the StaticEnvironmentSoil class."""


class CircuitInAirFromCableInputSchema(CircuitFromCableInputSchema[FDCableInAir], CircuitInAirProperties):
    """Input schema for the `add_circuit_from_cable` method of the StaticEnvironmentAir class."""


class CircuitInAirFromCableConstructionalInputSchema(CircuitFromCableConstructionalInputSchema, CircuitInAirProperties):
    """Input schema used in the StaticEnvironmentAir class.

    Input schema for the `add_circuit_from_cable_constructional_information` method.
    """


class CircuitInAirFromCableIdInputSchema(CircuitFromCableIdInputSchema, CircuitInAirProperties):
    """Input schema for the `add_circuit_from_cable_id` method of the StaticEnvironmentAir class."""
