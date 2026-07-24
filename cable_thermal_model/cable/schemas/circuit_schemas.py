# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, computed_field

from cable_thermal_model.cable.cable_builder import CableBuilder, CableT
from cable_thermal_model.cable.enums.circuit_enums import BondingType, CircuitType, CircuitYReference
from cable_thermal_model.cable.schemas.cable_input_schemas import CableConstructionalInputSchema
from cable_thermal_model.cable.schemas.pipe_schemas import PipeInputSchema
from cable_thermal_model.model.cables.cable import (
    Cable,
)
from cable_thermal_model.model.cables.cable_air import CableAir
from cable_thermal_model.model.cables.cable_trefoil_circuit_single_pipe import CableTrefoilCircuitSinglePipe


class BaseCircuitConfiguration(BaseModel):
    """Base schema for a cable configuration within a circuit."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    circuit_type: CircuitType = Field(description="Configuration of the cable(s)")
    dist: float | None = Field(default=None)
    length: float = Field(description="Length of the circuit in this configuration in meters", ge=0.0)


BaseCircuitConfigurationT = TypeVar("BaseCircuitConfigurationT", bound=BaseCircuitConfiguration)


class CircuitConfiguration(BaseCircuitConfiguration):
    """Circuit configuration that holds a pre-built Cable instance."""

    cable: Cable = Field(description="Cable object to use in this configuration.")


class CircuitConfigurationCableNotBuild(BaseCircuitConfiguration):
    """Circuit configuration where the cable is not pre-constructed."""

    pipe: PipeInputSchema | None = Field(default=None)

    @computed_field  # type: ignore[misc]
    @property
    def cable_class(self) -> type[Cable]:
        """Determine Cable implementation for this configuration.

        Returns:
            type[Cable]: `CableTrefoilCircuitSinglePipeInSoil` when a trefoil
                single-pipe configuration is requested, otherwise `CableSoil`.

        """
        if self.pipe is not None and self.pipe.trefoil_circuit_in_single_pipe:
            return CableTrefoilCircuitSinglePipe

        return Cable

    def _compute_circuit_configuration_from_cable(self, cable: Cable) -> CircuitConfiguration:
        """Compute a CircuitConfiguration from a pre-built Cable instance.

        Args:
            cable (Cable): Pre-built Cable instance to use in the configuration.

        Returns:
            CircuitConfiguration: Circuit configuration with the provided Cable instance.

        """
        return CircuitConfiguration(
            circuit_type=self.circuit_type,
            dist=self.dist,
            length=self.length,
            cable=cable,
        )


class CircuitConfigurationFromCableId(CircuitConfigurationCableNotBuild):
    """Circuit configuration where the cable is identified by a string cable ID."""

    cable_id: str = Field(
        description="Cable id to use in this configuration. The cable id should be present in the cable source file."
    )

    def _compute_circuit_configuration(self, cable_source_file_path: Path) -> CircuitConfiguration:
        """Compute a CircuitConfiguration from a cable ID and source file path.

        Args:
            cable_source_file_path (Path): Path to the source file containing the cable information.

        Returns:
            CircuitConfiguration: Circuit configuration with the Cable instance built.

        """
        cable = CableBuilder.build_cable_from_cable_id(
            cable_id=self.cable_id,
            cable_class=self.cable_class,
            pipe=self.pipe,
            cable_source_file_path=cable_source_file_path,
        )
        return self._compute_circuit_configuration_from_cable(cable)


class CircuitConfigurationFromCableConstructionalInputSchema(CircuitConfigurationCableNotBuild):
    """Circuit configuration where the cable is built from a constructional input schema."""

    cable_constructional_information: CableConstructionalInputSchema = Field(
        description="Cable constructional input schema to use in this configuration."
    )

    def _compute_circuit_configuration(self) -> CircuitConfiguration:
        """Compute a CircuitConfiguration from a cable constructional input schema.

        Returns:
            CircuitConfiguration: Circuit configuration with the Cable instance built.

        """
        cable = CableBuilder.build_cable(
            cable_constructional_input=self.cable_constructional_information,
            cable_class=self.cable_class,
            pipe=self.pipe,
        )
        return self._compute_circuit_configuration_from_cable(cable)


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


class CableInput(BaseModel, Generic[CableT]):
    """Schema carrying a pre-built cable instance."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    cable: CableT = Field(..., description="Cable instance to use in the circuit")


class CableId(BaseModel):
    """Schema identifying a cable by its string ID and source file path."""

    # Parameters that specify which cable to use and where to get the cable information from
    cable_id: str = Field(..., description="Identifier of the cable type to use in the circuit")
    cable_source_file_path: Path = Field(
        default=Path(__file__).resolve().parents[3] / "data" / "example_cables.csv",
        description="Path to the source file containing the cable information",
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


class CircuitFromCableInputSchema(BaseCircuitInputSchema[CircuitConfiguration], CableInput[CableT], Generic[CableT]):
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

    def _build_cable(self, cable_class: type[CableT]) -> CableT:
        """Build a cable instance based on the constructional information and cable class.

        Args:
            cable_class (type[CableT]): The class of the cable to build.

        Returns:
            CableT: An instance of the cable class.

        """
        return CableBuilder.build_cable(
            cable_constructional_input=self.cable_constructional_information,
            cable_class=cable_class,
            pipe=self.pipe,
        )


class CircuitFromCableIdInputSchema(BaseCircuitInputSchema[CircuitConfigurationFromCableId], CableId):
    """Input schema for the `add_circuit_from_cable_id` method of the StaticEnvironment class."""

    def _build_cable(self, cable_class: type[CableT]) -> CableT:
        """Build a cable instance based on the cable ID and source file path.

        Args:
            cable_class (type[CableT]): The class of the cable to build.

        Returns:
            CableT: An instance of the cable class.

        """
        return CableBuilder.build_cable_from_cable_id(
            cable_id=self.cable_id,
            cable_class=cable_class,
            pipe=self.pipe,
            cable_source_file_path=self.cable_source_file_path,
        )


class CircuitInSoilFromCableInputSchema(CircuitFromCableInputSchema[Cable], CircuitInSoilProperties):
    """Input schema for the `add_circuit_from_cable` method of the StaticEnvironmentSoil class."""


class CircuitInSoilFromCableConstructionalInputSchema(
    CircuitFromCableConstructionalInputSchema, CircuitInSoilProperties
):
    """Input schema used in the StaticEnvironmentSoil class.

    Input schema for the `add_circuit_from_cable_constructional_information` method.
    """


class CircuitInSoilFromCableIdInputSchema(CircuitFromCableIdInputSchema, CircuitInSoilProperties):
    """Input schema for the `add_circuit_from_cable_id` method of the StaticEnvironmentSoil class."""


class CircuitInAirFromCableInputSchema(CircuitFromCableInputSchema[CableAir], CircuitInAirProperties):
    """Input schema for the `add_circuit_from_cable` method of the StaticEnvironmentAir class."""


class CircuitInAirFromCableConstructionalInputSchema(CircuitFromCableConstructionalInputSchema, CircuitInAirProperties):
    """Input schema used in the StaticEnvironmentAir class.

    Input schema for the `add_circuit_from_cable_constructional_information` method.
    """


class CircuitInAirFromCableIdInputSchema(CircuitFromCableIdInputSchema, CircuitInAirProperties):
    """Input schema for the `add_circuit_from_cable_id` method of the StaticEnvironmentAir class."""
