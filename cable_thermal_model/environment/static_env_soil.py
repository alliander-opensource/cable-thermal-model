# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0


import pandas as pd

from cable_thermal_model.cable.cable_circuit import (
    CircuitBuilder,
    CircuitType,
)
from cable_thermal_model.cable.schemas.circuit_schemas import (
    BaseCircuitInputSchema,
    CircuitInSoilFromCableConstructionalInputSchema,
    CircuitInSoilFromCableIdInputSchema,
    CircuitInSoilFromCableInputSchema,
)
from cable_thermal_model.cable.schemas.pipe_schemas import PipeInputSchema
from cable_thermal_model.environment.measurement_point import MeasurementPointKey, MeasurementPointRegistry
from cable_thermal_model.environment.static_env import StaticEnv
from cable_thermal_model.model.cables.cable_soil import CableSoil
from cable_thermal_model.model.cables.cable_trefoil_circuit_single_pipe import CableTrefoilCircuitSinglePipeInSoil
from cable_thermal_model.model.cables.enum_classes_cable import PipeFillType
from data.settings import circuits_path


class StaticEnvSoil(
    StaticEnv[
        CableSoil,
        CircuitInSoilFromCableInputSchema,
        CircuitInSoilFromCableConstructionalInputSchema,
        CircuitInSoilFromCableIdInputSchema,
    ]
):
    """Class that builds a static environment for circuits in soil."""

    def __init__(self):
        """Initialize the StaticEnvSoil instance."""
        super().__init__()
        self._measurement_point_registry = MeasurementPointRegistry()

    def _determine_cable_class_from_circuit_input(self, circuit_input: BaseCircuitInputSchema) -> type[CableSoil]:
        return (
            CableTrefoilCircuitSinglePipeInSoil
            if CircuitBuilder._is_trefoil_circuit_in_single_pipe(circuit_input.circuit_type, circuit_input.pipe)
            else CableSoil
        )

    @property
    def _circuit_from_cable_input_schema_cls(self) -> type[CircuitInSoilFromCableInputSchema]:
        return CircuitInSoilFromCableInputSchema

    @staticmethod
    def _convert_circuit_type(circuit_type_string: str) -> CircuitType:
        return CircuitType[circuit_type_string]

    @staticmethod
    def _convert_pipe(pipe: int):
        """Convert legacy integer pipe flag to `PipeInputSchema` or `None`.

        Args:
            pipe: Legacy integer-like flag where truthy means pipe present.

        Returns:
            PipeInputSchema | None: Air-filled default pipe for truthy input,
                otherwise `None`.

        """
        if bool(pipe):
            return PipeInputSchema(fill_type=PipeFillType.Air)
        else:
            return None

    def _from_file(self, file_name: str):
        """Read circuit data from file. Additional columns are stored as attributes of the CableCircuit instance.

        This function is used internally only to quickly set up an environment.

         NO CUSTOM FILE PATHS ARE PROCESSED
         Args:
            file_name: A string representation of one of the files in the circuits folder
        """
        circuits_ = pd.read_csv(circuits_path() / file_name, sep=";")
        circuits_["circuit_type"] = circuits_["circuit_type"].apply(self._convert_circuit_type)
        circuits_["pipe"] = circuits_["pipe"].apply(self._convert_pipe)
        for _, row in circuits_.iterrows():
            self.add_circuit_from_cable_id(CircuitInSoilFromCableIdInputSchema(**dict(row)))
        return self

    def add_measurement_point(self, x: float, y: float, ndigits: int = 3) -> MeasurementPointKey:
        """Add a measurement point to the environment.

        Args:
            x: x-coordinate of the measurement point.
            y: y-coordinate of the measurement point.

            ndigits: Number of decimal places to round the coordinates for key generation.

        Returns:
            MeasurementPointKey: The identifier of the added measurement point.

        """
        return self._measurement_point_registry.add_measurement_point(
            x=x,
            y=y,
            ndigits=ndigits,
        )
