# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0


from cable_thermal_model.cable.cable_circuit import (
    CircuitBuilder,
    CircuitType,
)
from cable_thermal_model.cable.schemas.circuit_schemas import (
    BaseCircuitInputSchema,
    CircuitInAirFromCableConstructionalInputSchema,
    CircuitInAirFromCableIdInputSchema,
    CircuitInAirFromCableInputSchema,
)
from cable_thermal_model.environment.static_env import StaticEnv
from cable_thermal_model.model.cables.fd_cable import FDCableInAir, FDCableTrefoilCircuitInSinglePipeInAir


class StaticEnvAir(
    StaticEnv[
        FDCableInAir,
        CircuitInAirFromCableInputSchema,
        CircuitInAirFromCableConstructionalInputSchema,
        CircuitInAirFromCableIdInputSchema,
    ]
):
    """Class that builds a static environment for circuits in air."""

    @property
    def _circuit_from_cable_input_schema_cls(self) -> type[CircuitInAirFromCableInputSchema]:
        return CircuitInAirFromCableInputSchema

    def add_circuit_from_cable(
        self,
        circuit_input: CircuitInAirFromCableInputSchema,
    ):
        """Add the circuit to the environment based on a cable instance.

        Convection parameters are added to the circuit.

        Args:
            circuit_input: CircuitInputSchema containing the input parameters
                for the circuit in air, including a Cable instance.

        References:
            - NEN-IEC 60287-2-1 (2023) - [table 3]

        """
        if self.circuits:
            raise ValueError(
                "Environment already contains circuit(s). Cannot add multiple circuits to an air environment"
            )

        self.set_environment_convection_parameters(
            circuit_type=circuit_input.circuit_type,
            dist=circuit_input.dist,
            cable=circuit_input.cable,
            clipped_to_wall=circuit_input.clipped_to_wall,
        )

        return super().add_circuit_from_cable(circuit_input)

    def _determine_cable_class_from_circuit_input(self, circuit_input: BaseCircuitInputSchema) -> type[FDCableInAir]:
        return (
            FDCableTrefoilCircuitInSinglePipeInAir
            if CircuitBuilder._is_trefoil_circuit_in_single_pipe(circuit_input.circuit_type, circuit_input.pipe)
            else FDCableInAir
        )

    def set_environment_convection_parameters(
        self,
        circuit_type: CircuitType | None,
        dist: float | None,
        cable: FDCableInAir,
        clipped_to_wall: bool,
    ):
        """Adds convection parameters to the cables.

        Args:
            circuit_type: Type of circuit, one of 'single', 'trefoil', 'linear'
            dist: Distance between cables, relevant for 'linear' circuits
            cable: FDCable instance
            clipped_to_wall: Indicator if the circuit is clipped to a wall

        References:
            - NEN-IEC 60287-2-1 (2023) - [table 3]

        """
        Z, E, Cg = self._get_convection_parameters(circuit_type, dist, cable, clipped_to_wall)

        cable.set_convection_parameters(Z=Z, E=E, Cg=Cg)

    def _get_convection_parameters(
        self,
        circuit_type: CircuitType | None,
        dist: float | None,
        cable: FDCableInAir,
        clipped_to_wall: bool,
    ):
        if clipped_to_wall:
            return self._get_clipped_to_wall_params(circuit_type)

        match circuit_type:
            case CircuitType.Linear if dist is None or dist < cable.layer_metrics.outer_radius * 3.5:
                return 0.62, 1.95, 0.25
            case CircuitType.Single | CircuitType.Linear:
                return 0.21, 3.94, 0.60
            case CircuitType.Trefoil:
                return 0.96, 1.25, 0.20
            case CircuitType.LinearVertical:
                if dist is None or dist < cable.layer_metrics.outer_radius * 4:
                    return 1.61, 0.42, 0.20
                else:
                    return 1.31, 2.00, 0.20
            case _:
                raise ValueError(f"No convection parameters set for {circuit_type}")

    @staticmethod
    def _get_clipped_to_wall_params(circuit_type: CircuitType | None) -> tuple[float, float, float]:
        match circuit_type:
            case CircuitType.Single:
                return 1.69, 0.63, 0.25
            case CircuitType.Trefoil:
                return 0.94, 0.79, 0.20
            case _:
                raise ValueError(f"No convection parameters available for {circuit_type} when clipped to wall")
