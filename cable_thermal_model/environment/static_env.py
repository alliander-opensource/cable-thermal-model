# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np

from cable_thermal_model.cable.cable_builder import CableBuilder, CableT
from cable_thermal_model.cable.cable_circuit import (
    BondingType,
    CableCircuit,
    CableKey,
    CircuitBuilder,
    CircuitType,
    PosCable,
)
from cable_thermal_model.cable.schemas.circuit_schemas import (
    BaseCircuitInputSchema,
    CircuitConfiguration,
    CircuitConfigurationFromCableConstructionalInputSchema,
    CircuitConfigurationFromCableId,
    CircuitFromCableConstructionalInputSchema,
    CircuitFromCableIdInputSchema,
    CircuitFromCableInputSchema,
)
from cable_thermal_model.model.cables.abstract_cable import (
    ELECTRIC_RESISTANCE_REFERENCE_TEMPERATURE,
    AbstractCable,
    WeightedScreenImpedance,
)
from cable_thermal_model.model.cables.enum_classes_cable import CableConductorCount
from cable_thermal_model.model.cables.fd_cable import (
    FDCableTrefoilCircuitInSinglePipe,
    FDCableTrefoilCircuitInSinglePipeInAir,
)
from cable_thermal_model.utils.str_utils import tab_lines

CircuitFromCableInputSchemaT = TypeVar("CircuitFromCableInputSchemaT", bound=CircuitFromCableInputSchema)
CircuitFromCableConstructionalInputSchemaT = TypeVar(
    "CircuitFromCableConstructionalInputSchemaT", bound=CircuitFromCableConstructionalInputSchema
)
CircuitFromCableIdInputSchemaT = TypeVar("CircuitFromCableIdInputSchemaT", bound=CircuitFromCableIdInputSchema)


class StaticEnv(
    ABC,
    Generic[
        CableT,
        CircuitFromCableInputSchemaT,
        CircuitFromCableConstructionalInputSchemaT,
        CircuitFromCableIdInputSchemaT,
    ],
):
    """Class that builds a static environment."""

    def __init__(self) -> None:
        """Initialize the static environment with empty circuit and cable containers."""
        self.circuits: dict[str, CableCircuit] = {}
        self.circuit_cable_indices: dict[str, list[int]] = {}
        self.cables: dict[CableKey, PosCable] = {}
        self.number_of_cables: int = 0

        self.crossing_cables: bool = False

        self.n_phases: int = 3

    def __str__(self):
        """Generates a concise string representation of the environment."""
        return f"{self.__class__.__name__} with circuits: {', '.join(self.circuits.keys())}"

    def __repr__(self):
        """Generates an informative string representation of the environment."""
        str_information_circuits = [tab_lines(repr(circuit)) for circuit in self.circuits.values()]
        return f"{self.__class__.__name__}\n" + "\n".join(str_information_circuits)

    def get_cables(self) -> dict[CableKey, PosCable]:
        """Returns a dict of all cables in the static environment."""
        return self.cables

    def get_number_of_cables(self) -> int:
        """Returns the total number of cables in the static environment."""
        return self.number_of_cables

    @property
    @abstractmethod
    def _circuit_from_cable_input_schema_cls(self) -> type[CircuitFromCableInputSchemaT]:
        pass

    def add_circuit_from_cable_id(
        self,
        circuit_input: CircuitFromCableIdInputSchemaT,
    ):
        """Adds the circuit to the environment based on a Cable Id.

        Args:
            circuit_input: CircuitFromCableIdInputSchemaT containing the input parameters for the circuit.

        """
        # Build cable from cable id
        cable = self._build_cable_from_circuit_input_from_cable_id(circuit_input)
        multiple_configurations = self.multiple_configurations_from_cable_id(
            multiple_configurations_from_cable_id=circuit_input.multiple_configurations,
            cable_source_file_path=circuit_input.cable_source_file_path,
        )

        # Add circuit to environment from the constructed cable
        self.add_circuit_from_cable(
            self._circuit_from_cable_input_schema_cls(
                cable=cable,
                multiple_configurations=multiple_configurations,
                **circuit_input.model_dump(exclude={"multiple_configurations"}),
            )
        )
        return self

    def add_circuit_from_cable_constructional_information(
        self,
        circuit_input: CircuitFromCableConstructionalInputSchemaT,
    ):
        """Adds the circuit to the environment based on a Cable Constructional Input Schema.

        Args:
            circuit_input: CircuitFromCableConstructionalInputSchemaT containing the input parameters for the circuit.

        """
        # Build cable from cable constructional information
        cable = self._build_cable_from_circuit_input_from_cable_constructional_information(circuit_input)
        multiple_configurations = self.multiple_configurations_from_cable_constructional_input(
            multiple_configurations_from_cable_constructional_input=circuit_input.multiple_configurations,
        )

        # Add circuit to environment from the constructed cable
        self.add_circuit_from_cable(
            self._circuit_from_cable_input_schema_cls(
                cable=cable,
                multiple_configurations=multiple_configurations,
                **circuit_input.model_dump(exclude={"multiple_configurations"}),
            )
        )
        return self

    def add_circuit_from_cable(
        self,
        circuit_input: CircuitFromCableInputSchemaT,
    ):
        """Add a cable circuit consisting of one or three cables to the environment based on a given cable instance.

        Args:
           circuit_input: CircuitInputSchemaT containing the input
               parameters for the circuit, including a cable instance.

        """
        # If circuit_type is not provided, determine it
        if circuit_input.circuit_type is None:
            circuit_input.circuit_type = self.get_circuit_type(cable=circuit_input.cable)
        else:
            # Warning if unrealistic combination is chosen by user
            self._warn_for_unrealistic_circuits(circuit_input.cable, circuit_input.circuit_type)

        if (
            circuit_input.cable.layer_metrics.pipe
            and circuit_input.cable.layer_metrics.pipe.trefoil_circuit_in_single_pipe
            and circuit_input.circuit_type
            not in [
                CircuitType.Trefoil,
            ]
        ):
            raise NotImplementedError(
                f"Three cables in one pipe not implemented for circuit type: {circuit_input.circuit_type}"
            )

        if circuit_input.circuit_name in self.circuits:
            raise ValueError(
                f"{circuit_input.circuit_name} already exists in environment. Circuit names should be unique."
            )

        # Add circuit to domain based on cable object
        self.circuits[circuit_input.circuit_name] = CircuitBuilder().from_cable(circuit_input=circuit_input)

        if circuit_input.multiple_configurations:
            if self.circuits[circuit_input.circuit_name].bonding is not BondingType.TwoSided:
                raise NotImplementedError(
                    "Circuit is not bonded on two sides, no circulating currents are assumed. "
                    f"Multiple configurations are not implemented for bonding type {circuit_input.bonding_type}."
                )
            self._validate_local_configuration(
                self.circuits[circuit_input.circuit_name],
                [self._generate_circuit_configuration(config) for config in circuit_input.multiple_configurations],
            )
            weighted_screen_impedance = self._generate_weighted_screen_impedance(
                local_screen_resistance=self.circuits[circuit_input.circuit_name]
                .cables[0]
                .cable._get_resistance_screen(Ts=ELECTRIC_RESISTANCE_REFERENCE_TEMPERATURE),
                multiple_configurations=circuit_input.multiple_configurations,
            )
            self.circuits[circuit_input.circuit_name].set_weighted_screen_impedance(weighted_screen_impedance)

        circuit_cables = self.circuits[circuit_input.circuit_name].cables
        self.circuit_cable_indices[circuit_input.circuit_name] = [
            self.number_of_cables + n for n, _ in enumerate(circuit_cables)
        ]
        self._add_cables_to_cable_dict(circuit_cables)

        return self

    def _build_cable_from_circuit_input_from_cable_id(self, circuit_input: CircuitFromCableIdInputSchemaT) -> CableT:
        """Builds a cable from a circuit input schema.

        The cable is built based on the cable_id and cable_source_file provided in the circuit input schema.

        Args:
            circuit_input: Circuit input schema containing the input parameters for the circuit.

        Returns:
            A cable object built based on the input parameters.

        """
        # Determine appropriate FDCable class
        fd_cable_cls = self._determine_cable_class_from_circuit_input(circuit_input)

        return CableBuilder.build_cable_from_cable_id(
            cable_id=circuit_input.cable_id,
            fd_cable_class=fd_cable_cls,
            pipe=circuit_input.pipe,
            cable_source_file_path=circuit_input.cable_source_file_path,
        )

    def _build_cable_from_circuit_input_from_cable_constructional_information(
        self, circuit_input: CircuitFromCableConstructionalInputSchemaT
    ) -> CableT:
        """Builds a cable from a circuit input schema.

        The cable is built based on the cable constructional information provided in the circuit input schema.

        Args:
            circuit_input: Circuit input schema containing the input parameters for the circuit.

        Returns:
            A cable object built based on the input parameters.

        """
        # Determine appropriate FDCable class
        fd_cable_cls = self._determine_cable_class_from_circuit_input(circuit_input)

        return CableBuilder.build_cable(
            cable_constructional_input=circuit_input.cable_constructional_information,
            fd_cable_class=fd_cable_cls,
            pipe=circuit_input.pipe,
        )

    @abstractmethod
    def _determine_cable_class_from_circuit_input(self, circuit_input: BaseCircuitInputSchema) -> type[CableT]:
        """Determines the appropriate FDCable class based on the circuit input schema.

        This is implemented in the subclass since the cable class can differ per environment (Air/Soil).

        Args:
            circuit_input: Circuit input schema containing the input parameters for the circuit.

        Returns:
            The FDCable class that should be used to build the cable based on the input parameters.

        """
        raise NotImplementedError("This method should be implemented in the subclass of StaticEnv.")

    def _generate_circuit_configuration(self, configuration: CircuitConfiguration) -> CableCircuit:
        return CircuitBuilder().from_cable(
            CircuitFromCableInputSchema(
                circuit_type=configuration.circuit_type,
                circuit_name="config",
                cable=configuration.cable,
                dist=configuration.dist,
            )
        )

    def _generate_weighted_screen_impedance(
        self, local_screen_resistance: float, multiple_configurations: list[CircuitConfiguration]
    ) -> WeightedScreenImpedance:
        """Generate weighted impedance matrix.

        References:
            - IEC-60287-1-1 - section 5.3.5
            - IEC-60287-1-3

        """
        total_length = sum([config.length for config in multiple_configurations])
        weighted_reactance_matrix = np.zeros((self.n_phases - 1, self.n_phases))
        weighted_resistance = 0.0
        for config in multiple_configurations:
            reactance_matrix = self._generate_circuit_configuration(
                configuration=config
            ).get_relative_screen_reactances()

            weighted_reactance_matrix += reactance_matrix * config.length / total_length

            screen_resistance = config.cable._get_resistance_screen(Ts=ELECTRIC_RESISTANCE_REFERENCE_TEMPERATURE)
            weighted_resistance += screen_resistance * config.length / total_length

        weighted_resistance_factor = weighted_resistance / local_screen_resistance
        return WeightedScreenImpedance(
            weighted_reactance_matrix=weighted_reactance_matrix,
            weighted_resistance_factor=weighted_resistance_factor,
        )

    def _validate_local_configuration(
        self, local_configuration: CableCircuit, multiple_configurations: list[CableCircuit]
    ):
        local_relative_differences = local_configuration.get_relative_screen_distances()
        local_screen_resistance = local_configuration.cables[0].cable._get_resistance_screen(
            Ts=ELECTRIC_RESISTANCE_REFERENCE_TEMPERATURE
        )

        for config in multiple_configurations:
            valid_local_relative_differences = np.allclose(
                local_relative_differences, config.get_relative_screen_distances()
            )
            valid_local_screen_resistance = np.isclose(
                local_screen_resistance,
                config.cables[0].cable._get_resistance_screen(Ts=ELECTRIC_RESISTANCE_REFERENCE_TEMPERATURE),
            )
            if valid_local_relative_differences and valid_local_screen_resistance:
                return

        raise ValueError(
            "Local configuration does not match any of the provided configurations in multiple_configurations."
        )

    def _add_cables_to_cable_dict(self, cables: list[PosCable]):
        """Adds cables to the environment cables property.

        Args:
            cables: A list of cables to add to the static environment

        """
        for cable in cables:
            self.cables[cable.name] = cable
            self.number_of_cables += 1

    @staticmethod
    def _warn_for_unrealistic_circuits(cable: AbstractCable, circuit_type: CircuitType):
        """Checks if circuits are realistic and throws warnings if they're not.

        Args:
            cable: A cable object that is checked for number of conductors
            circuit_type: a string of the circuit type that can be checked against the cable

        """
        if circuit_type == CircuitType.Single and cable.conductor.number_of_conductors == CableConductorCount.One:
            raise ValueError(
                "Unrealistic combination of cable and circuit_type: cable has "
                f"{CableConductorCount.One.value} conductor and circuit_type '{CircuitType.Single.value}'."
                f" Did you mean circuit_type '{CircuitType.Trefoil.value}' or '{CircuitType.Linear.value}' instead?"
            )
        elif (
            circuit_type not in [CircuitType.Single]
            and cable.conductor.number_of_conductors == CableConductorCount.Three
        ):
            raise ValueError(
                "Unrealistic combination of cable and circuit_type: cable has "
                f"{CableConductorCount.Three.value} conductors and circuit_type is '{circuit_type.value}'."
                f" Did you mean circuit_type '{CircuitType.Single.value}' instead?"
            )

    @staticmethod
    def get_circuit_type(cable: AbstractCable) -> CircuitType:
        """Determines probable circuit type by number of conductors in the cable."""
        if isinstance(cable, FDCableTrefoilCircuitInSinglePipe | FDCableTrefoilCircuitInSinglePipeInAir):
            return CircuitType.Trefoil
        if cable.conductor.number_of_conductors == CableConductorCount.Three:
            return CircuitType.Single
        if cable.conductor.number_of_conductors == CableConductorCount.One:
            warnings.warn(
                '"trefoil" is assumed to be the circuit_type, since the cable has a single conductor. '
                'If "linear" was meant as circuit_type, please specify.',
                stacklevel=2,
            )
            return CircuitType.Trefoil

        raise NotImplementedError(f"Number of conductors '{cable.conductor.number_of_conductors}' not supported.")

    def get_cable(self, cable_key: CableKey) -> PosCable:
        """Gets the Cable-object corresponding to the cable_name from the environment."""
        return self.cables[cable_key]

    @staticmethod
    def multiple_configurations_from_cable_id(
        multiple_configurations_from_cable_id: list[CircuitConfigurationFromCableId], cable_source_file_path: Path
    ) -> list[CircuitConfiguration]:
        """Generates multiple circuit configurations based on a list of CircuitConfigurationFromCableId.

        Args:
            multiple_configurations_from_cable_id: A list of
                CircuitConfigurationFromCableId, specifying the cable ids and
                lengths of the different configurations.
            cable_source_file_path: Name of the file containing the cable
                specifications. This file has to be located in the data
                directory and must either be an excel or csv file.

        Returns:
            list[CircuitConfiguration]: A list of CircuitConfiguration objects that can be used in
                the multiple_configurations argument of add_circuit_from_cable or
                add_circuit_from_cable_id.

        """
        multiple_configurations: list[CircuitConfiguration] = []
        for config in multiple_configurations_from_cable_id:
            cable = CableBuilder.build_cable_from_cable_id(
                cable_id=config.cable_id,
                fd_cable_class=config.fd_cable_class,
                pipe=config.pipe,
                cable_source_file_path=cable_source_file_path,
            )
            multiple_configurations.append(
                CircuitConfiguration(
                    cable=cable,
                    length=config.length,
                    circuit_type=config.circuit_type,
                    dist=config.dist,
                )
            )
        return multiple_configurations

    @staticmethod
    def multiple_configurations_from_cable_constructional_input(
        multiple_configurations_from_cable_constructional_input: list[
            CircuitConfigurationFromCableConstructionalInputSchema
        ],
    ) -> list[CircuitConfiguration]:
        """Generate multiple configurations from constructional input schemas.

        Args:
            multiple_configurations_from_cable_constructional_input: A list
                of CircuitConfigurationFromCableConstructionalInputSchema,
                specifying the cable constructional input schemas and lengths
                of the different configurations.

        Returns:
            list[CircuitConfiguration]: A list of CircuitConfiguration objects that can be used in
                the multiple_configurations argument of add_circuit_from_cable or
                add_circuit_from_cable_id.

        """
        multiple_configurations: list[CircuitConfiguration] = []
        for config in multiple_configurations_from_cable_constructional_input:
            cable = CableBuilder.build_cable(
                cable_constructional_input=config.cable_constructional_information,
                fd_cable_class=config.fd_cable_class,
                pipe=config.pipe,
            )
            multiple_configurations.append(
                CircuitConfiguration(
                    cable=cable,
                    length=config.length,
                    circuit_type=config.circuit_type,
                    dist=config.dist,
                )
            )
        return multiple_configurations
