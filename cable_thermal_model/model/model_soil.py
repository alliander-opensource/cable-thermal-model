# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0


from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandera.typing import DataFrame

from cable_thermal_model.cable.cable_circuit import (
    CableKey,
    PosCable,
    add_soil_layer,
    return_mirror_cable,
)
from cable_thermal_model.environment.static_env_soil import StaticEnvSoil
from cable_thermal_model.model.cables.enum_classes_cable import CableLayer
from cable_thermal_model.model.model import Model
from cable_thermal_model.model.schemas import StateSoil
from cable_thermal_model.model.schemas.model_input_schemas import ScenarioSchemaSoil
from cable_thermal_model.model.schemas.run_options import ModelSoilRunOptions


@dataclass
class _SoilMatrices:
    """Finite-difference matrices for the cables with and without soil.

    Args:
        matrices_with_soil: Banded finite-difference matrices for the soil-extended cable representations.
        matrices_without_soil: Banded finite-difference matrices for the cable representations without soil.

    """

    matrices_with_soil: dict[CableKey, np.ndarray]
    matrices_without_soil: dict[CableKey, np.ndarray]


class ModelSoil(Model[ModelSoilRunOptions, StateSoil, ScenarioSchemaSoil, StaticEnvSoil, _SoilMatrices]):
    """ModelSoil computes temperatures for underground power cables using the finite-difference method.

    In most cases the model is instantiated with a StaticEnvSoil and a valid scenario, then executed via `run()`.
    """

    _run_options_class = ModelSoilRunOptions
    _state_class = StateSoil

    def __init__(self, static_env: StaticEnvSoil, scenario: DataFrame[ScenarioSchemaSoil]):
        """Initialize the ModelSoil instance with a static environment and scenario.

        Note: the scenario must contain one `load_<circuit_name>` column per circuit, plus ambient temperature and
        soil-property columns.

        Args:
            static_env: A StaticEnvSoil instance containing the soil thermal parameters and cable layout.
            scenario: A pandera DataFrame[ScenarioSchemaSoil] containing the dynamic load and soil data.

        Attributes:
            mirror_cables_with_soil:            A dict containing the mirror cables with soil for each cable in the
                                                environment
            logarithmic_soil_gridpoint_density: The density of grid points in the soil layers, this is used to compute
                                                the number of grid points in the soil layers based on their thickness.
                                                The default value is 20 grid points per factor 2 increase in soil layer
                                                thickness. For a cable with radius of 3.1 cm and a soil layer radius
                                                1 m,
                                                this would result in 100 grid points in the soil layer.
            minimal_soil_radius:                The minimal soil radius around a cable. For deeply buried cables, the
                                                soil radius is set to 2.5 times the cable depth, this parameter sets
                                                a lower bound to prevent very small soil layers for shallow cables.

        """
        if not isinstance(static_env, StaticEnvSoil):
            raise ValueError(
                f"Can not use model {self.__class__.__name__} if static "
                "environment is not an environment in soil. Please use "
                "ModelAir instead."
            )

        # Set up cables
        self.cables_with_soil: dict[CableKey, PosCable] = {}
        self.mirror_cables_with_soil: dict[CableKey, PosCable] = {}
        self.logarithmic_soil_gridpoint_density: float = 20
        self.minimal_soil_radius: float = 5.0
        self.last_soil_property_update_day: int = 0

        super().__init__(static_env=static_env, scenario=scenario)

    def _validate_scenario(self):
        """Validate the scenario dataframe for required columns.

        Raises:
            ValueError: If required columns are missing from the scenario dataframe.

        """
        super()._validate_scenario()
        ScenarioSchemaSoil.validate(self.scenario)

    def _initialize_cables(self):
        """Initialize cables with soil layers and mirror cables for the boundary condition."""
        # Start from the static cables without soil, then add a soil layer per cable.
        # Deep cables may get an extra outer soil layer to extend the domain.
        # The outer boundary is treated as ambient.

        super()._initialize_cables()
        cables_with_soil = {}

        for key, cable in self.cables.items():
            soil_radius = max(self.minimal_soil_radius, 2.5 * abs(cable.y))

            # Instantiate FDCable objects with the added soil layer.
            cables_with_soil[key] = add_soil_layer(
                deepcopy(cable),
                soil_rho=self.scenario[self.THERMAL_RESISTIVITY_COLUMN].iloc[0],
                soil_capacity=self.scenario[self.THERMAL_CAPACITY_COLUMN].iloc[0],
                soil_radius=soil_radius,
                logarithmic_soil_gridpoint_density=self.logarithmic_soil_gridpoint_density,
            )

        self.cables_with_soil = cables_with_soil

        # Create mirror cables to enforce the T=0 boundary condition on y=0.
        self.mirror_cables_with_soil = {
            key: return_mirror_cable(pos_cable) for key, pos_cable in self.cables_with_soil.items()
        }

    def _initialize_linear_system(self) -> tuple[_SoilMatrices, dict[CableKey, np.ndarray]]:
        """Initializes the linear system (matrices and vectors) for each cable in the model.

        Returns:
            tuple[_SoilMatrices, dict[CableKey, np.ndarray]]:
                A tuple containing the initialized matrices and vectors for each cable.
        """
        matrices_with_soil, vectors = self._build_linear_system_for_cables(cables=self.cables_with_soil)

        matrices_without_soil: dict[CableKey, np.ndarray] = {}
        for key, cable in self.cables.items():
            matrices_without_soil[key] = cable.cable.get_finite_difference_matrix()

        soil_matrices = _SoilMatrices(
            matrices_with_soil=matrices_with_soil,
            matrices_without_soil=matrices_without_soil,
        )

        return soil_matrices, vectors

    def _build_initial_thermal_state(self) -> StateSoil:
        """Builds the initial thermal state for the model.

        Returns:
            StateSoil: An instance of StateSoil containing the initialized temperature,
                            self-heating, and mutual-heating states for each cable.
        """
        return StateSoil(
            static_env_hash=self.static_env.compute_hash(),
            temperature=self._initialize_temperature_state(),
            self_heating=self._initialize_state_from_cables(cables=self.cables_with_soil),
            mutual_heating=self._initialize_state_from_cables(cables=self.cables),
        )

    def get_temp(self, x: float, y: float, time_sec: float, solutions: dict[CableKey, np.ndarray]) -> float:
        """Compute the temperature at a point and time in the environment.

        Args:
            x: x-coordinate of the point.
            y: y-coordinate of the point.
            time_sec: Time in seconds at which to evaluate the temperature.
            solutions: Cable solutions used to compute the temperature in the environment.

        Returns:
            float: Temperature in degrees Celsius.

        """
        time_grid = (self.scenario.index - self.scenario.index[0]).total_seconds()
        time_idx = np.nonzero(time_grid >= time_sec)[0][0]
        temp = self.scenario["ambient_temperature"].iloc[time_idx]
        for key, cable in self.cables_with_soil.items():
            dist = cable.distance_to_point(x=x, y=y)
            temp += self._compute_temp_contribution(cable, dist, solutions[key], is_mirror_cable=False)

        for key, mirror_cable in self.mirror_cables_with_soil.items():
            dist = mirror_cable.distance_to_point(x=x, y=y)
            temp += self._compute_temp_contribution(mirror_cable, dist, solutions[key], is_mirror_cable=True)

        return temp

    @staticmethod
    def _compute_temp_contribution(
        cable: PosCable,
        dist: float,
        solution: np.ndarray,
        is_mirror_cable: bool,
    ) -> float:
        """Compute the temperature contribution of one cable at a given distance."""
        if is_mirror_cable:
            solution = -solution
        return float(np.interp(x=[dist], xp=cable.cable.radii_grid, fp=solution)[0])

    def _get_dry_soil_radius_around_circuit(
        self,
        cables_with_soil: list[PosCable],
        temperature_state: list[np.ndarray],
    ) -> float:
        """Compute an approximation to the radius of dried-out soil around a cable circuit.

        This radius is determined through IEC/NPR norms: all soil with a temperature of 30 degrees or more is dried out.

        Notes:
            To improve computability we only use the heating due to the circuit itself to determine what soil is
            drying out. We determine the distance until where the temperature solution exceeds 30 degrees. This
            approach provides an approximate circular radius within which all soil is considered dried-out. Since
            there are multiple cables in the environment the actual shape of dried out soil surrounding a circuit
            is likely different.

        Args:
            cables_with_soil: Cables corresponding to the configuration.
            temperature_state: Internal heating solutions for cables in the circuit.

        Returns:
            float: The radius within which all soil is dried out around the cable.

        """
        # Base soil drying on one cable in the circuit, which for trefoil is the central 'top' cable
        cable_temperature_state = temperature_state[0]
        radii_grid = cables_with_soil[0].cable.radii_grid

        # Returns the radius of the last grid point where soil is dried out, or radii_grid[0] if none
        idxs = np.nonzero(cable_temperature_state >= self._SOIL_DRYING_TEMPERATURE)[0]
        if idxs.size == 0:
            return float(radii_grid[0])
        return float(radii_grid[idxs[-1]])

    def _get_dry_soil_radius_for_all_cables(
        self, temperature_state: dict[CableKey, np.ndarray]
    ) -> dict[CableKey, float]:
        """Compute the dry soil radii around the cables in the environment.

        Args:
            temperature_state: Temperature states for all cables in the environment.

        Returns:
            dict[CableKey, float]: Dry soil radius per cable.

        """
        dry_soil_radii = {}

        # determine the dried-out soil radius per circuit
        for circuit in self.static_env.circuits.values():
            circuit_cables = {cable.name: cable for cable in circuit.cables}
            circuit_temperature_state = [temperature_state[cable_key] for cable_key in circuit_cables]
            cables_with_soil = [self.cables_with_soil[cable_key] for cable_key in circuit_cables]

            dry_soil_radius = self._get_dry_soil_radius_around_circuit(
                cables_with_soil=cables_with_soil,
                temperature_state=circuit_temperature_state,
            )
            for cable_key in circuit_cables:
                dry_soil_radii[cable_key] = dry_soil_radius

        return dry_soil_radii

    def _compute_mutual_heating_effect(self, self_heating_state: dict[CableKey, np.ndarray]) -> dict[CableKey, float]:
        """Compute the heating of a cable due to other cables in the environment.

        These contributions are accumulated per cable and later added to the thermal state.

        Args:
            self_heating_state: Complete self-heating solutions for all cables at a given timestep.

        Returns:
            dict[CableKey, float]: Temperature increases due to mutual heating, one value per cable.

        """
        mutual_heating_effect = dict.fromkeys(self.cables, 0.0)

        for key, cable in self.cables.items():
            # Heating from other cables
            for other_key, other_cable in self.cables_with_soil.items():
                if key != other_key:  # skip self
                    dist = cable.distance_to(other_cable)
                    mutual_heating_effect[key] += self._compute_temp_contribution(
                        other_cable, dist, self_heating_state[other_key], is_mirror_cable=False
                    )

            # Cooling from mirror cables
            for mirror_key, mirror_cable in self.mirror_cables_with_soil.items():
                dist = cable.distance_to(mirror_cable)
                mutual_heating_effect[key] += self._compute_temp_contribution(
                    mirror_cable, dist, self_heating_state[mirror_key], is_mirror_cable=True
                )

        return mutual_heating_effect

    def _update_soil_resistivity_for_all_cables(
        self,
        soil_drying: bool,
        temperature_state: dict,
        soil_resistivity: float,
    ) -> set[CableKey]:
        """Updates soil resistivity for all cables if significantly different or if soil drying is taken into account.

        Args:
            soil_drying: Whether the scenario takes soil drying into account.
            temperature_state: Full temperature state per cable at the current timestep.
            soil_resistivity: Soil thermal resistivity for the current time step.

        Returns:
            set[CableKey]: Cables for which the soil resistivity was updated.

        """
        dry_soil_radii = (
            self._get_dry_soil_radius_for_all_cables(temperature_state=temperature_state) if soil_drying else None
        )

        # Update the soil layers when the soil resistivity changes or soil drying is enabled.
        updated_cables = set()
        for cable_key, cable in self.cables_with_soil.items():
            if soil_drying or not np.isclose(cable.cable.rho_grid[-1], soil_resistivity, rtol=1e-2):
                cable.cable.update_soil_resistivity(
                    soil_rho=soil_resistivity,
                    dry_soil_radius=dry_soil_radii[cable_key] if dry_soil_radii else None,
                )
                updated_cables.add(cable_key)

        return updated_cables

    def _update_soil_capacity_for_all_cables(self, soil_capacity: float) -> set[CableKey]:
        """Updates soil thermal capacity for all cables if significantly different.

        Args:
            soil_capacity: Soil thermal capacity for the current time step.

        Returns:
            set[CableKey]: Cables for which the soil capacity was updated.

        """
        updated_cables = set()
        for cable_key, cable in self.cables_with_soil.items():
            if not np.isclose(cable.cable.capacity_grid[-1], soil_capacity, rtol=1e-2):
                cable.cable.update_soil_capacity(soil_c=soil_capacity)
                updated_cables.add(cable_key)

        return updated_cables

    def _update_soil_properties_for_all_cables(
        self,
        soil_drying: bool,
        temperature_state: dict[CableKey, np.ndarray],
        soil_resistivity: float,
        soil_capacity: float,
    ) -> set[CableKey]:
        """Update soil properties for all cables if needed.

        Args:
            soil_drying: Whether the scenario takes soil drying into account.
            temperature_state: Full temperature state per cable at the current timestep.
            soil_resistivity: Soil thermal resistivity for the current time step.
            soil_capacity: Soil thermal capacity for the current time step.

        Returns:
            set[CableKey]: Cables for which the soil properties were updated.

        """
        updated_cables = self._update_soil_resistivity_for_all_cables(
            soil_resistivity=soil_resistivity,
            soil_drying=soil_drying,
            temperature_state=temperature_state,
        )

        updated_cables |= self._update_soil_capacity_for_all_cables(
            soil_capacity=soil_capacity,
        )

        return updated_cables

    def _update_pipe_resistivity_for_all_cables(
        self,
        temperature_state: dict[CableKey, np.ndarray],
    ) -> set[CableKey]:
        """Update pipe-fill resistivity for both no-soil and with-soil cable representations.

        Args:
            temperature_state: Full temperature state per cable at the current timestep.

        Returns:
            set[CableKey]: Cables for which the pipe-fill resistivity was updated.
        """
        updated_cables = super()._update_pipe_resistivity_for_all_cables(temperature_state=temperature_state)

        # Also update cables_with_soil because the pipe-fill resistivity is used in both finite-difference matrices.
        for cable_key in updated_cables:
            cable = self.cables_with_soil[cable_key].cable

            mean_pipe_fill_temp = cable.get_mean_temperature_cable_layer(
                temperature_grid=temperature_state[cable_key],
                layer=CableLayer.PipeFill,
            )
            cable.update_pipe_resistivity(Tfill=mean_pipe_fill_temp)

        return updated_cables

    def _check_if_daily_update_due(
        self, seconds_since_start_scenario: float, last_soil_property_update_day: int
    ) -> tuple[bool, int]:
        """Check if a daily update of soil properties is due based on the time elapsed since the start of the scenario.

        Args:
            seconds_since_start_scenario: The number of seconds that have passed since the start of the scenario.
            last_soil_property_update_day: Day counter indicating when the last soil-property update occurred.

        Returns:
            A tuple containing a boolean indicating whether a daily update is due and the updated day counter.

        """
        daily_update_due = False
        days = seconds_since_start_scenario / (60 * 60 * 24)
        if days > last_soil_property_update_day:
            daily_update_due = True
            last_soil_property_update_day = int(days)

        return daily_update_due, last_soil_property_update_day

    def _update_self_heating_state(
        self,
        self_heating_state: dict[CableKey, np.ndarray],
        matrices: dict[CableKey, np.ndarray],
        vectors: dict[CableKey, np.ndarray],
        time_step: float,
    ) -> dict[CableKey, np.ndarray]:
        """Update the self-heating state for all cables in the environment for a given time step.

        Args:
            self_heating_state: The current self-heating state.
            matrices: The matrices for the linear system.
            vectors: The vectors for the linear system.
            time_step: The time step for the integration.

        Returns:
            Updated self-heating state.

        """
        new_self_heating_state = {}
        for cable_key, cable in self.cables_with_soil.items():
            heat_equation_solution = cable.cable.integrate_timestep(
                s=self_heating_state[cable_key][:-1],
                A_banded=matrices[cable_key],
                b=vectors[cable_key],
                time_step=time_step,
                internal_heating=True,
            )
            # We assume the outer boundary of the soil is at ambient temperature
            new_self_heating_state[cable_key] = np.append(heat_equation_solution, 0.0)

        return new_self_heating_state

    def _update_mutual_heating_state(
        self,
        self_heating_state: dict[CableKey, np.ndarray],
        mutual_heating_state: dict[CableKey, np.ndarray],
        matrices_without_soil: dict[CableKey, np.ndarray],
        time_step: float,
    ) -> dict[CableKey, np.ndarray]:
        """Update the mutual heating state for all cables in the environment for a given time step.

        Args:
            self_heating_state: The current self-heating state.
            mutual_heating_state: The current mutual heating state.
            matrices_without_soil: The finite-difference matrices for the cable representations without soil layers.
            time_step: The time step for the integration.

        Returns:
            Updated mutual heating state.

        """
        # First compute the heating of a cable due to other cables in the environment
        mutual_heating_effect = self._compute_mutual_heating_effect(self_heating_state=self_heating_state)

        new_mutual_heating_state = {}
        for cable_key, cable in self.cables.items():
            matrix_without_soil = matrices_without_soil[cable_key]
            outer_boundary_coupling_coefficient = cable.cable.get_outer_boundary_coupling_coefficient_from_matrix(
                banded_matrix=matrix_without_soil
            )

            # Add the mutual heating to the outermost grid point of the vector
            vector_without_soil = np.zeros(cable.cable.radii_grid.size - 1)
            vector_without_soil[-1] = outer_boundary_coupling_coefficient * mutual_heating_effect[cable_key]

            heat_equation_solution = cable.cable.integrate_timestep(
                s=mutual_heating_state[cable_key][:-1],
                A_banded=matrix_without_soil,
                b=vector_without_soil,
                time_step=time_step,
                internal_heating=False,
            )
            new_mutual_heating_state[cable_key] = np.append(heat_equation_solution, mutual_heating_effect[cable_key])

        return new_mutual_heating_state

    def _update_temperature_state(
        self,
        self_heating_state: dict[CableKey, np.ndarray],
        mutual_heating_state: dict[CableKey, np.ndarray],
        ambient_temperature: float,
    ) -> dict[CableKey, np.ndarray]:
        """Update the temperature state for all cables by summing the different contributions.

        Args:
            self_heating_state: The current self-heating state for all cables.
            mutual_heating_state: The current mutual heating state for all cables.
            ambient_temperature: The ambient temperature for the current time step.

        Returns:
            dict[CableKey, np.ndarray]: Updated temperature state for all cables.
        """
        new_temperature_state = {}
        for cable_key in self.cables:
            mutual_heating_cable_state = mutual_heating_state[cable_key]
            self_heating_cable_state = self_heating_state[cable_key][: mutual_heating_cable_state.size]

            new_temperature_state[cable_key] = (
                self_heating_cable_state + mutual_heating_cable_state + ambient_temperature
            )

        return new_temperature_state

    def _refresh_matrices_if_needed(
        self,
        matrices: _SoilMatrices,
        temperature_state: dict[CableKey, np.ndarray],
        scenario_row: pd.Series,
        elapsed_seconds: float,
    ) -> tuple[_SoilMatrices, set[CableKey]]:
        """Update cable properties and refresh finite-difference matrices as needed."""
        soil_resistivity = scenario_row[self.THERMAL_RESISTIVITY_COLUMN]
        soil_capacity = scenario_row[self.THERMAL_CAPACITY_COLUMN]

        # Update pipe resistivity if it changes significantly based on the current temperature state.
        cables_with_updated_pipe_fill = self._update_pipe_resistivity_for_all_cables(
            temperature_state=temperature_state
        )

        daily_update_due, self.last_soil_property_update_day = self._check_if_daily_update_due(
            seconds_since_start_scenario=elapsed_seconds,
            last_soil_property_update_day=self.last_soil_property_update_day,
        )

        updated_cables = cables_with_updated_pipe_fill.copy()
        if daily_update_due:
            updated_cables |= self._update_soil_properties_for_all_cables(
                soil_drying=self.run_options.soil_drying,
                temperature_state=temperature_state,
                soil_resistivity=soil_resistivity,
                soil_capacity=soil_capacity,
            )

        for cable_key in updated_cables:
            # The soil-extended matrix always needs to be refreshed when pipe fill or soil layers change.
            matrices.matrices_with_soil[cable_key] = self.cables_with_soil[
                cable_key
            ].cable.get_finite_difference_matrix()

            if cable_key in cables_with_updated_pipe_fill:
                # If pipe-fill resistivity changed, the no-soil matrix also needs to be refreshed.
                matrix_without_soil = self.cables[cable_key].cable.get_finite_difference_matrix()
                matrices.matrices_without_soil[cable_key] = matrix_without_soil

        return matrices, updated_cables

    def _update_thermal_state(
        self,
        thermal_state: StateSoil,
        matrices: _SoilMatrices,
        vectors: dict[CableKey, np.ndarray],
        ambient_temperature: float,
        time_step: float,
    ) -> StateSoil:
        """Update thermal state for one timestep using extracted step variables."""
        new_self_heating_state = self._update_self_heating_state(
            self_heating_state=thermal_state.self_heating,
            matrices=matrices.matrices_with_soil,
            vectors=vectors,
            time_step=time_step,
        )

        new_mutual_heating_state = self._update_mutual_heating_state(
            self_heating_state=new_self_heating_state,
            mutual_heating_state=thermal_state.mutual_heating,
            matrices_without_soil=matrices.matrices_without_soil,
            time_step=time_step,
        )

        new_temperature_state = self._update_temperature_state(
            self_heating_state=new_self_heating_state,
            mutual_heating_state=new_mutual_heating_state,
            ambient_temperature=ambient_temperature,
        )

        thermal_state.temperature = new_temperature_state
        thermal_state.self_heating = new_self_heating_state
        thermal_state.mutual_heating = new_mutual_heating_state

        return thermal_state
