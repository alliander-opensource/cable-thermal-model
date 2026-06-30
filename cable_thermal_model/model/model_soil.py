# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0


from copy import deepcopy
from dataclasses import dataclass

import numpy as np
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
from cable_thermal_model.model.schemas import ModelOutputSchema, StateSoil
from cable_thermal_model.model.schemas.model_input_schemas import ScenarioSchemaSoil
from cable_thermal_model.model.schemas.run_options import ModelSoilRunOptions


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


class ModelSoil(Model[ModelSoilRunOptions, StateSoil, ScenarioSchemaSoil, StaticEnvSoil]):
    """ModelSoil is used to compute temperature of underground power cables using the finite differences methodology.

    A 1D approach is taken to modeling the environment and the cables, pipes and soil within it. The finite differences
    computations are fast and efficient.

    In most cases the model is used by instantiating it using a StaticEnvSoil and a valid scenario and calling the run()
    method.
        >> model = ModelSoil(environment, scenario)
        >> result = model.run()
    """

    def __init__(self, static_env: StaticEnvSoil, scenario: DataFrame[ScenarioSchemaSoil]):
        """To initialize a ModelSoil instance two inputs are required: a static environment and a scenario dataframe.

        N.B. the column names of 'scenario' should be as follows 'load_circuit_1' contains the load (in A) of the
        'circuit_1' object of static_env and column 'ambient_temperature' contains the ambient temperature (in degrees
        Celsius).

        Args:
            static_env: A StaticEnvSoil instance containing the soil thermal parameters and information of cables,
                        lying configuration
            scenario:   A pandera DataFrame[ScenarioSchemaSoil] containing the dynamic data i.e. loads of the
                        cable circuits and the soil temperature

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

        super().__init__(static_env=static_env, scenario=scenario)

    @dataclass
    class _ThermalState:
        """Thermal state tracked during the soil-model time simulation.

        Args:
            temperature: Full temperature solution per cable, combining self_heating, mutual_heating and ambient.
            self_heating: Temperature rise per cable caused by internal heating and soil conduction.
            mutual_heating: Temperature rise per cable caused by other cables and mirror cables.

        """

        temperature: dict[CableKey, np.ndarray]
        self_heating: dict[CableKey, np.ndarray]
        mutual_heating: dict[CableKey, np.ndarray]

    @dataclass
    class _MatrixState:
        """Finite-difference matrix state tracked during the soil-model time simulation.

        Args:
            matrices_with_soil: Banded finite-difference matrices for the soil-extended cable representations.
            matrices_without_soil: Banded finite-difference matrices for the cable representations without soil.
            outer_boundary_coupling_coefficients: Coupling coefficients for the known outer boundary of the cable.
            last_day_with_update: Day counter used to determine when soil properties must be refreshed.

        """

        matrices_with_soil: dict[CableKey, np.ndarray]
        matrices_without_soil: dict[CableKey, np.ndarray]
        outer_boundary_coupling_coefficients: dict[CableKey, float]
        last_day_with_update: int

    def _validate_scenario(self):
        """Validate the scenario DataFrame for required columns.

        Raises:
            ValueError: If required columns are missing from the scenario DataFrame.

        """
        super()._validate_scenario()
        ScenarioSchemaSoil.validate(self.scenario)

    def _initialize_cables(self):
        """Initialize cables with soil layers and mirror cables for boundary conditions.

        This method copies the cables from the static environment into the model, adding soil layers and creating
        mirror cables to enforce boundary conditions. It also sets properties such as pipes, number of cables, and
        conductor indices.
        """
        # The static_env comes with cable instances without soil, this is first added
        # We use two soil layers with each 100 grid points at 1m and 5m respectively,
        # for deeper cables an extra soil layer is added between 5m and 2 times the cable depth.
        # Boundary conditions are applied at the outermost soil radius.

        super()._initialize_cables()
        cables_with_soil = {}

        for key, cable in self.cables.items():
            soil_radius = max(self.minimal_soil_radius, 2.5 * abs(cable.y))

            # Instantiate FDCable objects with the added soil layer
            cables_with_soil[key] = add_soil_layer(
                deepcopy(cable),
                soil_rho=self.scenario[self.THERMAL_RESISTIVITY_COLUMN].iloc[0],
                soil_capacity=self.scenario[self.THERMAL_CAPACITY_COLUMN].iloc[0],
                soil_radius=soil_radius,
                logarithmic_soil_gridpoint_density=self.logarithmic_soil_gridpoint_density,
            )

        self.cables_with_soil = cables_with_soil

        # Create mirror cables with negative temperature solutions in order to enforce T=0 boundary condition on y=0
        self.mirror_cables_with_soil = {
            key: return_mirror_cable(pos_cable) for key, pos_cable in self.cables_with_soil.items()
        }

    def _initialize_matrix_state(self) -> _MatrixState:
        """Initialize the matrix state for the model.

        Returns:
            _MatrixState: An instance of _MatrixState containing the initialized finite-difference matrices and
                            outer boundary coupling coefficients for each cable.
        """
        matrices_with_soil = {
            key: cable.cable.get_finite_difference_matrix() for key, cable in self.cables_with_soil.items()
        }
        matrices_without_soil: dict[CableKey, np.ndarray] = {}
        outer_boundary_coupling_coefficients: dict[CableKey, float] = {}

        for key, cable in self.cables.items():
            matrix_without_soil, outer_boundary_coupling_coefficient = (
                cable.cable.get_finite_difference_matrix_with_outer_boundary_coupling()
            )
            matrices_without_soil[key] = matrix_without_soil
            outer_boundary_coupling_coefficients[key] = outer_boundary_coupling_coefficient

        return self._MatrixState(
            matrices_with_soil=matrices_with_soil,
            matrices_without_soil=matrices_without_soil,
            outer_boundary_coupling_coefficients=outer_boundary_coupling_coefficients,
            last_day_with_update=0,
        )

    def _initialize_mutual_heating_state(self, initial_state: StateSoil | None = None) -> dict[CableKey, np.ndarray]:
        """Initiate dicts that contain temperature solutions for each cable.

        These are pandas dataframes with:
        dimensions: [timegrid x gridpoints] per cable. Thus: number_cables x [timegrid x gridpoints]
        These dicts will be updated for each timestep the solving loop.
        The following dicts are initiated:
        - mutual_heating_state: a dict with temperature states of temperature rise inside a cable due to mutual
            heating from other cables.

        Args:
            initial_state: Optional StateSoil object containing mutual_heating_state to initialize from.

        Returns:
            dict[CableKey, np.ndarray]: A dictionary containing mutual heating states.

        """
        # Initiate a dict with temperature states of temperature rise inside a cable due to mutual heating from
        #  other cables
        mutual_heating_state = {
            key: np.zeros(pos_cable.cable.radii_grid.size) for key, pos_cable in self.cables.items()
        }

        # If state information was provided it is used to initialize the temperature solutions with
        if initial_state is not None:
            mutual_heating = initial_state.mutual_heating_solutions
            for key in mutual_heating:
                mutual_heating_state[key] += mutual_heating[key]

        return mutual_heating_state

    def _initialize_thermal_state(self, initial_state: StateSoil | None) -> _ThermalState:
        """Initialize the thermal state for the model.

        Args:
            initial_state: Optional StateSoil object containing temperature, self-heating, and mutual heating states.

        Returns:
            _ThermalState: An instance of _ThermalState containing the initialized temperature,
                            self-heating, and mutual-heating states for each cable.
        """
        temperature_state = self._initialize_temperature_state(initial_state=initial_state)
        self_heating_state = self._initialize_self_heating_state(
            cables=self.cables_with_soil, initial_state=initial_state
        )
        mutual_heating_state = self._initialize_mutual_heating_state(initial_state=initial_state)

        return self._ThermalState(
            temperature=temperature_state,
            self_heating=self_heating_state,
            mutual_heating=mutual_heating_state,
        )

    def _get_dry_soil_radius_around_circuit(
        self,
        cables_with_soil: list[PosCable],
        temperature_state: list[np.ndarray],
    ) -> float:
        """Computes an approximation to the radius of dried out soil around a cable circuit.

        This radius is determined through IEC/NPR norms: all soil with a temperature of 30 degrees or more is dried out.

        Notes:
            To improve computability we only use the heating due to the circuit itself to determine what soil is
            drying out. We determine the distance until where the temperature solution exceeds 30 degrees. This
            approach provides an approximate circular radius within which all soil is considered dried-out. Since
            there are multiple cables in the environment the actual shape of dried out soil surrounding a circuit
            is likely different.

        Args:
            cables_with_soil: list[PosCable]:     Cables corresponding to the configuration
            temperature_state (list[np.ndarray]):    Internal heating solutions for cables in circuit

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
        """This method computes the dry soil radii around the cables in the environment.

        Args:
            temperature_state (dict[CableKey, np.ndarray]): Temperature states for all cables in the environment

        Returns:
             dict[CableKey, float]: A dictionary of radii describing the amount of dried-out soil surrounding each
                                    cable in the same order as [self.cables].

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
        """Computes the heating of a cable due to other cables in the environment.

        These contributions are stored in an array "external_heat" and added as an
        instance variable to the environment cable.

        Args:
            self_heating_state (dict[CableKey, np.ndarray]): the complete dict of self-heating solutions
                at a given timestep corresponding to all cables in the environment.

        Returns:
            dict[CableKey, float]: a dictionary of the temperature increases
                (°C) due to mutual heating, one number per cable.

        """
        # dict to hold the temperature increase due to mutual heating for each cable, initialized with zeros
        mutual_heating_effect = dict.fromkeys(self.cables, 0.0)

        for key, cable in self.cables.items():
            # Heating from other cables
            for other_key, other_cable in self.cables_with_soil.items():
                if key != other_key:  # skip self
                    dist = cable.distance_to(other_cable)
                    mutual_heating_effect[key] += _compute_temp_contribution(
                        other_cable, dist, self_heating_state[other_key], is_mirror_cable=False
                    )

            # Cooling from mirror cables
            for mirror_key, mirror_cable in self.mirror_cables_with_soil.items():
                dist = cable.distance_to(mirror_cable)
                mutual_heating_effect[key] += _compute_temp_contribution(
                    mirror_cable, dist, self_heating_state[mirror_key], is_mirror_cable=True
                )

        return mutual_heating_effect

    def _update_solution(
        self,
        temp_solution: np.ndarray,
        mutual_heating_temp_solution: np.ndarray,
        ambient_temperature: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Update the internal and mutual heating dicts and combine into dict with full temperature solutions.

        Computes the full temperature solution for a cable at the given timestamp.
        The update is done given the solution (both internal heating and
        mutual heating) and the background temperature. These are added into the full solution.

        Args:
            temp_solution: The internal heating of the cable for the current time step.
            mutual_heating_temp_solution: The external heating of the cable for the current time step.
            ambient_temperature: The background/ambient temperature in degrees Celsius.

        Returns:
            The updated solutions

        """
        solution = temp_solution  # Internal heating
        mutual_heating_solution = mutual_heating_temp_solution  # external heating
        # Calculate the temperature of the cable at the current timestep for all grid points based on the inner heating
        # the external heating and the background temperature:
        full_solution = (
            temp_solution[: mutual_heating_temp_solution.size] + mutual_heating_temp_solution + ambient_temperature
        )
        return solution, mutual_heating_solution, full_solution

    def _update_soil_resistivity_for_all_cables(
        self,
        soil_drying: bool,
        temperature_state: dict,
        soil_resistivity: float,
    ) -> set[CableKey]:
        """Updates soil resistivity for all cables if significantly different or if soil drying is taken into account.

        Args:
            soil_drying: Boolean indicating if the scenario takes into account soil drying
            temperature_state: Full temperature state per cable at the current timestep.
            soil_resistivity: Soil thermal resistivity for the current time step.

        Returns:
            set[CableKey]: Set of cables for which the soil resistivity was updated.

        """
        dry_soil_radii = (
            self._get_dry_soil_radius_for_all_cables(temperature_state=temperature_state) if soil_drying else None
        )

        # If a dynamic series of soil thermal resistivity values is present, update the soil layers
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
        """Update soil thermal capacity if significantly different.

        Args:
            soil_capacity: Soil thermal capacity for the current time step.

        Returns:
            set[CableKey]: Set of cables for which the soil capacity was updated.

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
        """Update soil properties for all cables if significantly different.

        Args:
            soil_drying: Boolean indicating if the scenario takes into account soil drying
            temperature_state: Full temperature state per cable at the current timestep.
            soil_resistivity: Soil thermal resistivity for the current time step.
            soil_capacity: Soil thermal capacity for the current time step.

        Returns:
            set[CableKey]: Set of cables for which the soil properties were updated.

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
            set[CableKey]: Set of cables for which the pipe-fill resistivity was updated.
        """
        updated_cables = super()._update_pipe_resistivity_for_all_cables(temperature_state=temperature_state)

        # Also update cables_with_soil as the pipe fill resistivity is used in the finite-difference matrices of both
        for cable_key in updated_cables:
            cable = self.cables_with_soil[cable_key].cable

            mean_pipe_fill_temp = cable.get_mean_temperature_cable_layer(
                temperature_grid=temperature_state[cable_key],
                layer=CableLayer.PipeFill,
            )
            cable.update_pipe_resistivity(Tfill=mean_pipe_fill_temp)

        return updated_cables

    def get_temp(self, x: float, y: float, time_sec: float, solutions: dict[CableKey, np.ndarray]) -> float:
        """This method computes the temperature at a given point and time in the environment.

        Args:
            x (float): x-coordinate of point
            y (float): y-coordinate of point
            time_sec (float): time in seconds at which to evaluate the temperature
            solutions (dict[CableKey, np.ndarray]): overwrites time_sec and uses this dictionary of
                solutions for the cables to compute the temperature inside the environment.

        Returns:
            float: Temperature in degrees Celsius.

        """
        time_grid = (self.scenario.index - self.scenario.index[0]).total_seconds()
        time_idx = np.nonzero(time_grid >= time_sec)[0][0]
        temp = self.scenario.ambient_temperature[time_idx]
        for key, cable in self.cables_with_soil.items():
            dist = cable.distance_to_point(x=x, y=y)
            temp += _compute_temp_contribution(cable, dist, solutions[key], is_mirror_cable=False)

        for key, mirror_cable in self.mirror_cables_with_soil.items():
            dist = mirror_cable.distance_to_point(x=x, y=y)
            temp += _compute_temp_contribution(mirror_cable, dist, solutions[key], is_mirror_cable=True)

        return temp

    def _check_if_daily_update_due(
        self, seconds_since_start_scenario: float, last_day_with_update: int
    ) -> tuple[bool, int]:
        """Check if a daily update of soil properties is due based on the time elapsed since the start of the scenario.

        Args:
            seconds_since_start_scenario: The number of seconds that have passed since the start of the scenario.
            last_day_with_update: The last day counter indicating when the last update occurred.

        Returns:
            A tuple containing a boolean indicating whether a daily update is due and the updated day counter.

        """
        daily_update_due = False
        days = seconds_since_start_scenario / (60 * 60 * 24)
        if days > last_day_with_update:
            daily_update_due = True
            last_day_with_update = int(days)

        return daily_update_due, last_day_with_update

    def _update_matrix_state(
        self,
        matrix_state: _MatrixState,
        temperature_state: dict[CableKey, np.ndarray],
        soil_resistivity: float,
        soil_capacity: float,
        seconds_since_start_scenario: float,
    ) -> _MatrixState:
        """Update the finite-difference matrices for all cables if soil properties have changed.

        Args:
            matrix_state: The current state of the finite-difference matrices and coupling coefficients for all cables.
            temperature_state: The current temperature state for all cables.
            soil_resistivity: The current soil thermal resistivity.
            soil_capacity: The current soil thermal capacity.
            seconds_since_start_scenario: The number of seconds that have passed since the start of the
                scenario, used to determine if a daily update of soil properties is due.

        Returns:
            Updated _MatrixState containing the finite-difference matrices and coupling coefficients for all cables.
        """
        # Update pipe resistivity if changed significantly based on new temperature_state
        cables_with_updated_pipe_fill = self._update_pipe_resistivity_for_all_cables(
            temperature_state=temperature_state
        )

        daily_update_due, matrix_state.last_day_with_update = self._check_if_daily_update_due(
            seconds_since_start_scenario,
            last_day_with_update=matrix_state.last_day_with_update,
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
            # The matrix of cable_with_soil always needs to be updated if either pipe fill or soil layers are updated
            matrix_state.matrices_with_soil[cable_key] = self.cables_with_soil[
                cable_key
            ].cable.get_finite_difference_matrix()

            if cable_key in cables_with_updated_pipe_fill:
                # If the pipe fill resistivity was updated, the matrix of cable_without_soil also needs to be updated
                matrix_without_soil, outer_boundary_coupling_coefficient = self.cables[
                    cable_key
                ].cable.get_finite_difference_matrix_with_outer_boundary_coupling()
                matrix_state.matrices_without_soil[cable_key] = matrix_without_soil
                matrix_state.outer_boundary_coupling_coefficients[cable_key] = outer_boundary_coupling_coefficient

        return matrix_state

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
        outer_boundary_coupling_coefficients: dict[CableKey, float],
        time_step: float,
    ) -> dict[CableKey, np.ndarray]:
        """Update the mutual heating state for all cables in the environment for a given time step.

        Args:
            self_heating_state: The current self-heating state.
            mutual_heating_state: The current mutual heating state.
            matrices_without_soil: The finite-difference matrices for the cable representations without soil layers.
            outer_boundary_coupling_coefficients: Coefficients that couple the last solved node to the
                known outer boundary temperature per cable representation without soil layers.
            time_step: The time step for the integration.

        Returns:
            Updated mutual heating state.

        """
        # First compute the heating of a cable due to other cables in the environment
        mutual_heating_effect = self._compute_mutual_heating_effect(self_heating_state=self_heating_state)

        new_mutual_heating_state = {}
        for cable_key, cable in self.cables.items():
            # Add the mutual heating to the outermost grid point of the vector
            vector_without_soil = np.zeros(cable.cable.radii_grid.size - 1)
            vector_without_soil[-1] = outer_boundary_coupling_coefficients[cable_key] * mutual_heating_effect[cable_key]

            heat_equation_solution = cable.cable.integrate_timestep(
                s=mutual_heating_state[cable_key][:-1],
                A_banded=matrices_without_soil[cable_key],
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

    def _update_thermal_state(
        self,
        thermal_state: _ThermalState,
        matrix_state: _MatrixState,
        vectors: dict[CableKey, np.ndarray],
        time_step: float,
        ambient_temperature: float,
    ) -> _ThermalState:
        """Update the self-heating state, mutual heating state, and temperature state for the current time step.

        Args:
            thermal_state: The current thermal state containing self-heating and mutual heating states.
            matrix_state: The current matrix state containing the finite difference matrices and coupling coefficients.
            vectors: The vectors for the linear system for each cable.
            time_step: The time step for the integration.
            ambient_temperature: The ambient temperature for the current time step.

        Returns:
            _ThermalState containing the updated temperature state, self-heating state, and mutual heating state.
        """
        new_self_heating_state = self._update_self_heating_state(
            self_heating_state=thermal_state.self_heating,
            matrices=matrix_state.matrices_with_soil,
            vectors=vectors,
            time_step=time_step,
        )

        new_mutual_heating_state = self._update_mutual_heating_state(
            self_heating_state=new_self_heating_state,
            mutual_heating_state=thermal_state.mutual_heating,
            matrices_without_soil=matrix_state.matrices_without_soil,
            outer_boundary_coupling_coefficients=matrix_state.outer_boundary_coupling_coefficients,
            time_step=time_step,
        )

        new_temperature_state = self._update_temperature_state(
            self_heating_state=new_self_heating_state,
            mutual_heating_state=new_mutual_heating_state,
            ambient_temperature=ambient_temperature,
        )

        return self._ThermalState(
            temperature=new_temperature_state,
            self_heating=new_self_heating_state,
            mutual_heating=new_mutual_heating_state,
        )

    def _compute_temperature_solution(
        self,
        initial_state: StateSoil | None = None,
    ) -> ModelOutputSchema[StateSoil]:
        """Computes the temperature solutions for all cable objects.

        Args:
            initial_state: Heating information from a previous computation, if available.

        Returns:
            ModelOutputSchema: Temperature solutions for all cables.

        """
        # Initialize the cables, vectors, matrix state, thermal state, and temperature result
        vector_state = self._initialize_vector_state(self.cables_with_soil)
        matrix_state = self._initialize_matrix_state()
        thermal_state = self._initialize_thermal_state(initial_state=initial_state)
        temperature_result = self._initialize_temperature_result(temperature_state=thermal_state.temperature)

        time_grid = (self.scenario.index - self.scenario.index[0]).total_seconds().to_numpy()
        scenario_rows = self.scenario.iloc[1:].iterrows()

        for step_idx, (_, scenario_row) in enumerate(scenario_rows, start=1):
            time_step = time_grid[step_idx] - time_grid[step_idx - 1]

            # For the current time step, get variables from the scenario dataframe
            ambient_temperature = scenario_row["ambient_temperature"]
            soil_resistivity = scenario_row[self.THERMAL_RESISTIVITY_COLUMN]
            soil_capacity = scenario_row[self.THERMAL_CAPACITY_COLUMN]
            circuit_loads = self._get_circuit_loads_from_scenario_row(scenario_row)

            matrix_state = self._update_matrix_state(
                matrix_state=matrix_state,
                temperature_state=thermal_state.temperature,
                soil_resistivity=soil_resistivity,
                soil_capacity=soil_capacity,
                seconds_since_start_scenario=time_grid[step_idx],
            )

            vector_state = self._update_vector_state(
                vectors=vector_state,
                temperature_state=thermal_state.temperature,
                circuit_loads=circuit_loads,
            )

            thermal_state = self._update_thermal_state(
                thermal_state=thermal_state,
                matrix_state=matrix_state,
                vectors=vector_state,
                time_step=time_step,
                ambient_temperature=ambient_temperature,
            )

            temperature_result = self._update_temperature_result(
                temperature_result=temperature_result,
                temperature_state=thermal_state.temperature,
                step_idx=step_idx,
            )

        # Build the final temperature result dataframe from the temperature_result dict
        temperature_result_df = self._build_temperature_result_dataframe(temperature_result=temperature_result)

        # store heating information of final state
        cable_representations = list(self.static_env.get_cables().values())
        state = StateSoil(
            cable_representations=cable_representations,
            full_solution=thermal_state.temperature,
            internal_heating_solution=thermal_state.self_heating,
            mutual_heating_solutions=thermal_state.mutual_heating,
        )

        # Finalize the calculation by combining the results in the dataclass.
        return ModelOutputSchema[StateSoil](result=temperature_result_df, state=state)

    def _set_run_options(self, run_options: ModelSoilRunOptions | dict | None) -> None:
        """Define run options for ModelSoil.

        Run options that are not provided will be set to their default
        value.
        """
        if run_options is None:
            self.run_options = ModelSoilRunOptions()
        elif isinstance(run_options, ModelSoilRunOptions):
            self.run_options = run_options
        else:
            self.run_options = ModelSoilRunOptions(**run_options)

    def _validate_state_model_consistency(self, state: StateSoil | None):
        """Validate that the provided initial state is consistent with ModelSoil.

        Args:
            state: The state to validate.

        Raises:
            ValueError: If the provided state information does not match the used environment.

        """
        if state is not None and not isinstance(state, StateSoil):
            raise ValueError(
                f"{self.__class__.__name__} requires a {StateSoil.__name__} "
                f"instance, but received {type(state).__name__}."
            )
