# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0


import numpy as np
import pandas as pd
from pandera.typing import DataFrame

from cable_thermal_model.cable.cable_circuit import (
    CableKey,
    PosCable,
    add_soil_layer,
    remove_soil,
    return_mirror_cable,
)
from cable_thermal_model.environment.static_env_soil import StaticEnvSoil
from cable_thermal_model.model.model import Model
from cable_thermal_model.model.schemas import ModelOutputSchema, StateSoil, TemperatureResultSchema
from cable_thermal_model.model.schemas.model_input_schemas import ScenarioSchemaSoil
from cable_thermal_model.model.schemas.run_options import ModelSoilRunOptions


class ModelSoil(Model[ModelSoilRunOptions, StateSoil, ScenarioSchemaSoil]):
    """ModelSoil is used to compute temperature of underground power cables using the finite differences methodology.

    A 1D approach is taken to modelling the environment and the cables, pipes and soil within it. The finite differences
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
        self.mirror_cables_with_soil: dict[CableKey, PosCable] = {}
        self.logarithmic_soil_gridpoint_density: float = 20
        self.minimal_soil_radius: float = 5.0

        super().__init__(static_env=static_env, scenario=scenario)

    def _validate_scenario(self):
        """Validate the scenario DataFrame for required columns.

        Raises:
            ValueError: If required columns are missing from the scenario DataFrame.

        """
        super()._validate_scenario()
        ScenarioSchemaSoil.validate(self.scenario)

    def _initialize_cables(self):
        """This functions copies the cables as defined in the static_env into the model.

        A set of properties, such as pipes, number of cables and conductor indices are
        set as well.
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
                cable,
                soil_rho=self.scenario[self.THERMAL_RESISTIVITY_COLUMN].iloc[0],
                soil_capacity=self.scenario[self.THERMAL_CAPACITY_COLUMN].iloc[0],
                soil_radius=soil_radius,
                logarithmic_soil_gridpoint_density=self.logarithmic_soil_gridpoint_density,
            )

        self.cables = cables_with_soil
        self.cables_full_solutions = {key: remove_soil(pos_cable=pos_cable) for key, pos_cable in self.cables.items()}

        # Create mirror cables with negative temperature solutions in order to enforce T=0 boundary condition on y=0
        self.mirror_cables_with_soil = {key: return_mirror_cable(pos_cable) for key, pos_cable in self.cables.items()}

    def _initialize_solutions_lists_mutual_heating(
        self, initial_state: StateSoil | None = None
    ) -> dict[CableKey, np.ndarray]:
        """Initiate dicts that contain temperature solutions for each cable.

        These are pandas dataframes with:
        dimensions: [timegrid x gridpoints] per cable. Thus: number_cables x [timegrid x gridpoints]
        These dicts will be updated for each timestep the solving loop.
        The following dicts are initiated:
        - mutual_heating_solutions: a dict with temperature solutions of temperature rise inside a cable due to mutual
            heating from other cables.

        Args:
            initial_state: Optional StateSoil object containing mutual_heating_solutions to initialize from.

        Returns:
            dict[CableKey, np.ndarray]: A dictionary containing mutual heating solutions.

        """
        # Initiate a dict with temperature solutions of temperature rise inside a cable due to mutual heating from
        #  other cables
        mutual_heating_solutions = {
            key: np.zeros(pos_cable.cable.radii_grid.size) for key, pos_cable in self.cables_full_solutions.items()
        }

        # If state information was provided it is used to initialize the temperature solutions with
        if initial_state is not None:
            mutual_heating = initial_state.mutual_heating_solutions
            for key in mutual_heating:
                mutual_heating_solutions[key] += mutual_heating[key]

        return mutual_heating_solutions

    def _get_dry_soil_radius_for_all_cables(self, full_solutions: dict[CableKey, np.ndarray]) -> dict[CableKey, float]:
        """This method computes the dry soil radii around the cables in the environment.

        Args:
            full_solutions (dict[CableKey, np.ndarray]):    Full heating solutions for all cables in the environment

        Returns:
             dict[CableKey, float]: A dictionary of radii describing the amount of dried-out soil surrounding each
                                    cable in the same order as [self.cables].

        """
        dry_soil_radii = {}

        # determine the dried-out soil radius per circuit
        for circuit in self.static_env.circuits.values():
            circuit_cables = {cable.name: cable for cable in circuit.cables}
            circuit_solutions = [full_solutions[cable_key] for cable_key in circuit_cables]
            cables_with_soil = [self.cables[cable_key] for cable_key in circuit_cables]

            dry_soil_radius = self._get_dry_soil_radius_around_circuit(
                full_solutions=circuit_solutions, cables=cables_with_soil
            )
            for cable_key in circuit_cables:
                dry_soil_radii[cable_key] = dry_soil_radius

        return dry_soil_radii

    def _get_dry_soil_radius_around_circuit(
        self,
        cables: list[PosCable],
        full_solutions: list[np.ndarray],
    ) -> float:
        """Computes an approximation to the radius of dried out soil surrounding a selected cable.

        This radius is determined through IEC/NPR norms: all soil with a temperature of 30 degrees or more is dried out.

        Notes:
            To improve computability we only use the heating due to the circuit itself to determine what soil is
            drying out. We determine the distance until where the temperature solution exceeds 30 degrees. This
            approach provides an approximate circular radius within which all soil is considered dried-out. Since
            there are multiple cables in the environment the actual shape of dried out soil surrounding a circuit
            is likely different.

        Args:
            cables: list[PosCable]:     Cables corresponding to the configuration
            full_solutions (List[array]):    Internal heating solutions for cables in circuit

        Returns:
            float: The radius within which all soil is dried out around the cable.

        """
        # Base soil drying on one cable in the circuit, which for trefoil is the central 'top' cable
        full_solution = full_solutions[0]
        radii_grid = cables[0].cable.radii_grid

        # Returns the radius of the last grid point where soil is dried out, or radii_grid[0] if none
        idxs = np.nonzero(full_solution >= self._SOIL_DRYING_TEMPERATURE)[0]
        if idxs.size == 0:
            return radii_grid[0]
        return radii_grid[idxs[-1]]

    def _compute_external_temps(self, solutions: dict[CableKey, np.ndarray]) -> dict[CableKey, float]:
        """Computes the heating of a cable due to other cables in the environment.

        These contributions are stored in an array "external_heat" and added as an
        instance variable to the environment cable.

        Args:
            solutions (dict[CableKey, np.ndarray]): The complete dict of solutions at a given timestep corresponding to
                all cables in the environment.

        Returns:
            dict[CableKey, float]: A dictionary of the temperature increases
                (°C) due to mutual heating, one number per cable.

        """
        # dict to hold the external temperatures
        external_temps = dict.fromkeys(self.cables_full_solutions, 0.0)
        # Compute the external heating for all cables
        for key, cable in self.cables_full_solutions.items():
            # Heating from other cables
            for other_key, other_cable in self.cables.items():
                if key != other_key:  # skip self
                    dist = self.compute_distance_between_cables(cable, other_cable)
                    external_temps[key] += self._compute_temp_contribution(
                        other_cable, dist, solutions[other_key], is_mirror_cable=False
                    )

            # Cooling from mirror cables
            for mirror_key, mirror_cable in self.mirror_cables_with_soil.items():
                dist = self.compute_distance_between_cables(cable, mirror_cable)
                external_temps[key] += self._compute_temp_contribution(
                    mirror_cable, dist, solutions[mirror_key], is_mirror_cable=True
                )

        return external_temps

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
        daily_update: bool,
        update_matrices: dict[CableKey, bool],
        soil_drying: bool,
        full_solutions: dict,
        scenario_row: pd.Series,
    ) -> dict[CableKey, bool]:
        """Updates soil resistivity for applicable cables and returns whether linear system matrices should be updated.

        In case of soil drying conditions, any cable with a pipe, or dynamic soil thermal resistivity values,
        the soil resistivity is updated only once a day.

        Args:
            daily_update: Boolean indicating whether a day has passed since the last soil property update.
            update_matrices: Boolean per cable indicating if matrices of the linear system should be updated.
            soil_drying: Boolean indicating if the scenario takes into account soil drying
            full_solutions: Full temperature solutions per cable at the current timestep.
            scenario_row: Current row of the scenario DataFrame.

        Returns:
            Boolean indicator per cable whether to update the matrices.

        """
        if daily_update:
            dry_soil_radii = (
                self._get_dry_soil_radius_for_all_cables(full_solutions=full_solutions) if soil_drying else None
            )

            # If a dynamic series of soil thermal resistivity values is present, update the soil layers
            new_resistivity = scenario_row[self.THERMAL_RESISTIVITY_COLUMN]
            for cable_key, cable in self.cables.items():
                if not (np.isclose(cable.cable.rho_grid[-1], new_resistivity, rtol=1e-2) and dry_soil_radii is None):
                    cable.cable.update_soil_resistivity(
                        soil_rho=new_resistivity, dry_soil_radius=dry_soil_radii[cable_key] if dry_soil_radii else None
                    )
                    update_matrices[cable_key] = True

        return update_matrices

    def _update_soil_capacities_for_all_cables(
        self, daily_update: bool, update_matrices: dict[CableKey, bool], scenario_row: pd.Series
    ) -> dict[CableKey, bool]:
        """Updates soil thermal capacity.

        In case of dynamic soil thermal capacities, updates the soil thermal
        capacity and returns whether the matrix of the linear system should be
        updated.

        Args:
            daily_update: Boolean indicating whether a day has passed since the last soil property update.
            update_matrices: Boolean per cable indicating if matrices of the linear system should be updated.
            scenario_row: Current row of the scenario DataFrame.

        Returns:
            Boolean indicator per cable whether to update the matrices.

        """
        # Update soil thermal capacity for next time step
        if daily_update:
            new_capacity = scenario_row[self.THERMAL_CAPACITY_COLUMN]
            for cable_key, cable in self.cables.items():
                if not np.isclose(cable.cable.capacity_grid[-1], new_capacity, rtol=1e-2):
                    cable.cable.update_soil_capacity(soil_c=new_capacity)
                    update_matrices[cable_key] = True
        return update_matrices

    def _update_soil_properties_for_all_cables(
        self,
        seconds_since_start_scenario: float,
        day_counter: int,
        matrices: dict[CableKey, np.ndarray],
        matrices_without_soil: dict[CableKey, np.ndarray],
        update_matrices: dict[CableKey, bool],
        soil_drying: bool,
        full_solutions: dict[CableKey, np.ndarray],
        scenario_row: pd.Series,
    ) -> tuple[int, dict[CableKey, np.ndarray], dict[CableKey, np.ndarray]]:
        """Updates soil properties for all cables.

        This method calls two other functions to check if the thermal resistivity and thermal
        capacity should be updated. If needed, it updates the matrices of the linear systems.

        Args:
            seconds_since_start_scenario: Seconds passed since the start of the scenario.
            day_counter: Integer indicating how many days have been past.
            matrices: Matrix of the linear system for internal heating
            matrices_without_soil: Matrix of the linear system for external heating
            update_matrices: Boolean per cable indicating if matrices of the linear system should be updated.
            soil_drying: Boolean indicating if the scenario takes into account soil drying
            full_solutions: Full temperature solutions per cable at the current timestep.
            scenario_row: Current row of the scenario DataFrame.

        Returns:
            The possibly updated day counter and the finite-difference
            matrices with new soil thermal resistivity values.

        """
        daily_update = False

        # Check if the soil properties should be updated, typically this should happen only once per day
        days = seconds_since_start_scenario / (60 * 60 * 24)
        if days > day_counter:
            daily_update = True
            day_counter = int(days)

        # Update soil thermal resistivity based on new temperature for next timestep
        update_matrices = self._update_soil_resistivity_for_all_cables(
            daily_update=daily_update,
            update_matrices=update_matrices,
            soil_drying=soil_drying,
            full_solutions=full_solutions,
            scenario_row=scenario_row,
        )
        # Update soil thermal capacities
        update_matrices = self._update_soil_capacities_for_all_cables(
            daily_update=daily_update,
            update_matrices=update_matrices,
            scenario_row=scenario_row,
        )
        for key, update_matrix in update_matrices.items():
            if update_matrix:
                matrices[key] = self.cables[key].cable.get_finite_differences_matrix()
                matrices_without_soil[key] = self.cables_full_solutions[key].cable.get_finite_differences_matrix()
        return day_counter, matrices, matrices_without_soil

    def _compute_temp_contribution(
        self, cable: PosCable, dist: float, solution: np.ndarray, is_mirror_cable: bool
    ) -> float:
        """This method computes the temperature contribution of a cable at a given distance from the cable.

        Args:
            cable:              The cable object
            dist:               The distance from the cable
            solution:           The temperature solution for the cable
            is_mirror_cable:    Boolean indicating if the cable is a mirror cable

        Returns:
            The temperature contribution of the cable at the given distance

        """
        if is_mirror_cable:
            solution = -solution
        return np.interp(x=[dist], xp=cable.cable.radii_grid, fp=solution)[0]

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
        for key, cable in self.cables.items():
            dist = np.sqrt((x - cable.x) ** 2 + (y - cable.y) ** 2)
            temp += self._compute_temp_contribution(cable, dist, solutions[key], is_mirror_cable=False)

        for key, mirror_cable in self.mirror_cables_with_soil.items():
            dist = np.sqrt((x - mirror_cable.x) ** 2 + (y - mirror_cable.y) ** 2)
            temp += self._compute_temp_contribution(mirror_cable, dist, solutions[key], is_mirror_cable=True)

        return temp

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
        temperature_result = self._initialize_temperature_results()

        # Define lists to contain solutions, matrices, vectors, etc per cable
        matrices, vectors = self._initialize_linear_system()
        matrices_without_soil = {
            key: cable.cable.get_finite_differences_matrix() for key, cable in self.cables_full_solutions.items()
        }

        # Initiate lists with solutions per timestep per cable
        solutions, full_solutions = self._initialize_solutions_lists(initial_state=initial_state)
        mutual_heating_solutions = self._initialize_solutions_lists_mutual_heating(initial_state=initial_state)
        previous_scenario_index = self.scenario.index[0]

        temperature_result = self.update_temperature_result(
            temperature_result=temperature_result, full_solutions=full_solutions
        )

        day_counter = 0

        # The following loop solves the heat equation one timestep in the time grid at a time
        for scenario_index, scenario_row in self.scenario.iloc[1:].iterrows():
            # Set time step
            time_step = (scenario_index - previous_scenario_index).total_seconds()
            seconds_since_start_scenario = (scenario_index - self.scenario.index[0]).total_seconds()

            # First update the linear system based on the new state
            vectors = self._update_vectors_per_timestep(
                vectors=vectors,
                full_solutions=full_solutions,
                scenario_row=scenario_row,
            )

            # Compute the internal heating solutions of cables for the new timestep
            temp_solutions = {
                cable_key: self.integrate_timestep(
                    cable=cable,
                    solution=solutions[cable_key][:-1],
                    matrix=matrices[cable_key],
                    vector=vectors[cable_key],
                    time_step=time_step,
                    internal_heating=True,
                )
                for cable_key, cable in self.cables.items()
            }

            # We assume the outer boundary of the soil is at ambient temperature
            for cable_key, temp_solution in temp_solutions.items():
                temp_solutions[cable_key] = np.append(temp_solution, 0.0)
            # Compute the mutual heating solutions of cables for the new timestep
            # First compute the heating of a cable due to other cables in the environment
            external_temps = self._compute_external_temps(solutions=temp_solutions)

            # Enforce heat exchange from outer boundary (and inner boundary if cable has an inner boundary)
            mutual_heating_temp_solutions = {}
            for cable_key, cable in self.cables_full_solutions.items():
                upper_diagonal = self.cables_full_solutions[
                    cable_key
                ].cable._get_finite_differences_matrix_upper_diagonal()
                vector_without_soil = np.zeros(cable.cable.radii_grid.size - 1)
                vector_without_soil[-1] = upper_diagonal[-1] * external_temps[cable_key]

                mutual_heating_temp_solution = self.integrate_timestep(
                    cable=cable,
                    solution=mutual_heating_solutions[cable_key][:-1],
                    matrix=matrices_without_soil[cable_key],
                    vector=vector_without_soil,
                    time_step=time_step,
                    internal_heating=False,
                )
                mutual_heating_temp_solutions[cable_key] = np.append(
                    mutual_heating_temp_solution, external_temps[cable_key]
                )
            # Update the internal and mutual heating lists and combine into list with full temperature solutions for
            #  each cable
            for cable_key, _ in self.cables.items():
                solutions[cable_key], mutual_heating_solutions[cable_key], full_solutions[cable_key] = (
                    self._update_solution(
                        temp_solution=temp_solutions[cable_key],
                        mutual_heating_temp_solution=mutual_heating_temp_solutions[cable_key],
                        ambient_temperature=scenario_row["ambient_temperature"],
                    )
                )

            update_matrices = dict.fromkeys(self.cables, False)
            # Update pipe resistivity
            if self.pipes_present:
                update_matrices = self._update_pipe_resistivity_for_all_cables(
                    full_solutions=full_solutions,
                    update_matrices=update_matrices,
                )
            # Update soil properties
            day_counter, matrices, matrices_without_soil = self._update_soil_properties_for_all_cables(
                seconds_since_start_scenario=seconds_since_start_scenario,
                day_counter=day_counter,
                matrices=matrices,
                matrices_without_soil=matrices_without_soil,
                update_matrices=update_matrices,
                soil_drying=self.run_options.soil_drying,
                full_solutions=full_solutions,
                scenario_row=scenario_row,
            )

            temperature_result = self.update_temperature_result(
                temperature_result=temperature_result, full_solutions=full_solutions
            )

            previous_scenario_index = scenario_index

        temperature_result_dfs = {
            (cable_key.circuit_name, cable_key.cable_position): pd.DataFrame(
                temperature_result[cable_key], index=self.scenario.index
            )
            for cable_key in temperature_result
        }

        # Combine the individual temperature result dataframes into a single
        # dataframe with a MultiIndex of
        # (circuit_name, cable_position, cable_layer) for the columns:
        combined_temperature_result_df = pd.concat(
            temperature_result_dfs.values(), keys=temperature_result_dfs.keys(), axis=1
        )

        # Validate and cast to typed DataFrame
        temperature_result_df = TemperatureResultSchema(combined_temperature_result_df)
        # store heating information of final state
        cable_representations = list(self.static_env.get_cables().values())
        state = StateSoil(
            cable_representations=cable_representations,
            full_solution=full_solutions,
            internal_heating_solution=solutions,
            mutual_heating_solutions=mutual_heating_solutions,
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
