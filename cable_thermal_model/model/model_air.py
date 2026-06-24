# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

import warnings

import numpy as np
import pandas as pd
from pandera.typing import DataFrame

from cable_thermal_model.environment.static_env_air import StaticEnvAir
from cable_thermal_model.model.model import Model
from cable_thermal_model.model.schemas import ModelOutputSchema, StateAir, TemperatureResultSchema
from cable_thermal_model.model.schemas.model_input_schemas import ScenarioSchemaAir
from cable_thermal_model.model.schemas.run_options import ModelAirRunOptions


class ModelAir(Model[ModelAirRunOptions, StateAir, ScenarioSchemaAir]):
    """The ModelAir is used to compute temperature of using the finite differences methodology.

    In finite differences a 1D approach is taken to modelling the environment
    and the cables, pipes and soil within it. The finite differences
    computation are fast and efficient.

    In most cases the model is used by instantiating it using a StaticEnvAir and a valid scenario and calling the run()
    method.
        >> model = ModelAir(environment, scenario)
        >> result = model.run()
    """

    def __init__(self, static_env: StaticEnvAir, scenario: DataFrame[ScenarioSchemaAir]):
        """Initializes the ModelAir instance with a static environment and scenario.

        To initialize a ModelAir instance two inputs are required: a static
        environment and a scenario dataframe.

        N.B. the column names of 'scenario' should be as follows:
        'load_circuit_1' contains the load (in A) of the 'circuit_1' object of
        static_env and column 'ambient_temperature' contains the ambient
        temperature (in degrees Celsius)

        Args:
            static_env: A StaticEnvAir instance containing the circuit configuration and cable properties.
            scenario:   A pandera DataFrame[ScenarioSchemaAir] containing the dynamic data i.e. loads of the
                        cable circuits and the ambient temperature

        """
        if not isinstance(static_env, StaticEnvAir):
            raise ValueError(
                f"Can not use model '{self.__class__.__name__}' if static "
                "environment is not an environment in air. Please use "
                "ModelSoil instead."
            )

        super().__init__(static_env=static_env, scenario=scenario)

    def _validate_scenario(self):
        """Validates the scenario DataFrame.

        Ensures that the scenario contains the required columns
        for the model to operate correctly. Issues warnings if unused columns are present.
        """
        super()._validate_scenario()

        for column in [self.THERMAL_RESISTIVITY_COLUMN, self.THERMAL_CAPACITY_COLUMN]:
            if column in self.scenario.columns:
                warnings.warn(
                    message=f"{column} is provided in the scenario, but is not used in {self.__class__.__name__}",
                    stacklevel=2,
                )

    def _update_solution(
        self,
        temp_solution: np.ndarray,
        ambient_temperature: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update the internal heating list and combine it into the full temperature solution.

        Computes the full temperature solution for a cable at the given time step.
        The update is performed using the computed internal heating
        and the ambient temperature from the scenario.

        Args:
            temp_solution: Array with the internal heating for the current time step.
            ambient_temperature: Ambient temperature (in degrees Celsius) for the current time step.

        Returns:
            A tuple containing the updated solutions.

        """
        solution = temp_solution  # Internal heating
        full_solution = temp_solution + ambient_temperature  # Full solution includes ambient temperature
        return solution, full_solution

    def _compute_temperature_solution(
        self,
        initial_state: StateAir | None = None,
    ) -> ModelOutputSchema[StateAir]:
        """Computes the temperature solutions for all cable objects.

        Args:
            initial_state: Heating information from a previous computation, if available.

        Returns:
            ModelOutputSchema: Temperature solutions for all cables.

        """
        temperature_result = self._initialize_temperature_results()

        # Define dicts to contain solutions, matrices, vectors, etc per cable
        matrices, vectors = self._initialize_linear_system()

        # Initiate dicts with solutions per timestep per cable
        solutions, full_solutions = self._initialize_solutions_lists(initial_state=initial_state)
        previous_scenario_index = self.scenario.index[0]

        temperature_result = self.update_temperature_result(
            temperature_result=temperature_result, full_solutions=full_solutions
        )

        # The following loop solves the heat equation one timestep in the time grid at a time
        for scenario_index, scenario_row in self.scenario.iloc[1:].iterrows():
            # Set time step
            time_step = (scenario_index - previous_scenario_index).total_seconds()

            # First update the linear system based on the new state
            vectors = self._update_vectors_per_timestep(
                vectors=vectors,
                full_solutions=full_solutions,
                scenario_row=scenario_row,
            )

            for cable_key, cable in self.cables.items():
                temp_solution = self.integrate_timestep(
                    cable=cable,
                    solution=solutions[cable_key],
                    matrix=matrices[cable_key],
                    vector=vectors[cable_key],
                    time_step=time_step,
                    internal_heating=True,  # For cable in air we condsider internal heating exclusively
                )

                # Update the internal and ambient temperatures and combine into dict with full temperature solutions for
                #  each cable
                solutions[cable_key], full_solutions[cable_key] = self._update_solution(
                    temp_solution=temp_solution,
                    ambient_temperature=scenario_row["ambient_temperature"],
                )

            # Update pipe resistivity
            if self.pipes_present:
                update_matrices = self._update_pipe_resistivity_for_all_cables(
                    full_solutions=full_solutions,
                    update_matrices=dict.fromkeys(self.cables_full_solutions.keys(), False),
                )

                for cable_key, update_matrix in update_matrices.items():
                    if update_matrix:
                        matrices[cable_key] = self.cables[cable_key].cable.get_finite_differences_matrix()

            temperature_result = self.update_temperature_result(
                temperature_result=temperature_result, full_solutions=full_solutions
            )
            previous_scenario_index = scenario_index

        temperature_result_dfs = {
            # To prevent introducing breaking changes we keep using the tuple (circuit_name, cable_position) as key
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

        state = StateAir(
            cable_representations=cable_representations,
            full_solution=full_solutions,
            internal_heating_solution=solutions,
        )

        # Finalize the calculation by combining the results in the dataclass.
        return ModelOutputSchema[StateAir](result=temperature_result_df, state=state)

    def _set_run_options(self, run_options: ModelAirRunOptions | dict | None) -> None:
        """Define run options for ModelAir.

        Run options that are not provided will be set to their default
        value.
        """
        if run_options is None:
            self.run_options = ModelAirRunOptions()
        elif isinstance(run_options, ModelAirRunOptions):
            self.run_options = run_options
        else:
            self.run_options = ModelAirRunOptions(**run_options)

    def _validate_state_model_consistency(self, state: StateAir | None):
        """Validate that the provided initial state is consistent with ModelAir.

        Args:
            state: The state to validate.

        Raises:
            ValueError: If the provided state information does not match the used environment.

        """
        if state is not None and not isinstance(state, StateAir):
            raise ValueError(
                f"{self.__class__.__name__} requires a {StateAir.__name__} "
                f"instance, but received {type(state).__name__}."
            )
