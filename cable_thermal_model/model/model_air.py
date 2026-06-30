# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

import warnings
from dataclasses import dataclass

import numpy as np
from pandera.typing import DataFrame

from cable_thermal_model.cable.cable_circuit import CableKey
from cable_thermal_model.environment.static_env_air import StaticEnvAir
from cable_thermal_model.model.model import Model
from cable_thermal_model.model.schemas import ModelOutputSchema, StateAir
from cable_thermal_model.model.schemas.model_input_schemas import ScenarioSchemaAir
from cable_thermal_model.model.schemas.run_options import ModelAirRunOptions


class ModelAir(Model[ModelAirRunOptions, StateAir, ScenarioSchemaAir, StaticEnvAir]):
    """The ModelAir is used to compute temperature of using the finite differences methodology.

    In finite differences a 1D approach is taken to modelling the environment
    and the cables, pipes and soil within it. The finite differences
    computation are fast and efficient.

    In most cases the model is used by instantiating it using a StaticEnvAir and a valid scenario and calling the run()
    method.
        >> model = ModelAir(environment, scenario)
        >> result = model.run()
    """

    _run_options_class = ModelAirRunOptions
    _state_class = StateAir

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

    @dataclass
    class _ThermalState:
        """Thermal state tracked during the air-model time simulation.

        Args:
            self_heating: Temperature rise per cable caused by internal heating and air conduction.
            temperature: Full temperature solution per cable, combining self_heating and ambient.

        """

        self_heating: dict[CableKey, np.ndarray]
        temperature: dict[CableKey, np.ndarray]

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

    def _initialize_linear_system(self) -> tuple[dict[CableKey, np.ndarray], dict[CableKey, np.ndarray]]:
        """Initializes the matrices and vectors that define the linear system for each cable.

        Returns:
            tuple: (matrices, vectors) where each is a dict mapping CableKey to np.ndarray for each cable.

        """
        # Define lists to contain solutions, matrices, vectors, etc per cable
        matrices = {}
        vectors = {}

        for cable_key, cable in self.cables.items():
            matrices[cable_key], vectors[cable_key] = cable.cable.get_linear_system(
                neglect_dielectric_loss=self.run_options.neglect_dielectric_loss
            )

        return matrices, vectors

    def _initialize_thermal_state(self, initial_state: StateAir | None) -> _ThermalState:
        """Initialize the thermal state for the model.

        Args:
            initial_state: Optional StateAir object containing temperature and self-heating states.

        Returns:
            _ThermalState: An instance of _ThermalState containing the initialized temperature,
                            self-heating, and mutual-heating states for each cable.
        """
        temperature_state = self._initialize_temperature_state(initial_state=initial_state)
        self_heating_state = self._initialize_self_heating_state(cables=self.cables, initial_state=initial_state)

        return self._ThermalState(
            temperature=temperature_state,
            self_heating=self_heating_state,
        )

    def _update_matrix_state(
        self,
        matrices: dict[CableKey, np.ndarray],
        temperature_state: dict[CableKey, np.ndarray],
    ) -> dict[CableKey, np.ndarray]:
        """Updates the matrices for each cable based on the current temperature state.

        Args:
            matrices: Current matrices for each cable.
            temperature_state: Current temperature state for each cable.

        Returns:
            dict[CableKey, np.ndarray]: Updated matrices for each cable based on current pipe-fill resistivity.
        """
        cables_with_updated_pipe_fill = self._update_pipe_resistivity_for_all_cables(
            temperature_state=temperature_state,
        )

        for cable_key in cables_with_updated_pipe_fill:
            matrices[cable_key] = self.cables[cable_key].cable.get_finite_difference_matrix()

        return matrices

    def _update_self_heating_state(
        self,
        self_heating_state: dict[CableKey, np.ndarray],
        matrices: dict[CableKey, np.ndarray],
        vectors: dict[CableKey, np.ndarray],
        time_step: float,
    ) -> dict[CableKey, np.ndarray]:
        """Updates the self-heating state for each cable based on the current matrices, vectors, and time step.

        Args:
            self_heating_state: Current self-heating state for each cable.
            matrices: Current matrices for each cable.
            vectors: Current vectors for each cable.
            time_step: Time step for the update.

        Returns:
            Updated self-heating state for each cable.
        """
        new_self_heating_state = {}
        for cable_key, cable in self.cables.items():
            new_self_heating_state[cable_key] = cable.cable.integrate_timestep(
                s=self_heating_state[cable_key],
                A_banded=matrices[cable_key],
                b=vectors[cable_key],
                time_step=time_step,
                internal_heating=True,
            )

        return new_self_heating_state

    def _update_temperature_state(
        self,
        self_heating_state: dict[CableKey, np.ndarray],
        ambient_temperature: float,
    ) -> dict[CableKey, np.ndarray]:
        """Updates the temperature state for each cable based on the current self-heating state and ambient temperature.

        Args:
            self_heating_state: Current self-heating state for each cable.
            ambient_temperature: Ambient temperature (in degrees Celsius) for the current time step.

        Returns:
            Updated temperature state for each cable.
        """
        new_temperature_state = {}
        for cable_key in self.cables:
            new_temperature_state[cable_key] = self_heating_state[cable_key] + ambient_temperature

        return new_temperature_state

    def _update_thermal_state(
        self,
        thermal_state: _ThermalState,
        matrix_state: dict[CableKey, np.ndarray],
        vectors: dict[CableKey, np.ndarray],
        time_step: float,
        ambient_temperature: float,
    ) -> _ThermalState:
        """Update the self-heating state and temperature state for the current time step.

        Args:
            thermal_state: The current thermal state containing self-heating states.
            matrix_state: The matrices for the linear system for each cable.
            vectors: The vectors for the linear system for each cable.
            time_step: The time step for the integration.
            ambient_temperature: The ambient temperature for the current time step.

        Returns:
            _ThermalState containing the updated temperature state and self-heating state.
        """
        new_self_heating_state = self._update_self_heating_state(
            self_heating_state=thermal_state.self_heating,
            matrices=matrix_state,
            vectors=vectors,
            time_step=time_step,
        )

        new_temperature_state = self._update_temperature_state(
            self_heating_state=new_self_heating_state,
            ambient_temperature=ambient_temperature,
        )

        return self._ThermalState(
            self_heating=new_self_heating_state,
            temperature=new_temperature_state,
        )

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
        # Initialize the cables, vectors, matrix state, thermal state, and temperature result
        matrix_state, vector_state = self._initialize_linear_system()
        thermal_state = self._initialize_thermal_state(initial_state=initial_state)
        temperature_result = self._initialize_temperature_result(temperature_state=thermal_state.temperature)

        time_grid = (self.scenario.index - self.scenario.index[0]).total_seconds().to_numpy()
        scenario_rows = self.scenario.iloc[1:].iterrows()

        for step_idx, (_, scenario_row) in enumerate(scenario_rows, start=1):
            time_step = time_grid[step_idx] - time_grid[step_idx - 1]

            # For the current time step, get variables from the scenario dataframe
            ambient_temperature = scenario_row["ambient_temperature"]
            circuit_loads = self._get_circuit_loads_from_scenario_row(scenario_row)

            matrix_state = self._update_matrix_state(
                matrices=matrix_state,
                temperature_state=thermal_state.temperature,
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

        temperature_result_df = self._build_temperature_result_dataframe(temperature_result=temperature_result)

        # store heating information of final state
        cable_representations = list(self.static_env.get_cables().values())

        state = StateAir(
            cable_representations=cable_representations,
            full_solution=thermal_state.temperature,
            internal_heating_solution=thermal_state.self_heating,
        )

        # Finalize the calculation by combining the results in the dataclass.
        return ModelOutputSchema[StateAir](result=temperature_result_df, state=state)
