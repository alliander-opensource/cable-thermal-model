# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from copy import deepcopy
from typing import Generic, TypeVar, cast

import numpy as np
import pandas as pd
from pandera.typing import DataFrame

from cable_thermal_model.cable.cable_circuit import CableKey, PosCable
from cable_thermal_model.model.abstract_model import AbstractModel, StaticEnvT
from cable_thermal_model.model.cables.enum_classes_cable import CableLayer
from cable_thermal_model.model.schemas import ModelOutputSchema
from cable_thermal_model.model.schemas.model_input_schemas import ScenarioSchemaT
from cable_thermal_model.model.schemas.model_output_schemas import TemperatureResultSchema
from cable_thermal_model.model.schemas.run_options import ModelRunOptionsT
from cable_thermal_model.model.schemas.state_schemas import StateT

MatricesT = TypeVar("MatricesT")


class Model(
    AbstractModel[ModelRunOptionsT, StateT, ScenarioSchemaT, StaticEnvT],
    Generic[ModelRunOptionsT, StateT, ScenarioSchemaT, StaticEnvT, MatricesT],
):
    """Finite Difference Model for Thermal Cable Model.

    This class implements the finite-difference orchestration shared by concrete models such as ModelAir and
    ModelSoil.
    """

    _run_options_class: type[ModelRunOptionsT]
    _state_class: type[StateT]

    def __init__(self, static_env: StaticEnvT, scenario: DataFrame[ScenarioSchemaT]):
        """Initialize the model with a static environment and a scenario DataFrame.

        Args:
            static_env: The static environment containing cable circuits.
            scenario: The scenario DataFrame describing load conditions over time.

        """
        super().__init__(static_env, scenario)

        self.cables: dict[CableKey, PosCable] = {}
        self._initialize_cables()

        self.extra_solution_layers: list[CableLayer] = []
        self.solution_ = None

    def add_solution_location(self, layer_name: CableLayer):
        """Select an additional solution layer.

        The chosen layer is included in the returned temperature results when calling `run()`.

        Args:
            layer_name: Cable layer to include in the returned results.

        Returns:
            Self: The model instance.

        """
        if not isinstance(layer_name, CableLayer):
            raise TypeError("The layer argument must be of type CableLayer!")
        self.extra_solution_layers.append(layer_name)
        return self

    def _set_run_options(self, run_options: ModelRunOptionsT | dict | None) -> None:
        """Define run options for the model.

        Run options that are not provided are set to their default values.
        """
        if run_options is None:
            self.run_options = self._run_options_class()
        elif isinstance(run_options, self._run_options_class):
            self.run_options = run_options
        elif isinstance(run_options, dict):
            self.run_options = self._run_options_class(**run_options)
        else:
            raise TypeError("run_options must be None, a dict, or an instance of the model run-options schema")

    def _validate_state_model_consistency(self, state: StateT | None):
        """Validate that the provided initial state is consistent with the model type."""
        if state is not None and not isinstance(state, self._state_class):
            raise ValueError(
                f"{self.__class__.__name__} requires a {self._state_class.__name__} "
                f"instance, but received {type(state).__name__}."
            )

    def _initialize_cables(self):
        """Copies the cables as defined in the static_env into the model and initializes cable-related indices.

        This method sets up:
            - The cables dictionary from the static environment.
            - Indices for conductor and screen layers for each cable, using the dict-based CableLayerProperties.
            - A flag indicating whether any cable contains a pipe.
        """
        self.cables = deepcopy(self.static_env.get_cables())
        self.number_of_cables = len(self.cables)

        # Track whether any cable contains a pipe so the run loop can refresh pipe resistivity when needed.
        self.pipes_present = False
        for pos_cable in self.cables.values():
            if pos_cable.cable.layer_metrics.pipe:
                self.pipes_present = True
                break

    def _initialize_thermal_state(
        self,
        initial_state: StateT | None = None,
    ) -> StateT:
        """Initializes the thermal state for the model, either from a provided initial state or by creating a new state.

        Args:
            initial_state: An optional initial state to use for the thermal state.

        Returns:
            StateT: The initialized thermal state for the model.
        """
        if initial_state is not None:
            return initial_state.model_copy(deep=True)

        return self._build_initial_thermal_state()

    def _build_initial_thermal_state(self) -> StateT:
        """Builds the initial thermal state for the model.

        Returns:
            StateT: The initial thermal state for the model.
        """
        raise NotImplementedError("Subclasses of Model must implement _build_initial_thermal_state().")

    def _initialize_linear_system(self) -> tuple[MatricesT, dict[CableKey, np.ndarray]]:
        """Initializes the linear system (matrices and vectors) for each cable in the model.

        Returns:
            A tuple containing the initialized matrices and vectors for each cable.

        """
        raise NotImplementedError("Subclasses of Model must implement _initialize_linear_system().")

    def _build_linear_system_for_cables(
        self, cables: dict[CableKey, PosCable]
    ) -> tuple[dict[CableKey, np.ndarray], dict[CableKey, np.ndarray]]:
        """Initializes the linear system (matrices and vectors) for each cable.

        Args:
            cables: A dictionary of positioned cables for which to initialize the linear system.

        Returns:
            A tuple containing the initialized matrices and vectors for each cable.
        """
        matrices = {}
        vectors = {}

        for cable_key, cable in cables.items():
            matrices[cable_key], vectors[cable_key] = cable.cable.get_linear_system(
                neglect_dielectric_loss=self.run_options.neglect_dielectric_loss
            )

        return matrices, vectors

    def _initialize_state_from_cables(self, cables: dict[CableKey, PosCable]) -> dict[CableKey, np.ndarray]:
        """Initialize a zero-valued state grid for each provided cable.

        Args:
            cables (dict[CableKey, PosCable]): A dictionary of positioned cables.

        Returns:
            dict[CableKey, np.ndarray]: A dictionary with one zero-initialized array per cable,
                sized from each cable radii grid.
        """
        return {cable_key: np.zeros(cable.cable.radii_grid.size) for cable_key, cable in cables.items()}

    def _initialize_temperature_state(self) -> dict[CableKey, np.ndarray]:
        """Initializes the temperature state for each cable based on the ambient temperature from the scenario.

        Returns:
            dict[CableKey, np.ndarray]: A dictionary containing the initialized temperature state for each cable.
        """
        temperature_state = {}

        ambient_temperature = self.scenario["ambient_temperature"].iloc[0]
        for cable_key in self.cables:
            grid_size = self.cables[cable_key].cable.radii_grid.size
            temperature_state[cable_key] = np.full(grid_size, ambient_temperature)

        return temperature_state

    def _initialize_temperature_result(
        self,
        temperature_state: dict[CableKey, np.ndarray],
    ) -> dict[CableKey, dict[CableLayer, np.ndarray]]:
        """Initializes a nested dictionary to store temperature results for each cable and each relevant layer.

        Args:
            temperature_state (dict[CableKey, np.ndarray]):
                A dictionary containing the initial temperature state for each cable, used to initialize the results.

        Returns:
            dict[CableKey, dict[CableLayer, np.ndarray]]:
                Outer dict maps CableKey to an inner dict, which maps
                CableLayer to a numpy array of temperature values over time.

        """
        n_steps = len(self.scenario.index)
        temperature_result: dict[CableKey, dict[CableLayer, np.ndarray]] = {}

        for cable_key, _ in self.cables.items():
            temperature_result[cable_key] = {}
            for layer in [CableLayer.Conductor, CableLayer.Sheath, CableLayer.Pipe] + self.extra_solution_layers:
                if layer in self.cables[cable_key].cable.layers:
                    temperature_result[cable_key][layer] = np.full(n_steps, np.nan, dtype=float)

        # Add initial temperature state to the results
        self._update_temperature_result(
            temperature_result=temperature_result,
            temperature_state=temperature_state,
            step_idx=0,
        )

        return temperature_result

    def _get_circuit_loads_from_scenario_row(self, scenario_row) -> dict[str, float]:
        """Extract circuit loads from a scenario row produced by iterrows()."""
        return {name: scenario_row[f"load_{name}"] for name in self.static_env.circuits}

    def _refresh_matrices_if_needed(
        self,
        matrices: MatricesT,
        temperature_state: dict[CableKey, np.ndarray],
        scenario_row: pd.Series,
        elapsed_seconds: float,
    ) -> tuple[MatricesT, set[CableKey]]:
        """Update cables and refresh matrices in one step.

        Returns:
            tuple[MatricesT, set[CableKey]]: Updated matrices and keys for which matrices were refreshed.
        """
        raise NotImplementedError("Subclasses of Model must implement _refresh_matrices_if_needed().")

    def _update_vectors(
        self,
        vectors: dict[CableKey, np.ndarray],
        temperature_state: dict[CableKey, np.ndarray],
        circuit_loads: dict[str, float],
    ) -> dict[CableKey, np.ndarray]:
        """Updates the vectors (right-hand side) of the linear system for each cable at a given timestep.

        Args:
            vectors (dict[CableKey, np.ndarray]):
                The current vectors for each cable, to be updated in-place.
            temperature_state (dict[CableKey, np.ndarray]):
                The temperature state for each cable at the current timestep.
            circuit_loads (dict[str, float]):
                The load for each circuit at the current timestep.

        Returns:
            dict[CableKey, np.ndarray]:
                The updated vectors for each cable.

        """
        for cable_key, cable in self.cables.items():
            circuit_name = cable_key.circuit_name
            conductor_load = circuit_loads[circuit_name]

            conductor_temperature = cable.cable.get_mean_temperature_cable_layer(
                temperature_grid=temperature_state[cable_key], layer=CableLayer.Conductor
            )

            if CableLayer.Screen in cable.cable.layers and self.run_options.ac_current:
                screen_temperature = cable.cable.get_mean_temperature_cable_layer(
                    temperature_grid=temperature_state[cable_key], layer=CableLayer.Screen
                )

                # Compute the heat that is generated in the conductor and screen
                heat_generation_conductor, heat_generation_screen = (
                    cable.cable.get_heat_generation_conductor_and_screen(
                        ac_current=self.run_options.ac_current,
                        load=conductor_load,
                        conductor_temperature=conductor_temperature,
                        screen_temperature=screen_temperature,
                        temperature_dependent_electric_resistance=self.run_options.temperature_dependent_electric_resistance,
                    )
                )

                vectors[cable_key] = cable.cable.update_vector_with_heat_generation_for_layer(
                    vector=vectors[cable_key],
                    heat_generation=heat_generation_screen,
                    layer=CableLayer.Screen,
                )
            else:
                heat_generation_conductor = cable.cable.get_heat_generation_conductor(
                    ac_current=self.run_options.ac_current,
                    load=conductor_load,
                    conductor_temperature=conductor_temperature,
                    temperature_dependent_electric_resistance=self.run_options.temperature_dependent_electric_resistance,
                )

            # Distribute the heat generation over the conductor and screen layer grid points
            vectors[cable_key] = cable.cable.update_vector_with_heat_generation_for_layer(
                vector=vectors[cable_key],
                heat_generation=heat_generation_conductor,
                layer=CableLayer.Conductor,
            )

        return vectors

    def _update_thermal_state(
        self,
        thermal_state: StateT,
        matrices: MatricesT,
        vectors: dict[CableKey, np.ndarray],
        ambient_temperature: float,
        time_step: float,
    ) -> StateT:
        """Update thermal state for the current timestep.

        Args:
            thermal_state: Current thermal state.
            matrices: Current matrices for each cable.
            vectors: Current vectors for each cable.
            ambient_temperature: Current ambient temperature.
            time_step: Time step for the current iteration.

        Returns:
            Updated thermal state for the current timestep.
        """
        raise NotImplementedError("Subclasses of Model must implement _update_thermal_state().")

    def _update_pipe_resistivity_for_all_cables(
        self,
        temperature_state: dict[CableKey, np.ndarray],
    ) -> set[CableKey]:
        """Update pipe-fill resistivity for all cables based on the current temperature state.

        Args:
            temperature_state: Full temperature state per cable at the current timestep.

        Returns:
            set[CableKey]: Set of cables for which the pipe-fill resistivity was updated.
        """
        updated_cables = set()
        for cable_key, cable in self.cables.items():
            if cable.cable.layer_metrics.pipe is None:
                continue

            mean_pipe_fill_temp = cable.cable.get_mean_temperature_cable_layer(
                temperature_grid=temperature_state[cable_key],
                layer=CableLayer.PipeFill,
            )

            cable_updated = cable.cable.update_pipe_resistivity(Tfill=mean_pipe_fill_temp)
            if cable_updated:
                updated_cables.add(cable_key)

        return updated_cables

    def _update_temperature_result(
        self,
        temperature_result: dict[CableKey, dict[CableLayer, np.ndarray]],
        temperature_state: dict[CableKey, np.ndarray],
        step_idx: int,
    ) -> dict[CableKey, dict[CableLayer, np.ndarray]]:
        """Store one timestep of temperatures in the accumulated result dictionary.

        For layers other than the conductor, sheath, and pipe, the
        temperature is fetched for the center of the layer. The conductor
        temperature is taken from the first grid point of the conductor
        (center or inside for hollow conductors).

        Args:
            temperature_result (dict[CableKey, dict[CableLayer, np.ndarray]]):
                The dictionary to store temperature results for each cable and layer.
            temperature_state (dict[CableKey, np.ndarray]):
                The full temperature solution for each cable at the current timestep.
            step_idx (int):
                The index of the current timestep in the scenario.

        Returns:
            dict[CableKey, dict[CableLayer, np.ndarray]]:
                The updated temperature result dictionary.

        """
        for cable_key, cable in self.cables.items():
            cable_temperatures = temperature_state[cable_key]

            conductor_index_inner = cable.cable.get_layer_indices_for_layer(CableLayer.Conductor)[0]
            sheath_index_outer = cable.cable.get_layer_indices_for_layer(CableLayer.Sheath)[-1]

            temperature_result[cable_key][CableLayer.Conductor][step_idx] = cable_temperatures[conductor_index_inner]
            temperature_result[cable_key][CableLayer.Sheath][step_idx] = cable_temperatures[sheath_index_outer]
            for extra_solution_layer in self.extra_solution_layers:
                if extra_solution_layer in cable.cable.layers:
                    layer_start_index, layer_end_index = cable.cable.get_layer_indices_for_layer(extra_solution_layer)
                    layer_index_center = int((layer_start_index + layer_end_index) / 2)

                    temperature_result[cable_key][extra_solution_layer][step_idx] = cable_temperatures[
                        layer_index_center
                    ]
            if cable.cable.layer_metrics.pipe is not None:
                # Fetch temperature of pipe sheath
                pipe_index_outer = cable.cable.get_layer_indices_for_layer(CableLayer.Pipe)[-1]
                temperature_result[cable_key][CableLayer.Pipe][step_idx] = cable_temperatures[pipe_index_outer]

        return temperature_result

    def _build_temperature_result_dataframe(
        self, temperature_result: dict[CableKey, dict[CableLayer, np.ndarray]]
    ) -> DataFrame[TemperatureResultSchema]:
        """Builds a DataFrame from the temperature results for each cable and layer.

        Args:
            temperature_result (dict[CableKey, dict[CableLayer, np.ndarray]]):
                A nested dictionary containing temperature results for each cable and layer.

        Returns:
            DataFrame[TemperatureResultSchema]:
                A Pandas DataFrame with a MultiIndex of (circuit_name, cable_position, cable_layer) for the columns,
                    containing the temperature results over time.
        """
        temperature_result_dfs = {
            (cable_key.circuit_name, cable_key.cable_position): pd.DataFrame(
                temperature_result[cable_key], index=self.scenario.index
            )
            for cable_key in temperature_result
        }

        combined_temperature_result_df = pd.concat(
            temperature_result_dfs.values(),
            keys=temperature_result_dfs.keys(),
            axis=1,
        )

        return TemperatureResultSchema.validate(combined_temperature_result_df)

    def _compute_temperature_solution(
        self,
        initial_state: StateT | None = None,
    ) -> ModelOutputSchema[StateT]:
        """Compute the temperature solution over the entire scenario.

        Args:
            initial_state: Optional previously computed state to initialize the simulation.

        Returns:
            ModelOutputSchema[StateT]: The computed temperature solution and final thermal state.
        """
        matrices, vectors = self._initialize_linear_system()
        thermal_state = self._initialize_thermal_state(initial_state=initial_state)
        temperature_result = self._initialize_temperature_result(temperature_state=thermal_state.temperature)

        time_grid = (self.scenario.index - self.scenario.index[0]).total_seconds().to_numpy()
        scenario_rows = self.scenario.iloc[1:].iterrows()

        for step_idx, (_, scenario_row) in enumerate(scenario_rows, start=1):
            time_step = time_grid[step_idx] - time_grid[step_idx - 1]

            matrices, _ = self._refresh_matrices_if_needed(
                matrices=matrices,
                temperature_state=thermal_state.temperature,
                scenario_row=scenario_row,
                elapsed_seconds=time_grid[step_idx],
            )

            vectors = self._update_vectors(
                vectors=vectors,
                temperature_state=thermal_state.temperature,
                circuit_loads=self._get_circuit_loads_from_scenario_row(scenario_row),
            )

            thermal_state = self._update_thermal_state(
                thermal_state=thermal_state,
                matrices=matrices,
                vectors=vectors,
                ambient_temperature=scenario_row["ambient_temperature"],
                time_step=time_step,
            )

            temperature_result = self._update_temperature_result(
                temperature_result=temperature_result,
                temperature_state=thermal_state.temperature,
                step_idx=step_idx,
            )

        temperature_result_df = self._build_temperature_result_dataframe(temperature_result=temperature_result)

        output_schema_cls = cast(
            type[ModelOutputSchema[StateT]],
            ModelOutputSchema.__class_getitem__(self._state_class),
        )
        return output_schema_cls(result=temperature_result_df, state=thermal_state)
