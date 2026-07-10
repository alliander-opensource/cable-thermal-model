# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from abc import abstractmethod
from copy import deepcopy
from typing import Generic

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


class Model(
    AbstractModel[ModelRunOptionsT, StateT, ScenarioSchemaT, StaticEnvT],
    Generic[ModelRunOptionsT, StateT, ScenarioSchemaT, StaticEnvT],
):
    """Finite Difference Model for Thermal Cable Model.

    This class implements the finite difference orchestration shared by concrete models such as ModelAir and
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

        self.temperature_result: dict[CableKey, dict[CableLayer, np.ndarray]] = {}

    def add_solution_location(
        self,
        layer_name: CableLayer,
    ) -> None:
        """Select an additional solution layer.

        The chosen layer is included in the returned temperature results when calling `run()`.

        Args:
            layer_name: Cable layer to include in the returned results.

        """
        if not isinstance(layer_name, CableLayer):
            raise TypeError("The layer argument must be of type CableLayer!")

        self.extra_solution_layers.append(layer_name)

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

    @property
    @abstractmethod
    def _cables_for_heat_vectors(self) -> dict[CableKey, PosCable]:
        """Return the cables used to assemble finite difference vectors."""
        pass

    def _initialize_heat_vectors(self) -> dict[CableKey, np.ndarray]:
        """Initialize the heat vectors for each cable.

        Args:
            cables (dict[CableKey, PosCable]): A dictionary of positioned cables.
            neglect_dielectric_loss (bool): Whether to neglect dielectric losses in the initial vectors.

        Returns:
            dict[CableKey, np.ndarray]: A dictionary with one initialized vector per cable,
                sized from each cable finite difference representation.
        """
        return {
            cable_key: pos_cable.cable.get_finite_difference_vector(self.run_options.neglect_dielectric_loss)
            for cable_key, pos_cable in self._cables_for_heat_vectors.items()
        }

    def _initialize_state(
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

        return self._build_initial_state()

    @abstractmethod
    def _build_initial_state(self) -> StateT:
        """Builds the initial thermal state for the model.

        Returns:
            StateT: The initial thermal state for the model.
        """
        pass

    def _initialize_state_from_cables(
        self,
        cables: dict[CableKey, PosCable],
        fill_value: float = 0.0,
    ) -> dict[CableKey, np.ndarray]:
        """Initialize a constant-valued state grid for each provided cable.

        Args:
            cables (dict[CableKey, PosCable]): A dictionary of positioned cables.
            fill_value (float): Value used to fill every cable state grid point. Defaults to 0.0.

        Returns:
            dict[CableKey, np.ndarray]: A dictionary with one initialized array per cable,
                sized from each cable finite difference representation.
        """
        return {
            cable_key: np.full(pos_cable.cable.grid_size, fill_value, dtype=float)
            for cable_key, pos_cable in cables.items()
        }

    def _initialize_temperature_result(
        self,
        state: StateT,
    ) -> None:
        """Initializes a nested dictionary to store temperature results for each cable and each relevant layer.

        Args:
            state (StateT):
                The initial thermal state for each cable, used to initialize the results.

        Returns:
            dict[CableKey, dict[CableLayer, np.ndarray]]:
                Outer dict maps CableKey to an inner dict, which maps
                CableLayer to a numpy array of temperature values over time.

        """
        self._initialize_empty_temperature_result()

        # Add initial temperature state to the results
        self._update_temperature_result(
            state=state,
            step_idx=0,
        )

        return

    def _initialize_empty_temperature_result(self) -> None:
        """Initializes an empty nested dictionary.

        Dictionary is used to store temperature results for each cable and each relevant layer.
        """
        for cable_key, _ in self.cables.items():
            self.temperature_result[cable_key] = {}
            for layer in [CableLayer.Conductor, CableLayer.Sheath, CableLayer.Pipe] + self.extra_solution_layers:
                if layer in self.cables[cable_key].cable.layers:
                    self.temperature_result[cable_key][layer] = np.full(self.scenario_length, np.nan, dtype=float)

    def _get_circuit_loads_from_scenario_row(self, scenario_row) -> dict[str, float]:
        """Extract circuit loads from a scenario row produced by iterrows()."""
        return {name: scenario_row[f"load_{name}"] for name in self.static_env.circuits}

    @abstractmethod
    def _update_thermal_properties_if_needed(
        self,
        temperature_state: dict[CableKey, np.ndarray],
        scenario_row: pd.Series,
        elapsed_seconds: float,
    ) -> None:
        """Update cables and refresh matrices in one step."""
        pass

    def _update_heat_vectors(
        self,
        heat_vectors: dict[CableKey, np.ndarray],
        temperature_state: dict[CableKey, np.ndarray],
        circuit_loads: dict[str, float],
    ) -> dict[CableKey, np.ndarray]:
        """Updates the vectors (right-hand side) of the linear system for each cable at a given timestep.

        Args:
            heat_vectors (dict[CableKey, np.ndarray]):
                The current vectors for each cable, to be updated in-place.
            temperature_state (dict[CableKey, np.ndarray]):
                The temperature state for each cable at the current timestep.
            circuit_loads (dict[str, float]):
                The load for each circuit at the current timestep.

        Returns:
            dict[CableKey, np.ndarray]:
                The updated vectors for each cable.

        """
        for cable_key, pos_cable in self._cables_for_heat_vectors.items():
            circuit_name = cable_key.circuit_name
            conductor_load = circuit_loads[circuit_name]

            heat_vectors[cable_key] = pos_cable.cable.update_finite_difference_vector(
                vector=heat_vectors[cable_key],
                temperature_grid=temperature_state[cable_key],
                load=conductor_load,
                ac_current=self.run_options.ac_current,
                temperature_dependent_electric_resistance=self.run_options.temperature_dependent_electric_resistance,
            )

        return heat_vectors

    @abstractmethod
    def _update_state(
        self,
        state: StateT,
        heat_vectors: dict[CableKey, np.ndarray],
        ambient_temperature: float,
        time_step: float,
    ) -> StateT:
        """Update thermal state for the current timestep.

        Args:
            state: Current state of the model.
            heat_vectors: Current heat vectors for each cable.
            ambient_temperature: Current ambient temperature.
            time_step: Time step for the current iteration.

        Returns:
            Updated state for the current timestep.
        """
        pass

    def _update_pipe_fill_resistivity(
        self,
        temperature_state: dict[CableKey, np.ndarray],
        cables: dict[CableKey, PosCable],
    ) -> None:
        """Update pipe-fill resistivity for given cables based on the current temperature state.

        Args:
            temperature_state: Full temperature state per cable at the current timestep.
            cables: Dictionary of cables to update.
        """
        for cable_key, cable in cables.items():
            if cable.cable.layer_metrics.pipe is None:
                continue

            cable.cable.update_pipe_fill_resistivity(temperature_grid=temperature_state[cable_key])

    def _update_temperature_result(
        self,
        state: StateT,
        step_idx: int,
    ) -> None:
        """Store one timestep of temperatures in the accumulated result dictionary.

        For layers other than the conductor, sheath, and pipe, the
        temperature is fetched for the center of the layer. The conductor
        temperature is taken from the first grid point of the conductor
        (center or inside for hollow conductors).

        Args:
            state (StateT): Current state of the model.
            step_idx (int): The index of the current timestep in the scenario.

        """
        for cable_key, pos_cable in self.cables.items():
            cable = pos_cable.cable
            cable_temperatures = state.temperature[cable_key]

            conductor_index_inner = cable.get_layer_indices_for_layer(CableLayer.Conductor)[0]
            sheath_index_outer = cable.get_layer_indices_for_layer(CableLayer.Sheath)[-1]

            self.temperature_result[cable_key][CableLayer.Conductor][step_idx] = cable_temperatures[
                conductor_index_inner
            ]
            self.temperature_result[cable_key][CableLayer.Sheath][step_idx] = cable_temperatures[sheath_index_outer]

            for extra_solution_layer in self.extra_solution_layers:
                if extra_solution_layer in cable.layers:
                    layer_start_index, layer_end_index = cable.get_layer_indices_for_layer(extra_solution_layer)
                    layer_index_center = int((layer_start_index + layer_end_index) / 2)

                    self.temperature_result[cable_key][extra_solution_layer][step_idx] = cable_temperatures[
                        layer_index_center
                    ]

            if cable.layer_metrics.pipe is not None:
                # Fetch temperature of pipe sheath
                pipe_index_outer = cable.get_layer_indices_for_layer(CableLayer.Pipe)[-1]
                self.temperature_result[cable_key][CableLayer.Pipe][step_idx] = cable_temperatures[pipe_index_outer]

        return

    def _build_temperature_result_dataframe(
        self,
    ) -> DataFrame[TemperatureResultSchema]:
        """Builds a DataFrame from the temperature results for each cable and layer.

        Args:
            temperature_result (dict[CableKey, dict[CableLayer, np.ndarray]]):
                A nested dictionary containing temperature results for each cable and layer.

        Returns:
            DataFrame[TemperatureResultSchema]:
                A Pandas DataFrame with a MultiIndex of (circuit_name, pos_cableition, cable_layer) for the columns,
                    containing the temperature results over time.
        """
        temperature_result_dfs = {
            (cable_key.circuit_name, cable_key.cable_position): pd.DataFrame(
                self.temperature_result[cable_key], index=self.scenario.index
            )
            for cable_key in self.temperature_result
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
        heat_vectors = self._initialize_heat_vectors()
        state = self._initialize_state(initial_state=initial_state)
        self._initialize_temperature_result(state=state)

        time_grid = (self.scenario.index - self.scenario.index[0]).total_seconds().to_numpy()
        scenario_rows = self.scenario.iloc[1:].iterrows()

        for step_idx, (_, scenario_row) in enumerate(scenario_rows, start=1):
            time_step = time_grid[step_idx] - time_grid[step_idx - 1]

            self._update_thermal_properties_if_needed(
                temperature_state=state.temperature,
                scenario_row=scenario_row,
                elapsed_seconds=time_grid[step_idx],
            )

            heat_vectors = self._update_heat_vectors(
                heat_vectors=heat_vectors,
                temperature_state=state.temperature,
                circuit_loads=self._get_circuit_loads_from_scenario_row(scenario_row),
            )

            state = self._update_state(
                state=state,
                heat_vectors=heat_vectors,
                ambient_temperature=scenario_row["ambient_temperature"],
                time_step=time_step,
            )

            self._update_temperature_result(
                state=state,
                step_idx=step_idx,
            )

        temperature_result_df = self._build_temperature_result_dataframe()

        return ModelOutputSchema(result=temperature_result_df, state=state)
