# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from copy import deepcopy
from typing import Generic

import numpy as np
import pandas as pd
from pandera.typing import DataFrame

from cable_thermal_model.cable.cable_circuit import CableKey, PosCable
from cable_thermal_model.model.abstract_model import AbstractModel, StaticEnvT
from cable_thermal_model.model.cables.enum_classes_cable import CableLayer
from cable_thermal_model.model.schemas.model_input_schemas import ScenarioSchemaT
from cable_thermal_model.model.schemas.model_output_schemas import TemperatureResultSchema
from cable_thermal_model.model.schemas.run_options import ModelRunOptionsT
from cable_thermal_model.model.schemas.state_schemas import StateT


class Model(
    AbstractModel[ModelRunOptionsT, StateT, ScenarioSchemaT, StaticEnvT],
    Generic[ModelRunOptionsT, StateT, ScenarioSchemaT, StaticEnvT],
):
    """Finite Difference Model for Thermal Cable Model.

    This class implements a model based on the finite difference (FD) method for simulating
    thermal behavior in cable models. It inherits from the base `AbstractModel` class and uses
    a static environment and a scenario DataFrame to set up the simulation.

    This class contains shared functionality for different Models, such as ModelAir and ModelSoil.
    """

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
        """Selects additional solution layer.

        This method is used to select a cable layer for which the temperature solution will also be returned when
        using the run method. Used to add, for example, the insulation layer as solution location.

        Args:
            layer_name: The name of a cable layer, the temperatures of added layers will be returned in the
                        ModelOutputSchema under the layer name
        Returns:
            self

        """
        if not isinstance(layer_name, CableLayer):
            raise TypeError("The layer argument must be of type CableLayer!")
        self.extra_solution_layers.append(layer_name)
        return self

    def _initialize_cables(self):
        """Copies the cables as defined in the static_env into the model and initializes cable-related indices.

        This method sets up:
            - The cables dictionary from the static environment.
            - Indices for conductor and screen layers for each cable, using the dict-based CableLayerProperties.
            - A flag for the presence of pipes in any cable.
        """
        self.cables = deepcopy(self.static_env.get_cables())
        self.number_of_cables = len(self.cables)

        # Check for pipes. This boolean is used in the run-loop to update the resistivity of the pipe
        self.pipes_present = False
        for pos_cable in self.cables.values():
            if pos_cable.cable.layer_metrics.pipe:
                self.pipes_present = True
                break

    def _initialize_vector_state(self, cables: dict[CableKey, PosCable]) -> dict[CableKey, np.ndarray]:
        """Initialize the vectors for the linear system for each cable.

        Args:
            cables (dict[CableKey, PosCable]): A dictionary of positioned cables.

        Returns:
            dict[CableKey, np.ndarray]: A dictionary containing the initialized vector for each cable.
        """
        vectors = {}

        for cable_key, cable in cables.items():
            vectors[cable_key] = cable.cable.get_finite_difference_vector(
                neglect_dielectric_loss=self.run_options.neglect_dielectric_loss
            )

        return vectors

    def _initialize_self_heating_state(
        self, cables: dict[CableKey, PosCable], initial_state: StateT | None
    ) -> dict[CableKey, np.ndarray]:
        self_heating_state = {cable_key: np.zeros(cable.cable.radii_grid.size) for cable_key, cable in cables.items()}

        if initial_state is not None:
            for cable_key, internal_heating_solution in initial_state.internal_heating_solution.items():
                self_heating_state[cable_key] += internal_heating_solution

        return self_heating_state

    def _initialize_temperature_state(self, initial_state: StateT | None) -> dict[CableKey, np.ndarray]:
        temperature_state = {}

        if initial_state is None:
            # If no initial state is provided, initialize temperature_state with the ambient temperature for each cable
            ambient_temperature = self.scenario["ambient_temperature"].iloc[0]
            for cable_key in self.cables:
                grid_size = self.cables[cable_key].cable.radii_grid.size
                temperature_state[cable_key] = np.full(grid_size, ambient_temperature)
        else:
            # If an initial state is provided, initialize temperature_state with the initial_state
            for cable_key, full_heating_solution in initial_state.full_solution.items():
                temperature_state[cable_key] = full_heating_solution

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

    def _update_vector_state(
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
            temperature_result_dfs.values(), keys=temperature_result_dfs.keys(), axis=1
        )

        return TemperatureResultSchema.validate(combined_temperature_result_df)

    def integrate_timestep(
        self,
        cable: PosCable,
        solution: np.ndarray,
        matrix: np.ndarray,
        vector: np.ndarray,
        time_step: float,
        internal_heating: bool | None = None,
    ) -> np.ndarray:
        """Computes the temperature solution for the next timestep using the finite-difference matrix and vector.

        Args:
            cable (PosCable):
                The cable object for which to compute the new temperature solution.
            solution (np.ndarray):
                The temperature solution at the previous timestep.
            matrix (np.ndarray):
                The finite-difference matrix.
            vector (np.ndarray):
                The finite-difference vector.
            time_step (float):
                Duration of the current time step in seconds.
            internal_heating (bool | None):
                Whether to compute internal heating. Defaults to None.

        Returns:
            np.ndarray: The temperature solution at the next timestep.

        """
        return cable.cable.integrate_timestep(
            solution, matrix, vector, time_step=time_step, internal_heating=internal_heating
        )

    @staticmethod
    def compute_distance_between_cables(cable: PosCable, other_cable: PosCable) -> float:
        """Compute the heart-to-heart distance (m) between two cables.

        Args:
            cable:          Positioned cable object
            other_cable:    Second positioned cable object

        Returns:
            float: Distance between two cable objects in meters.

        """
        return np.sqrt((cable.x - other_cable.x) ** 2 + (cable.y - other_cable.y) ** 2)

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
