# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from copy import deepcopy
from typing import Generic

import numpy as np
import pandas as pd
from pandera.typing import DataFrame

from cable_thermal_model.cable.cable_circuit import CableKey, PosCable
from cable_thermal_model.environment.static_env import StaticEnv
from cable_thermal_model.model.abstract_model import AbstractModel
from cable_thermal_model.model.cables.enum_classes_cable import CableLayer
from cable_thermal_model.model.schemas import State
from cable_thermal_model.model.schemas.model_input_schemas import ScenarioSchemaT
from cable_thermal_model.model.schemas.run_options import ModelRunOptionsT
from cable_thermal_model.model.schemas.state_schemas import StateT


class Model(
    AbstractModel[ModelRunOptionsT, StateT, ScenarioSchemaT], Generic[ModelRunOptionsT, StateT, ScenarioSchemaT]
):
    """Finite Difference Model for Thermal Cable Model.

    This class implements a model based on the finite difference (FD) method for simulating
    thermal behavior in cable models. It inherits from the base `AbstractModel` class and uses
    a static environment and a scenario DataFrame to set up the simulation.

    This class contains shared functionality for different Models, such as ModelAir and ModelSoil.
    """

    def __init__(self, static_env: StaticEnv, scenario: DataFrame[ScenarioSchemaT]):
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
        # For the different cables in a circuit, the same thermal behaviour is assumed.
        self.cables = self.static_env.get_cables()
        self.cables_full_solutions = deepcopy(self.cables)
        self.number_of_cables = len(self.cables)

        self.conductor_start_end_indices = {
            cable_key: cable.cable.get_layer_indices_for_layer(CableLayer.Conductor)
            for cable_key, cable in self.cables.items()
        }
        self.screen_start_end_indices = {
            cable_key: cable.cable.get_layer_indices_for_layer(CableLayer.Screen)
            for cable_key, cable in self.cables.items()
            if CableLayer.Screen in cable.cable.layers
        }

        # Check for pipes. This boolean is used in the run-loop to update the resistivity of the pipe
        self.pipes_present = False
        for pos_cable in self.cables.values():
            if pos_cable.cable.layer_metrics.pipe:
                self.pipes_present = True
                break

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

    def _update_vectors_per_timestep(
        self,
        vectors: dict[CableKey, np.ndarray],
        full_solutions: dict[CableKey, np.ndarray],
        scenario_row: pd.Series,
    ) -> dict[CableKey, np.ndarray]:
        """Updates the vectors (right-hand side) of the linear system for each cable at a given timestep.

        Args:
            vectors (dict[CableKey, np.ndarray]):
                The current vectors for each cable, to be updated in-place.
            full_solutions (dict[CableKey, np.ndarray]):
                The temperature solution for each cable at the current timestep.
            temperature_dependent_electric_resistance (bool):
                Whether to use temperature-dependent electric resistance.
            ac_resistance (bool):
                Whether to use AC resistance (skin/proximity effects) or only DC resistance.
            current_in_screen (bool):
                Whether to include induced current in the screen layer.
            scenario_row (pd.Series):
                The scenario data for the current timestep.

        Returns:
            dict[CableKey, np.ndarray]:
                The updated vectors for each cable.

        """
        for cable_key, cable in self.cables_full_solutions.items():
            circuit_name = cable_key.circuit_name
            conductor_load = scenario_row["load_" + circuit_name]

            # Get layer start and end indices
            conductor_start_index, conductor_end_index = self.conductor_start_end_indices[cable_key]

            # Calculate conductor and screen average temperature
            conductor_temperature = (
                full_solutions[cable_key][conductor_start_index] + full_solutions[cable_key][conductor_end_index]
            ) / 2

            if CableLayer.Screen in cable.cable.layers and self.run_options.ac_current:
                screen_start_index, screen_end_index = self.screen_start_end_indices[cable_key]
                screen_temperature = (
                    full_solutions[cable_key][screen_start_index] + full_solutions[cable_key][screen_end_index]
                ) / 2

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

                vectors[cable_key] = cable.cable.update_vector_with_heat_generation(
                    vector=vectors[cable_key],
                    heat_generation=heat_generation_screen,
                    start_index=screen_start_index,
                    end_index=screen_end_index,
                )
            else:
                heat_generation_conductor = cable.cable.get_heat_generation_conductor(
                    ac_current=self.run_options.ac_current,
                    load=conductor_load,
                    conductor_temperature=conductor_temperature,
                    temperature_dependent_electric_resistance=self.run_options.temperature_dependent_electric_resistance,
                )

            # Distribute the heat generation over the conductor and screen layer grid points
            vectors[cable_key] = cable.cable.update_vector_with_heat_generation(
                vector=vectors[cable_key],
                heat_generation=heat_generation_conductor,
                start_index=conductor_start_index,
                end_index=conductor_end_index,
            )

        return vectors

    def _initialize_solutions_lists(
        self, initial_state: State | None = None
    ) -> tuple[dict[CableKey, np.ndarray], dict[CableKey, np.ndarray]]:
        """Initializes solution arrays for each cable for the time integration loop.

        Args:
            initial_state (State | None):
                Optional initial state to initialize the temperature arrays.

        Returns:
            tuple[dict[CableKey, np.ndarray], dict[CableKey, np.ndarray]]:
                (solutions, full_solutions) where each dict maps CableKey to
                a numpy array of temperatures for each grid point.

        """
        # Initiate a dict to contain the temperature solutions due to solely internal Ohmic heating
        solutions = {cable_key: np.zeros(cable.cable.radii_grid.size) for cable_key, cable in self.cables.items()}

        # Initiate a dict with temperature solutions of actual temperature inside a cable i.e. combining internal
        # heating and background ambient temperature
        full_solutions = {
            cable_key: np.zeros(cable.cable.radii_grid.size) for cable_key, cable in self.cables_full_solutions.items()
        }

        # If state information was provided it is used to initialize the temperature solutions with
        if initial_state is not None:
            full_heating = initial_state.full_solution
            internal_heating = initial_state.internal_heating_solution
            for cable_key, full_heating_solution in full_heating.items():
                full_solutions[cable_key] += full_heating_solution
                solutions[cable_key] += internal_heating[cable_key]
        else:
            for cable_key in full_solutions:
                full_solutions[cable_key] += self.scenario["ambient_temperature"].iloc[0]

        return solutions, full_solutions

    def _initialize_temperature_results(
        self,
    ) -> dict[CableKey, dict[CableLayer, list[float]]]:
        """Initializes a nested dictionary to store temperature results for each cable and each relevant layer.

        Returns:
            dict[CableKey, dict[CableLayer, list[float]]]:
                Outer dict maps CableKey to an inner dict, which maps
                CableLayer to a list of temperature values over time.

        """
        temperature_result: dict[CableKey, dict[CableLayer, list[float]]] = {}
        for cable_key, _ in self.cables_full_solutions.items():
            temperature_result[cable_key] = {}
            for layer in [CableLayer.Conductor, CableLayer.Sheath, CableLayer.Pipe] + self.extra_solution_layers:
                if layer in self.cables_full_solutions[cable_key].cable.layers:
                    temperature_result[cable_key][layer] = []
        return temperature_result

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
        full_solutions: dict[CableKey, np.ndarray],
        update_matrices: dict[CableKey, bool],
    ) -> dict[CableKey, bool]:
        """Updates the pipe fill resistivity for all cables that have a pipe, based on the current temperature solution.

        Args:
            full_solutions (dict[CableKey, np.ndarray]):
                The full temperature solution for each cable at the current timestep.
            update_matrices (dict[CableKey, bool]):
                Dictionary to track which cables require their matrices to be updated.

        Returns:
            dict[CableKey, bool]:
                Updated dictionary indicating which cables require matrix updates.

        """
        for key, cable in self.cables_full_solutions.items():
            if cable.cable.layer_metrics.pipe is not None:
                # We want to update the pipe resistivity based on the average temperature of the outer sheath and the
                # inside of the physical pipe. This method is described in TB880 case 0-2.
                pipe_fill_start_index, pipe_fill_end_index = cable.cable.get_layer_indices_for_layer(
                    CableLayer.PipeFill
                )

                cable_sheath_temperature = full_solutions[key][pipe_fill_start_index]
                pipe_inside_temperature = full_solutions[key][pipe_fill_end_index]

                mean_pipe_fill_temp = (cable_sheath_temperature + pipe_inside_temperature) / 2

                cable.cable.update_pipe_resistivity(Tfill=mean_pipe_fill_temp)
                update_matrices[key] = self.cables[key].cable.update_pipe_resistivity(Tfill=mean_pipe_fill_temp)

        return update_matrices

    def update_temperature_result(
        self,
        temperature_result: dict[CableKey, dict[CableLayer, list[float]]],
        full_solutions: dict[CableKey, np.ndarray],
    ) -> dict[CableKey, dict[CableLayer, list[float]]]:
        """Stores the temperature over time for a set of layers in a nested dictionary.

        For layers other than the conductor, sheath, and pipe, the
        temperature is fetched for the center of the layer. The conductor
        temperature is taken from the first grid point of the conductor
        (center or inside for hollow conductors).

        Args:
            temperature_result (dict[CableKey, dict[CableLayer, list[float]]]):
                The dictionary to store temperature results for each cable and layer.
            full_solutions (dict[CableKey, np.ndarray]):
                The full temperature solution for each cable at the current timestep.

        Returns:
            dict[CableKey, dict[CableLayer, list[float]]]:
                The updated temperature result dictionary.

        """
        for cable_key, cable in self.cables_full_solutions.items():
            # Fetch conductor temperatures
            temperature_result[cable_key][CableLayer.Conductor].append(
                full_solutions[cable_key][cable.cable.get_layer_indices_for_layer(CableLayer.Conductor)[0]]
            )
            temperature_result[cable_key][CableLayer.Sheath].append(
                full_solutions[cable_key][cable.cable.get_layer_indices_for_layer(CableLayer.Sheath)[-1]]
            )
            for extra_solution_layer in self.extra_solution_layers:
                if extra_solution_layer in cable.cable.layers:
                    layer_start_index, layer_end_index = cable.cable.get_layer_indices_for_layer(extra_solution_layer)
                    # Get index of the center of the layer:
                    layer_center_index = int((layer_start_index + layer_end_index) / 2)
                    temperature_result[cable_key][extra_solution_layer].append(
                        full_solutions[cable_key][layer_center_index]
                    )
            if cable.cable.layer_metrics.pipe is not None:
                # Fetch temperature of pipe sheath
                temperature_result[cable_key][CableLayer.Pipe].append(
                    full_solutions[cable_key][cable.cable.get_layer_indices_for_layer(CableLayer.Pipe)[-1]]
                )
        return temperature_result
