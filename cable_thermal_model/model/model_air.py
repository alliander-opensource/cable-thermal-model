# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

import warnings

import numpy as np
import pandas as pd
from pandera.typing import DataFrame

from cable_thermal_model.cable.cable_circuit import CableKey
from cable_thermal_model.environment.static_env_air import StaticEnvAir
from cable_thermal_model.model.model import Model
from cable_thermal_model.model.schemas import StateAir
from cable_thermal_model.model.schemas.model_input_schemas import ScenarioSchemaAir
from cable_thermal_model.model.schemas.run_options import ModelAirRunOptions


class ModelAir(Model[ModelAirRunOptions, StateAir, ScenarioSchemaAir, StaticEnvAir, dict[CableKey, np.ndarray]]):
    """ModelAir computes cable temperatures for installations in air using the finite-difference method.

    In most cases the model is instantiated with a StaticEnvAir and a valid scenario, then executed via `run()`.
    """

    _run_options_class = ModelAirRunOptions
    _state_class = StateAir

    def __init__(self, static_env: StaticEnvAir, scenario: DataFrame[ScenarioSchemaAir]):
        """Initialize the ModelAir instance with a static environment and scenario.

        Note: the scenario must contain one `load_<circuit_name>` column per circuit and an
        `ambient_temperature` column.

        Args:
            static_env: A StaticEnvAir instance containing the circuit configuration and cable properties.
            scenario: A pandera DataFrame[ScenarioSchemaAir] containing the dynamic load data and ambient
                temperature.

        """
        if not isinstance(static_env, StaticEnvAir):
            raise ValueError(
                f"Can not use model '{self.__class__.__name__}' if static "
                "environment is not an environment in air. Please use "
                "ModelSoil instead."
            )

        super().__init__(static_env=static_env, scenario=scenario)

    def _validate_scenario(self):
        """Validate the scenario dataframe and warn about unused soil columns."""
        super()._validate_scenario()

        for column in [self.THERMAL_RESISTIVITY_COLUMN, self.THERMAL_CAPACITY_COLUMN]:
            if column in self.scenario.columns:
                warnings.warn(
                    message=f"{column} is provided in the scenario, but is not used in {self.__class__.__name__}",
                    stacklevel=2,
                )

    def _initialize_linear_system(self) -> tuple[dict[CableKey, np.ndarray], dict[CableKey, np.ndarray]]:
        """Initializes the linear system (matrices and vectors) for each cable in the model.

        Returns:
            tuple[dict[CableKey, np.ndarray], dict[CableKey, np.ndarray]]: A tuple containing two dictionaries:
                - The first dictionary maps each cable key to its corresponding finite difference matrix (A).
                - The second dictionary maps each cable key to its corresponding right-hand side vector (b).
        """
        return self._build_linear_system_for_cables(cables=self.cables)

    def _build_initial_thermal_state(self) -> StateAir:
        """Builds the initial thermal state for the model.

        Returns:
            StateAir: The initialized thermal state for the model.
        """
        return StateAir(
            static_env_hash=self.static_env.compute_hash(),
            temperature=self._initialize_temperature_state(),
            self_heating=self._initialize_state_from_cables(cables=self.cables),
        )

    def _refresh_matrices_if_needed(
        self,
        matrices: dict[CableKey, np.ndarray],
        temperature_state: dict[CableKey, np.ndarray],
        scenario_row: pd.Series,
        elapsed_seconds: float,
    ) -> tuple[dict[CableKey, np.ndarray], set[CableKey]]:
        """Refresh matrices when cable properties change.

        Args:
            matrices: Current finite-difference matrices.
            temperature_state: Current temperature state for all cables.
            scenario_row: Current scenario row.
            elapsed_seconds: Time elapsed since the start of the scenario in seconds.

        Notes:
            `scenario_row` and `elapsed_seconds` are accepted for interface compatibility with other model types.

        Returns:
            tuple[dict[CableKey, np.ndarray], set[CableKey]]: Updated matrices and cable keys that were refreshed.
        """
        _ = (scenario_row, elapsed_seconds)  # Unused in this subclass

        updated_cables = self._update_pipe_resistivity_for_all_cables(
            temperature_state=temperature_state,
        )

        for cable_key in updated_cables:
            matrices[cable_key] = self.cables[cable_key].cable.get_finite_difference_matrix()

        return matrices, updated_cables

    def _update_thermal_state(
        self,
        thermal_state: StateAir,
        matrices: dict[CableKey, np.ndarray],
        vectors: dict[CableKey, np.ndarray],
        ambient_temperature: float,
        time_step: float,
    ) -> StateAir:
        """Update the self-heating and temperature state for the current time step."""
        new_self_heating_state = {
            cable_key: cable.cable.integrate_timestep(
                s=thermal_state.self_heating[cable_key],
                A_banded=matrices[cable_key],
                b=vectors[cable_key],
                time_step=time_step,
                internal_heating=True,
            )
            for cable_key, cable in self.cables.items()
        }

        new_temperature_state = {
            cable_key: self_heating + ambient_temperature for cable_key, self_heating in new_self_heating_state.items()
        }

        thermal_state.temperature = new_temperature_state
        thermal_state.self_heating = new_self_heating_state
        return thermal_state
