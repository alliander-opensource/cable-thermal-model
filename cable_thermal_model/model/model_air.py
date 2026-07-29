# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd

from cable_thermal_model.cable.cable_circuit import CableKey, PosCable
from cable_thermal_model.environment.static_env_air import StaticEnvAir
from cable_thermal_model.model.cables.cable_air import CableAir
from cable_thermal_model.model.model import Model
from cable_thermal_model.model.schemas import StateAir
from cable_thermal_model.model.schemas.model_input_schemas import ScenarioSchemaAir
from cable_thermal_model.model.schemas.run_options import ModelAirRunOptions


class ModelAir(Model[ModelAirRunOptions, StateAir, ScenarioSchemaAir, StaticEnvAir, CableAir]):
    """ModelAir computes cable temperatures for installations in air using the finite difference method.

    In most cases the model is instantiated with a StaticEnvAir and executed with a scenario via `run()`.
    """

    _run_options_class = ModelAirRunOptions
    _state_class = StateAir
    _scenario_schema_class = ScenarioSchemaAir

    def __init__(self, static_env: StaticEnvAir):
        """Initialize the ModelAir instance with a static environment.

        Args:
            static_env: A StaticEnvAir instance containing the circuit configuration and cable properties.

        """
        if not isinstance(static_env, StaticEnvAir):
            raise ValueError(
                f"Can not use model '{self.__class__.__name__}' if static "
                "environment is not an environment in air. Please use "
                "ModelSoil instead."
            )

        super().__init__(static_env=static_env)

    @property
    def _cables_for_heat_vectors(self) -> dict[CableKey, PosCable[CableAir]]:
        """Return the cables used to assemble finite difference vectors."""
        return self.cables

    def _build_initial_state(self, ambient_temperature: float) -> StateAir:
        """Builds the initial thermal state for the model.

        Args:
            ambient_temperature: The ambient temperature to initialize the model state.

        Returns:
            An instance of StateAir containing the initialized temperature,
                and self-heating states for each cable.
        """
        return StateAir(
            static_env_hash=self.static_env.compute_hash(),
            temperature=self._initialize_state_from_cables(cables=self.cables, fill_value=ambient_temperature),
            self_heating_contribution=self._initialize_state_from_cables(cables=self.cables),
            ambient_temperature=ambient_temperature,
        )

    def _update_thermal_properties_if_needed(
        self,
        temperature_state: dict[CableKey, np.ndarray],
        scenario_row: pd.Series,
    ) -> None:
        """Update the pipe-fill resistivity if changed.

        Args:
            temperature_state: Current temperature state for all cables.
            scenario_row: Current scenario row.

        Notes:
            `scenario_row` is accepted for interface compatibility with other model types.
        """
        _ = scenario_row  # Unused in this subclass

        self._update_pipe_fill_resistivity(
            temperature_state=temperature_state,
            cables=self.cables,
        )

    def _update_state(
        self,
        state: StateAir,
        ambient_temperature: float,
        time_step: float,
    ) -> StateAir:
        """Update the self-heating and temperature state for the current time step."""
        new_self_heating_contribution = {
            cable_key: pos_cable.cable.integrate_timestep(
                previous_solution=state.self_heating_contribution[cable_key],
                time_step=time_step,
            )
            for cable_key, pos_cable in self.cables.items()
        }

        new_temperature_state = {
            cable_key: self_heating + ambient_temperature
            for cable_key, self_heating in new_self_heating_contribution.items()
        }

        new_state = StateAir(
            static_env_hash=state.static_env_hash,
            temperature=new_temperature_state,
            self_heating_contribution=new_self_heating_contribution,
            ambient_temperature=ambient_temperature,
        )
        return new_state
