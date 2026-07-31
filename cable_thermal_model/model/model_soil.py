# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
from pandera.typing import DataFrame

from cable_thermal_model import CableKey, StaticEnvSoil
from cable_thermal_model.cable.cable_circuit import PosCable, return_mirror_cable
from cable_thermal_model.environment.measurement_point import (
    MeasurementPoint,
    MeasurementPointKey,
)
from cable_thermal_model.model.cables.cable_soil import CableSoil
from cable_thermal_model.model.cables.enum_classes_cable import CableLayer
from cable_thermal_model.model.model import Model
from cable_thermal_model.model.schemas import ScenarioSchemaSoil, StateSoil
from cable_thermal_model.model.schemas.model_input_schemas import THERMAL_CAPACITY_COLUMN, THERMAL_RESISTIVITY_COLUMN
from cable_thermal_model.model.schemas.model_output_schemas import TemperatureResultSchema
from cable_thermal_model.model.schemas.run_options import ModelSoilRunOptions


class ModelSoil(Model[ModelSoilRunOptions, StateSoil, ScenarioSchemaSoil, StaticEnvSoil, CableSoil]):
    """ModelSoil computes temperatures for underground power cables using the finite difference method.

    In most cases the model is instantiated with a StaticEnvSoil and executed with a scenario via `run()`.

    Class Attributes:
        _run_options_class: Run-options schema class.
        _state_class: State schema class.
        _scenario_schema_class: Scenario schema class.

    Internal Runtime State:
        _cables_with_soil: Per-run cable representations extended with soil layers and updated during simulation.
        _mirror_cables_with_soil: Per-run mirrored soil cable representations for image-source calculations.
        _measurement_point_temperature_result: Per-run cache of measurement-point temperatures.

    Configuration Parameters:
        logarithmic_soil_gridpoint_density: Soil-grid point density factor used for discretization.
            Default is 20.
        minimal_soil_radius: Minimum soil radius around each cable in meters. Default is 5.0.
            The effective soil radius is max(minimal_soil_radius, 2.5 * abs(cable depth)).
    """

    _run_options_class = ModelSoilRunOptions
    _state_class = StateSoil
    _scenario_schema_class = ScenarioSchemaSoil

    def __init__(self, static_env: StaticEnvSoil):
        """Initialize the ModelSoil instance with a static environment.

        Args:
            static_env: A StaticEnvSoil instance containing the soil thermal parameters and cable layout.

        """
        if not isinstance(static_env, StaticEnvSoil):
            raise ValueError(
                f"Can not use model {self.__class__.__name__} if static "
                "environment is not an environment in soil. Please use "
                "ModelAir instead."
            )

        # Set up cables
        self._cables_with_soil: dict[CableKey, PosCable[CableSoil]] = {}
        self._mirror_cables_with_soil: dict[CableKey, PosCable[CableSoil]] = {}

        self.logarithmic_soil_gridpoint_density: float = 20
        self.minimal_soil_radius: float = 5.0

        self._measurement_point_temperature_result: dict[MeasurementPointKey, np.ndarray] = {}

        super().__init__(static_env=static_env)

    @property
    def cables_in_environment(self) -> dict[CableKey, PosCable[CableSoil]]:
        """Return per-run soil-extended cable instances for the model.

        Runtime cable properties may deviate from their static baseline defaults. For soil cables, this concerns:
        - updated soil thermal resistivity and capacity from the active scenario (optionally with soil-drying behavior);
        - temperature-dependent pipe-fill resistivity where a pipe layer exists.
        """
        return self._cables_with_soil

    def get_measurement_point_temp(
        self,
        state: StateSoil,
        measurement_point: MeasurementPoint,
    ) -> float:
        """Compute the temperature at a point in the environment given the state.

        Args:
            measurement_point: The measurement point object containing coordinates and distances to cables.
            state: The state of the soil model containing time and self-heating contributions.

        Returns:
            float: Temperature in degrees Celsius.

        """
        measurement_point_temp = state.ambient_temperature

        for cable_key, cable in self._cables_with_soil.items():
            distance_to_cable = measurement_point.distances_to_cables[cable_key]
            measurement_point_temp += cable.cable.get_heating_contribution_at_radius(
                radius=distance_to_cable, self_heating_contribution=state.self_heating_contribution[cable_key]
            )
        for cable_key, mirror_cable in self._mirror_cables_with_soil.items():
            distance_to_mirror_cable = measurement_point.distances_to_mirror_cables[cable_key]
            measurement_point_temp -= mirror_cable.cable.get_heating_contribution_at_radius(
                radius=distance_to_mirror_cable, self_heating_contribution=state.self_heating_contribution[cable_key]
            )

        return measurement_point_temp

    def _prepare_for_run(self, first_scenario_row: pd.Series) -> None:
        """Reset cable state and rebuild the soil-extended representation for the active scenario."""
        super()._prepare_for_run(first_scenario_row=first_scenario_row)

        self._initialize_cables_with_soil(
            soil_rho=first_scenario_row[THERMAL_RESISTIVITY_COLUMN],
            soil_capacity=first_scenario_row[THERMAL_CAPACITY_COLUMN],
        )

        self._mirror_cables_with_soil = {
            key: return_mirror_cable(pos_cable) for key, pos_cable in self._cables_with_soil.items()
        }

        for measurement_point in self.static_env._measurement_point_registry.points:
            measurement_point.distances_to_cables = {
                key: pos_cable.distance_to_point(x=measurement_point.x, y=measurement_point.y)
                for key, pos_cable in self._cables_with_soil.items()
            }
            measurement_point.distances_to_mirror_cables = {
                key: pos_cable.distance_to_point(x=measurement_point.x, y=measurement_point.y)
                for key, pos_cable in self._mirror_cables_with_soil.items()
            }

    def _build_initial_state(self, ambient_temperature: float) -> StateSoil:
        """Builds the initial state for the model.

        Args:
            ambient_temperature: The ambient temperature to initialize the model state.

        Returns:
            StateSoil: An instance of StateSoil containing the initialized temperature,
                            self-heating, and mutual-heating states for each cable.
        """
        return StateSoil(
            static_env_hash=self.static_env.compute_hash(),
            temperature=self._initialize_state_from_cables(cables=self._cables, fill_value=ambient_temperature),
            self_heating_contribution=self._initialize_state_from_cables(cables=self._cables_with_soil),
            mutual_heating_contribution=self._initialize_state_from_cables(cables=self._cables),
            ambient_temperature=ambient_temperature,
        )

    @staticmethod
    def _sum_heating_contributions(
        cables: dict[CableKey, PosCable[CableSoil]],
        self_heating_contribution: dict[CableKey, np.ndarray],
        x: float,
        y: float,
    ) -> float:
        """Sum the heating contributions from a list of cables at a given point in space.

        Args:
            cables: A dictionary of cable keys and their corresponding PosCable instances.
            self_heating_contribution: A dictionary containing the self-heating contributions for each cable.
            x: The x-coordinate of the point in space.
            y: The y-coordinate of the point in space.

        Returns:
            float: The total heating contribution at the specified point from all cables.
        """
        return sum(
            pos_cable.cable.get_heating_contribution_at_radius(
                radius=pos_cable.distance_to_point(x=x, y=y),
                self_heating_contribution=self_heating_contribution[key],
            )
            for key, pos_cable in cables.items()
        )

    def _compute_mutual_heating_effect(
        self,
        self_heating_contribution: dict[CableKey, np.ndarray],
    ) -> dict[CableKey, float]:
        """Compute the heating of a cable due to other cables in the environment.

        These contributions are accumulated per cable and later added to the thermal state.

        Args:
            self_heating_contribution: Self-heating contributions for all cables at a given timestep.

        Returns:
            dict[CableKey, float]: Temperature increases due to mutual heating, one value per cable.

        """
        mutual_heating_effect = dict.fromkeys(self._cables, 0.0)

        for key, pos_cable in self._cables_with_soil.items():
            other_cables = self._cables_with_soil.copy()
            other_cables.pop(key)

            mutual_heating_effect[key] += self._sum_heating_contributions(
                cables=other_cables,
                self_heating_contribution=self_heating_contribution,
                x=pos_cable.x,
                y=pos_cable.y,
            )

            mutual_heating_effect[key] -= self._sum_heating_contributions(
                cables=self._mirror_cables_with_soil,
                self_heating_contribution=self_heating_contribution,
                x=pos_cable.x,
                y=pos_cable.y,
            )

        return mutual_heating_effect

    def _update_soil_properties_for_all_cables(
        self,
        soil_drying: bool,
        temperature_state: dict[CableKey, np.ndarray],
        soil_resistivity: float,
        soil_capacity: float,
    ) -> None:
        """Update soil properties for all cables if needed.

        Args:
            soil_drying: Whether the scenario takes soil drying into account.
            temperature_state: Full temperature state per cable at the current timestep.
            soil_resistivity: Soil thermal resistivity for the current time step.
            soil_capacity: Soil thermal capacity for the current time step.

        """
        for cable_key, pos_cable in self._cables_with_soil.items():
            pos_cable.cable.update_soil_properties(
                soil_rho=soil_resistivity,
                soil_c=soil_capacity,
                temperature_grid=temperature_state[cable_key],
                soil_drying=soil_drying,
            )

    def _update_self_heating_contribution(
        self,
        self_heating_contribution: dict[CableKey, np.ndarray],
        time_step: float,
    ) -> dict[CableKey, np.ndarray]:
        """Update the self-heating contribution for all cables in the environment for a given time step.

        Args:
            self_heating_contribution: The current self-heating contribution.
            time_step: The time step for the integration.

        Returns:
            Updated self-heating contribution.

        """
        # We assume the outer boundary of the soil is at ambient temperature
        solution_at_boundary = 0.0

        new_self_heating_contribution = {}
        for cable_key, pos_cable in self._cables_with_soil.items():
            new_self_heating_contribution[cable_key] = pos_cable.cable.integrate_timestep(
                previous_solution=self_heating_contribution[cable_key],
                time_step=time_step,
                solution_at_boundary=solution_at_boundary,
            )

        return new_self_heating_contribution

    def _update_mutual_heating_contribution(
        self,
        self_heating_contribution: dict[CableKey, np.ndarray],
        mutual_heating_contribution: dict[CableKey, np.ndarray],
        time_step: float,
    ) -> dict[CableKey, np.ndarray]:
        """Update the mutual heating contribution for all cables in the environment for a given time step.

        Args:
            self_heating_contribution: The current self-heating contribution.
            mutual_heating_contribution: The current mutual heating contribution.
            time_step: The time step for the integration.

        Returns:
            Updated mutual heating contribution.

        """
        # First compute the heating of a cable due to other cables in the environment
        mutual_heating_effect = self._compute_mutual_heating_effect(self_heating_contribution=self_heating_contribution)

        new_mutual_heating_contribution = {}
        for cable_key, pos_cable in self._cables.items():
            new_mutual_heating_contribution[cable_key] = pos_cable.cable.integrate_timestep(
                previous_solution=mutual_heating_contribution[cable_key],
                time_step=time_step,
                solution_at_boundary=mutual_heating_effect[cable_key],
            )

        return new_mutual_heating_contribution

    def _update_temperature_state(
        self,
        self_heating_contribution: dict[CableKey, np.ndarray],
        mutual_heating_contribution: dict[CableKey, np.ndarray],
        ambient_temperature: float,
    ) -> dict[CableKey, np.ndarray]:
        """Update the temperature state for all cables by summing the different contributions.

        Args:
            self_heating_contribution: The current self-heating contribution for all cables.
            mutual_heating_contribution: The current mutual heating contribution for all cables.
            ambient_temperature: The ambient temperature for the current time step.

        Returns:
            dict[CableKey, np.ndarray]: Updated temperature state for all cables.
        """
        new_temperature_state = {}
        for cable_key in self._cables:
            mutual_heat = mutual_heating_contribution[cable_key]
            self_heat = self_heating_contribution[cable_key][: mutual_heat.size]
            new_temperature_state[cable_key] = self_heat + mutual_heat + ambient_temperature

        return new_temperature_state

    def _update_thermal_properties_if_needed(
        self,
        temperature_state: dict[CableKey, np.ndarray],
        scenario_row: pd.Series,
    ) -> None:
        """Update pipe-fill resistivity and soil properties if needed.

        Args:
            temperature_state: Current temperature state for all cables.
            scenario_row: Current scenario row.
            elapsed_seconds: Time elapsed since the start of the scenario in seconds.

        """
        self._update_pipe_fill_resistivity(temperature_state=temperature_state, cables=self._cables)
        self._update_pipe_fill_resistivity(temperature_state=temperature_state, cables=self._cables_with_soil)

        soil_resistivity = scenario_row[THERMAL_RESISTIVITY_COLUMN]
        soil_capacity = scenario_row[THERMAL_CAPACITY_COLUMN]

        self._update_soil_properties_for_all_cables(
            soil_drying=self.run_options.soil_drying,
            temperature_state=temperature_state,
            soil_resistivity=soil_resistivity,
            soil_capacity=soil_capacity,
        )

    def _update_state(
        self,
        state: StateSoil,
        ambient_temperature: float,
        time_step: float,
    ) -> StateSoil:
        """Update thermal state for one timestep using extracted step variables."""
        new_self_heating_contribution = self._update_self_heating_contribution(
            self_heating_contribution=state.self_heating_contribution,
            time_step=time_step,
        )

        new_mutual_heating_contribution = self._update_mutual_heating_contribution(
            self_heating_contribution=new_self_heating_contribution,
            mutual_heating_contribution=state.mutual_heating_contribution,
            time_step=time_step,
        )

        new_temperature_state = self._update_temperature_state(
            self_heating_contribution=new_self_heating_contribution,
            mutual_heating_contribution=new_mutual_heating_contribution,
            ambient_temperature=ambient_temperature,
        )

        new_state = StateSoil(
            static_env_hash=state.static_env_hash,
            temperature=new_temperature_state,
            self_heating_contribution=new_self_heating_contribution,
            mutual_heating_contribution=new_mutual_heating_contribution,
            ambient_temperature=ambient_temperature,
        )
        return new_state

    def _initialize_empty_temperature_result(
        self,
        n_scenario_rows: int,
    ) -> dict[CableKey, dict[CableLayer, np.ndarray]]:
        """Initializes an empty nested dictionary.

        Dictionary is used to store temperature results for each cable and each relevant layer.
        """
        temperature_result = super()._initialize_empty_temperature_result(n_scenario_rows=n_scenario_rows)

        self._measurement_point_temperature_result = {
            mp.key: np.full(n_scenario_rows, np.nan, dtype=float)
            for mp in self.static_env._measurement_point_registry.points
        }

        return temperature_result

    def _update_temperature_result(
        self,
        temperature_result: dict[CableKey, dict[CableLayer, np.ndarray]],
        state: StateSoil,
        step_idx: int,
    ) -> None:
        """Update the temperature result dictionary with the current state for a given timestep."""
        super()._update_temperature_result(
            temperature_result=temperature_result,
            state=state,
            step_idx=step_idx,
        )

        for measurement_point in self.static_env._measurement_point_registry.points:
            self._measurement_point_temperature_result[measurement_point.key][step_idx] = (
                self.get_measurement_point_temp(state=state, measurement_point=measurement_point)
            )

    def _build_temperature_result_dataframe(
        self,
        temperature_result: dict[CableKey, dict[CableLayer, np.ndarray]],
        scenario: DataFrame[ScenarioSchemaSoil],
    ) -> DataFrame[TemperatureResultSchema]:
        """Builds a DataFrame from the temperature result dictionary.

        Returns:
            pd.DataFrame: A DataFrame containing the temperature results for all cables and layers.

        """
        df = super()._build_temperature_result_dataframe(
            temperature_result=temperature_result,
            scenario=scenario,
        )

        # Add measurement point temperatures to the DataFrame
        for measurement_point in self.static_env._measurement_point_registry.points:
            df[measurement_point.key] = self._measurement_point_temperature_result[measurement_point.key]

        return df

    def _initialize_cables_with_soil(
        self,
        soil_rho: float,
        soil_capacity: float,
    ) -> None:
        """Build the _cables_with_soil attribute from the existing cables using the provided soil properties.

        Make sure initialize_cables() is called before this method to ensure that the cables are set up correctly.

        Args:
            soil_rho: Soil thermal resistivity for the start of the scenario.
            soil_capacity: Soil thermal capacity for the start of the scenario.

        """
        self._cables_with_soil = {}
        for key, pos_cable in self._cables.items():
            soil_radius = max(self.minimal_soil_radius, 2.5 * abs(pos_cable.y))

            cable_in_soil = pos_cable.cable.from_cable_with_added_soil_layer(
                cable=pos_cable.cable,
                soil_rho=soil_rho,
                soil_capacity=soil_capacity,
                soil_radius=soil_radius,
                logarithmic_soil_gridpoint_density=self.logarithmic_soil_gridpoint_density,
            )

            self._cables_with_soil[key] = PosCable[CableSoil](
                cable=cable_in_soil,
                x=pos_cable.x,
                y=pos_cable.y,
                circuit_name=pos_cable.circuit_name,
                cable_position=pos_cable.cable_position,
            )
