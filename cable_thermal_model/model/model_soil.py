# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0


from copy import deepcopy

import numpy as np
import pandas as pd
from pandera.typing import DataFrame

from cable_thermal_model.cable.cable_circuit import (
    CableKey,
    PosCable,
    add_soil_layer,
    return_mirror_cable,
)
from cable_thermal_model.environment.static_env_soil import StaticEnvSoil
from cable_thermal_model.model.model import Model
from cable_thermal_model.model.schemas import StateSoil
from cable_thermal_model.model.schemas.model_input_schemas import ScenarioSchemaSoil
from cable_thermal_model.model.schemas.run_options import ModelSoilRunOptions


class ModelSoil(Model[ModelSoilRunOptions, StateSoil, ScenarioSchemaSoil, StaticEnvSoil]):
    """ModelSoil computes temperatures for underground power cables using the finite difference method.

    In most cases the model is instantiated with a StaticEnvSoil and a valid scenario, then executed via `run()`.
    """

    _run_options_class = ModelSoilRunOptions
    _state_class = StateSoil

    def __init__(self, static_env: StaticEnvSoil, scenario: DataFrame[ScenarioSchemaSoil]):
        """Initialize the ModelSoil instance with a static environment and scenario.

        Note: the scenario must contain one `load_<circuit_name>` column per circuit, plus ambient temperature and
        soil-property columns.

        Args:
            static_env: A StaticEnvSoil instance containing the soil thermal parameters and cable layout.
            scenario: A pandera DataFrame[ScenarioSchemaSoil] containing the dynamic load and soil data.

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
        self.cables_with_soil: dict[CableKey, PosCable] = {}
        self.mirror_cables_with_soil: dict[CableKey, PosCable] = {}
        self.logarithmic_soil_gridpoint_density: float = 20
        self.minimal_soil_radius: float = 5.0
        self.last_soil_property_update_day: int = 0

        super().__init__(static_env=static_env, scenario=scenario)

    def _validate_scenario(self):
        """Validate the scenario dataframe for required columns.

        Raises:
            ValueError: If required columns are missing from the scenario dataframe.

        """
        super()._validate_scenario()
        ScenarioSchemaSoil.validate(self.scenario)

    def _initialize_cables(self):
        """Initialize cables with soil layers and mirror cables for the boundary condition."""
        # Start from the static cables without soil, then add a soil layer per cable.
        # Deep cables may get an extra outer soil layer to extend the domain.
        # The outer boundary is treated as ambient.

        super()._initialize_cables()
        cables_with_soil = {}

        for key, pos_cable in self.cables.items():
            soil_radius = max(self.minimal_soil_radius, 2.5 * abs(pos_cable.y))

            # Instantiate FDCable objects with the added soil layer.
            cables_with_soil[key] = add_soil_layer(
                deepcopy(pos_cable),
                soil_rho=self.scenario[self.THERMAL_RESISTIVITY_COLUMN].iloc[0],
                soil_capacity=self.scenario[self.THERMAL_CAPACITY_COLUMN].iloc[0],
                soil_radius=soil_radius,
                logarithmic_soil_gridpoint_density=self.logarithmic_soil_gridpoint_density,
            )

        self.cables_with_soil = cables_with_soil

        # Create mirror cables to enforce the T=0 boundary condition on y=0.
        self.mirror_cables_with_soil = {
            key: return_mirror_cable(pos_cable) for key, pos_cable in self.cables_with_soil.items()
        }

    def _get_vector_cables(self) -> dict[CableKey, PosCable]:
        """Return the soil-extended cables used to assemble finite-difference vectors."""
        return self.cables_with_soil

    def _build_initial_thermal_state(self) -> StateSoil:
        """Builds the initial thermal state for the model.

        Returns:
            StateSoil: An instance of StateSoil containing the initialized temperature,
                            self-heating, and mutual-heating states for each cable.
        """
        ambient_temperature = self.scenario["ambient_temperature"].iloc[0]

        return StateSoil(
            static_env_hash=self.static_env.compute_hash(),
            temperature=self._initialize_state_from_cables(cables=self.cables, fill_value=ambient_temperature),
            self_heating_contribution=self._initialize_state_from_cables(cables=self.cables_with_soil),
            mutual_heating_contribution=self._initialize_state_from_cables(cables=self.cables),
        )

    def _sum_heating_contributions(
        self,
        cables: dict[CableKey, PosCable],
        self_heating_contribution: dict[CableKey, np.ndarray],
        x: float,
        y: float,
    ) -> float:
        """Sum the heating contributions from a list of cables at a given point in space.

        Args:
            cables: A dictionary of cables to consider for heating contributions.
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

    def get_temp(
        self, x: float, y: float, time_sec: float, self_heating_contribution: dict[CableKey, np.ndarray]
    ) -> float:
        """Compute the temperature at a point and time in the environment.

        Args:
            x: x-coordinate of the point.
            y: y-coordinate of the point.
            time_sec: Time in seconds at which to evaluate the temperature.
            self_heating_contribution: Self-heating contributions for each cable at the specified time.

        Returns:
            float: Temperature in degrees Celsius.

        """
        time_grid = (self.scenario.index - self.scenario.index[0]).total_seconds()
        time_idx = np.nonzero(time_grid >= time_sec)[0][0]
        ambient_temperature = self.scenario["ambient_temperature"].iloc[time_idx]

        heating_from_cables = self._sum_heating_contributions(
            cables=self.cables_with_soil,
            self_heating_contribution=self_heating_contribution,
            x=x,
            y=y,
        )

        cooling_from_mirror_cables = self._sum_heating_contributions(
            cables=self.mirror_cables_with_soil,
            self_heating_contribution=self_heating_contribution,
            x=x,
            y=y,
        )

        return ambient_temperature + heating_from_cables - cooling_from_mirror_cables

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
        mutual_heating_effect = dict.fromkeys(self.cables, 0.0)

        for key, pos_cable in self.cables_with_soil.items():
            other_cables = {k: v for k, v in self.cables_with_soil.items() if k != key}

            heating_from_other_cables = self._sum_heating_contributions(
                cables=other_cables,
                self_heating_contribution=self_heating_contribution,
                x=pos_cable.x,
                y=pos_cable.y,
            )

            cooling_from_mirror_cables = self._sum_heating_contributions(
                cables=self.mirror_cables_with_soil,
                self_heating_contribution=self_heating_contribution,
                x=pos_cable.x,
                y=pos_cable.y,
            )

            mutual_heating_effect[key] = heating_from_other_cables - cooling_from_mirror_cables

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
        for cable_key, pos_cable in self.cables_with_soil.items():
            pos_cable.cable.update_soil_properties(
                soil_rho=soil_resistivity,
                soil_c=soil_capacity,
                temperature_grid=temperature_state[cable_key],
                soil_drying=soil_drying,
            )

    def _check_if_daily_update_due(
        self, seconds_since_start_scenario: float, last_soil_property_update_day: int
    ) -> tuple[bool, int]:
        """Check if a daily update of soil properties is due based on the time elapsed since the start of the scenario.

        Args:
            seconds_since_start_scenario: The number of seconds that have passed since the start of the scenario.
            last_soil_property_update_day: Day counter indicating when the last soil-property update occurred.

        Returns:
            A tuple containing a boolean indicating whether a daily update is due and the updated day counter.

        """
        daily_update_due = False
        days = seconds_since_start_scenario / (60 * 60 * 24)
        if days > last_soil_property_update_day:
            daily_update_due = True
            last_soil_property_update_day = int(days)

        return daily_update_due, last_soil_property_update_day

    def _update_self_heating_contribution(
        self,
        self_heating_contribution: dict[CableKey, np.ndarray],
        vectors: dict[CableKey, np.ndarray],
        time_step: float,
    ) -> dict[CableKey, np.ndarray]:
        """Update the self-heating contribution for all cables in the environment for a given time step.

        Args:
            self_heating_contribution: The current self-heating contribution.
            vectors: The vectors for the linear system.
            time_step: The time step for the integration.

        Returns:
            Updated self-heating contribution.

        """
        new_self_heating_contribution = {}
        for cable_key, pos_cable in self.cables_with_soil.items():
            heat_equation_solution = pos_cable.cable.integrate_timestep(
                s=self_heating_contribution[cable_key][:-1],
                b=vectors[cable_key],
                time_step=time_step,
                internal_heating=True,
            )
            # We assume the outer boundary of the soil is at ambient temperature
            new_self_heating_contribution[cable_key] = np.append(heat_equation_solution, 0.0)

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
        for cable_key, pos_cable in self.cables.items():
            cable = pos_cable.cable
            outer_boundary_coupling_coefficient = cable.outer_boundary_coupling_coefficient

            # Add the mutual heating to the outermost grid point of the vector
            vector_without_soil = np.zeros(cable.grid_size - 1)
            vector_without_soil[-1] = outer_boundary_coupling_coefficient * mutual_heating_effect[cable_key]

            heat_equation_solution = cable.integrate_timestep(
                s=mutual_heating_contribution[cable_key][:-1],
                b=vector_without_soil,
                time_step=time_step,
                internal_heating=False,
            )
            new_mutual_heating_contribution[cable_key] = np.append(
                heat_equation_solution, mutual_heating_effect[cable_key]
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
        for cable_key in self.cables:
            mutual_heating_cable_state = mutual_heating_contribution[cable_key]
            self_heating_cable_state = self_heating_contribution[cable_key][: mutual_heating_cable_state.size]

            new_temperature_state[cable_key] = (
                self_heating_cable_state + mutual_heating_cable_state + ambient_temperature
            )

        return new_temperature_state

    def _update_thermal_properties_if_needed(
        self,
        temperature_state: dict[CableKey, np.ndarray],
        scenario_row: pd.Series,
        elapsed_seconds: float,
    ) -> None:
        """Update pipe-fill resistivity and soil properties if needed.

        Args:
            temperature_state: Current temperature state for all cables.
            scenario_row: Current scenario row.
            elapsed_seconds: Time elapsed since the start of the scenario in seconds.

        """
        soil_resistivity = scenario_row[self.THERMAL_RESISTIVITY_COLUMN]
        soil_capacity = scenario_row[self.THERMAL_CAPACITY_COLUMN]

        self._update_pipe_fill_resistivity(temperature_state=temperature_state, cables=self.cables)
        self._update_pipe_fill_resistivity(temperature_state=temperature_state, cables=self.cables_with_soil)

        daily_update_due, self.last_soil_property_update_day = self._check_if_daily_update_due(
            seconds_since_start_scenario=elapsed_seconds,
            last_soil_property_update_day=self.last_soil_property_update_day,
        )

        if daily_update_due:
            self._update_soil_properties_for_all_cables(
                soil_drying=self.run_options.soil_drying,
                temperature_state=temperature_state,
                soil_resistivity=soil_resistivity,
                soil_capacity=soil_capacity,
            )

    def _update_thermal_state(
        self,
        thermal_state: StateSoil,
        heat_vectors: dict[CableKey, np.ndarray],
        ambient_temperature: float,
        time_step: float,
    ) -> StateSoil:
        """Update thermal state for one timestep using extracted step variables."""
        new_self_heating_contribution = self._update_self_heating_contribution(
            self_heating_contribution=thermal_state.self_heating_contribution,
            vectors=heat_vectors,
            time_step=time_step,
        )

        new_mutual_heating_contribution = self._update_mutual_heating_contribution(
            self_heating_contribution=new_self_heating_contribution,
            mutual_heating_contribution=thermal_state.mutual_heating_contribution,
            time_step=time_step,
        )

        new_temperature_state = self._update_temperature_state(
            self_heating_contribution=new_self_heating_contribution,
            mutual_heating_contribution=new_mutual_heating_contribution,
            ambient_temperature=ambient_temperature,
        )

        thermal_state.temperature = new_temperature_state
        thermal_state.self_heating_contribution = new_self_heating_contribution
        thermal_state.mutual_heating_contribution = new_mutual_heating_contribution

        return thermal_state
