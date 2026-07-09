# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod
from typing import Generic

from pandera.typing import DataFrame

from cable_thermal_model.environment.static_env import StaticEnvT
from cable_thermal_model.model.schemas import ModelOutputSchema
from cable_thermal_model.model.schemas.model_input_schemas import AbstractScenarioSchema, ScenarioSchemaT
from cable_thermal_model.model.schemas.run_options import ModelRunOptionsT
from cable_thermal_model.model.schemas.state_schemas import StateT
from cable_thermal_model.utils.str_utils import tab_lines


class AbstractModel(ABC, Generic[ModelRunOptionsT, StateT, ScenarioSchemaT, StaticEnvT]):
    """Abstract base class for thermal cable models."""

    static_env: StaticEnvT
    scenario: DataFrame[ScenarioSchemaT]

    THERMAL_RESISTIVITY_COLUMN = "soil_thermal_resistivity"
    THERMAL_CAPACITY_COLUMN = "soil_thermal_capacity"

    def __str__(self):
        """Generates a concise string representation of the model."""
        num_circuits = len(self.static_env.circuits)
        num_days = (self.scenario.index[-1] - self.scenario.index[0]).days
        num_days = round(num_days, 1) if num_days < 7 else int(num_days)  # round for readability  # noqa: PLR2004
        return f"Model with {num_circuits} circuit environment and {num_days} day scenario"

    def __repr__(self):
        """Generates an informative string representation of the model."""
        return (
            "Environment\n\n"
            + f"{tab_lines(repr(self.static_env))}\n"
            + "\n\tScenario\n"
            + f"{tab_lines(tab_lines(repr(self.scenario.describe())))}\n"
        )

    def __init__(self, static_env: StaticEnvT, scenario: DataFrame[ScenarioSchemaT]):
        """Initialise the model with a static environment and scenario DataFrame."""
        # Validate that the scenario dataframe provides the required cable loads and ambient temperature.
        self.static_env = static_env
        self._set_scenario(scenario=scenario)
        self._set_run_options(run_options=None)

    def _validate_scenario(self):
        """Validates that the scenario DataFrame contains all required columns and no missing values.

        This method validates the scenario against the AbstractScenarioSchema and also checks that the static
        environment has a matching load column for each circuit.

        Raises:
            ValueError: If the index type is incorrect, required columns are
                missing, or there are missing values in the scenario.

        """
        for circuit in self.static_env.circuits:
            if "load_" + circuit not in self.scenario.columns:
                raise ValueError(f"Scenario dataframe does not contain a load column for circuit '{circuit}'.")

        AbstractScenarioSchema.validate(self.scenario)

    def _set_scenario(self, scenario: DataFrame[ScenarioSchemaT]):
        """Sets a new scenario and validates it.

        Args:
            scenario: The new scenario dataframe

        """
        self.scenario = scenario
        self._validate_scenario()

        # Set up time grids
        self.time_max: float = (scenario.index[-1] - scenario.index[0]).total_seconds()
        self.time_grid: list[float] = list((scenario.index - scenario.index[0]).total_seconds())
        self.time_samples: int = len(self.time_grid)

    def run(
        self,
        initial_state: StateT | None = None,
        run_options: ModelRunOptionsT | dict | None = None,
    ) -> ModelOutputSchema[StateT]:
        """Computes the temperature solutions for all cable objects.

        Notes:
            Be careful about changing default run option values. The following settings affect the
            temperature outcome and should only be changed if you understand the implications:

            - temperature_dependent_electric_resistance
            - ac_current
            - soil_drying
            - initial_state

        Args:
            initial_state: Heating information from a previous computation.
            run_options: Run options for the model. If `None` or a dictionary is provided, the
                options are validated and default values are applied.

        Returns:
            ModelOutputSchema: Temperature solutions for all cables.

        Raises:
            ValueError: If the provided initial state does not match the model environment.

        """
        self._set_run_options(run_options=run_options)

        self._validate_initial_state(initial_state=initial_state)

        # Compute the temperature solution.
        result = self._compute_temperature_solution(
            initial_state=initial_state,
        )

        return result

    @abstractmethod
    def _set_run_options(self, run_options: ModelRunOptionsT | dict | None) -> None:
        self.run_options: ModelRunOptionsT

    @abstractmethod
    def _compute_temperature_solution(
        self,
        initial_state: StateT | None = None,
    ) -> ModelOutputSchema[StateT]:
        """Compute and return the full temperature solution for the configured scenario."""
        raise NotImplementedError("This method should be implemented by subclasses of AbstractModel.")

    @abstractmethod
    def _validate_state_model_consistency(self, state: StateT | None) -> None:
        """Validate that the given state is consistent with the current model configuration."""

    def _validate_initial_state(self, initial_state: StateT | None) -> None:
        """Check the provided initial state against the current static environment.

        Args:
            initial_state: A State object containing runtime solutions and environment metadata.

        Raises:
            ValueError: If the provided state information does not match the used environment.

        """
        self._validate_state_model_consistency(initial_state)

        if initial_state is not None:
            expected_keys = self.static_env.get_cables().keys()
            expected_hash = self.static_env.compute_hash()

            found_keys = initial_state.temperature.keys()
            found_hash = initial_state.static_env_hash

            if found_keys != expected_keys:
                raise ValueError(
                    "Provided state cable keys do not match the used environment. "
                    f"Found keys: {found_keys}, expected keys: {expected_keys}."
                )

            if found_hash != expected_hash:
                raise ValueError("Provided state environment hash does not match the used environment.")
