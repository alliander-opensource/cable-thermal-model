# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from typing import cast, overload

from pandera.typing import DataFrame

from cable_thermal_model.environment.static_env import StaticEnv
from cable_thermal_model.environment.static_env_air import StaticEnvAir
from cable_thermal_model.environment.static_env_soil import StaticEnvSoil
from cable_thermal_model.model.model import Model
from cable_thermal_model.model.model_air import ModelAir
from cable_thermal_model.model.model_soil import ModelSoil
from cable_thermal_model.model.schemas.model_input_schemas import (
    ScenarioSchemaAir,
    ScenarioSchemaSoil,
)


class ModelFactory:
    """Factory class for creating model instances based on the environment."""

    @staticmethod
    @overload
    def create_model(static_env: StaticEnvAir, scenario: DataFrame[ScenarioSchemaAir]) -> ModelAir: ...

    @staticmethod
    @overload
    def create_model(static_env: StaticEnvSoil, scenario: DataFrame[ScenarioSchemaSoil]) -> ModelSoil: ...

    @staticmethod
    def create_model(
        static_env: StaticEnv,
        scenario: DataFrame[ScenarioSchemaAir] | DataFrame[ScenarioSchemaSoil],
    ) -> Model:
        """Create a model instance based on the environment type.

        Args:
            static_env (StaticEnv): Static environment configuration for the model.
            scenario (DataFrame[ScenarioSchemaAir] | DataFrame[ScenarioSchemaSoil]):
                Scenario data used by the model.

        Returns:
            Model: An instance of ModelAir or ModelSoil.

        Raises:
            ValueError: If static_env is not a supported environment type.
        """
        if isinstance(static_env, StaticEnvAir):
            return ModelAir(static_env=static_env, scenario=cast(DataFrame[ScenarioSchemaAir], scenario))
        elif isinstance(static_env, StaticEnvSoil):
            return ModelSoil(static_env=static_env, scenario=cast(DataFrame[ScenarioSchemaSoil], scenario))
        else:
            raise ValueError(
                f"Unsupported static environment type: {type(static_env).__name__}. "
                f"Expected {StaticEnvAir.__name__} or {StaticEnvSoil.__name__}."
            )
