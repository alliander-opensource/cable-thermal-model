# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from typing import overload

import pandas as pd

from cable_thermal_model.environment.static_env import StaticEnv, StaticEnvT
from cable_thermal_model.environment.static_env_air import StaticEnvAir
from cable_thermal_model.environment.static_env_soil import StaticEnvSoil
from cable_thermal_model.model.model import Model
from cable_thermal_model.model.model_air import ModelAir
from cable_thermal_model.model.model_soil import ModelSoil


class ModelFactory:
    """Factory class for creating model instances based on the environment."""

    # Overloaded methods for type checking. Used to infer the return type based on the input static environment type.
    @staticmethod
    @overload
    def create_model(static_env: StaticEnvAir, scenario: pd.DataFrame) -> ModelAir: ...

    @staticmethod
    @overload
    def create_model(static_env: StaticEnvSoil, scenario: pd.DataFrame) -> ModelSoil: ...

    @staticmethod
    @overload
    def create_model(static_env: StaticEnv, scenario: pd.DataFrame) -> Model: ...

    @staticmethod
    def create_model(
        static_env: StaticEnvT,
        scenario: pd.DataFrame,
    ) -> Model:
        """Create a model instance based on the environment type.

        Args:
            static_env (StaticEnvT): Static environment configuration for the model.
            scenario (pd.DataFrame): Scenario data used by the model.

        Returns:
            Model: An instance of ModelAir or ModelSoil, depending on the type of static_env.

        Raises:
            ValueError: If static_env is not a supported environment type.
        """
        if isinstance(static_env, StaticEnvAir):
            return ModelAir(static_env=static_env, scenario=scenario)  # type: ignore
        elif isinstance(static_env, StaticEnvSoil):
            return ModelSoil(static_env=static_env, scenario=scenario)  # type: ignore
        else:
            raise ValueError(
                f"Unsupported static environment type: {type(static_env).__name__}. "
                f"Expected {StaticEnvAir.__name__} or {StaticEnvSoil.__name__}."
            )
