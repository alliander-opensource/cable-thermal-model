# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from typing import TypeVar

import pandas as pd
import pandera.pandas as pa

THERMAL_RESISTIVITY_COLUMN = "soil_thermal_resistivity"
THERMAL_CAPACITY_COLUMN = "soil_thermal_capacity"


# Input model for scenario dataframe:
class AbstractScenarioModel(pa.DataFrameModel):
    """Base schema for scenario dataframe as used when creating a model.

    Structure:
    - Index: datetime (time series)
    - Columns:
        - load_<circuit_name> (float): load in Amperes for each circuit (e.g., load_circuit1, load_circuit2, etc.)
        - ambient_temperature (float): ambient temperature in degrees Celsius
    """

    ambient_temperature: pa.typing.Series[float]  # Ambient temperature in degrees Celsius

    class Config:
        """Configuration for the schema model."""

        strict = True

    @pa.dataframe_check(error="Scenario index must be either datetime-like or timedelta-like..")
    @classmethod
    def check_datetime_index(cls, df: pd.DataFrame):
        """Ensure index is datetime-like or timedelta-like."""
        return pd.api.types.is_datetime64_any_dtype(df.index) or pd.api.types.is_timedelta64_dtype(df.index)

    @pa.dataframe_check(error="Scenario dataframe must not contain missing values.")
    @classmethod
    def check_no_missing_values(cls, df: pd.DataFrame):
        """Ensure there are no missing values in the scenario dataframe."""
        return not df.isna().any().any()


class ScenarioModelAir(AbstractScenarioModel):
    """Air scenario schema extending the base scenario schema."""


class ScenarioModelSoil(AbstractScenarioModel):
    """Soil scenario schema extending the base scenario schema."""

    soil_thermal_resistivity: pa.typing.Series[float]
    soil_thermal_capacity: pa.typing.Series[float]


ScenarioModelT = TypeVar("ScenarioModelT", bound=AbstractScenarioModel)
