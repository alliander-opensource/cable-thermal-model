# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from typing import TYPE_CHECKING, TypeVar

import pandas as pd
import pandera.pandas as pa

if TYPE_CHECKING:
    from cable_thermal_model.cable_configuration.cable_circuit import CableKey

THERMAL_RESISTIVITY_COLUMN = "soil_thermal_resistivity"
THERMAL_CAPACITY_COLUMN = "soil_thermal_capacity"


def get_screen_current_column(cable_key: "CableKey") -> str:
    """Return the optional scenario column for a cable's RMS screen current."""
    return f"screen_current_{cable_key.circuit_name}_{cable_key.cable_position.value}"


# Input model for scenario dataframe:
class AbstractScenarioModel(pa.DataFrameModel):
    """Base schema for scenario dataframe as used when creating a model.

    Structure:
    - Index: datetime (time series)
    - Columns:
        - load_<circuit_name> (float): load in Amperes for each circuit (e.g., load_circuit1, load_circuit2, etc.)
        - screen_current_<circuit_name>_<position> (optional float): RMS screen current in amperes for a cable.
        - ambient_temperature (float): ambient temperature in degrees Celsius

    Load and screen-current values describe the interval ending at their timestamp.
    The first row initializes the simulation; it does not drive a preceding interval.
    """

    ambient_temperature: pa.typing.Series[float]

    class Config:
        """Configuration for the schema model."""

        strict = False
        coerce = True

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

    solar_radiation_intensity: pa.typing.Series[float] = pa.Field(
        description="Solar radiation intensity in W/m^2", ge=0
    )


class ScenarioModelSoil(AbstractScenarioModel):
    """Soil scenario schema extending the base scenario schema."""

    soil_thermal_resistivity: pa.typing.Series[float]
    soil_thermal_capacity: pa.typing.Series[float]


ScenarioModelT = TypeVar("ScenarioModelT", bound=AbstractScenarioModel)
