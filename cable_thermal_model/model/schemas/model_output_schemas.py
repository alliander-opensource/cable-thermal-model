# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from typing import Generic

import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame
from pydantic import BaseModel, ConfigDict

from cable_thermal_model.cable.cable_circuit import CablePosition
from cable_thermal_model.model.cables.enum_classes_cable import CableLayer
from cable_thermal_model.model.schemas.state_schemas import StateT


# OutputSchema for temperatureResult dataframe:
class TemperatureResultSchema(pa.DataFrameModel):
    """Schema for temperature result DataFrame with MultiIndex columns.

    Structure:
    - Index: datetime (time series)
    - Columns: MultiIndex with 3 levels:
        - Level 0: circuit_name (str)
        - Level 1: cable_position (CablePosition enum values)
        - Level 2: cable_layer (CableLayer enum values)
    - Values: temperature in degrees Celsius (float)
    """

    @pa.dataframe_check(error="Temperature result index must be datetime-like.")
    @classmethod
    def check_datetime_index(cls, df: pd.DataFrame):
        """Ensure index is datetime-like."""
        return pd.api.types.is_datetime64_any_dtype(df.index) or pd.api.types.is_timedelta64_dtype(df.index)

    @pa.dataframe_check(
        error="Temperature result columns must be a 3-level MultiIndex: (circuit_name, cable_position, cable_layer)."
    )
    @classmethod
    def check_multiindex_columns(cls, df: pd.DataFrame):
        """Ensure columns are a MultiIndex with 3 levels."""
        expected_nlevels = 3
        return isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == expected_nlevels

    # Column level 0: circuit_name (any non-empty string)
    @pa.dataframe_check
    @classmethod
    def check_circuit_names(cls, df: pd.DataFrame) -> bool:
        """Ensure circuit names are non-empty strings."""
        circuit_names = df.columns.get_level_values(0).unique()
        for name in circuit_names:
            if not isinstance(name, str) or len(name) == 0:
                raise ValueError(f"Circuit name '{name}' is not a valid non-empty string.")
        return True

    # Column level 1: cable_position (valid enum values)
    @pa.dataframe_check
    @classmethod
    def check_cable_positions(cls, df: pd.DataFrame) -> bool:
        """Ensure cable positions are valid CablePosition enum values."""
        positions = df.columns.get_level_values(1).unique()
        return bool([CablePosition(pos) for pos in positions])

    # Column level 2: cable_layer (valid enum values)
    @pa.dataframe_check
    @classmethod
    def check_cable_layers(cls, df: pd.DataFrame) -> bool:
        """Ensure cable layers are valid CableLayer enum values."""
        layers = df.columns.get_level_values(2).unique()
        return bool([CableLayer(layer) for layer in layers])

    # Data values: temperature (float)
    @pa.dataframe_check
    @classmethod
    def check_temperature_values(cls, df: pd.DataFrame) -> bool:
        """Ensure temperature values are floats."""
        for dtype in df.dtypes:
            if not pd.api.types.is_float_dtype(dtype):
                raise ValueError("All temperature values must be of float type.")
        return True


class ModelOutputSchema(BaseModel, Generic[StateT]):
    """Schema for the output of the thermal cable model, containing the temperature results and the final state."""

    result: DataFrame[TemperatureResultSchema]
    state: StateT

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
