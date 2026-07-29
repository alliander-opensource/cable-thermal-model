# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from typing import Generic

import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame
from pydantic import BaseModel, ConfigDict

from cable_thermal_model.cable.cable_circuit import CablePosition
from cable_thermal_model.environment.measurement_point import MEASUREMENT_POINT_KEY_PREFIX
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

    @pa.dataframe_check
    @classmethod
    def check_circuit_names(cls, df: pd.DataFrame) -> bool:
        """Ensure level-0 names are valid circuit names or the measurement-point prefix."""
        circuit_names = df.columns.get_level_values(0).unique()
        for name in circuit_names:
            if name == MEASUREMENT_POINT_KEY_PREFIX:
                continue  # Skip validation for measurement-point columns
            if not isinstance(name, str) or len(name) == 0:
                raise ValueError(f"Circuit name '{name}' is not a valid non-empty string.")
        return True

    @pa.dataframe_check
    @classmethod
    def check_cable_positions(cls, df: pd.DataFrame) -> bool:
        """Validate level-1 and level-2 values by column type.

        - For cable result columns: level 1 must be a valid CablePosition. Level 2 must be a valid CableLayer.
        - For measurement-point columns: level 1 must be a string starting with 'x='. Level 2 must be a string
            starting with 'y='.
        """
        level0_values = df.columns.get_level_values(0)
        level1_values = df.columns.get_level_values(1)
        level2_values = df.columns.get_level_values(2)

        for level0, level1, level2 in zip(level0_values, level1_values, level2_values, strict=True):
            if level0 == MEASUREMENT_POINT_KEY_PREFIX:
                if not isinstance(level1, str) or not level1.startswith("x="):
                    raise ValueError("Measurement-point columns must have level 1 as a string starting with 'x='.")
                if not isinstance(level2, str) or not level2.startswith("y="):
                    raise ValueError("Measurement-point columns must have level 2 as a string starting with 'y='.")
            else:
                CablePosition(level1)
                CableLayer(level2)

        return True

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
