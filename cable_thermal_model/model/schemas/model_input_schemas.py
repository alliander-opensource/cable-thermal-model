# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from typing import TypeVar

import pandas as pd
import pandera.pandas as pa


# Output schema for scenario dataframe:
class AbstractScenarioSchema(pa.DataFrameModel):
    """Base schema for scenario dataframe as used when creating a model.

    Structure:
    - Index: datetime (time series)
    - Columns:
        - load_<circuit_name> (float): load in Amperes for each circuit (e.g., load_circuit1, load_circuit2, etc.)
        - ambient_temperature (float): ambient temperature in degrees Celsius
    """

    @pa.dataframe_check(error="Scenario index must be either datetime-like or timedelta-like..")
    @classmethod
    def check_datetime_index(cls, df: pd.DataFrame):
        """Ensure index is datetime-like or timedelta-like."""
        return pd.api.types.is_datetime64_any_dtype(df.index) or pd.api.types.is_timedelta64_dtype(df.index)

    @pa.dataframe_check(error="Scenario columns must include load_<circuit_name> and ambient_temperature.")
    @classmethod
    def check_required_columns(cls, df: pd.DataFrame):
        """Ensure shared required columns are present."""
        req_columns = {"ambient_temperature"}
        load_columns = [col for col in df.columns if col.startswith("load_")]
        if not all(col in df.columns for col in req_columns) or len(load_columns) == 0:
            raise ValueError(
                f"Scenario dataframe must include columns: {req_columns} and at least one load_<circuit_name> column."
            )
        return True

    @pa.dataframe_check(error="Load columns must be in the format load_<circuit_name> and contain numeric values.")
    @classmethod
    def check_load_columns(cls, df: pd.DataFrame):
        """Ensure load columns are in the correct format and contain numeric values."""
        load_columns = [col for col in df.columns if col.startswith("load_")]
        for col in load_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Load column {col} must contain numeric values.")
        return True

    @pa.dataframe_check(error="Ambient temperature must contain numeric values.")
    @classmethod
    def check_numeric_columns(cls, df: pd.DataFrame):
        """Ensure ambient temperature column contains numeric values."""
        numeric_columns = ["ambient_temperature"]
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column {col} must contain numeric values.")
        return True

    @pa.dataframe_check(error="Scenario dataframe must not contain missing values.")
    @classmethod
    def check_no_missing_values(cls, df: pd.DataFrame):
        """Ensure there are no missing values in the scenario dataframe."""
        if any(df.isna().sum()):
            raise ValueError("Scenario contains nan values.")
        return True


class ScenarioSchemaAir(AbstractScenarioSchema):
    """Air scenario schema extending the base scenario schema.

    This schema currently does not add extra checks beyond AbstractScenarioSchema,
    but exists to allow air-specific evolution in the future.
    """


class ScenarioSchemaSoil(AbstractScenarioSchema):
    """Soil scenario schema extending the base scenario with required soil properties."""

    @pa.dataframe_check(error="Scenario columns must include soil_thermal_resistivity and soil_thermal_capacity.")
    @classmethod
    def check_required_soil_columns(cls, df: pd.DataFrame):
        """Ensure soil-specific required columns are present."""
        req_columns = {"soil_thermal_resistivity", "soil_thermal_capacity"}
        if not all(col in df.columns for col in req_columns):
            raise ValueError(f"Scenario dataframe must include columns: {req_columns}.")
        return True

    @pa.dataframe_check(error="Soil thermal columns must contain numeric values.")
    @classmethod
    def check_numeric_soil_columns(cls, df: pd.DataFrame):
        """Ensure soil thermal columns contain numeric values."""
        numeric_columns = ["soil_thermal_resistivity", "soil_thermal_capacity"]
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column {col} must contain numeric values.")
        return True


ScenarioSchemaT = TypeVar("ScenarioSchemaT", bound=AbstractScenarioSchema)
