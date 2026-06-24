# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError

from cable_thermal_model.model.schemas.model_output_schemas import TemperatureResultSchema


@pytest.fixture(scope="function")
def valid_output_temperature_result_df():
    """Fixture providing a valid temperature result DataFrame for testing."""
    # Create a sample MultiIndex for columns
    index = pd.date_range(start="2024-01-01", periods=3, freq="D")
    columns = pd.MultiIndex.from_tuples(
        [
            ("circuit1", "trefoil_top", "Conductor"),
            ("circuit1", "trefoil_top", "Sheath"),
            ("circuit1", "trefoil_left", "Conductor"),
            ("circuit1", "trefoil_left", "Sheath"),
            ("circuit1", "trefoil_right", "Conductor"),
            ("circuit1", "trefoil_right", "Sheath"),
            ("circuit2", "single", "Conductor"),
            ("circuit2", "single", "Sheath"),
        ],
    )
    # Create a sample DataFrame with valid data
    data = 15.0 * np.ones((len(index), len(columns)))
    return pd.DataFrame(data, index=index, columns=columns)


def test_valid_temperature_result_schema(valid_output_temperature_result_df):
    TemperatureResultSchema(valid_output_temperature_result_df)


def test_two_level_multiindex(valid_output_temperature_result_df):
    # Create a DataFrame with only 2 levels in the MultiIndex
    invalid_df = valid_output_temperature_result_df
    invalid_df.columns = invalid_df.columns.droplevel(0)

    with pytest.raises(SchemaError, match="Temperature result columns must be a 3-level MultiIndex"):
        TemperatureResultSchema(invalid_df)


def test_four_level_multiindex(valid_output_temperature_result_df):
    # Create a DataFrame with 4 levels in the MultiIndex
    invalid_df = valid_output_temperature_result_df
    new_level = pd.Index(["extra"] * len(invalid_df.columns), name="extra_level")
    invalid_df.columns = pd.MultiIndex.from_arrays(
        [
            invalid_df.columns.get_level_values(0),
            invalid_df.columns.get_level_values(1),
            invalid_df.columns.get_level_values(2),
            new_level,
        ]
    )

    with pytest.raises(SchemaError, match="Temperature result columns must be a 3-level MultiIndex"):
        TemperatureResultSchema(invalid_df)


def test_non_str_level_0(valid_output_temperature_result_df):
    invalid_df = valid_output_temperature_result_df
    new_columns = list(invalid_df.columns)
    new_columns[0] = (None, "trefoil_top", "Conductor")
    invalid_df.columns = pd.MultiIndex.from_tuples(new_columns)

    with pytest.raises(SchemaError, match="not a valid non-empty string"):
        TemperatureResultSchema(invalid_df)


def test_invalid_cable_position(valid_output_temperature_result_df):
    invalid_df = valid_output_temperature_result_df
    new_columns = list(invalid_df.columns)
    new_columns[1] = ("circuit1", "invalid_position", "Conductor")
    invalid_df.columns = pd.MultiIndex.from_tuples(new_columns)

    with pytest.raises(SchemaError, match="is not a valid CablePosition"):
        TemperatureResultSchema(invalid_df)


def test_invalid_cable_layer(valid_output_temperature_result_df):
    invalid_df = valid_output_temperature_result_df
    new_columns = list(invalid_df.columns)
    new_columns[2] = ("circuit1", "trefoil_top", "invalid_layer")
    invalid_df.columns = pd.MultiIndex.from_tuples(new_columns)

    with pytest.raises(SchemaError, match="is not a valid CableLayer"):
        TemperatureResultSchema(invalid_df)


def test_non_numeric_values(valid_output_temperature_result_df):
    invalid_df = valid_output_temperature_result_df.copy()
    invalid_df.iloc[0, 0] = "not a number"

    with pytest.raises(SchemaError, match="All temperature values must be of float type."):
        TemperatureResultSchema(invalid_df)


def test_non_datetime_index(valid_output_temperature_result_df):
    invalid_df = valid_output_temperature_result_df.copy()
    invalid_df.index = ["not a datetime"] * len(invalid_df)

    with pytest.raises(SchemaError, match="Temperature result index must be datetime-like."):
        TemperatureResultSchema(invalid_df)
