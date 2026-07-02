# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import pytest

from cable_thermal_model.model.model import Model
from cable_thermal_model.model.model_air import ModelAir
from cable_thermal_model.model.model_soil import ModelSoil


@pytest.mark.parametrize("model_class", [ModelAir, ModelSoil])
@pytest.mark.parametrize(
    "run_options,expected_exception",
    [
        [None, None],
        [{"temperature_dependent_electric_resistance": False}, None],
        [{"ac_current": False}, None],
        [{"neglect_dielectric_loss": True}, None],
        [{"soil_drying": True}, ModelAir],
        [{}, None],
        [{"non_existing_option": 123}, True],
    ],
)
def test_set_run_options(
    model_class: type[Model], run_options: dict | None, expected_exception: type[Model] | bool | None
):
    """Test the _set_run_options method of the model classes."""
    model = model_class.__new__(model_class)  # Create an instance of the model class without calling __init__

    if expected_exception is True or expected_exception == model_class:
        with pytest.raises(ValueError):
            model._set_run_options(run_options=run_options)
        return

    model._set_run_options(run_options=run_options)
    if run_options is not None:
        for key, value in run_options.items():
            assert getattr(model.run_options, key) == value


def test_initialize_thermal_state_returns_deep_copy(model: ModelSoil):
    """Ensure provided initial state is deep-copied before reuse."""
    initial_state = model.run().state

    initialized_state = model._initialize_thermal_state(initial_state=initial_state)

    assert initialized_state is not initial_state

    cable_key = next(iter(initialized_state.temperature))
    original_value = initial_state.temperature[cable_key][0]
    initialized_state.temperature[cable_key][0] = original_value + 1.0

    assert np.isclose(initial_state.temperature[cable_key][0], original_value)


def test_build_linear_system_for_cables_keys_match(model: ModelSoil):
    """Ensure linear-system helper returns matrices and vectors for all provided cables."""
    matrices, vectors = model._build_linear_system_for_cables(cables=model.cables_with_soil)

    expected_keys = set(model.cables_with_soil.keys())
    assert set(matrices.keys()) == expected_keys
    assert set(vectors.keys()) == expected_keys


@pytest.mark.parametrize("model_class", [ModelAir, ModelSoil])
def test_refresh_matrices_if_needed_returns_expected_contract(
    model_class: type[Model],
    single_circuit_in_air_env,
    single_circuit_env,
    load_series_constant,
):
    """Ensure refresh_matrices_if_needed returns (matrices, updated_cables_set)."""
    if model_class is ModelAir:
        scenario = pd.DataFrame(
            data={"load_c1": load_series_constant, "ambient_temperature": 10},
            index=load_series_constant.index,
        )
        model = ModelAir(single_circuit_in_air_env, scenario)
    else:
        scenario = pd.DataFrame(
            data={
                "load_c1": load_series_constant,
                "ambient_temperature": 10,
                "soil_thermal_resistivity": 0.75,
                "soil_thermal_capacity": 2e6,
            },
            index=load_series_constant.index,
        )
        model = ModelSoil(single_circuit_env, scenario)

    matrices, _ = model._initialize_linear_system()
    temperature_state = model._initialize_temperature_state()

    refreshed_matrices, updated_cables = model._refresh_matrices_if_needed(
        matrices=matrices,
        temperature_state=temperature_state,
        scenario_row=model.scenario.iloc[0],
        elapsed_seconds=0.0,
    )

    assert refreshed_matrices is matrices
    assert isinstance(updated_cables, set)
    assert updated_cables.issubset(set(model.cables.keys()))
