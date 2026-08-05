# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from cable_thermal_model.model.cables.enum_classes_cable import CableLayer
from cable_thermal_model.model.model import Model
from cable_thermal_model.model.model_air import ModelAir
from cable_thermal_model.model.model_soil import ModelSoil
from cable_thermal_model.model.schemas import StateAir


@pytest.mark.parametrize("model_class", [ModelAir, ModelSoil])
@pytest.mark.parametrize(
    "run_options,expected_exception",
    [
        (None, None),
        ({"temperature_dependent_electric_resistance": False}, None),
        ({"ac_current": False}, None),
        ({"neglect_dielectric_loss": True}, None),
        ({"soil_drying": True}, ModelAir),
        ({}, None),
        ({"non_existing_option": 123}, ValueError),
        (123, TypeError),
    ],
)
def test_set_run_options(
    model_class: type[Model], run_options: dict | int | None, expected_exception: type[Exception] | type[Model] | None
):
    """Test the _set_run_options method of the model classes."""
    model = model_class.__new__(model_class)  # Create an instance of the model class without calling __init__

    if expected_exception == model_class:
        with pytest.raises(ValueError):
            model._set_run_options(run_options=run_options)
        return

    if isinstance(expected_exception, type) and issubclass(expected_exception, Exception):
        with pytest.raises(expected_exception):
            model._set_run_options(run_options=run_options)
        return

    model._set_run_options(run_options=run_options)
    if run_options is not None:
        for key, value in run_options.items():
            assert getattr(model.run_options, key) == value


@pytest.mark.parametrize("model_class", [ModelAir, ModelSoil])
def test_set_run_options_accepts_model_run_options_instance(model_class: type[Model]):
    """Ensure _set_run_options accepts already-instantiated run options objects."""
    model = model_class.__new__(model_class)
    run_options_instance = model_class._run_options_class(ac_current=False)

    model._set_run_options(run_options=run_options_instance)

    assert model.run_options is run_options_instance


def test_run_rejects_invalid_extra_solution_layer(model: ModelSoil, scenario_constant):
    """Ensure invalid extra solution layers are rejected through run-option validation."""
    with pytest.raises(ValueError, match="extra_solution_layers"):
        model.run(scenario_constant, run_options={"extra_solution_layers": ["NotALayer"]})


@pytest.mark.parametrize(
    "layer",
    [
        CableLayer.Conductor,
        CableLayer.Sheath,
        CableLayer.Pipe,
    ],
)
def test_run_rejects_standard_layers_in_extra_solution_layers(model: ModelSoil, scenario_constant, layer: CableLayer):
    """Ensure dedicated output layers cannot be requested through extra_solution_layers."""
    with pytest.raises(ValueError, match="extra_solution_layers"):
        model.run(scenario_constant, run_options={"extra_solution_layers": [layer]})


def test_initialize_state_from_cables_uses_fill_value(model: ModelSoil):
    """Ensure helper initializes all cable arrays with the provided fill value."""
    fill_value = 42.5
    initialized = model._initialize_state_from_cables(cables=model._cables, fill_value=fill_value)

    assert set(initialized) == set(model._cables)
    for cable_key in model._cables:
        assert np.all(np.isclose(initialized[cable_key], fill_value))


def test_get_circuit_loads_from_scenario_row(model: ModelSoil, scenario_constant):
    """Ensure scenario row is mapped to circuit load dict using load_<circuit_name> keys."""
    _, scenario_row = next(scenario_constant.iterrows())

    loads = model._get_circuit_loads_from_scenario_row(scenario_row)

    assert loads == {"c1": scenario_row["load_c1"]}


def test_initialize_temperature_result_contains_expected_layers(model: ModelSoil, scenario_constant):
    """Ensure initialized result includes standard and requested extra layers, and excludes absent layers."""
    run_options = {"extra_solution_layers": [CableLayer.Insulation]}
    model.run(scenario_constant, run_options=run_options)
    initial_state = model._build_initial_state(scenario_constant["ambient_temperature"].iloc[0])

    temperature_result = model._initialize_temperature_result(
        state=initial_state,
        n_scenario_rows=len(scenario_constant.index),
    )

    for cable_key in model._cables:
        assert CableLayer.Conductor in temperature_result[cable_key]
        assert CableLayer.Sheath in temperature_result[cable_key]
        assert CableLayer.Insulation in temperature_result[cable_key]
        assert CableLayer.Pipe not in temperature_result[cable_key]
        assert np.isfinite(temperature_result[cable_key][CableLayer.Conductor][0])


def test_update_pipe_fill_resistivity_skips_cables_without_pipe(model: ModelSoil, scenario_constant):
    """Ensure no pipe-fill updates happen for cables without a pipe layer."""
    model.run(scenario_constant)
    temperature_state = model._build_initial_state(scenario_constant["ambient_temperature"].iloc[0]).temperature

    mocked_update_methods = {}
    for cable_key, pos_cable in model._cables.items():
        mocked_update_methods[cable_key] = MagicMock()
        pos_cable.cable.update_pipe_fill_resistivity = mocked_update_methods[cable_key]

    model._update_pipe_fill_resistivity(temperature_state=temperature_state, cables=model._cables)

    for cable_key in mocked_update_methods:
        mocked_update_methods[cable_key].assert_not_called()


def test_update_pipe_fill_resistivity_updates_pipe_cables(model_with_pipe: ModelSoil, scenario_constant):
    """Ensure pipe-fill resistivity is updated with the mean PipeFill temperature when a pipe exists."""
    model_with_pipe.run(scenario_constant)
    temperature_state = model_with_pipe._build_initial_state(
        scenario_constant["ambient_temperature"].iloc[0]
    ).temperature

    for cable_key, pos_cable in model_with_pipe._cables.items():
        if pos_cable.cable.layer_metrics.pipe is None:
            continue

        pos_cable.cable.update_pipe_fill_resistivity = MagicMock()

        model_with_pipe._update_pipe_fill_resistivity(
            temperature_state=temperature_state,
            cables={cable_key: pos_cable},
        )

        pos_cable.cable.update_pipe_fill_resistivity.assert_called_once_with(
            temperature_grid=temperature_state[cable_key]
        )


def test_validate_state_model_consistency_rejects_wrong_state_type(model: ModelSoil):
    """Ensure model type check rejects states from a different model class."""
    cable_key = next(iter(model.static_env.get_cables()))
    wrong_state = StateAir(
        static_env_hash=model.static_env.compute_hash(),
        temperature={cable_key: np.array([20.0])},
        self_heating_contribution={cable_key: np.array([20.0])},
        ambient_temperature=5.0,
    )

    with pytest.raises(ValueError, match="ModelSoil requires a StateSoil instance, but received StateAir"):
        model._validate_state_model_consistency(wrong_state)


def test_initialize_thermal_state_returns_deep_copy(model: ModelSoil, scenario_constant):
    """Ensure provided initial state is deep-copied before reuse."""
    initial_state = model.run(scenario_constant).state

    initialized_state = model._initialize_state(scenario_constant, initial_state=initial_state)

    assert initialized_state is not initial_state

    cable_key = next(iter(initialized_state.temperature))
    original_value = initial_state.temperature[cable_key][0]
    initialized_state.temperature[cable_key][0] = original_value + 1.0

    assert np.isclose(initial_state.temperature[cable_key][0], original_value)


def test_validate_scenario_accepts_integer_numeric_columns(model: ModelSoil):
    """Tests whether integer-valued numeric scenario columns are accepted via schema coercion."""
    scenario = pd.DataFrame(
        index=pd.date_range("2020-01-01", "2020-01-03", freq="1h"),
        data={
            "load_c1": 100,
            "ambient_temperature": 10,
            "soil_thermal_resistivity": 1,
            "soil_thermal_capacity": 2_000_000,
        },
    )

    result = model.run(scenario)
    assert result.result.shape[0] == scenario.shape[0]
