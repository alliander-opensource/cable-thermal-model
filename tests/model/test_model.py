# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import MagicMock

import numpy as np
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


def test_add_solution_location_rejects_invalid_layer_type(model: ModelSoil):
    """Ensure a clear TypeError is raised for non-CableLayer input."""
    with pytest.raises(TypeError, match="The layer argument must be of type CableLayer"):
        model.add_solution_location(layer_name="Conductor")  # type: ignore[arg-type]


def test_initialize_state_from_cables_uses_fill_value(model: ModelSoil):
    """Ensure helper initializes all cable arrays with the provided fill value."""
    fill_value = 42.5
    initialized = model._initialize_state_from_cables(cables=model.cables, fill_value=fill_value)

    assert set(initialized) == set(model.cables)
    for cable_key in model.cables:
        assert np.all(np.isclose(initialized[cable_key], fill_value))


def test_get_circuit_loads_from_scenario_row(model: ModelSoil):
    """Ensure scenario row is mapped to circuit load dict using load_<circuit_name> keys."""
    _, scenario_row = next(model.scenario.iterrows())

    loads = model._get_circuit_loads_from_scenario_row(scenario_row)

    assert loads == {"c1": scenario_row["load_c1"]}


def test_initialize_temperature_result_contains_expected_layers(model: ModelSoil):
    """Ensure initialized result includes standard and requested extra layers, and excludes absent layers."""
    model.add_solution_location(CableLayer.Insulation)
    initial_state = model._build_initial_state()

    model._initialize_temperature_result(state=initial_state)
    temperature_result = model.temperature_result

    for cable_key in model.cables:
        assert CableLayer.Conductor in temperature_result[cable_key]
        assert CableLayer.Sheath in temperature_result[cable_key]
        assert CableLayer.Insulation in temperature_result[cable_key]
        assert CableLayer.Pipe not in temperature_result[cable_key]
        assert np.isfinite(temperature_result[cable_key][CableLayer.Conductor][0])


def test_update_pipe_fill_resistivity_skips_cables_without_pipe(model: ModelSoil):
    """Ensure no pipe-fill updates happen for cables without a pipe layer."""
    temperature_state = model._build_initial_state().temperature

    mocked_update_methods = {}
    for cable_key, pos_cable in model.cables.items():
        mocked_update_methods[cable_key] = MagicMock()
        pos_cable.cable.update_pipe_fill_resistivity = mocked_update_methods[cable_key]

    model._update_pipe_fill_resistivity(temperature_state=temperature_state, cables=model.cables)

    for cable_key in mocked_update_methods:
        mocked_update_methods[cable_key].assert_not_called()


def test_update_pipe_fill_resistivity_updates_pipe_cables(model_with_pipe: ModelSoil):
    """Ensure pipe-fill resistivity is updated with the mean PipeFill temperature when a pipe exists."""
    temperature_state = model_with_pipe._build_initial_state().temperature

    for cable_key, pos_cable in model_with_pipe.cables.items():
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
    cable_key = next(iter(model.cables))
    wrong_state = StateAir(
        static_env_hash=model.static_env.compute_hash(),
        temperature={cable_key: np.array([20.0])},
        self_heating_contribution={cable_key: np.array([20.0])},
        ambient_temperature=5.0,
    )

    with pytest.raises(ValueError, match="ModelSoil requires a StateSoil instance, but received StateAir"):
        model._validate_state_model_consistency(wrong_state)


def test_initialize_thermal_state_returns_deep_copy(model: ModelSoil):
    """Ensure provided initial state is deep-copied before reuse."""
    initial_state = model.run().state

    initialized_state = model._initialize_state(initial_state=initial_state)

    assert initialized_state is not initial_state

    cable_key = next(iter(initialized_state.temperature))
    original_value = initial_state.temperature[cable_key][0]
    initialized_state.temperature[cable_key][0] = original_value + 1.0

    assert np.isclose(initial_state.temperature[cable_key][0], original_value)
