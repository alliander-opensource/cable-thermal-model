# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from cable_thermal_model.model.cables.cable import CableSoil
from cable_thermal_model.model.cables.enum_classes_cable import CableScreenLossType
from cable_thermal_model.model.model import Model
from cable_thermal_model.model.model_air import ModelAir
from cable_thermal_model.model.model_soil import ModelSoil


def test_get_heat_generation_conductor_and_screen(
    three_core_cable_pilc: CableSoil,
):
    # Set the screen loss function
    three_core_cable_pilc.layer_metrics.screen_loss_type = CableScreenLossType.TwoSidedBondingLinearCenter
    no_load_heat_generation_conductor, no_load_heat_generation_screen = (
        three_core_cable_pilc.get_heat_generation_conductor_and_screen(
            load=0.0,
            conductor_temperature=50.0,
            screen_temperature=40.0,
            temperature_dependent_electric_resistance=True,
            ac_current=True,
        )
    )

    assert np.isclose(no_load_heat_generation_conductor, 0.0)
    assert np.isclose(no_load_heat_generation_screen, 0.0)

    # Check that more heat is generated when incorporating AC effects
    load = 500.0  # Amperes
    ac_heat_generation_conductor, ac_heat_generation_screen = (
        three_core_cable_pilc.get_heat_generation_conductor_and_screen(
            load=load,
            conductor_temperature=50.0,
            screen_temperature=40.0,
            temperature_dependent_electric_resistance=True,
            ac_current=True,
        )
    )

    dc_heat_generation_conductor, dc_heat_generation_screen = (
        three_core_cable_pilc.get_heat_generation_conductor_and_screen(
            load=load,
            conductor_temperature=50.0,
            screen_temperature=40.0,
            temperature_dependent_electric_resistance=True,
            ac_current=False,
        )
    )

    # Check that AC heat generation is higher than DC heat generation
    assert ac_heat_generation_conductor > dc_heat_generation_conductor

    # Check that the heta generated in the screen is strictly positive in the AC case
    assert ac_heat_generation_screen > 0.0

    # Check that no heat is generated in the screen in the DC case, where we set current_in_screen=False
    assert np.isclose(dc_heat_generation_screen, 0.0)


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
