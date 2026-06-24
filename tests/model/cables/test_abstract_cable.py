# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

"""Testing Notes.

NOT TO TEST:
    There are a number of simple Enum classes. These do not need to be tested.
    The same goes for the __main__() method as it is not considered for now.
    The same goes for the ConductorType class, as it solely sets values, making it an extremely simple dataclass.

TO TEST:
    All non-abstract [Cable] class methods

Notes:
    In the future tests should be expanded upon.

"""

import numpy as np
import pytest

from cable_thermal_model.model.cables.enum_classes_cable import CableScreenLossType
from cable_thermal_model.model.cables.fd_cable import FDCable


def test_dielectric_loss(
    three_core_cable_xlpe: FDCable,
    three_core_cable_pilc: FDCable,
):
    """Tests whether no dielectric losses are computed for cables where they should be neglected."""
    assert np.isclose(three_core_cable_xlpe.get_dielectric_loss_for_cable(), 0.0)
    assert np.isclose(three_core_cable_pilc.get_dielectric_loss_for_cable(), 0.0)


def test_compute_dielectric_loss_per_conductor(single_core_cable_xlpe: FDCable):
    """Tests whether the calculated dielectric losses per conductor are meeting the manually calculated losses."""
    # Generation:
    dielectric_loss = single_core_cable_xlpe._compute_dielectric_loss_per_conductor()

    # Evaluation:
    assert np.isclose(dielectric_loss, 0.020190704723891475, atol=0.02)


def test_compute_dielectric_loss_per_conductor_three_core(three_core_cable_xlpe: FDCable):
    """Tests whether NotImplementedError is raised for three-core cables in the dielectric loss calculation."""
    with pytest.raises(NotImplementedError):
        three_core_cable_xlpe._compute_dielectric_loss_per_conductor()


@pytest.mark.skip("Not yet implemented")
def test_abstract_cable_select_ks_for_hollow_helical_stranded(cable_builder):
    # TODO: Implement. Could not properly find a cable with attribute "inner_oil_duct_layer_index"
    ...


@pytest.mark.parametrize(
    "cable_fixture, expected_k_s, expected_k_p",
    [
        ("single_core_cable_xlpe", 1.0, 1.0),
        ("three_core_cable_xlpe", 1.0, 1.0),
        ("three_core_cable_pilc", 1.0, 0.8),
    ],
)
def test_abstract_cable_select_ks_and_kp(cable_fixture, expected_k_s, expected_k_p, request):
    fd_cable = request.getfixturevalue(cable_fixture)
    k_s, k_p = fd_cable._select_k_s_and_k_p()

    # Evaluation:
    assert np.isclose(k_s, expected_k_s, rtol=0.02)
    assert np.isclose(k_p, expected_k_p, rtol=0.02)


def test_abstract_cable_calculate_y_p(single_core_cable_xlpe):
    # Generation:
    y_p = single_core_cable_xlpe._calculate_y_p_from_x_p(d=1, s=2, x_p=5.4, factor=2)

    # Evaluation:
    assert np.isclose(y_p, 0.5036, rtol=0.02)


def test_abstract_cable_calculate_y_s(single_core_cable_xlpe):
    # Generation:
    y_s = single_core_cable_xlpe._calculate_y_s_from_x_s(x_s=1)

    # Evaluation:
    assert np.isclose(y_s, 0.005187, rtol=0.02)


def test_abstract_cable_ac_resistance_conductor(single_core_cable_xlpe: FDCable):
    # Generation:
    ac_resistance = single_core_cable_xlpe.get_ac_resistance_conductor(Tc=20, s=0.02)

    # Evaluation:
    assert np.isclose(ac_resistance, 6.338e-05, rtol=0.02)


def test_abstract_cable_resistance_screen(single_core_cable_xlpe):
    # Generation:
    resistance_screen = single_core_cable_xlpe._get_resistance_screen(Ts=20)

    # Evaluation:
    assert np.isclose(resistance_screen, 0.000344, rtol=0.02)


def test_abstract_cable_dc_resistance_conductor(single_core_cable_xlpe):
    # Generation:
    dc_resistance_conductor = single_core_cable_xlpe.get_dc_resistance_conductor(Tc=20)

    # Evaluation:
    assert np.isclose(dc_resistance_conductor, 4.69e-05, rtol=0.02)


def test_get_heat_generation_conductor_and_screen(
    three_core_cable_pilc: FDCable,
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
