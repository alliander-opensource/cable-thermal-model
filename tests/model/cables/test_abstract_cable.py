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

from cable_thermal_model.model.cables.cable_soil import CableSoil
from cable_thermal_model.model.cables.enum_classes_cable import (
    CableConductorSurfaceType,
    CableScreenCirculatingLossType,
    CableScreenEddyLossType,
)


def test_dielectric_loss(
    three_core_cable_xlpe: CableSoil,
    three_core_cable_pilc: CableSoil,
):
    """Tests whether no dielectric losses are computed for cables where they should be neglected."""
    assert np.isclose(three_core_cable_xlpe.get_dielectric_loss_for_cable(), 0.0)
    assert np.isclose(three_core_cable_pilc.get_dielectric_loss_for_cable(), 0.0)


def test_compute_dielectric_loss_per_conductor(single_core_cable_xlpe: CableSoil):
    """Tests whether the calculated dielectric losses per conductor are meeting the manually calculated losses."""
    # Generation:
    dielectric_loss = single_core_cable_xlpe._compute_dielectric_loss_per_conductor()

    # Evaluation:
    assert np.isclose(dielectric_loss, 0.020190704723891475, atol=0.02)


def test_compute_dielectric_loss_per_conductor_three_core(three_core_cable_xlpe: CableSoil):
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


def test_abstract_cable_ac_resistance_conductor(single_core_cable_xlpe: CableSoil):
    # Generation:
    ac_resistance = single_core_cable_xlpe.get_ac_resistance_conductor(Tc=20, s=0.02)

    # Evaluation:
    assert np.isclose(ac_resistance, 6.338e-05, rtol=0.02)


def test_abstract_cable_resistance_screen(single_core_cable_xlpe: CableSoil):
    # Generation:
    resistance_screen = single_core_cable_xlpe._get_resistance_screen(Ts=20)

    # Evaluation:
    assert np.isclose(resistance_screen, 0.000344, rtol=0.02)


def test_abstract_cable_dc_resistance_conductor(single_core_cable_xlpe: CableSoil):
    # Generation:
    dc_resistance_conductor = single_core_cable_xlpe.get_dc_resistance_conductor(Tc=20)

    # Evaluation:
    assert np.isclose(dc_resistance_conductor, 4.69e-05, rtol=0.02)


def test_get_heat_generation_conductor_and_screen(
    three_core_cable_pilc: CableSoil,
):
    three_core_cable_pilc.layer_metrics.screen_circulating_loss_type = CableScreenCirculatingLossType.LinearCenter
    three_core_cable_pilc.layer_metrics.screen_eddy_loss_type = CableScreenEddyLossType.TwoSidedLinearCenter
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


@pytest.mark.parametrize(
    "circulating_loss_type, expected",
    [
        (CableScreenCirculatingLossType.LinearLeading, 11 / 13 - 5 * np.sqrt(3) / 26),
        (CableScreenCirculatingLossType.LinearCenter, 0.5),
        (CableScreenCirculatingLossType.LinearLagging, 11 / 13 + 5 * np.sqrt(3) / 26),
    ],
)
def test_linear_circulating_loss_phase_expression(single_core_cable_xlpe, circulating_loss_type, expected, monkeypatch):
    cable = single_core_cable_xlpe
    cable.layer_metrics.conductor_distance = 0.1
    cable.layer_metrics.screen_circulating_loss_type = circulating_loss_type
    monkeypatch.setattr(type(cable), "X", 2.0)
    monkeypatch.setattr(type(cable), "Xm", 3.0)
    monkeypatch.setattr(cable, "get_ac_resistance_conductor", lambda **kwargs: 1.0)
    monkeypatch.setattr(cable, "_get_resistance_screen", lambda Ts: 1.0)

    assert cable.get_screen_circulating_loss_factor(Ts=40.0, Tc=50.0) == pytest.approx(expected)


@pytest.mark.parametrize(
    "eddy_loss_type",
    [
        CableScreenEddyLossType.TwoSidedTrefoil,
        CableScreenEddyLossType.TwoSidedLinearLeading,
        CableScreenEddyLossType.TwoSidedLinearCenter,
        CableScreenEddyLossType.TwoSidedLinearLagging,
        CableScreenEddyLossType.GeneralizedTwoSided,
    ],
)
def test_two_sided_eddy_factor_without_circulating_solve(single_core_cable_xlpe, eddy_loss_type, monkeypatch):
    cable = single_core_cable_xlpe
    cable.layer_metrics.conductor_distance = 0.1
    cable.conductor.surface_type = CableConductorSurfaceType.Stranded
    cable.layer_metrics.screen_eddy_loss_type = eddy_loss_type
    cable.layer_metrics.screen_circulating_loss_type = None
    cable.weighted_screen_impedance = None

    def reject_solve(*args, **kwargs):
        pytest.fail("Eddy loss must not evaluate circulating currents")

    monkeypatch.setattr(cable, "_calculate_relative_circulating_screen_currents", reject_solve)

    def reject_base_calculation(*args, **kwargs):
        pytest.fail("Non-Milliken two-sided eddy loss must return zero before evaluating a base formula")

    for method_name in (
        "_get_screen_eddy_loss_trefoil",
        "_get_screen_eddy_loss_linear_leading",
        "_get_screen_eddy_loss_linear_center",
        "_get_screen_eddy_loss_linear_lagging",
        "_get_resistance_screen",
    ):
        monkeypatch.setattr(cable, method_name, reject_base_calculation)

    assert cable.get_screen_eddy_loss_factor(Ts=40.0, Tc=50.0) == 0.0


@pytest.mark.parametrize(
    "eddy_loss_type",
    [
        CableScreenEddyLossType.Zero,
        CableScreenEddyLossType.Trefoil,
        CableScreenEddyLossType.LinearLeading,
        CableScreenEddyLossType.LinearCenter,
        CableScreenEddyLossType.LinearLagging,
    ],
)
def test_eddy_only_factor_matches_total(single_core_cable_xlpe, eddy_loss_type):
    cable = single_core_cable_xlpe
    cable.layer_metrics.conductor_distance = 0.1
    cable.layer_metrics.screen_eddy_loss_type = eddy_loss_type
    cable.layer_metrics.screen_circulating_loss_type = CableScreenCirculatingLossType.Zero

    assert cable.get_screen_eddy_loss_factor(Ts=40.0, Tc=50.0) == pytest.approx(
        cable.get_cable_screen_loss_factor(Ts=40.0, Tc=50.0)
    )


@pytest.mark.parametrize(
    "circulating_loss_type, eddy_loss_type, base_eddy_method, linear",
    [
        (
            CableScreenCirculatingLossType.Trefoil,
            CableScreenEddyLossType.TwoSidedTrefoil,
            CableSoil._get_screen_eddy_loss_trefoil,
            False,
        ),
        (
            CableScreenCirculatingLossType.LinearLeading,
            CableScreenEddyLossType.TwoSidedLinearLeading,
            CableSoil._get_screen_eddy_loss_linear_leading,
            True,
        ),
        (
            CableScreenCirculatingLossType.LinearCenter,
            CableScreenEddyLossType.TwoSidedLinearCenter,
            CableSoil._get_screen_eddy_loss_linear_center,
            True,
        ),
        (
            CableScreenCirculatingLossType.LinearLagging,
            CableScreenEddyLossType.TwoSidedLinearLagging,
            CableSoil._get_screen_eddy_loss_linear_lagging,
            True,
        ),
    ],
)
def test_milliken_eddy_factor_applies_correction_to_base_factor(
    single_core_cable_xlpe, circulating_loss_type, eddy_loss_type, base_eddy_method, linear
):
    cable = single_core_cable_xlpe
    cable.layer_metrics.conductor_distance = 0.1
    cable.conductor.surface_type = CableConductorSurfaceType.Milliken
    cable.layer_metrics.screen_circulating_loss_type = circulating_loss_type
    cable.layer_metrics.screen_eddy_loss_type = eddy_loss_type
    resistance = cable._get_resistance_screen(Ts=40.0)
    correction = cable._get_correction_factor_milliken(
        CM1=resistance / (cable.X + cable.Xm if linear else cable.X),
        CN=resistance / (cable.X - cable.Xm / 3 if linear else cable.X),
    )
    eddy_factor = base_eddy_method(cable, Tc=50.0, Ts=40.0)
    expected = correction * eddy_factor

    assert cable.get_screen_eddy_loss_factor(Ts=40.0, Tc=50.0) == pytest.approx(expected)
    circulating_factor = cable.get_screen_circulating_loss_factor(Ts=40.0, Tc=50.0)
    assert circulating_factor + expected == pytest.approx(cable.get_cable_screen_loss_factor(Ts=40.0, Tc=50.0))


@pytest.mark.parametrize(
    "circulating_loss_type, eddy_loss_type",
    [
        (CableScreenCirculatingLossType.Zero, CableScreenEddyLossType.Trefoil),
        (CableScreenCirculatingLossType.Trefoil, CableScreenEddyLossType.TwoSidedTrefoil),
    ],
)
def test_calculated_screen_heating_is_circulating_plus_eddy(
    single_core_cable_xlpe, circulating_loss_type, eddy_loss_type
):
    cable = single_core_cable_xlpe
    cable.layer_metrics.conductor_distance = 0.1
    cable.layer_metrics.screen_circulating_loss_type = circulating_loss_type
    cable.layer_metrics.screen_eddy_loss_type = eddy_loss_type
    cable.conductor.surface_type = CableConductorSurfaceType.Stranded

    conductor_heat = 100.0
    screen_heat = cable.get_heat_generation_screen(
        heat_generation_conductor=conductor_heat,
        screen_temperature=40.0,
        conductor_temperature=50.0,
        temperature_dependent_electric_resistance=True,
    )
    circulating_factor = cable.get_screen_circulating_loss_factor(Ts=40.0, Tc=50.0)
    eddy_factor = cable.get_screen_eddy_loss_factor(Ts=40.0, Tc=50.0)

    assert screen_heat == pytest.approx(conductor_heat * (circulating_factor + eddy_factor))
    assert circulating_factor + eddy_factor == pytest.approx(cable.get_cable_screen_loss_factor(Ts=40.0, Tc=50.0))


@pytest.mark.parametrize("measured_current", [-25.0, 0.0, 25.0, 50.0])
@pytest.mark.parametrize("load", [0.0, 500.0])
@pytest.mark.parametrize("temperature_dependent", [False, True])
@pytest.mark.parametrize(
    "circulating_loss_type, eddy_loss_type",
    [
        (CableScreenCirculatingLossType.Zero, CableScreenEddyLossType.Zero),
        (CableScreenCirculatingLossType.Zero, CableScreenEddyLossType.Trefoil),
        (CableScreenCirculatingLossType.Trefoil, CableScreenEddyLossType.TwoSidedTrefoil),
        (CableScreenCirculatingLossType.GeneralizedLeading, CableScreenEddyLossType.GeneralizedTwoSided),
    ],
)
def test_measured_screen_heating(
    single_core_cable_xlpe,
    measured_current,
    load,
    temperature_dependent,
    circulating_loss_type,
    eddy_loss_type,
    monkeypatch,
):
    cable = single_core_cable_xlpe
    cable.layer_metrics.conductor_distance = 0.1
    cable.layer_metrics.screen_circulating_loss_type = circulating_loss_type
    cable.layer_metrics.screen_eddy_loss_type = eddy_loss_type
    cable.conductor.surface_type = CableConductorSurfaceType.Stranded

    def reject_circulating_loss(*args, **kwargs):
        pytest.fail("Measured heating must not evaluate calculated circulating loss")

    monkeypatch.setattr(cable, "get_screen_circulating_loss_factor", reject_circulating_loss)
    conductor_heat, screen_heat = cable.get_heat_generation_conductor_and_screen(
        load=load,
        conductor_temperature=60.0,
        screen_temperature=40.0,
        temperature_dependent_electric_resistance=temperature_dependent,
        ac_current=True,
        custom_screen_current=measured_current,
    )
    screen_temperature = 40.0 if temperature_dependent else 20.0
    conductor_temperature = 60.0 if temperature_dependent else 20.0
    eddy_heat = conductor_heat * cable.get_screen_eddy_loss_factor(Ts=screen_temperature, Tc=conductor_temperature)

    assert screen_heat == pytest.approx(
        measured_current**2 * cable._get_resistance_screen(Ts=screen_temperature) + eddy_heat
    )
    assert conductor_heat == pytest.approx(
        cable.get_heat_generation_conductor(
            ac_current=True,
            load=load,
            conductor_temperature=60.0,
            temperature_dependent_electric_resistance=temperature_dependent,
        )
    )


def test_measured_common_screen_heating_is_not_multiplied_by_conductor_count(three_core_cable_pilc):
    cable = three_core_cable_pilc
    cable.layer_metrics.screen_eddy_loss_type = CableScreenEddyLossType.Sector
    screen_heat = cable.get_heat_generation_screen(
        heat_generation_conductor=0.0,
        screen_temperature=40.0,
        conductor_temperature=60.0,
        temperature_dependent_electric_resistance=True,
        custom_screen_current=25.0,
    )

    assert screen_heat == pytest.approx(25.0**2 * cable._get_resistance_screen(Ts=40.0))


def test_custom_screen_current_rejected_in_dc_mode(single_core_cable_xlpe):
    with pytest.raises(ValueError, match="DC mode"):
        single_core_cable_xlpe.get_heat_generation_conductor_and_screen(
            load=500.0,
            conductor_temperature=60.0,
            screen_temperature=40.0,
            temperature_dependent_electric_resistance=True,
            ac_current=False,
            custom_screen_current=0.0,
        )


@pytest.mark.parametrize("circulating_loss_type", list(CableScreenCirculatingLossType))
def test_circulating_loss_does_not_require_eddy_case(single_core_cable_xlpe, circulating_loss_type, monkeypatch):
    cable = single_core_cable_xlpe
    cable.layer_metrics.conductor_distance = 0.1
    cable.layer_metrics.screen_circulating_loss_type = circulating_loss_type
    cable.layer_metrics.screen_eddy_loss_type = None
    cable.conductor.surface_type = CableConductorSurfaceType.Stranded

    def reject_eddy_loss(*args, **kwargs):
        pytest.fail("Circulating loss must not evaluate eddy loss")

    monkeypatch.setattr(cable, "get_screen_eddy_loss_factor", reject_eddy_loss)
    monkeypatch.setattr(cable, "_calculate_relative_circulating_screen_currents", lambda **kwargs: np.ones((3, 1)))
    assert cable.get_screen_circulating_loss_factor(Ts=40.0, Tc=50.0) >= 0.0


@pytest.mark.parametrize(
    "circulating_loss_type, expected",
    [
        (CableScreenCirculatingLossType.GeneralizedLeading, 1.0),
        (CableScreenCirculatingLossType.GeneralizedCenter, 4.0),
        (CableScreenCirculatingLossType.GeneralizedLagging, 9.0),
    ],
)
def test_generalized_circulating_loss_uses_phase_index(
    single_core_cable_xlpe, circulating_loss_type, expected, monkeypatch
):
    cable = single_core_cable_xlpe
    cable.layer_metrics.conductor_distance = 0.1
    cable.layer_metrics.screen_circulating_loss_type = circulating_loss_type
    cable.conductor.surface_type = CableConductorSurfaceType.Stranded
    monkeypatch.setattr(cable, "get_ac_resistance_conductor", lambda **kwargs: 1.0)
    monkeypatch.setattr(cable, "_get_resistance_screen", lambda Ts: 1.0)
    monkeypatch.setattr(
        cable,
        "_calculate_relative_circulating_screen_currents",
        lambda **kwargs: np.array([[1], [2], [3]]),
    )

    assert cable.get_screen_circulating_loss_factor(Ts=40.0, Tc=50.0) == pytest.approx(expected)


def test_eddy_loss_does_not_require_circulating_case(single_core_cable_xlpe):
    cable = single_core_cable_xlpe
    cable.layer_metrics.conductor_distance = 0.1
    cable.layer_metrics.screen_eddy_loss_type = CableScreenEddyLossType.Trefoil
    cable.layer_metrics.screen_circulating_loss_type = None
    expected = cable.get_screen_eddy_loss_factor(Ts=40.0, Tc=50.0)
    assert expected > 0.0

    cable.layer_metrics.screen_circulating_loss_type = CableScreenCirculatingLossType.GeneralizedLeading
    cable.weighted_screen_impedance = None
    assert cable.get_screen_eddy_loss_factor(Ts=40.0, Tc=50.0) == expected


def test_each_loss_component_requires_its_own_case(single_core_cable_xlpe):
    cable = single_core_cable_xlpe
    cable.layer_metrics.screen_eddy_loss_type = None
    cable.layer_metrics.screen_circulating_loss_type = CableScreenCirculatingLossType.Zero
    with pytest.raises(ValueError, match="Screen eddy loss method is not set"):
        cable.get_screen_eddy_loss_factor(Ts=40.0, Tc=50.0)

    cable.layer_metrics.screen_eddy_loss_type = CableScreenEddyLossType.Zero
    cable.layer_metrics.screen_circulating_loss_type = None
    with pytest.raises(ValueError, match="Screen circulating loss method is not set"):
        cable.get_screen_circulating_loss_factor(Ts=40.0, Tc=50.0)


@pytest.mark.parametrize("measured_current", [None, 0.0, 25.0])
def test_generalized_milliken_remains_unsupported(single_core_cable_xlpe, measured_current):
    cable = single_core_cable_xlpe
    cable.conductor.surface_type = CableConductorSurfaceType.Milliken
    cable.layer_metrics.screen_circulating_loss_type = CableScreenCirculatingLossType.GeneralizedLeading
    cable.layer_metrics.screen_eddy_loss_type = CableScreenEddyLossType.GeneralizedTwoSided

    with pytest.raises(NotImplementedError, match="Generalized screen loss not implemented for Milliken"):
        cable.get_heat_generation_screen(
            heat_generation_conductor=100.0,
            screen_temperature=40.0,
            conductor_temperature=50.0,
            temperature_dependent_electric_resistance=True,
            custom_screen_current=measured_current,
        )
