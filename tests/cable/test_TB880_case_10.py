# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from cable_thermal_model.cable.cable_circuit import CableKey, CablePosition
from cable_thermal_model.model.cables.enum_classes_cable import CableLayer, CableScreenLossType
from cable_thermal_model.model.cables.fd_cable import FDCable
from cable_thermal_model.model.model_soil import ModelSoil
from cable_thermal_model.validation.cable_analysis import CableAnalysis

RELATIVE_TOLERANCE = 0.001


def test_build_cable_fixture(TB880_case_10_fd_cable: FDCable):
    """Test the build of the FDCable fixture for TB880 case 0.

    This is a test of the build process up to section 14-2, not the
    results of the calculations.
    """
    k_s, k_p = TB880_case_10_fd_cable._select_k_s_and_k_p()
    frequency = TB880_case_10_fd_cable.omega / (2 * np.pi)
    layer_properties = TB880_case_10_fd_cable.layer_properties
    layer_metrics = TB880_case_10_fd_cable.layer_metrics

    # Table 14-1
    assert np.isclose(TB880_case_10_fd_cable.get_dc_resistance_conductor(Tc=20), 0.320e-3, rtol=RELATIVE_TOLERANCE)
    assert np.isclose(k_s, 1.0, rtol=RELATIVE_TOLERANCE)
    assert np.isclose(k_p, 0.8, rtol=RELATIVE_TOLERANCE)
    assert np.isclose(layer_properties[CableLayer.Conductor].alpha, 4.03e-3, rtol=RELATIVE_TOLERANCE)
    assert np.isclose(frequency, 50.0, rtol=RELATIVE_TOLERANCE)

    conductor_operating_temperature = 50
    # 14.2 step 1.
    R_dc = TB880_case_10_fd_cable.get_dc_resistance_conductor(Tc=conductor_operating_temperature)
    assert np.isclose(R_dc, 3.58688e-4, rtol=RELATIVE_TOLERANCE)

    # 14.2 step 2.
    x_coefficient = TB880_case_10_fd_cable._calculate_x_coefficient(Rdc=R_dc)
    x_s_squared = x_coefficient * k_s
    assert np.isclose(x_s_squared, 0.3503426547, rtol=RELATIVE_TOLERANCE)

    # 14.2 step 3.
    y_s = TB880_case_10_fd_cable._calculate_y_s_from_x_s(x_s=x_s_squared**0.5)
    assert np.isclose(y_s, 6.3894394042e-4, rtol=RELATIVE_TOLERANCE)

    # 14.2 step 4.
    d_x = 2 * layer_metrics.conductor_radius_original
    assert np.isclose(d_x, 11.0e-3, rtol=RELATIVE_TOLERANCE)

    # 14.2 step 5.
    s = d_x + TB880_case_10_fd_cable.s
    assert np.isclose(s, 15.7e-3, rtol=RELATIVE_TOLERANCE)

    # 14.2 step 6.
    x_p_squared = x_coefficient * k_p
    assert np.isclose(x_p_squared, 0.2802741238, rtol=RELATIVE_TOLERANCE)

    # 14.2 step 7.
    y_p = TB880_case_10_fd_cable._calculate_y_p_from_x_p(d=d_x, s=s, x_p=x_p_squared**0.5, factor=2 / 3)
    assert np.isclose(y_p, 6.0458825885e-4, rtol=RELATIVE_TOLERANCE)

    # 14.2 step 8.
    R_ac = TB880_case_10_fd_cable._get_ac_resistance_conductor_from_dc_resistance(Rdc=R_dc, s=TB880_case_10_fd_cable.s)
    assert np.isclose(R_ac, 3.5935706012e-4, rtol=RELATIVE_TOLERANCE)


def test_calculate_loss_for_lead_sheath(TB880_case_10_fd_cable: FDCable):
    TB880_case_10_fd_cable.layer_metrics.screen_loss_type = CableScreenLossType.SingleCablePILC

    # 14.4 step 1.
    assert np.isclose(TB880_case_10_fd_cable._get_resistance_screen(Ts=20), 8.9629362689e-4, rtol=RELATIVE_TOLERANCE)

    # 14.4 step 2. Skipped as this requires rating calculations.
    Tc = 50.0
    Ts = 42.0424550288

    # 14.4 step 3.
    assert np.isclose(TB880_case_10_fd_cable._get_resistance_screen(Ts=Ts), 9.7531967474e-4, rtol=RELATIVE_TOLERANCE)

    # 14.4 step 5.
    assert np.isclose(TB880_case_10_fd_cable.d, 38e-3, rtol=RELATIVE_TOLERANCE)
    lambda_1 = TB880_case_10_fd_cable.get_cable_screen_loss_factor(Ts=Ts, Tc=Tc)
    lambda_1_eddy = lambda_1 / TB880_case_10_fd_cable.Ft
    assert np.isclose(lambda_1_eddy, 1.9821141851e-3, rtol=RELATIVE_TOLERANCE)

    # 14.4 step 6.
    assert TB880_case_10_fd_cable.layer_metrics.armour_cross_section is not None
    assert np.isclose(
        TB880_case_10_fd_cable.layer_metrics.armour_cross_section, 159.5091309983e-6, rtol=RELATIVE_TOLERANCE
    )
    assert np.isclose(TB880_case_10_fd_cable.Ft, 2.4173685240, rtol=RELATIVE_TOLERANCE)

    # 14.4 step 7.
    assert np.isclose(lambda_1, 4.7915004419e-3, rtol=RELATIVE_TOLERANCE)


def test_calculate_thermal_resistances(
    TB880_case_10_model: ModelSoil, TB880_case_10_steady_state_full_solution: np.ndarray
):
    TB880_case_10_fd_cable = TB880_case_10_model.cables[
        CableKey(circuit_name="TB880_case_10", cable_position=CablePosition.Single)
    ].cable
    assert CableLayer.ConductorScreen not in TB880_case_10_fd_cable.layers
    assert CableLayer.InsulationScreen not in TB880_case_10_fd_cable.layers
    analysis = CableAnalysis(cable=TB880_case_10_fd_cable, solution=TB880_case_10_steady_state_full_solution)

    t1 = analysis.get_thermal_resistance_cable_layer(layer=CableLayer.Insulation)
    assert np.isclose(3 * t1, 0.804568981, rtol=RELATIVE_TOLERANCE)

    t2 = analysis.get_thermal_resistance_cable_layer(layer=CableLayer.Bedding)
    assert np.isclose(t2, 0.1334628239, rtol=RELATIVE_TOLERANCE)

    t3 = analysis.get_thermal_resistance_cable_layer(layer=CableLayer.Sheath)
    assert np.isclose(t3, 0.0886807855, rtol=RELATIVE_TOLERANCE)

    t4 = analysis.get_thermal_resistance_external_medium(
        ambient_temperature=TB880_case_10_model.scenario["ambient_temperature"].iloc[-1]
    )
    assert np.isclose(t4, 0.6850633170, rtol=RELATIVE_TOLERANCE)
