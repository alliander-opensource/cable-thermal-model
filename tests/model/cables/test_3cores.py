# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import pytest

from cable_thermal_model.cable.enums.circuit_enums import CircuitType
from cable_thermal_model.cable.schemas.circuit_schemas import CircuitInSoilFromCableInputSchema
from cable_thermal_model.environment.static_env_soil import StaticEnvSoil
from cable_thermal_model.model.cables.cable_soil import CableSoil
from cable_thermal_model.model.cables.enum_classes_cable import CableLayer
from cable_thermal_model.model.model_factory import ModelFactory


def test_single_core_xlpe(single_core_cable_xlpe: CableSoil):
    expected_number_of_cable_layers = 7
    assert len(single_core_cable_xlpe.layers) == expected_number_of_cable_layers
    assert (
        single_core_cable_xlpe.layer_metrics.conductor_radius_original
        == single_core_cable_xlpe.layer_properties[CableLayer.Conductor].outer_radius
    )


def test_single_core_pilc(single_core_cable_pilc: CableSoil):
    expected_number_of_cable_layers = 7
    assert len(single_core_cable_pilc.layers) == expected_number_of_cable_layers
    assert (
        single_core_cable_pilc.layer_metrics.conductor_radius_original
        == single_core_cable_pilc.layer_properties[CableLayer.Conductor].outer_radius
    )


def test_three_core(three_core_cable_pilc: CableSoil):
    expected_number_of_cable_layers = 6
    assert len(three_core_cable_pilc.layers) == expected_number_of_cable_layers
    assert (
        three_core_cable_pilc.layer_properties[CableLayer.Conductor].outer_radius
        > three_core_cable_pilc.layer_metrics.conductor_radius_original
    )
    assert (
        three_core_cable_pilc.layer_properties[CableLayer.Conductor].outer_radius
        < three_core_cable_pilc.layer_properties[CableLayer.Insulation].outer_radius
    )


@pytest.mark.parametrize(
    "load, expected_temperature_conductor, expected_temperature_sheath, soil_thermal_resistivity",
    [
        [50.0, 20.7, 20.2, 0.25],
        [100.0, 23.0, 20.8, 0.25],
        [150.0, 26.9, 21.8, 0.25],
        [200.0, 32.5, 23.3, 0.25],
        [250.0, 40.1, 25.3, 0.25],
        [50.0, 21.1, 20.6, 0.75],
        [100.0, 24.6, 22.4, 0.75],
        [150.0, 30.6, 25.5, 0.75],
        [200.0, 39.5, 30.1, 0.75],
        [250.0, 51.9, 36.4, 0.75],
        [50.0, 21.7, 21.2, 1.50],
        [100.0, 27.1, 24.8, 1.50],
        [150.0, 36.5, 31.2, 1.50],
        [200.0, 50.8, 40.9, 1.50],
        [250.0, 71.5, 55.0, 1.50],
    ],
)
def test_3core_pilc_run(
    load: float,
    expected_temperature_conductor: float,
    expected_temperature_sheath: float,
    soil_thermal_resistivity: float,
    three_core_cable_pilc: CableSoil,
    max_absolute_temperature_error: float,
):
    scenario = pd.DataFrame(index=pd.timedelta_range("0 days", "30000 days", periods=5))
    scenario["ambient_temperature"] = 20.0
    scenario["soil_thermal_capacity"] = 2e6
    scenario["load_c"] = load
    scenario["soil_thermal_resistivity"] = soil_thermal_resistivity

    environment = StaticEnvSoil()
    environment.add_circuit_from_cable(
        CircuitInSoilFromCableInputSchema(
            x=0, y=-0.8, circuit_name="c", cable=three_core_cable_pilc, circuit_type=CircuitType.Single
        )
    )
    model = ModelFactory.create_model(environment, scenario)
    solution = model.run()

    assert np.isclose(
        solution.result[("c", "single")].iloc[-1][CableLayer.Conductor],
        expected_temperature_conductor,
        atol=max_absolute_temperature_error,
    )

    assert np.isclose(
        solution.result[("c", "single")].iloc[-1][CableLayer.Sheath],
        expected_temperature_sheath,
        atol=max_absolute_temperature_error,
    )


@pytest.mark.parametrize(
    "load, expected_temperature_conductor, expected_temperature_sheath, soil_thermal_resistivity",
    [
        [50.0, 20.4, 20.1, 0.25],
        [100.0, 21.6, 20.6, 0.25],
        [200.0, 26.6, 22.4, 0.25],
        [300.0, 35.4, 25.5, 0.25],
        [450.0, 57.5, 33.3, 0.25],
        [50.0, 20.7, 20.4, 0.75],
        [100.0, 22.8, 21.7, 0.75],
        [200.0, 31.6, 27.2, 0.75],
        [300.0, 47.5, 37.1, 0.75],
        [450.0, 91.3, 64.4, 0.75],
        [50.0, 21.1, 20.9, 1.5],
        [100.0, 24.6, 23.5, 1.5],
        [200.0, 39.3, 34.8, 1.5],
        [300.0, 67.8, 56.7, 1.5],
    ],
)
def test_3core_xlpe_run(
    load: float,
    expected_temperature_conductor: float,
    expected_temperature_sheath: float,
    soil_thermal_resistivity: float,
    three_core_cable_xlpe: CableSoil,
    max_absolute_temperature_error: float,
):
    scenario = pd.DataFrame(index=pd.timedelta_range("0 days", "30000 days", periods=5))
    scenario["ambient_temperature"] = 20.0
    scenario["soil_thermal_capacity"] = 2e6
    scenario["load_c"] = load
    scenario["soil_thermal_resistivity"] = soil_thermal_resistivity

    environment = StaticEnvSoil()
    environment.add_circuit_from_cable(
        CircuitInSoilFromCableInputSchema(
            x=0, y=-0.8, circuit_name="c", cable=three_core_cable_xlpe, circuit_type=CircuitType.Single
        )
    )
    model = ModelFactory.create_model(environment, scenario)
    solution = model.run()

    assert np.isclose(
        solution.result[("c", "single")].iloc[-1][CableLayer.Conductor],
        expected_temperature_conductor,
        atol=max_absolute_temperature_error,
    )

    assert np.isclose(
        solution.result[("c", "single")].iloc[-1][CableLayer.Sheath],
        expected_temperature_sheath,
        atol=max_absolute_temperature_error,
    )
