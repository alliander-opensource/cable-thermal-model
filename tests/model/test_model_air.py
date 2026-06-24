# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import pytest
from pandera.typing import DataFrame

from cable_thermal_model import CableLayer, CircuitType, ModelFactory, StaticEnvAir, StaticEnvSoil
from cable_thermal_model.cable.cable_circuit import (
    CableKey,
    CablePosition,
    PosCable,
)
from cable_thermal_model.cable.schemas.circuit_schemas import (
    CircuitInAirFromCableIdInputSchema,
    CircuitInSoilFromCableIdInputSchema,
)
from cable_thermal_model.model.model_air import ModelAir, StateAir
from cable_thermal_model.model.model_soil import StateSoil
from cable_thermal_model.model.schemas.model_input_schemas import ScenarioSchemaAir, ScenarioSchemaSoil
from cable_thermal_model.validation.cable_analysis import CableAnalysis


@pytest.mark.parametrize(
    "load,cable_id,circuit_type,expected_temperature",
    [
        (1000.0, "YMeKrvaslqwd 12/20kV 1x630 Alrm + as50", "linear", 112.7),
        (800, "YMeKrvaslqwd 12/20kV 1x630 Alrm + as50", "linear_vertical", 82.9),
        (445, "YMeKrvaslqwd 12/20kV 3x240 Alrm + as50", "single", 90.4),
    ],
)
def test_model_steady_state(
    load: float,
    cable_id: str,
    circuit_type: CircuitType,
    expected_temperature: float,
    max_absolute_temperature_error: int,
):
    """Test whether steady state temperature matches VCA for a circuit in air."""
    env = StaticEnvAir()
    env.add_circuit_from_cable_id(
        CircuitInAirFromCableIdInputSchema(
            cable_id=cable_id,
            circuit_name="c",
            circuit_type=circuit_type,
        )
    )

    scenario = pd.DataFrame(
        index=pd.timedelta_range("0 days", "24 hours", periods=97),
        data={
            "ambient_temperature": 30,
            "load_c": load,
        },
    )

    model = ModelAir(env, ScenarioSchemaAir.validate(scenario))
    solution = model.run()
    result = solution.result
    # First we get all the cables for test circuit 'c'
    circuit_c_cables = list(set(list(result.columns.get_level_values(1))))
    ctm_temp = max([result["c"][cable_key][CableLayer.Conductor].iloc[-1] for cable_key in circuit_c_cables])
    assert np.isclose(expected_temperature, ctm_temp, atol=max_absolute_temperature_error)

    cable_key = list(model.cables.keys())[0]
    cable = model.cables[cable_key].cable
    cable_full_solution = solution.state.full_solution[cable_key]
    conductor_start_index, conductor_end_index = cable.get_layer_indices_for_layer(CableLayer.Conductor)
    screen_start_index, screen_end_index = cable.get_layer_indices_for_layer(CableLayer.Screen)
    Tc = (cable_full_solution[conductor_start_index] + cable_full_solution[conductor_end_index]) / 2
    Ts = (cable_full_solution[screen_start_index] + cable_full_solution[screen_end_index]) / 2
    heat_generation_conductor, heat_generation_screen = cable.get_heat_generation_conductor_and_screen(
        load=load,
        conductor_temperature=Tc,
        screen_temperature=Ts,
        temperature_dependent_electric_resistance=True,
        ac_current=True,
    )
    heat_generation_insulation = cable.get_dielectric_loss_for_cable()

    total_heat_generation = heat_generation_conductor + heat_generation_screen + heat_generation_insulation

    # Determine the heat flow for the sheath layer, which should equal the total heat generation in steady state
    heat_flow_for_sheath = CableAnalysis(cable=cable, solution=cable_full_solution).get_heat_flow_cable_layer(
        CableLayer.Sheath
    )

    assert np.isclose(total_heat_generation, heat_flow_for_sheath)


def test_single_cable_in_air_compare_to_soil(scenario_steady_state: DataFrame[ScenarioSchemaSoil]):
    """Compare single cables in air and soil.

    When we ignore the effect of temperature-dependent resistance, the heat flux at the cable boundary should be
    similar for cables in air and soil.
    """
    # Create two static environments: one in soil, one in air
    static_env_soil = StaticEnvSoil()
    static_env_soil.add_circuit_from_cable_id(
        CircuitInSoilFromCableIdInputSchema(
            x=0.0,
            y=-1,
            circuit_name="c1",
            cable_id="YMeKrvaslqwd 12/20kV 3x240 Alrm + as50",
            circuit_type=CircuitType.Single,
        )
    )

    static_env_air = StaticEnvAir()
    static_env_air.add_circuit_from_cable_id(
        CircuitInAirFromCableIdInputSchema(
            circuit_name="c1",
            cable_id="YMeKrvaslqwd 12/20kV 3x240 Alrm + as50",
            circuit_type=CircuitType.Single,
        )
    )

    # Add load to the scenario
    load = 575.0
    scenario_steady_state["load_c1"] = load

    # Compute the steady state solution for both environments
    model_soil = ModelFactory.create_model(static_env_soil, scenario_steady_state)
    steady_state_soil = model_soil.run(run_options={"temperature_dependent_electric_resistance": False}).state

    model_air = ModelFactory.create_model(static_env_air, ScenarioSchemaAir.validate(scenario_steady_state))
    steady_state_air = model_air.run(run_options={"temperature_dependent_electric_resistance": False}).state

    # Select the single cable from both circuits and collect their steady state solutions
    cable_key = CableKey(circuit_name="c1", cable_position=CablePosition.Single)

    cable_soil = model_soil.cables[cable_key].cable
    steady_state_solution_soil = steady_state_soil.internal_heating_solution[cable_key]

    cable_air = model_air.cables[cable_key].cable
    steady_state_solution_air = steady_state_air.internal_heating_solution[cable_key]

    cable_analysis_soil = CableAnalysis(cable=cable_soil, solution=steady_state_solution_soil)
    cable_analysis_air = CableAnalysis(cable=cable_air, solution=steady_state_solution_air)

    # Compare the heat flow just outside the cable conductor for both cables
    conductor_radius = cable_soil.layer_properties[CableLayer.Conductor].outer_radius
    heat_flow_conductor_soil = cable_analysis_soil.get_heat_flow_at_radius(r=conductor_radius)
    heat_flow_conductor_air = cable_analysis_air.get_heat_flow_at_radius(r=conductor_radius)
    assert np.isclose(heat_flow_conductor_soil, heat_flow_conductor_air, atol=0.01)

    # Compare the heat flow at the sheath layer for both cables, should be almost identical
    heat_flow_soil = cable_analysis_soil.get_heat_flow_cable_layer(CableLayer.Sheath)
    heat_flow_air = cable_analysis_air.get_heat_flow_cable_layer(CableLayer.Sheath)

    assert np.isclose(heat_flow_soil, heat_flow_air, atol=0.01)


def test_stateair_validate_single_circuit(single_core_cable_xlpe):
    """Test that StateAir validator allows single circuit and rejects multiple circuits."""
    # Test 1: Single circuit should pass
    pos_cable_single = PosCable(
        circuit_name="circuit_1", cable_position=CablePosition.Single, cable=single_core_cable_xlpe, x=0.0, y=0.0
    )

    # Create keys for the solution dictionaries
    cable_key_single = pos_cable_single.name

    StateAir(
        cable_representations=[pos_cable_single],
        full_solution={cable_key_single: np.array([20.0])},
        internal_heating_solution={cable_key_single: np.array([20.0])},
    )

    # Test 2: Multiple circuits should fail
    pos_cable_1 = PosCable(
        circuit_name="circuit_1", cable_position=CablePosition.TrefoilLeft, cable=single_core_cable_xlpe, x=0.0, y=0.0
    )
    pos_cable_2 = PosCable(
        circuit_name="circuit_2", cable_position=CablePosition.TrefoilRight, cable=single_core_cable_xlpe, x=1.0, y=0.0
    )

    # Create keys for the solution dictionaries
    cable_key_1 = pos_cable_1.name
    cable_key_2 = pos_cable_2.name

    # This should fail - there are multiple circuits in the cable representations
    with pytest.raises(ValueError, match="StateAir should only contain one circuit"):
        StateAir(
            cable_representations=[pos_cable_1, pos_cable_2],
            full_solution={cable_key_1: np.array([20.0]), cable_key_2: np.array([25.0])},
            internal_heating_solution={cable_key_1: np.array([20.0]), cable_key_2: np.array([25.0])},
        )


def test_model_air_validate_state(single_core_cable_xlpe):
    """Test the _validate_state method of ModelAir."""
    # Create a minimal ModelAir instance for testing
    env = StaticEnvAir()
    circuit_name = "test_circuit"
    env.add_circuit_from_cable_id(
        CircuitInAirFromCableIdInputSchema(
            cable_id="YMeKrvaslqwd 12/20kV 3x240 Alrm + as50",
            circuit_name=circuit_name,
            circuit_type=CircuitType.Single,
        )
    )

    scenario = pd.DataFrame(
        index=pd.timedelta_range("0 days", "1 hour", periods=2),
        data={"ambient_temperature": 30, f"load_{circuit_name}": 100.0},
    )

    model = ModelAir(env, ScenarioSchemaAir.validate(scenario))
    # Mock the output of model.compute_temperature_result() to prevent the need for a full model run
    model._compute_temperature_solution = lambda initial_state: None

    # Test 1: state=None should pass
    model.run(initial_state=None)

    # Test 2: state=StateAir instance should pass
    pos_cable = env.cables[CableKey(circuit_name=circuit_name, cable_position=CablePosition.Single)]
    cable_key = pos_cable.name

    valid_state = StateAir(
        cable_representations=[pos_cable],
        full_solution={cable_key: np.array([20.0])},
        internal_heating_solution={cable_key: np.array([20.0])},
    )

    model.run(initial_state=valid_state)

    # Test 3: state=StateSoil instance should raise ValueError
    invalid_state_soil = StateSoil(
        cable_representations=[pos_cable],
        full_solution={cable_key: np.array([20.0])},
        internal_heating_solution={cable_key: np.array([20.0])},
        mutual_heating_solutions={cable_key: np.array([15.0])},
    )

    with pytest.raises(ValueError, match="ModelAir requires a StateAir instance, but received StateSoil"):
        model.run(initial_state=invalid_state_soil)


def test_use_wrong_static_env_type():
    """Test that using a wrong static environment type raises an error."""
    with pytest.raises(
        ValueError,
        match=(
            "Can not use model 'ModelAir' if static environment is not an "
            "environment in air. Please use ModelSoil instead."
        ),
    ):
        ModelAir(static_env=StaticEnvSoil(), scenario=pd.DataFrame())
