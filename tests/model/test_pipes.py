# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest
from pandera.typing import DataFrame

from cable_thermal_model import CircuitType
from cable_thermal_model.cable.cable_circuit import (
    CableKey,
    CablePosition,
    SingleCable,
)
from cable_thermal_model.cable.enums.circuit_enums import CircuitYReference
from cable_thermal_model.cable.schemas.circuit_schemas import (
    CircuitInAirFromCableIdInputSchema,
    CircuitInSoilFromCableIdInputSchema,
)
from cable_thermal_model.cable.schemas.pipe_schemas import PipeInputSchema
from cable_thermal_model.environment.static_env_air import StaticEnvAir
from cable_thermal_model.environment.static_env_soil import StaticEnvSoil
from cable_thermal_model.model.cables.enum_classes_cable import CableLayer, PipeFillType
from cable_thermal_model.model.cables.fd_cable import FDCableInAir
from cable_thermal_model.model.model_factory import ModelFactory
from cable_thermal_model.model.schemas.model_input_schemas import ScenarioSchemaAir, ScenarioSchemaSoil
from cable_thermal_model.validation.cable_analysis import CableAnalysis
from tests.conftest import vca_pipe_results


def test_trefoil_in_single_pipe_heat_flow(scenario_steady_state: DataFrame[ScenarioSchemaSoil]):
    """Test that a trefoil cable in a single pipe in soil behaves as expected."""
    load = 575.0

    static_env = StaticEnvSoil()
    static_env.add_circuit_from_cable_id(
        CircuitInSoilFromCableIdInputSchema(
            x=0.0,
            y=-1,
            circuit_name="c1",
            cable_id="YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
            pipe=PipeInputSchema(
                fill_type=PipeFillType.Water, outer_radius=0.1, sdr=11, trefoil_circuit_in_single_pipe=True
            ),
            circuit_type=CircuitType.Trefoil,
        )
    )

    # Add load to the scenario
    scenario_steady_state["load_c1"] = load

    # Compute the steady state solution
    model = ModelFactory.create_model(static_env, scenario_steady_state)
    steady_state = model.run().state

    # Select a cable from the circuit
    cable_key = list(model.cables.keys())[0]
    cable = model.cables[cable_key].cable
    steady_state_solution = steady_state.internal_heating_solution[cable_key]
    steady_state_full_solution = steady_state.full_solution[cable_key]

    # Get conductor and screen temperatures
    conductor_temperature = steady_state_full_solution[0]
    screen_start_index, screen_end_index = cable.get_layer_indices_for_layer(CableLayer.Screen)
    screen_temperature = (
        steady_state_full_solution[screen_start_index] + steady_state_full_solution[screen_end_index]
    ) / 2

    # Calculate the heat generated in the conductor and screen
    heat_generation_conductor, heat_generation_screen = cable.get_heat_generation_conductor_and_screen(
        load=load,
        conductor_temperature=conductor_temperature,
        screen_temperature=screen_temperature,
        temperature_dependent_electric_resistance=True,
        ac_current=True,
    )

    # In steady state, the heat generation in the conductor should equal the heat flow outside the conductor
    analysis = CableAnalysis(cable=cable, solution=steady_state_solution)
    assert np.isclose(
        heat_generation_conductor,
        analysis.get_heat_flow_cable_layer(CableLayer.ConductorScreen),
        atol=0.1,
    )

    # In steady state, the heat flow through the pipe should equal three
    # times the sum of the heat generated in the conductor and the screen.
    total_heat_generation = 3 * (heat_generation_conductor + heat_generation_screen)
    assert np.isclose(
        total_heat_generation,
        analysis.get_heat_flow_cable_layer(CableLayer.Pipe),
        atol=0.1,
    )


def test_trefoil_in_single_pipe_in_air_compare_to_soil(scenario_steady_state: DataFrame[ScenarioSchemaSoil]):
    """Compare trefoil circuits in single pipes in air and soil.

    When we ignore the effect of temperature-dependent resistance, the heat flow at the cable boundary should be
    similar for cables in air and soil.
    """
    # Create two static environments: one in soil, one in air
    static_env_soil = StaticEnvSoil()
    static_env_soil.add_circuit_from_cable_id(
        CircuitInSoilFromCableIdInputSchema(
            x=0.0,
            y=-1,
            circuit_name="c1",
            cable_id="YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
            pipe=PipeInputSchema(
                fill_type=PipeFillType.Air, outer_radius=0.1, sdr=11, trefoil_circuit_in_single_pipe=True
            ),
            circuit_type=CircuitType.Trefoil,
        )
    )

    static_env_air = StaticEnvAir()
    static_env_air.add_circuit_from_cable_id(
        CircuitInAirFromCableIdInputSchema(
            circuit_name="c1",
            cable_id="YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
            pipe=PipeInputSchema(
                fill_type=PipeFillType.Air, outer_radius=0.1, sdr=11, trefoil_circuit_in_single_pipe=True
            ),
            circuit_type=CircuitType.Trefoil,
        )
    )

    # Add load to the scenario
    load = 575.0
    scenario_steady_state["load_c1"] = load

    run_options = {"temperature_dependent_electric_resistance": False}

    # Compute the steady state solution for both environments
    model_soil = ModelFactory.create_model(static_env_soil, scenario_steady_state)
    steady_state_soil = model_soil.run(run_options=run_options).state

    model_air = ModelFactory.create_model(static_env_air, ScenarioSchemaAir.validate(scenario_steady_state))
    steady_state_air = model_air.run(run_options=run_options).state

    # Select the single cable from both circuits and collect their steady state solutions
    cable_key = CableKey(circuit_name="c1", cable_position=CablePosition.TrefoilCircuitInSinglePipe)

    cable_soil = model_soil.cables[cable_key].cable
    steady_state_solution_soil = steady_state_soil.internal_heating_solution[cable_key]

    cable_air = model_air.cables[cable_key].cable
    steady_state_solution_air = steady_state_air.internal_heating_solution[cable_key]
    analysis_soil = CableAnalysis(cable=cable_soil, solution=steady_state_solution_soil)
    analysis_air = CableAnalysis(cable=cable_air, solution=steady_state_solution_air)

    # Compare the heat flow just outside the cable conductor for both cables
    heat_flow_conductor_soil = analysis_soil.get_heat_flow_cable_layer(CableLayer.ConductorScreen)
    heat_flow_conductor_air = analysis_air.get_heat_flow_cable_layer(CableLayer.ConductorScreen)
    assert np.isclose(heat_flow_conductor_soil, heat_flow_conductor_air, atol=0.1)

    # Compare the heat flow at the cable boundary for both cables
    heat_flow_soil = analysis_soil.get_heat_flow_cable_layer(CableLayer.Pipe)
    heat_flow_air = analysis_air.get_heat_flow_cable_layer(CableLayer.Pipe)
    assert np.isclose(heat_flow_soil, heat_flow_air, atol=0.1)


def test_trefoil_in_single_pipe_in_air_heat_flow(scenario_steady_state: DataFrame[ScenarioSchemaSoil]):
    """Test that a trefoil cable in a single pipe in air behaves as expected."""
    load = 575.0

    static_env = StaticEnvAir()
    static_env.add_circuit_from_cable_id(
        CircuitInAirFromCableIdInputSchema(
            circuit_name="c1",
            cable_id="YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
            pipe=PipeInputSchema(
                fill_type=PipeFillType.Air, outer_radius=0.1, sdr=11, trefoil_circuit_in_single_pipe=True
            ),
            circuit_type=CircuitType.Trefoil,
        )
    )

    # Add load to the scenario
    scenario_steady_state["load_c1"] = load

    # Compute the steady state solution
    model = ModelFactory.create_model(static_env, ScenarioSchemaAir.validate(scenario_steady_state))
    steady_state = model.run().state

    # Select a cable from the circuit
    cable_key = CableKey(circuit_name="c1", cable_position=CablePosition.TrefoilCircuitInSinglePipe)
    cable = model.cables[cable_key].cable
    steady_state_solution = steady_state.internal_heating_solution[cable_key]
    steady_state_full_solution = steady_state.full_solution[cable_key]

    # Get conductor and screen temperatures
    conductor_temperature = steady_state_full_solution[0]
    screen_start_index, screen_end_index = cable.get_layer_indices_for_layer(CableLayer.Screen)
    screen_temperature = (
        steady_state_full_solution[screen_start_index] + steady_state_full_solution[screen_end_index]
    ) / 2

    # Calculate the heat generated in the conductor and screen
    heat_generation_conductor, heat_generation_screen = cable.get_heat_generation_conductor_and_screen(
        load=load,
        conductor_temperature=conductor_temperature,
        screen_temperature=screen_temperature,
        temperature_dependent_electric_resistance=True,
        ac_current=True,
    )

    # In steady state, the heat generated in the conductor should equal the heat flow for conductor screen
    analysis = CableAnalysis(cable=cable, solution=steady_state_solution)
    assert np.isclose(
        heat_generation_conductor,
        analysis.get_heat_flow_cable_layer(CableLayer.ConductorScreen),
        atol=0.1,
    )

    # In steady state, the heat flow for the pipe should equal three times
    # the sum of the heat generated in the conductor and the screen.
    total_heat_generation = (heat_generation_conductor + heat_generation_screen) * 3
    assert np.isclose(
        total_heat_generation,
        analysis.get_heat_flow_cable_layer(CableLayer.Pipe),
        atol=0.1,
    )


def test_trefoil_in_single_pipe_in_air_norm(scenario_steady_state: DataFrame[ScenarioSchemaSoil]):
    """Test that a trefoil cable in a single pipe in air behaves as expected under standard operation."""
    load = 575.0
    pipe_input_schema = PipeInputSchema(
        fill_type=PipeFillType.Air, outer_radius=0.1, sdr=11, trefoil_circuit_in_single_pipe=True
    )

    static_env = StaticEnvAir()
    static_env.add_circuit_from_cable_id(
        CircuitInAirFromCableIdInputSchema(
            circuit_name="c1",
            cable_id="YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
            pipe=pipe_input_schema,
            circuit_type=CircuitType.Trefoil,
        )
    )

    # Add load to the scenario
    scenario_steady_state["load_c1"] = load

    # Compute the steady state solution
    model = ModelFactory.create_model(static_env, ScenarioSchemaAir.validate(scenario_steady_state))
    steady_state = model.run().state

    # Select a cable from the circuit
    cable_key = list(model.cables.keys())[0]
    cable = model.cables[cable_key].cable
    assert isinstance(cable, FDCableInAir)
    assert cable.convection_coefficient is not None
    steady_state_solution = steady_state.internal_heating_solution[cable_key]

    # The expected heat exchange at the boundary is given by the IEC norm as theta_N/T_4
    theta_N = steady_state_solution[-1]
    T4 = 1 / (np.pi * cable.layer_metrics.outer_radius * 2 * cable.convection_coefficient * theta_N ** (1 / 4))

    expected_heat_exchange = theta_N / T4

    # The heat flow for the pipe should equal the norm heat exchange
    heat_flow = CableAnalysis(cable=cable, solution=steady_state_solution).get_heat_flow_cable_layer(CableLayer.Pipe)
    assert np.isclose(heat_flow, expected_heat_exchange, atol=0.01)


@pytest.mark.parametrize(
    (
        "cable_id, pipe_outer_radius, pipe_fill_type, load, "
        "conductor_temperature, pipe_temperature, "
        "trefoil_circuit_in_single_pipe, cable_position"
    ),
    [
        (
            "YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
            0.055,
            PipeFillType.Air,
            661.7,
            90.0,
            58.1,
            False,
            CablePosition.TrefoilLeft,
        ),  # Single cable, three air-filled 110mm pipe case
        # Three cables in one pipe case
        (
            "YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
            0.1,
            PipeFillType.Water,
            575.0,
            56.0,
            38.7,
            True,
            CablePosition.TrefoilCircuitInSinglePipe,
        ),
        (
            "YMeKrvaslqwd 12/20kV 3x240 Alrm + as50",
            0.080,
            PipeFillType.Water,
            427.8,
            90.0,
            50.5,
            False,
            CablePosition.Single,
        ),
    ],
)
def test_pipe_b5901_cases(
    b5901_scenario_steady_state: DataFrame[ScenarioSchemaSoil],
    max_absolute_temperature_error: float,
    cable_id: str,
    pipe_outer_radius: float,
    pipe_fill_type: PipeFillType,
    load: float,
    conductor_temperature: float,
    pipe_temperature: float,
    trefoil_circuit_in_single_pipe: bool,
    cable_position: CablePosition,
):
    """Test the pipe model for the B5901 cases.

    Three cases should be tested:
    1. single core cable, three air-filled 110mm pipe
    2. single core cable, one water-filled 200mm pipe
    3. three-core cable, one 160mm water-filled pipe
    """
    pipe = PipeInputSchema(
        fill_type=pipe_fill_type,
        outer_radius=pipe_outer_radius,
        sdr=11,
        trefoil_circuit_in_single_pipe=trefoil_circuit_in_single_pipe,
    )

    environment = StaticEnvSoil()
    environment.add_circuit_from_cable_id(
        CircuitInSoilFromCableIdInputSchema(
            x=0, y=-1.0, cable_id=cable_id, circuit_name="c1", pipe=pipe, y_ref=CircuitYReference.Top
        )
    )

    # Compute the steady state solution
    b5901_scenario_steady_state["load_c1"] = load
    model = ModelFactory.create_model(environment, b5901_scenario_steady_state)

    temperature_solution = model.run().result[("c1", cable_position)]
    steady_state_temperatures = temperature_solution.iloc[-1]

    # Check that the temperatures match the VCA results
    assert np.isclose(
        steady_state_temperatures["Conductor"], conductor_temperature, atol=max_absolute_temperature_error
    )
    assert np.isclose(steady_state_temperatures["Pipe"], pipe_temperature, atol=max_absolute_temperature_error)


@pytest.mark.parametrize(
    "cable_id, pipe_outer_radius, sdr, pipe_fill_type, load, conductor_temperature, pipe_temperature",
    vca_pipe_results(),
)
def test_pipe_model_steady_state_vca(
    b5901_scenario_steady_state: DataFrame[ScenarioSchemaSoil],
    cable_id: str,
    pipe_outer_radius: float,
    sdr: float,
    pipe_fill_type: PipeFillType,
    load: float,
    conductor_temperature: float,
    pipe_temperature: float,
    max_absolute_temperature_error: float,
):
    """Test that the pipe model matches the VCA results for a given cable and pipe configuration."""
    # Setup test environment
    pipe = PipeInputSchema(
        fill_type=pipe_fill_type,
        outer_radius=pipe_outer_radius,
        sdr=sdr,
    )

    environment = StaticEnvSoil()
    environment.add_circuit_from_cable_id(
        CircuitInSoilFromCableIdInputSchema(x=0, y=-1.0, cable_id=cable_id, circuit_name="c1", pipe=pipe)
    )
    b5901_scenario_steady_state["load_c1"] = load

    # Compute the steady state solution
    cable_position = (
        CablePosition.Single if isinstance(environment.circuits["c1"], SingleCable) else CablePosition.TrefoilLeft
    )

    model = ModelFactory.create_model(environment, b5901_scenario_steady_state)
    temperature_solution = model.run().result[("c1", cable_position.value)]
    steady_state_temperatures = temperature_solution.iloc[-1]

    # Check that the temperatures match the VCA results
    assert np.isclose(
        steady_state_temperatures["Conductor"], conductor_temperature, atol=max_absolute_temperature_error
    )
    assert np.isclose(steady_state_temperatures["Pipe"], pipe_temperature, atol=max_absolute_temperature_error)


def test_two_trefoil_circuits_in_single_pipes_vca(
    b5901_scenario_steady_state: DataFrame[ScenarioSchemaSoil], max_absolute_temperature_error: float
):
    load = 575.0

    # Add two trefoil circuits in single pipes to the environment
    static_env = StaticEnvSoil()
    static_env.add_circuit_from_cable_id(
        CircuitInSoilFromCableIdInputSchema(
            x=0.0,
            y=-1,
            circuit_name="c1",
            cable_id="YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
            pipe=PipeInputSchema(
                fill_type=PipeFillType.Water, outer_radius=0.1, sdr=11, trefoil_circuit_in_single_pipe=True
            ),
            circuit_type=CircuitType.Trefoil,
        )
    )
    static_env.add_circuit_from_cable_id(
        CircuitInSoilFromCableIdInputSchema(
            x=0.5,
            y=-1,
            circuit_name="c2",
            cable_id="YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
            pipe=PipeInputSchema(
                fill_type=PipeFillType.Water, outer_radius=0.1, sdr=11, trefoil_circuit_in_single_pipe=True
            ),
            circuit_type=CircuitType.Trefoil,
        )
    )

    # Add load to the scenario
    b5901_scenario_steady_state["load_c1"] = load
    b5901_scenario_steady_state["load_c2"] = load

    # Compute the steady state solution
    model = ModelFactory.create_model(static_env, b5901_scenario_steady_state)
    result = model.run().result

    conductor_temperature_1 = result[("c1", CablePosition.TrefoilCircuitInSinglePipe)]["Conductor"].iloc[-1]
    conductor_temperature_2 = result[("c2", CablePosition.TrefoilCircuitInSinglePipe)]["Conductor"].iloc[-1]

    pipe_temperature_1 = result[("c1", CablePosition.TrefoilCircuitInSinglePipe)]["Pipe"].iloc[-1]
    pipe_temperature_2 = result[("c2", CablePosition.TrefoilCircuitInSinglePipe)]["Pipe"].iloc[-1]

    # Check that the temperatures match the VCA results
    expected_conductor_temperature = 67.8  # VCA result for conductor temperature
    expected_pipe_temperature = 50.1  # VCA result for pipe temperature

    assert np.isclose(conductor_temperature_1, expected_conductor_temperature, atol=max_absolute_temperature_error)
    assert np.isclose(conductor_temperature_2, expected_conductor_temperature, atol=max_absolute_temperature_error)
    assert np.isclose(pipe_temperature_1, expected_pipe_temperature, atol=max_absolute_temperature_error)
    assert np.isclose(pipe_temperature_2, expected_pipe_temperature, atol=max_absolute_temperature_error)
