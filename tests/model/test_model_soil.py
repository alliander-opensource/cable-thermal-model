# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

import os
from typing import Any, cast
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError
from pandera.typing import DataFrame
from pydantic import ValidationError

from cable_thermal_model.cable.cable_circuit import (
    BondingType,
    CableKey,
    CablePosition,
    CircuitType,
)
from cable_thermal_model.cable.schemas.circuit_schemas import (
    CircuitConfigurationFromCableId,
    CircuitInSoilFromCableIdInputSchema,
    CircuitInSoilFromCableInputSchema,
)
from cable_thermal_model.cable.schemas.pipe_schemas import PipeInputSchema
from cable_thermal_model.environment.static_env_air import StaticEnvAir
from cable_thermal_model.environment.static_env_soil import StaticEnvSoil
from cable_thermal_model.model.abstract_model import ModelOutputSchema
from cable_thermal_model.model.cables.enum_classes_cable import CableLayer, PipeFillType
from cable_thermal_model.model.cables.fd_cable import FDCable
from cable_thermal_model.model.model import Model
from cable_thermal_model.model.model_air import StateAir
from cable_thermal_model.model.model_soil import ModelSoil, StateSoil
from cable_thermal_model.model.schemas.model_input_schemas import ScenarioSchemaSoil
from cable_thermal_model.model.schemas.run_options import ModelSoilRunOptions
from cable_thermal_model.validation.cable_analysis import CableAnalysis


def test_scenario_validation(single_circuit_env: StaticEnvSoil, scenario_constant: DataFrame[ScenarioSchemaSoil]):
    """Test whether scenario is correctly validated when instantiating a Model instance."""
    # Check whether standard scenario passes the validation
    ModelSoil(single_circuit_env, scenario_constant)

    # check whether error is raised if ambient temperature column is missing
    with pytest.raises(SchemaError):
        ModelSoil(
            single_circuit_env,
            cast(DataFrame[ScenarioSchemaSoil], scenario_constant.drop("ambient_temperature", axis=1)),
        )

    # check whether error is raised if circuit load column is missing
    with pytest.raises(ValueError):
        ModelSoil(single_circuit_env, cast(DataFrame[ScenarioSchemaSoil], scenario_constant.drop("load_c1", axis=1)))

    # check whether error is raised if circuit load column is misspelled
    with pytest.raises(ValueError):
        misspelled_column_scenario = scenario_constant.copy()
        misspelled_column_scenario.columns = ["ambient_temprature", "load_c2"]  # type: ignore[assignment]
        ModelSoil(single_circuit_env, cast(DataFrame[ScenarioSchemaSoil], misspelled_column_scenario))

    # check whether error is raised if there are missing values
    with pytest.raises(SchemaError):
        missing_value_scenario = scenario_constant.copy()
        missing_value_scenario.iloc[4, 1] = np.nan  # set a random value to NaN
        ModelSoil(single_circuit_env, cast(DataFrame[ScenarioSchemaSoil], missing_value_scenario))


@pytest.mark.parametrize(
    "load,conductor_distance,expected_temperatures",
    [(650.0, 0, [80.0, 82.8, 80.1]), (650.0, 0.2, [79.9, 81.8, 80.5])],
)
def test_model_steady_state_linear_circuit(
    load: float,
    conductor_distance: float,
    expected_temperatures: list[float],
    max_absolute_temperature_error: float,
):
    """Test whether steady state temperature matches VCA for a circuits in flat formation."""
    env = StaticEnvSoil()
    env.add_circuit_from_cable_id(
        CircuitInSoilFromCableIdInputSchema(
            x=0.0,
            y=-1.0,
            cable_id="OD 50kV 1x400Cu",
            circuit_name="c",
            circuit_type=CircuitType.Linear,
            dist=conductor_distance,
        )
    )

    scenario = pd.DataFrame(
        index=pd.timedelta_range("0 days", "30000 days", periods=5),
        data={
            "ambient_temperature": 20.0,
            "load_c": load,
            "soil_thermal_resistivity": 1.0,
            "soil_thermal_capacity": 2e6,
        },
    )

    model = ModelSoil(env, ScenarioSchemaSoil.validate(scenario))
    result = model.run(run_options={"neglect_dielectric_loss": True}).result
    # take steady state temperature of the conductor
    for vca_temp, pos in zip(expected_temperatures, ["left", "center", "right"], strict=True):
        ctm_temp = result[("c", f"linear_{pos}")].Conductor.iloc[-1]
        assert np.isclose(vca_temp, ctm_temp, atol=max_absolute_temperature_error)


def test_model_validate_steady_state(scenario_steady_state: DataFrame[ScenarioSchemaSoil]):
    """Test whether the steady state solution matches the heat generation at different radii."""
    env = StaticEnvSoil()
    load = 575.0
    env.add_circuit_from_cable_id(
        CircuitInSoilFromCableIdInputSchema(
            x=0.0,
            y=-1.0,
            cable_id="YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
            circuit_name="c1",
            circuit_type=CircuitType.Trefoil,
        )
    )
    scenario_steady_state["load_c1"] = load

    model = ModelSoil(env, scenario_steady_state)
    steady_state = model.run().state

    # Select a cable from the circuit
    cable_key = next(iter(model.cables_with_soil.keys()))
    cable = model.cables_with_soil[cable_key].cable
    steady_state_solution = steady_state.self_heating_contribution[cable_key]
    steady_state_full_solution = steady_state.temperature[cable_key]

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

    analysis = CableAnalysis(cable=cable, solution=steady_state_solution)

    # In steady state, here the heat flow for conductor screen should be equal to the heat generated at the conductor.
    assert np.isclose(
        heat_generation_conductor,
        analysis.get_heat_flow_cable_layer(CableLayer.ConductorScreen),
        atol=1e-1,
    )

    # In steady state, the heat flowing through the cable boundary should
    # equal the heat generated in the conductor and screen.
    sheath_start_index, sheath_end_index = cable.get_layer_indices_for_layer(CableLayer.Sheath)
    analysis = CableAnalysis(cable=cable, solution=steady_state_solution)
    assert np.isclose(
        heat_generation_conductor + heat_generation_screen,
        analysis.get_heat_flow(inner_index=sheath_end_index - 1),
        atol=5e-1,
    )

    # The heat flux should be constant through every layer outside of the cable in steady state
    assert np.isclose(
        analysis.get_heat_flow(inner_index=sheath_start_index),
        analysis.get_heat_flow(inner_index=sheath_end_index - 1),
        atol=1e-1,
    )


@pytest.mark.parametrize(
    "load, vca_conductor_temperature, rho",
    [
        (100, 9.6, 0.25),
        (200, 11.3, 0.25),
        (400, 18.3, 0.25),
        (600, 30.1, 0.25),
        (800, 49.6, 0.25),
        (1000, 77.1, 0.25),
        (100, 10.4, 0.75),
        (200, 14.5, 0.75),
        (400, 31.9, 0.75),
        (600, 65.0, 0.75),
        (800, 123.0, 0.75),
        (100, 11.5, 1.5),
        (200, 19.4, 1.5),
        (400, 54.2, 1.5),
        (600, 130.2, 1.5),
    ],
)
def test_model_steady_state_vca(
    elst_five_static_env: StaticEnvSoil,
    load: float,
    vca_conductor_temperature: float,
    rho: float,
    max_absolute_temperature_error: int,
):
    """Test Elst 5 situation.

    Test whether the conductor steady state temperatures are correct in the Elst 5 situation where cables of both
    circuits lay 23cm from each other at 729mm depth. Refer to elst_five.csv for more information on the environment.
    """
    sdf = pd.DataFrame(
        index=pd.timedelta_range("0 days", "30000 days", periods=5),
        data={
            "ambient_temperature": 9,
            "load_ELT2.24": float(load),
            "load_ELT2.26": float(load),
            "soil_thermal_resistivity": float(rho),
            "soil_thermal_capacity": 2e6,
        },
    )
    model = ModelSoil(elst_five_static_env, ScenarioSchemaSoil.validate(sdf))
    solution = model.run()

    # 'trefoil_right' is the hottest cable in circuit 'ELT2.24', since it is
    # closest to circuit 'ELT2.26'. The vca_conductor_temperatures are the
    # hottest of the cables as calculated by VCA. Therefore we make the right
    # comparison below.
    assert np.isclose(
        solution.result[("ELT2.24", "trefoil_right")].iloc[-1][CableLayer.Conductor],
        vca_conductor_temperature,
        atol=max_absolute_temperature_error,
    )


@pytest.mark.parametrize(
    "rho,vca_temp,y,pipe_fill_type",
    [
        (0.25, 35.5, -1.15, PipeFillType.Air),
        (0.75, 58.0, -1.15, PipeFillType.Air),
        (1.25, 82.3, -1.15, PipeFillType.Air),
        (0.25, 26.5, -1, PipeFillType.Water),
        (1.25, 71.9, -1, PipeFillType.Water),
        (0.5, 58.0, -5, PipeFillType.Air),
        (0.5, 50.1, -5, PipeFillType.Water),
        (0.5, 55.8, -10, PipeFillType.Water),
        (0.5, 65.1, -30, PipeFillType.Water),
    ],
)
def test_model_steady_state_pipes_vca(
    max_absolute_temperature_error: int, rho: float, y: float, vca_temp: float, pipe_fill_type: PipeFillType
):
    """Test Elst 4 situation.

    Test whether the conductor steady state temperatures are correct in the Elst 4 situation where cables of both
    circuits lay in pipes. Refer to elst_four.csv for more information on the environment.
    """
    scenario = pd.DataFrame(index=pd.timedelta_range("0 days", "30000 days", periods=5))
    scenario["ambient_temperature"] = 9
    scenario["load_ELT2.24"] = 450
    scenario["load_ELT2.26"] = 450
    scenario["soil_thermal_capacity"] = 1

    static_env = StaticEnvSoil()
    static_env.add_circuit_from_cable_id(
        CircuitInSoilFromCableIdInputSchema(
            circuit_name="ELT2.24",
            cable_id="YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
            x=0,
            y=y,
            circuit_type=CircuitType.Trefoil,
            dist=0,
            pipe=PipeInputSchema(
                fill_type=pipe_fill_type,
                inner_radius=0.045,
                outer_radius=0.055,
            ),
        )
    )
    static_env.add_circuit_from_cable_id(
        CircuitInSoilFromCableIdInputSchema(
            circuit_name="ELT2.26",
            cable_id="YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
            x=0.23,
            y=y,
            circuit_type=CircuitType.Trefoil,
            dist=0,
            pipe=PipeInputSchema(
                fill_type=pipe_fill_type,
                inner_radius=0.045,
                outer_radius=0.055,
            ),
        )
    )

    # Constructing static env from the elst_four cable file.

    scenario["soil_thermal_resistivity"] = rho

    # Use the model
    model = ModelSoil(static_env, ScenarioSchemaSoil.validate(scenario))
    solution = model.run(
        run_options=ModelSoilRunOptions(
            ac_current=True,
            temperature_dependent_electric_resistance=True,
            soil_drying=False,
        )
    )

    # 'trefoil_right' is the hottest cable in circuit 'ELT2.24', since it is closest to circuit 'ELT2.26'. The
    # temperatures in VCA_results_Elst4 are also taken from the hottest cable, so we make the correct comparison.
    conductor_temperature = solution.result[("ELT2.24", "trefoil_right")].Conductor.iloc[-1]
    assert np.isclose(conductor_temperature, vca_temp, atol=max_absolute_temperature_error), (
        f"Computed conductor temperature ({conductor_temperature} C) differs from VCA temperature ({vca_temp})."
    )


def test_model_soil_thermal_resistivity_series(single_circuit_env: StaticEnvSoil):
    """Test time varying soil thermal resistivity.

    Test whether the cases with a thermal resistivity that is time varying leads to a higher temperature compared to
    the static soil resistivity.
    """
    static_env = single_circuit_env

    # Solve the heat equation over a time-period of 7 days with time intervals of one hour
    datetime_index = pd.timedelta_range(start="0 days", end="2 days", freq="1h")
    scenario = pd.DataFrame(
        index=datetime_index,
        data={
            "ambient_temperature": 20,
            "soil_thermal_resistivity": 0.75,
            "soil_thermal_capacity": 2e6,
        },
    )
    daily_sine_seconds = datetime_index.total_seconds() / (3600 * 24) * 2 * np.pi
    scenario["load_c1"] = 500 + 200 * np.sin(daily_sine_seconds)
    scenario = ScenarioSchemaSoil.validate(scenario)

    # Taking a static soil resistivity
    model = ModelSoil(static_env, scenario)

    solution = model.run()

    # Set the soil thermal resistivity in this scenario
    # Create a dynamic soil thermal resistivity series starting at 0.75,
    # peaking at 2.0 midway, and going back to 0.75 within 7 days
    scenario["soil_thermal_resistivity"] = 0.75 + 1.25 * np.sin(daily_sine_seconds / 14)

    model_dynamic_soil_thermal_resistivity = ModelSoil(static_env, scenario)
    solution_dynamic_soil_thermal_resistivity = model_dynamic_soil_thermal_resistivity.run()

    # Take the resulting temperatures
    conductor_temperature_base = solution.result[("c1", "trefoil_top")].Conductor
    conductor_temperature_soil_thermal_resistivity = solution_dynamic_soil_thermal_resistivity.result[
        ("c1", "trefoil_top")
    ].Conductor
    # Check if the temperatures are equal or higher everywhere
    assert all(conductor_temperature_base <= conductor_temperature_soil_thermal_resistivity), (
        "Computed conductor temperature with thermal resistivity dynamic has at least the temperature of "
        "constant soil resistivity."
    )

    # Check whether the temperatures are higher at some moments
    assert any(conductor_temperature_base < conductor_temperature_soil_thermal_resistivity), (
        "Computed conductor temperature with thermal resistivity dynamic is greater at some point in time."
    )


def test_add_measurement_point_to_model_soil(model: ModelSoil):
    """Test adding a measurement point to the model."""
    x, y = 1.0, -2.0
    key = model.add_measurement_point(x=x, y=y)

    # Check that the key is in the model's measurement points
    assert key in model.measurement_point_registry.measurement_point_keys

    # Check that the measurement point has the correct coordinates and ndigits
    measurement_point = next((mp for mp in model.measurement_point_registry.points if mp.key == key), None)
    assert measurement_point is not None
    assert measurement_point.key == ("measurement_point", f"x={x:.3f}m", f"y={y:.3f}m")

    # Check that all CableKeys in the model occur in the distances_to_cables of the measurement point
    for cable_key in model.cables_with_soil:
        assert cable_key in measurement_point.distances_to_cables
    for cable_key in model.mirror_cables_with_soil:
        assert cable_key in measurement_point.distances_to_mirror_cables


def test_run_model_soil_with_measurement_points(model: ModelSoil):
    """Test running the model with measurement points."""
    # Add a measurement point
    key1 = model.add_measurement_point(x=0.2, y=-0.8)
    key2 = model.add_measurement_point(x=0.5, y=-0.8)

    # Run the model
    temperature_result = model.run().result

    # Check that the result contains the measurement point keys
    assert key1 in temperature_result.columns
    assert key2 in temperature_result.columns

    # Check that the measurement point results are not empty
    assert not temperature_result[key1].empty
    assert not temperature_result[key2].empty

    # Check that the values exceed the ambient temperature except for the first time step
    ambient_temperature = model.scenario.ambient_temperature.iloc[0]
    assert temperature_result[key1].iloc[0] == ambient_temperature
    assert temperature_result[key2].iloc[0] == ambient_temperature
    assert (temperature_result[key1].iloc[1:] > ambient_temperature).all()
    assert (temperature_result[key2].iloc[1:] > ambient_temperature).all()

    # Check that the values of point 1 are higher than the values of point 2
    # since point 1 is closer to the circuit
    assert (temperature_result[key1].iloc[1:] > temperature_result[key2].iloc[1:]).all()


@pytest.mark.parametrize("cable_id", ["GPLK 10/10 kV 3x185 Al", "YMeKrvaslqwd 12/20kV 1x630 Alrm + as50"])
@pytest.mark.parametrize("temperature_dependent_electric_resistance", [True, False])
@pytest.mark.parametrize("soil_drying", [True, False])
@pytest.mark.parametrize("ac_current", [True, False])
@pytest.mark.parametrize("initial_state", [True, False])
@pytest.mark.parametrize("neglect_dielectric_loss", [True, False])
def test_compute_temperature_solution(
    cable_id: str,
    scenario_constant: DataFrame[ScenarioSchemaSoil],
    temperature_dependent_electric_resistance: bool,
    soil_drying: bool,
    ac_current: bool,
    initial_state: bool,
    neglect_dielectric_loss: bool,
):
    """Performs an end-to-end test for a multitude of cable/model configurations to ensure the output hasn't changed."""
    # Constructing the Static Env and model
    static_env = StaticEnvSoil()
    static_env.add_circuit_from_cable_id(
        CircuitInSoilFromCableIdInputSchema(
            x=0,
            y=-0.8,
            circuit_name="c1",
            cable_id=cable_id,
        )
    )

    model = ModelSoil(static_env, scenario_constant)
    model.run_options = ModelSoilRunOptions(
        temperature_dependent_electric_resistance=temperature_dependent_electric_resistance,
        soil_drying=soil_drying,
        ac_current=ac_current,
        neglect_dielectric_loss=neglect_dielectric_loss,
    )

    # Fill the initial state variable with either the state or None depending on what we are testing.
    initial_state_val = model._compute_temperature_solution().state if initial_state is True else None

    result = model._compute_temperature_solution(initial_state=initial_state_val)

    # Loop over the cable results (e.g. cable_key could be "(c1, trefoil_top)"")
    for column in result.result.columns.droplevel(2).unique():
        circuit_name, cable_position = cast(tuple[str, CablePosition], column)
        # Construct path of the file we are reading
        path_base = (
            f"{neglect_dielectric_loss}_{temperature_dependent_electric_resistance}_{soil_drying}_{ac_current}_"
            f"{initial_state}_{cable_id}_{circuit_name}_{cable_position}.csv"
        )
        path_stripped = (
            path_base.replace("'", "")
            .replace("(", "")
            .replace(")", "")
            .replace(",", "")
            .replace(" ", "")
            .replace("True", "y")
            .replace("False", "n")
            .replace("-", "_")
            .replace("/", "_")
        )
        filepath = os.path.join("test_results/test_compute_temperature_solution", path_stripped)

        # Uncomment this line temporarily if the files need to be updated due to changes in the model
        # result.result[(circuit_name, cable_position)].reset_index(drop=True).to_csv(filepath, index=False)

        # Read in the stored test results
        expected_df = pd.read_csv(filepath)

        # Comparing the results from computation with the expected results from the file.
        # Ignoring the indices as this does not work well when reading/writing to files
        actual_df = cast(pd.DataFrame, result.result[(circuit_name, cable_position)])
        pd.testing.assert_frame_equal(actual_df.reset_index(drop=True), expected_df.reset_index(drop=True))


def test_initializing_thermal_state(model: ModelSoil):
    # Check whether the thermal state components have the correct sizes.
    initial_state = model._build_initial_state()

    self_heating_state = initial_state.self_heating_contribution
    temperature_state = initial_state.temperature
    mutual_heating_state = initial_state.mutual_heating_contribution

    cable_count = len(model.cables_with_soil)
    assert len(self_heating_state) == cable_count
    assert len(mutual_heating_state) == cable_count
    assert len(temperature_state) == cable_count

    for cable_key in model.cables_with_soil:
        assert self_heating_state[cable_key].size == model.cables_with_soil[cable_key].cable._radii_grid.size
        assert temperature_state[cable_key].size == model.cables[cable_key].cable._radii_grid.size
        assert mutual_heating_state[cable_key].size == temperature_state[cable_key].size


@pytest.mark.parametrize("time_idx", [1])
@pytest.mark.parametrize("self_heating_state", [5.0 * np.ones(4)])
@pytest.mark.parametrize("mutual_heating_state", [2.0 * np.ones(3)])
@pytest.mark.parametrize("expected_self_heating_state", [5.0 * np.ones(4)])
@pytest.mark.parametrize("expected_mutual_heating_state", [2.0 * np.ones(3)])
@pytest.mark.parametrize("expected_temperature_state", [(10.0 + 5.0 + 2.0) * np.ones(3)])
def test_update_thermal_state(
    model: ModelSoil,
    time_idx: int,
    self_heating_state: np.ndarray,
    mutual_heating_state: np.ndarray,
    expected_self_heating_state: np.ndarray,
    expected_mutual_heating_state: np.ndarray,
    expected_temperature_state: np.ndarray,
):
    """Simple test to check if all cable states are updated correctly in one call."""
    self_heating_state_map = {cable_key: self_heating_state.copy() for cable_key in model.cables_with_soil}
    mutual_heating_state_map = {cable_key: mutual_heating_state.copy() for cable_key in model.cables}

    current_state = StateSoil(
        static_env_hash=model.static_env.compute_hash(),
        temperature={cable_key: np.zeros_like(mutual_heating_state_map[cable_key]) for cable_key in model.cables},
        self_heating_contribution=self_heating_state_map,
        mutual_heating_contribution=mutual_heating_state_map,
        ambient_temperature=model.scenario["ambient_temperature"].iloc[time_idx],
    )

    model._update_self_heating_contribution = mock.Mock(return_value=self_heating_state_map)
    model._update_mutual_heating_contribution = mock.Mock(return_value=mutual_heating_state_map)

    vectors = {
        cable_key: np.zeros(model.cables_with_soil[cable_key].cable._radii_grid.size - 1)
        for cable_key in model.cables_with_soil
    }

    state = model._update_state(
        state=current_state,
        heat_vectors=vectors,
        time_step=1.0,
        ambient_temperature=model.scenario["ambient_temperature"].iloc[time_idx],
    )

    for cable_key in model.cables:
        assert np.array_equal(state.self_heating_contribution[cable_key], expected_self_heating_state)
        assert np.array_equal(state.mutual_heating_contribution[cable_key], expected_mutual_heating_state)
        assert np.array_equal(state.temperature[cable_key], expected_temperature_state)


def test_get_vector_cables_returns_cables_with_soil(model: ModelSoil):
    """Test that _get_vector_cables returns the soil-extended cable mapping."""
    assert model._cables_for_heat_vectors is model.cables_with_soil


@pytest.mark.parametrize(
    "seconds_since_start,last_update_day,expected_due,expected_day",
    [
        (0.0, 0, False, 0),
        (24 * 60 * 60, 0, True, 1),
        (24 * 60 * 60, 1, False, 1),
        (2.9 * 24 * 60 * 60, 1, True, 2),
    ],
)
def test_check_if_daily_update_due(
    model: ModelSoil,
    seconds_since_start: float,
    last_update_day: int,
    expected_due: bool,
    expected_day: int,
):
    """Test daily-update decision logic around boundaries and multi-day jumps."""
    is_due, updated_day = model._check_if_daily_update_due(
        seconds_since_start_scenario=seconds_since_start,
        last_soil_property_update_day=last_update_day,
    )

    assert is_due is expected_due
    assert updated_day == expected_day


def test_update_soil_properties_for_all_cables_calls_each_cable(model: ModelSoil):
    """Test whether soil property update is forwarded to every soil-extended cable."""
    temperature_state = {
        cable_key: np.ones(pos_cable.cable._radii_grid.size) for cable_key, pos_cable in model.cables_with_soil.items()
    }

    update_mocks = {}
    for pos_cable in model.cables_with_soil.values():
        update_mock = mock.Mock()
        pos_cable.cable.update_soil_properties = update_mock
        update_mocks[pos_cable.name] = update_mock

    model._update_soil_properties_for_all_cables(
        soil_drying=True,
        temperature_state=temperature_state,
        soil_resistivity=1.6,
        soil_capacity=2.5e6,
    )

    for cable_key in model.cables_with_soil:
        update_mocks[cable_key].assert_called_once_with(
            soil_rho=1.6,
            soil_c=2.5e6,
            temperature_grid=temperature_state[cable_key],
            soil_drying=True,
        )


@pytest.mark.parametrize("daily_update_due", [False, True])
def test_update_thermal_properties_if_needed_conditional_soil_update(model: ModelSoil, daily_update_due: bool):
    """Test that soil-property updates are only applied when the daily-update condition is met."""
    temperature_state = {key: np.ones_like(model.cables[key].cable._radii_grid) for key in model.cables}
    scenario_row = model.scenario.iloc[0]

    model._update_pipe_fill_resistivity = mock.Mock()
    model._update_soil_properties_for_all_cables = mock.Mock()
    model._check_if_daily_update_due = mock.Mock(return_value=(daily_update_due, 7))
    model.last_soil_property_update_day = 3

    model._update_thermal_properties_if_needed(
        temperature_state=temperature_state,
        scenario_row=scenario_row,
        elapsed_seconds=12.0,
    )

    assert model._update_pipe_fill_resistivity.call_count == 2
    first_call = model._update_pipe_fill_resistivity.call_args_list[0]
    second_call = model._update_pipe_fill_resistivity.call_args_list[1]
    assert first_call.kwargs["temperature_state"] is temperature_state
    assert second_call.kwargs["temperature_state"] is temperature_state
    assert first_call.kwargs["cables"] is model.cables
    assert second_call.kwargs["cables"] is model.cables_with_soil

    model._check_if_daily_update_due.assert_called_once_with(
        seconds_since_start_scenario=12.0,
        last_soil_property_update_day=3,
    )
    assert model.last_soil_property_update_day == 7

    if daily_update_due:
        model._update_soil_properties_for_all_cables.assert_called_once_with(
            soil_drying=model.run_options.soil_drying,
            temperature_state=temperature_state,
            soil_resistivity=scenario_row[model.THERMAL_RESISTIVITY_COLUMN],
            soil_capacity=scenario_row[model.THERMAL_CAPACITY_COLUMN],
        )
    else:
        model._update_soil_properties_for_all_cables.assert_not_called()


@pytest.mark.parametrize(
    "circuit_fix,scenario_fix,has_pipe,expected_number_of_cables",
    [
        ("single_circuit_env", "scenario_constant", False, 3),
        ("single_circuit_with_pipe_env", "scenario_constant", True, 3),
        ("two_circuit_with_pipe_env", "scenario_constant_multi", True, 6),
        ("two_circuit_env", "scenario_constant_multi", False, 6),
    ],
)
def test_initialize_cables(
    circuit_fix: str, scenario_fix: str, has_pipe: bool, expected_number_of_cables: int, request: pytest.FixtureRequest
):
    """Test the cable initialization in the model to refer to if all params are set correctly."""
    circuit = request.getfixturevalue(circuit_fix)
    scenario = request.getfixturevalue(scenario_fix)
    model = ModelSoil(circuit, scenario)
    assert model.number_of_cables == expected_number_of_cables
    assert model.cables is not None
    assert len(model.cables) == expected_number_of_cables
    assert model.cables_with_soil is not None
    assert len(model.cables_with_soil) == expected_number_of_cables
    assert model.mirror_cables_with_soil is not None
    assert len(model.mirror_cables_with_soil) == expected_number_of_cables


def test_non_uniform_scenario(single_circuit_env: StaticEnvSoil):
    """Test that model works as expected for a scenario with a non-uniform time index.

    Longer time steps lead to less frequent updating of thermal resistance.
    When using a constant load, having a larger step in between should lead to
    lower temperatures following that step.
    """
    data = {
        "ambient_temperature": 10,
        "load_c1": 575,
        "soil_thermal_resistivity": 0.75,
        "soil_thermal_capacity": 2e6,
    }
    uniform_index = pd.timedelta_range("0 min", "40 min", freq="10 min")
    uniform_scenario = ScenarioSchemaSoil.validate(pd.DataFrame(index=uniform_index, data=data))

    # create scenario where length of time steps decreases during scenario, shortening the duration of the scenario.
    # the final temperature should be lower
    longer_non_uniform_index = pd.timedelta_range("0 min", "20 min", freq="10 min").append(
        pd.timedelta_range("25 min", "30 min", freq="5 min")
    )
    longer_scenario = ScenarioSchemaSoil.validate(pd.DataFrame(index=longer_non_uniform_index, data=data))

    # create scenario where length of time steps decreases during scenario, keeping the time of the scenario equal.
    # the final temperature should be higher
    same_length_non_uniform_index = pd.timedelta_range("0 min", "20 min", freq="10 min").append(
        pd.timedelta_range("25 min", "40 min", freq="5 min")
    )
    same_length_scenario = ScenarioSchemaSoil.validate(pd.DataFrame(index=same_length_non_uniform_index, data=data))

    # compute temperatures using both all three scenarios then compare
    temps = {}
    for name, scenario in [
        ("uniform", uniform_scenario),
        ("non_uniform_longer", longer_scenario),
        ("non_uniform_equal", same_length_scenario),
    ]:
        model = ModelSoil(single_circuit_env, scenario)
        temps[name] = model.run().result[("c1", "trefoil_left")]["Conductor"].iloc[-1]
    assert temps["uniform"] < temps["non_uniform_equal"]


def test_add_extra_solution_layer(model: ModelSoil):
    """Test if solution layer is added and is found in the solution of the model."""
    model.add_solution_location(CableLayer.Insulation)
    assert CableLayer.Insulation in model.extra_solution_layers
    solution = model.run()
    assert CableLayer.Insulation in solution.result[("c1", "trefoil_left")].columns


def test_compare_multiple_configs(
    model: Model,
    model_single_config: Model,
    model_multiple_configs: Model,
):
    """Test if solution layer is added and is found in the solution of the model."""
    solution = model.run().result[("c1", "trefoil_right")]
    solution_single_config = model_single_config.run().result[("c1", "trefoil_right")]
    solution_multiple_configs = model_multiple_configs.run().result[("c1", "trefoil_right")]

    assert isinstance(solution, pd.DataFrame)
    assert isinstance(solution_single_config, pd.DataFrame)
    assert isinstance(solution_multiple_configs, pd.DataFrame)

    pd.testing.assert_frame_equal(
        solution,
        solution_single_config,
        atol=1e-10,
    )

    assert (solution_multiple_configs.iloc[-1] > solution.iloc[-1]).all()


def test_model_trefoil_in_single_pipe_two_configurations():
    """Test if model raises error when trefoil in single pipe configuration is invalid."""
    # We should be able to calculate the temperature of the below
    # environment consisting of two configurations at both sections.
    cable_id = "YMeKrvaslqwd 12/20kV 1x630 Alrm + as50"
    multiple_configurations_from_cable_id = [
        CircuitConfigurationFromCableId(
            circuit_type=CircuitType.Trefoil,
            length=1000,
            cable_id=cable_id,
        ),
        CircuitConfigurationFromCableId(
            circuit_type=CircuitType.Trefoil,
            pipe=PipeInputSchema(fill_type=PipeFillType.Air, trefoil_circuit_in_single_pipe=True),
            length=1000,
            cable_id=cable_id,
        ),
    ]

    StaticEnvSoil().add_circuit_from_cable_id(
        CircuitInSoilFromCableIdInputSchema(
            x=0,
            y=-0.8,
            circuit_name="c1",
            cable_id=cable_id,
            circuit_type=CircuitType.Trefoil,
            bonding_type=BondingType.TwoSided,
            multiple_configurations=multiple_configurations_from_cable_id,
        )
    )

    StaticEnvSoil().add_circuit_from_cable_id(
        CircuitInSoilFromCableIdInputSchema(
            x=0,
            y=-0.8,
            circuit_name="c1",
            cable_id=cable_id,
            circuit_type=CircuitType.Trefoil,
            bonding_type=BondingType.TwoSided,
            pipe=PipeInputSchema(fill_type=PipeFillType.Air, trefoil_circuit_in_single_pipe=True),
            multiple_configurations=multiple_configurations_from_cable_id,
        )
    )


def test_model_trefoil_in_single_pipe_three_configurations():
    """Test if model raises error when trefoil in single pipe configuration is invalid."""
    # We should be able to calculate the temperature of the below environment consisting of three configurations
    # at the sections where the trefoil is not in a single pipe.
    cable_id = "YMeKrvaslqwd 12/20kV 1x630 Alrm + as50"
    multiple_configurations_from_cable_id = [
        CircuitConfigurationFromCableId(
            circuit_type=CircuitType.Trefoil,
            length=1000,
            cable_id=cable_id,
        ),
        CircuitConfigurationFromCableId(
            circuit_type=CircuitType.Trefoil,
            pipe=PipeInputSchema(fill_type=PipeFillType.Air, trefoil_circuit_in_single_pipe=True),
            length=1000,
            cable_id=cable_id,
        ),
        CircuitConfigurationFromCableId(
            circuit_type=CircuitType.Linear,
            length=1000,
            cable_id=cable_id,
        ),
    ]

    StaticEnvSoil().add_circuit_from_cable_id(
        CircuitInSoilFromCableIdInputSchema(
            x=0,
            y=-0.8,
            circuit_name="c1",
            cable_id=cable_id,
            circuit_type=CircuitType.Trefoil,
            bonding_type=BondingType.TwoSided,
            multiple_configurations=multiple_configurations_from_cable_id,
        )
    )
    with pytest.raises(
        NotImplementedError,
        match="Non-symmetric sheath currents are not supported for trefoil circuit in a single pipe.",
    ):
        StaticEnvSoil().add_circuit_from_cable_id(
            CircuitInSoilFromCableIdInputSchema(
                x=0,
                y=-0.8,
                circuit_name="c1",
                cable_id=cable_id,
                circuit_type=CircuitType.Trefoil,
                bonding_type=BondingType.TwoSided,
                pipe=PipeInputSchema(fill_type=PipeFillType.Air, trefoil_circuit_in_single_pipe=True),
                multiple_configurations=multiple_configurations_from_cable_id,
            )
        )

    StaticEnvSoil().add_circuit_from_cable_id(
        CircuitInSoilFromCableIdInputSchema(
            x=0,
            y=-0.8,
            circuit_name="c1",
            cable_id=cable_id,
            circuit_type=CircuitType.Linear,
            bonding_type=BondingType.TwoSided,
            multiple_configurations=multiple_configurations_from_cable_id,
        )
    )


cable_ids_with_different_screen_resistance = [
    "YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
    "YMeKrvaslqwd 12/20kV 1x630 Alrm + as35",
]


@pytest.mark.parametrize(
    "local_cable_id",
    cable_ids_with_different_screen_resistance,
)
@pytest.mark.parametrize(
    "first_cable_id",
    cable_ids_with_different_screen_resistance,
)
@pytest.mark.parametrize(
    "second_cable_id",
    cable_ids_with_different_screen_resistance,
)
def test_different_screen_resistance_in_multiple_configurations(
    local_cable_id: str,
    first_cable_id: str,
    second_cable_id: str,
):
    """Test equivalent screen resistance across multiple configurations."""
    multiple_configurations_from_cable_id = [
        CircuitConfigurationFromCableId(
            circuit_type=CircuitType.Trefoil,
            length=1000,
            cable_id=first_cable_id,
        ),
        CircuitConfigurationFromCableId(
            circuit_type=CircuitType.Trefoil,
            length=1000,
            cable_id=second_cable_id,
        ),
    ]
    static_env = StaticEnvSoil()

    if local_cable_id not in [first_cable_id, second_cable_id]:
        with pytest.raises(
            ValueError,
            match="Local configuration does not match any of the provided configurations in multiple_configurations.",
        ):
            static_env.add_circuit_from_cable_id(
                CircuitInSoilFromCableIdInputSchema(
                    x=0,
                    y=-0.8,
                    circuit_name="c1",
                    cable_id=local_cable_id,
                    circuit_type=CircuitType.Trefoil,
                    bonding_type=BondingType.TwoSided,
                    multiple_configurations=multiple_configurations_from_cable_id,
                )
            )

    else:
        if local_cable_id == first_cable_id == second_cable_id:

            def check_function(cable: FDCable):
                assert cable.weighted_screen_impedance is not None
                assert np.isclose(cable.weighted_screen_impedance.weighted_resistance_factor, 1.0)

        elif local_cable_id == "YMeKrvaslqwd 12/20kV 1x630 Alrm + as50":

            def check_function(cable: FDCable):
                assert cable.weighted_screen_impedance is not None
                assert cable.weighted_screen_impedance.weighted_resistance_factor > 1.0

        elif local_cable_id == "YMeKrvaslqwd 12/20kV 1x630 Alrm + as35":

            def check_function(cable: FDCable):
                assert cable.weighted_screen_impedance is not None
                assert cable.weighted_screen_impedance.weighted_resistance_factor < 1.0

        else:
            raise ValueError("Unexpected cable id")

        static_env.add_circuit_from_cable_id(
            CircuitInSoilFromCableIdInputSchema(
                x=0,
                y=-0.8,
                circuit_name="c1",
                cable_id=local_cable_id,
                circuit_type=CircuitType.Trefoil,
                bonding_type=BondingType.TwoSided,
                multiple_configurations=multiple_configurations_from_cable_id,
            )
        )
        for _, cable in static_env.cables.items():
            check_function(cable.cable)


def test_statesoil_validate_mutual_heating_solutions(single_circuit_env, scenario_constant):
    """Test the validate_mutual_heating_solutions validator."""
    # Create an ModelSoil to get real cable representations
    model = ModelSoil(single_circuit_env, scenario_constant)

    # Get cable keys from the model.
    cable_keys = list(model.cables.keys())

    # Create valid mutual heating solutions
    valid_mutual_heating_solutions = {key: np.array([1.0, 2.0, 3.0]) for key in cable_keys}

    # Test case 1: Valid StateSoil should pass upon initialization
    StateSoil(
        static_env_hash=model.static_env.compute_hash(),
        temperature={key: np.array([10.0]) for key in cable_keys},
        self_heating_contribution={key: np.array([10.0]) for key in cable_keys},
        mutual_heating_contribution=valid_mutual_heating_solutions,
        ambient_temperature=5.0,
    )

    # Test case 2: Invalid keys should fail
    wrong_key = CableKey(circuit_name="wrong_circuit", cable_position=CablePosition.Single)
    invalid_mutual_heating = {wrong_key: np.array([1.0, 2.0, 3.0])}

    env_hash = model.static_env.compute_hash()
    temperature = {key: np.array([10.0]) for key in cable_keys}
    self_heating = {key: np.array([10.0]) for key in cable_keys}

    with pytest.raises(ValidationError, match="CableKeys of mutual_heating_contribution should match"):
        StateSoil(
            static_env_hash=env_hash,
            temperature=temperature,
            self_heating_contribution=self_heating,
            mutual_heating_contribution=invalid_mutual_heating,
            ambient_temperature=0.0,
        )


def test_model_soil_validate_state(three_core_cable_xlpe):
    """Test the _validate_state method of ModelSoil."""
    circuit_name = "test_circuit"

    # Create a minimal ModelSoil instance for testing
    env = StaticEnvSoil()
    env.add_circuit_from_cable(
        CircuitInSoilFromCableInputSchema(
            x=0.0,
            y=-1.0,
            circuit_name=circuit_name,
            cable=three_core_cable_xlpe,
        )
    )

    scenario = pd.DataFrame(
        index=pd.timedelta_range("0 days", "1 hour", periods=2),
        data={
            "ambient_temperature": 30,
            "load_test_circuit": 100.0,
            "soil_thermal_resistivity": 1.0,
            "soil_thermal_capacity": 2.0e6,
        },
    )

    model = ModelSoil(env, ScenarioSchemaSoil.validate(scenario))

    # Test 1: state=None should pass
    model._validate_initial_state(None)

    # Test 2: state=StateSoil instance should pass
    pos_cable = env.cables[CableKey(circuit_name=circuit_name, cable_position=CablePosition.Single)]
    cable_key = pos_cable.name

    valid_state = StateSoil(
        static_env_hash=env.compute_hash(),
        temperature={cable_key: np.array([20.0])},
        self_heating_contribution={cable_key: np.array([20.0])},
        mutual_heating_contribution={cable_key: np.array([15.0])},
        ambient_temperature=5.0,
    )

    model._validate_initial_state(valid_state)

    # Test 3: state=StateAir instance should raise ValueError
    invalid_state_air = StateAir(
        static_env_hash=env.compute_hash(),
        temperature={cable_key: np.array([20.0])},
        self_heating_contribution={cable_key: np.array([20.0])},
        ambient_temperature=5.0,
    )

    invalid_state = cast(Any, invalid_state_air)

    with pytest.raises(ValueError, match="ModelSoil requires a StateSoil instance, but received StateAir"):
        model._validate_initial_state(invalid_state)


def test_cable_without_screen(simple_cable: FDCable):
    """Test that when adding a cable without screen, the bonding type is set to NoBonding."""
    # No screen input provided, should be able to create a cable without
    # screen and model should set bonding type to NoBonding.
    static_env = StaticEnvSoil()
    static_env.add_circuit_from_cable(
        CircuitInSoilFromCableInputSchema(
            x=0,
            y=-0.8,
            circuit_name="c1",
            cable=simple_cable,
        )
    )
    static_env.add_circuit_from_cable(
        CircuitInSoilFromCableInputSchema(
            x=0,
            y=-0.8,
            circuit_name="c2",
            cable=simple_cable,
            bonding_type=BondingType.NoBonding,
        )
    )
    static_env.add_circuit_from_cable(
        CircuitInSoilFromCableInputSchema(
            x=0,
            y=-0.8,
            circuit_name="c3",
            cable=simple_cable,
            bonding_type=BondingType.TwoSided,
        )
    )

    for circuit in static_env.circuits.values():
        assert circuit.bonding == BondingType.NoBonding

    scenario = pd.DataFrame(
        index=pd.timedelta_range("0 days", "1 hour", periods=5),
        data={
            "ambient_temperature": 30,
            "load_c1": 100.0,
            "load_c2": 100.0,
            "load_c3": 100.0,
            "soil_thermal_resistivity": 1.0,
            "soil_thermal_capacity": 2.0e6,
        },
    )

    solution = ModelSoil(static_env, ScenarioSchemaSoil.validate(scenario)).run()
    assert isinstance(solution, ModelOutputSchema)


def test_use_wrong_static_env_type():
    """Test that using a wrong static environment type raises an error."""
    with pytest.raises(
        ValueError,
        match=(
            "Can not use model ModelSoil if static environment is not an "
            "environment in soil. Please use ModelAir instead."
        ),
    ):
        ModelSoil(
            static_env=cast(StaticEnvSoil, StaticEnvAir()),
            scenario=cast(DataFrame[ScenarioSchemaSoil], pd.DataFrame()),
        )
