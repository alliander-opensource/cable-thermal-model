# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

import os
from typing import cast
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError
from pandera.typing import DataFrame

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
    cable_key = next(iter(model.cables.keys()))
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


@pytest.mark.parametrize(
    "temperature_solution_max, temperature_solution_min",
    [
        (10.0, 0.0),
        (50.0, 0.0),
        (100.0, 0.0),
        (50.0, 40.0),
    ],
)
def test_get_dry_soil_radius_around_circuit(
    model: ModelSoil,
    temperature_solution_min: float,
    temperature_solution_max: float,
):
    """Tests whether the soil dries out around if the temperature is above a certain limit."""
    assert temperature_solution_max >= temperature_solution_min

    cable_key = list(model.cables.keys())[0]
    radii_grid = model.cables[cable_key].cable.radii_grid
    full_solutions = [
        np.linspace(temperature_solution_max, temperature_solution_min, len(radii_grid))
        for _ in range(len(model.cables))
    ]

    if temperature_solution_max < model._SOIL_DRYING_TEMPERATURE:
        expected_soil_radius = 0.0
    elif temperature_solution_min >= model._SOIL_DRYING_TEMPERATURE:
        expected_soil_radius = radii_grid[-1]
    else:
        expected_soil_radius_index = int(
            (temperature_solution_max - model._SOIL_DRYING_TEMPERATURE)
            / (temperature_solution_max - temperature_solution_min)
            * (len(radii_grid) - 1)
        )
        expected_soil_radius = radii_grid[expected_soil_radius_index]

    # Check the result for the first cable
    dry_soil_result = model._get_dry_soil_radius_around_circuit(
        full_solutions=full_solutions,
        cables=[cable for cable in model.cables.values()],
    )
    assert np.isclose(dry_soil_result, expected_soil_radius)


def test_get_temp(model: ModelSoil):
    """Test get_temp function returns a temperature value."""
    # Simple coordinates
    x, y = 0.0, 1.0

    # Use a time that exists in the scenario (first time step in seconds)
    time_sec = 0.0

    # Create simple solutions for all cable keys
    solutions = {}
    for cable_key in model.cables:
        # Create a simple solution array matching the cable's radii grid size
        grid_size = model.cables[cable_key].cable.radii_grid.size
        solutions[cable_key] = np.full(grid_size, 10.0)  # 10°C heating everywhere

    # Call the function
    temperature = model.get_temp(x, y, time_sec, solutions)

    # Check that we get a temperature value
    assert isinstance(temperature, float)


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
        actual_df = result.result[(circuit_name, cable_position)]
        assert isinstance(actual_df, pd.DataFrame)
        pd.testing.assert_frame_equal(actual_df.reset_index(drop=True), expected_df.reset_index(drop=True))


def test_initializing_solutions(model: ModelSoil):
    # Test the shape of the initial solutions lists
    solutions, full_solutions = model._initialize_solutions_lists(initial_state=None)
    mutual_heating_solutions = model._initialize_solutions_lists_mutual_heating(initial_state=None)
    # Environment with one circuit with trefoil formation: 3 cables:
    cable_count = len(model.cables)
    assert len(solutions) == cable_count
    assert len(mutual_heating_solutions) == cable_count
    assert len(full_solutions) == cable_count
    for cable_key in model.cables:
        assert full_solutions[cable_key].size == model.cables_full_solutions[cable_key].cable.radii_grid.size
        assert mutual_heating_solutions[cable_key].size == full_solutions[cable_key].size
        assert solutions[cable_key].size > full_solutions[cable_key].size
        assert solutions[cable_key].size == model.cables[cable_key].cable.radii_grid.size


def test_initializing_linear_system(model: ModelSoil):
    # Check whether the output matrices have the correct sizes.
    matrices, vectors = model._initialize_linear_system()
    matrices_without_soil = {
        key: cable.cable.get_finite_differences_matrix() for key, cable in model.cables_full_solutions.items()
    }
    # Environment with one circuit with trefoil formation: 3 cables:
    cable_count = len(model.cables)
    assert len(matrices) == cable_count
    assert len(vectors) == cable_count
    assert len(matrices_without_soil) == cable_count
    for cable_key in model.cables:
        grid_size_count = model.cables[cable_key].cable.radii_grid.size
        grid_size_full_solutions = model.cables_full_solutions[cable_key].cable.radii_grid.size
        assert matrices[cable_key].shape == (3, grid_size_count - 1)
        assert vectors[cable_key].shape == (grid_size_count - 1,)
        assert matrices_without_soil[cable_key].shape == (3, grid_size_full_solutions - 1)


@pytest.mark.parametrize("conductor_load", [100.0, 500.0, 1000.0])
@pytest.mark.parametrize("temperature_dependent_electric_resistance", [True, False])
@pytest.mark.parametrize("ac_current", [True, False])
def test_update_vectors_per_timestep(
    model: ModelSoil,
    conductor_load: float,
    temperature_dependent_electric_resistance: bool,
    ac_current: bool,
):
    """Test for different combinations of settings and time index if the vectors are correctly updated."""
    screen_loss_factor = 0.1
    dc_resistance_conductor = 0.0001
    ac_resistance_conductor = 0.00015
    conductor_temperature = 30.0
    screen_temperature = 25.0

    vectors = {cable_key: np.zeros(model.cables[cable_key].cable.radii_grid.size - 1) for cable_key in model.cables}
    full_solutions = {
        cable_key: np.zeros(cable_full_solution.cable.radii_grid.size)
        for cable_key, cable_full_solution in model.cables_full_solutions.items()
    }

    for cable_key, cable in model.cables_full_solutions.items():
        conductor_start_index, conductor_end_index = cable.cable.get_layer_indices_for_layer(CableLayer.Conductor)
        screen_start_index, screen_end_index = cable.cable.get_layer_indices_for_layer(CableLayer.Screen)
        full_solutions[cable_key][conductor_start_index : conductor_end_index + 1] = conductor_temperature
        full_solutions[cable_key][screen_start_index : screen_end_index + 1] = screen_temperature

        cable.cable.get_dc_resistance_conductor = mock.Mock(return_value=dc_resistance_conductor)
        cable.cable._get_ac_resistance_conductor_from_dc_resistance = mock.Mock(return_value=ac_resistance_conductor)
        cable.cable.get_cable_screen_loss_factor = mock.Mock(return_value=screen_loss_factor)

    model._set_run_options(
        run_options=ModelSoilRunOptions(
            temperature_dependent_electric_resistance=temperature_dependent_electric_resistance,
            ac_current=ac_current,
        ),
    )

    vectors = model._update_vectors_per_timestep(
        vectors=vectors,
        full_solutions=full_solutions,
        scenario_row=pd.Series({"load_c1": conductor_load}),
    )

    conductor_temperature_electrical_resistance = (
        conductor_temperature if temperature_dependent_electric_resistance else 20.0
    )

    # Assert get_dc_resistance_conductor was called for each cable
    for cable in model.cables_full_solutions.values():
        cable.cable.get_dc_resistance_conductor.assert_called_with(Tc=conductor_temperature_electrical_resistance)
        if ac_current:
            cable.cable._get_ac_resistance_conductor_from_dc_resistance.assert_called_with(
                Rdc=dc_resistance_conductor, s=cable.cable.layer_metrics.conductor_distance
            )
        else:
            cable.cable._get_ac_resistance_conductor_from_dc_resistance.assert_not_called()
        if not ac_current:
            cable.cable.get_cable_screen_loss_factor.assert_not_called()
            screen_loss_factor = 0.0

    for cable_key, cable in model.cables_full_solutions.items():
        resistance_conductor = ac_resistance_conductor if ac_current else dc_resistance_conductor
        expected_conductor_loss = resistance_conductor * conductor_load**2
        expected_screen_loss = screen_loss_factor * expected_conductor_loss

        conductor_start_index, conductor_end_index = cable.cable.get_layer_indices_for_layer(CableLayer.Conductor)
        screen_start_index, screen_end_index = cable.cable.get_layer_indices_for_layer(CableLayer.Screen)

        assert all(
            vectors[cable_key][conductor_start_index : conductor_end_index + 1]
            == expected_conductor_loss
            / cable.cable.surface_area_grid[conductor_start_index : conductor_end_index + 1].sum()
        )
        assert all(
            vectors[cable_key][screen_start_index : screen_end_index + 1]
            == expected_screen_loss / cable.cable.surface_area_grid[screen_start_index : screen_end_index + 1].sum()
        )


@pytest.mark.parametrize("time_idx", [1])
@pytest.mark.parametrize("temp_solution", [5.0 * np.ones(4)])
@pytest.mark.parametrize("mutual_heating_temp_solution", [2.0 * np.ones(3)])
@pytest.mark.parametrize("expected_solution", [5.0 * np.ones(4)])
@pytest.mark.parametrize("expected_mutual_heating_solution", [2.0 * np.ones(3)])
@pytest.mark.parametrize("expected_full_solution", [(10.0 + 5.0 + 2.0) * np.ones(3)])
def test_update_solutions(
    model: ModelSoil,
    time_idx: int,
    temp_solution: np.ndarray,
    mutual_heating_temp_solution: np.ndarray,
    expected_solution: np.ndarray,
    expected_mutual_heating_solution: np.ndarray,
    expected_full_solution: np.ndarray,
):
    """Simple test to check if the solution matrices are updated correctly."""
    upd_solution, upd_mutual_heating_solution, upd_full_solution = model._update_solution(
        temp_solution=temp_solution,
        mutual_heating_temp_solution=mutual_heating_temp_solution,
        ambient_temperature=model.scenario["ambient_temperature"].iloc[time_idx],
    )
    assert np.array_equal(upd_solution, expected_solution)
    assert np.array_equal(upd_mutual_heating_solution, expected_mutual_heating_solution)
    assert np.array_equal(upd_full_solution, expected_full_solution)


@pytest.mark.parametrize(
    "circuit_fixture,has_pipe", [("single_circuit_env", False), ("single_circuit_with_pipe_env", True)]
)
def test_update_pipe_resistivity_for_all_cables(
    scenario_constant: DataFrame[ScenarioSchemaSoil],
    circuit_fixture: str,
    has_pipe: bool,
    request: pytest.FixtureRequest,
):
    """Test to check if the resistivity values of a cable are changed if the cable has a pipe or not."""
    environment = request.getfixturevalue(circuit_fixture)
    model = ModelSoil(environment, scenario_constant)
    rhos_pre = {
        key: [cable.cable.layer_properties[layer].rho for layer in cable.cable.layers]
        for key, cable in model.cables.items()
    }
    model._update_pipe_resistivity_for_all_cables(
        full_solutions={
            key: np.ones(model.cables_full_solutions[key].cable.radii_grid.size) for key in model.cables
        },  # For this test the specific value of this function is not relevant.
        update_matrices={key: False for key in model.cables},
    )
    for cable_key, cable in model.cables.items():
        rho_post = [cable.cable.layer_properties[layer].rho for layer in cable.cable.layers]
        # If the cables have a pipe, the rhos are changed, so pre and post ar NOT equal.
        assert np.array_equal(rhos_pre[cable_key], rho_post) != has_pipe


@pytest.mark.parametrize(
    "model_fixture,daily_update,update_matrices_all_cables,expected",
    [
        ("model", False, False, False),
        ("model", False, True, True),
        ("model", True, False, False),
        ("model", True, True, True),
        ("model_dynamic_soil", False, False, False),
        ("model_dynamic_soil", False, True, True),
        ("model_dynamic_soil", True, False, True),
        ("model_dynamic_soil", True, True, True),
    ],
)
def test_update_soil_capacities_for_all_cables(
    model_fixture: str,
    daily_update: bool,
    update_matrices_all_cables: bool,
    expected: bool,
    request: pytest.FixtureRequest,
):
    """For different circuit types test whether the matrices are updated correctly."""
    model: ModelSoil = request.getfixturevalue(model_fixture)
    update_matrices = model._update_soil_capacities_for_all_cables(
        daily_update=daily_update,
        update_matrices={cable_key: update_matrices_all_cables for cable_key in model.cables},
        scenario_row=model.scenario.iloc[1],
    )
    assert update_matrices == {cable_key: expected for cable_key in model.cables}


@pytest.mark.parametrize(
    "model_fixture,time_idx,daily_update,update_matrices_all_cables,soil_drying,expected_update_matrices",
    [
        ("model", 1, False, False, False, False),
        ("model", 1, False, False, True, False),
        ("model", 1, False, True, False, True),
        ("model", 1, False, True, True, True),
        ("model", 25, True, False, True, True),
        ("model", 25, True, True, True, True),
        ("model", 25, True, True, False, True),
        ("model_dynamic_soil", 1, False, False, False, False),
        ("model_dynamic_soil", 1, False, False, True, False),
        ("model_dynamic_soil", 1, False, True, False, True),
        ("model_dynamic_soil", 1, False, True, True, True),
        ("model_dynamic_soil", 25, True, False, False, True),
        ("model_dynamic_soil", 25, True, False, True, True),
        ("model_dynamic_soil", 25, True, True, True, True),
        ("model_dynamic_soil", 25, True, True, False, True),
        ("model_with_pipe", 1, False, False, False, False),
        ("model_with_pipe", 1, True, True, True, True),
        ("model_with_pipe", 25, False, False, False, False),
        ("model_with_pipe", 25, True, True, True, True),
    ],
)
def test_update_soil_resistivity_for_all_cables(
    model_fixture: str,
    time_idx: int,
    daily_update: bool,
    update_matrices_all_cables: bool,
    soil_drying: bool,
    expected_update_matrices: bool,
    request: pytest.FixtureRequest,
):
    """Test for different circuit types if the matrices are updated correctly."""
    model: ModelSoil = request.getfixturevalue(model_fixture)
    # Get solutions for all cables with correct shape
    solutions, _ = model._initialize_solutions_lists(initial_state=None)
    result_update_matr = model._update_soil_resistivity_for_all_cables(
        daily_update,
        {key: update_matrices_all_cables for key in model.cables},
        soil_drying,
        solutions,
        scenario_row=model.scenario.iloc[time_idx],
    )
    assert {key: expected_update_matrices for key in model.cables} == result_update_matr

    if daily_update:
        # Check if the soil resistivity values are updated correctly in the cable properties
        expected_soil_thermal_resistivity = model.scenario.iloc[time_idx]["soil_thermal_resistivity"]
        first_cable = model.cables[list(model.cables.keys())[0]]
        assert np.isclose(expected_soil_thermal_resistivity, first_cable.cable.rho_grid[-1], rtol=1e-2)
        assert np.isclose(
            expected_soil_thermal_resistivity, first_cable.cable.layer_properties[CableLayer.SoilOne].rho, rtol=1e-2
        )


@pytest.mark.parametrize(
    "cable1_index,cable2_index,expected_distance",
    [
        (0, 1, 0.04904325387870013),
        (0, 3, 1),
    ],
)
def compute_distance_between_cables(two_circuit_fd_env, cable1_index, cable2_index, expected_distance):
    """Test that distance computation between cables works for FD environments.

    Tests both within a circuit and between two circuits using both distance methods.

    Args:
        two_circuit_fd_env: FD environment with two trefoil circuits
        cable1_index: The index of the first cable to use in distance computation
        cable2_index: The index of the second cable to use in distance computation
        expected_distance: Expected computed distance

    """
    cables = list(two_circuit_fd_env.cables())
    cable1 = cables[cable1_index]
    cable2 = cables[cable2_index]
    distance = two_circuit_fd_env.compute_distance(cable1, cable2)
    assert np.isclose(distance, expected_distance, atol=1e-8)


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
    assert model.pipes_present == has_pipe
    assert model.cables_full_solutions is not None
    assert model.cables is not None
    assert len(model.cables) == expected_number_of_cables
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
    model = model.add_solution_location(CableLayer.Insulation)
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

    # Get cable representations and corresponding keys
    cable_representations = list(model.cables.values())
    cable_keys = [cable.name for cable in cable_representations]

    # Create valid mutual heating solutions
    valid_mutual_heating_solutions = {key: np.array([1.0, 2.0, 3.0]) for key in cable_keys}

    # Test case 1: Valid StateSoil should pass upon initialization
    StateSoil(
        cable_representations=cable_representations,
        full_solution={key: np.array([10.0]) for key in cable_keys},
        internal_heating_solution={key: np.array([10.0]) for key in cable_keys},
        mutual_heating_solutions=valid_mutual_heating_solutions,
    )

    # Test case 2: Invalid keys should fail
    wrong_key = CableKey(circuit_name="wrong_circuit", cable_position=CablePosition.Single)
    invalid_mutual_heating_solutions = {wrong_key: np.array([1.0, 2.0, 3.0])}

    with pytest.raises(ValueError, match="CableKeys of mutual_heating_solutions should match"):
        StateSoil(
            cable_representations=cable_representations,
            full_solution={key: np.array([10.0]) for key in cable_keys},
            internal_heating_solution={key: np.array([10.0]) for key in cable_keys},
            mutual_heating_solutions=invalid_mutual_heating_solutions,
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

    model = ModelSoil(env, scenario)

    # Test 1: state=None should pass
    model._validate_initial_state(None)

    # Test 2: state=StateSoil instance should pass
    pos_cable = env.cables[CableKey(circuit_name=circuit_name, cable_position=CablePosition.Single)]
    cable_key = pos_cable.name

    valid_state = StateSoil(
        cable_representations=[pos_cable],
        full_solution={cable_key: np.array([20.0])},
        internal_heating_solution={cable_key: np.array([20.0])},
        mutual_heating_solutions={cable_key: np.array([15.0])},
    )

    model._validate_initial_state(valid_state)

    # Test 3: state=StateAir instance should raise ValueError
    invalid_state_air = StateAir(
        cable_representations=[pos_cable],
        full_solution={cable_key: np.array([20.0])},
        internal_heating_solution={cable_key: np.array([20.0])},
    )

    with pytest.raises(ValueError, match="ModelSoil requires a StateSoil instance, but received StateAir"):
        model._validate_initial_state(invalid_state_air)


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
        ModelSoil(static_env=StaticEnvAir(), scenario=pd.DataFrame())
