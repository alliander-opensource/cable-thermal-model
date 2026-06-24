# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from typing import cast

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError
from pandera.typing import DataFrame

from cable_thermal_model.cable.cable_circuit import CableKey, CablePosition, PosCable
from cable_thermal_model.environment.static_env_soil import StaticEnvSoil
from cable_thermal_model.model.abstract_model import AbstractModel
from cable_thermal_model.model.model_factory import ModelFactory
from cable_thermal_model.model.schemas.model_input_schemas import ScenarioSchemaSoil
from cable_thermal_model.model.schemas.state_schemas import State


def test_model_init_without_arguments():
    """Tests whether the construction fails if no arguments are supplied."""
    with pytest.raises(TypeError) as exc_info:
        # construct model without arguments, should fail as the methods are not defined.
        _ = AbstractModel()

    assert exc_info is not None


@pytest.mark.parametrize(
    "new_scenario",
    [
        pd.DataFrame(
            index=pd.date_range("2020-01-01", "2020-01-03", freq="2h"),
            data={
                "load_c1": np.linspace(-25, 25, 25) + 100,
                "ambient_temperature": 10,
                "soil_thermal_resistivity": 1.0,
                "soil_thermal_capacity": 2e6,
            },
        ),
        pd.DataFrame(
            index=pd.date_range("2020-01-01", "2020-01-03", freq="1h"),
            data={
                "load_c1": np.linspace(-25, 25, 49) + 100 + 50 * np.sin(np.linspace(0, 4 * np.pi, 49)),
                "ambient_temperature": 10,
                "soil_thermal_resistivity": 1.0,
                "soil_thermal_capacity": np.linspace(1.5e6, 3e6, 49),
            },
        ),
        pd.DataFrame(
            index=pd.timedelta_range("0 days", "48 days", freq=pd.Timedelta("1 days")) + pd.Timestamp("2020-01-01"),
            data={
                "load_c1": np.linspace(-25, 25, 49) + 100 + 50 * np.sin(np.linspace(0, 4 * np.pi, 49)),
                "ambient_temperature": np.linspace(-25, 25, 49) + 100 + 50 * np.sin(np.linspace(0, 4 * np.pi, 49)),
                "soil_thermal_resistivity": np.linspace(0.2, 2.5, 49),
                "soil_thermal_capacity": 2e6,
            },
        ),
    ],
)
def test_set_scenario(model, new_scenario):
    """Tests whether the updated scenario is set in the model object."""
    model.run()
    model._set_scenario(new_scenario)
    assert model.scenario.equals(new_scenario)

    model.run()


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param(
            pd.DataFrame(
                index=pd.date_range("2020-01-01", "2020-01-03", freq="1h"),
                data={"load_wrong_cable_name": np.linspace(-25, 25, 49) + 100, "ambient_temperature": 10},
            ),
            marks=pytest.mark.xfail(reason="Fails as load_c1 is not available in the scenario."),
        ),
        pd.DataFrame(
            index=pd.date_range("2020-01-01", "2020-01-03", freq="1h"),
            data={
                "load_c1": np.linspace(-25, 25, 49) + 100,
                "soil_thermal_resistivity": 1.0,
                "soil_thermal_capacity": 2e6,
            },
        ),
        pd.DataFrame(
            index=pd.date_range("2020-01-01", "2020-01-03", freq="1h"),
            data={
                "load_c1": np.linspace(-25, 25, 49) + 100,
                "ambient_temperature": 10,
                "soil_thermal_capacity": 2e6,
            },
        ),
        pd.DataFrame(
            index=pd.date_range("2020-01-01", "2020-01-03", freq="1h"),
            data={
                "load_c1": np.linspace(-25, 25, 49) + 100,
                "ambient_temperature": 10,
                "soil_thermal_resistivity": 1.0,
            },
        ),
        pd.DataFrame(
            index=pd.timedelta_range("0 days", "48 days", freq=pd.Timedelta("1 days")) + pd.Timestamp("2020-01-01"),
            data={
                "load_c1": pd.Series(np.array(list(np.linspace(-25, 0, 24)) + [None] + list(np.linspace(0, 25, 24)))),
                "ambient_temperature": 10,
                "soil_thermal_resistivity": 1.0,
                "soil_thermal_capacity": 2e6,
            },
        ),
        pd.DataFrame(
            index=pd.timedelta_range("0 days", "48 days", freq=pd.Timedelta("1 days")) + pd.Timestamp("2020-01-01"),
            data={
                "load_c1": pd.Series(np.array(list(np.linspace(-25, 0, 24)) + [np.nan] + list(np.linspace(0, 25, 24)))),
                "ambient_temperature": 10,
                "soil_thermal_resistivity": 1.0,
                "soil_thermal_capacity": 2e6,
            },
        ),
        pd.DataFrame(
            index=pd.timedelta_range("0 days", "48 days", freq=pd.Timedelta("1 days")) + pd.Timestamp("2020-01-01"),
            data={
                "load_c1": pd.Series(
                    np.array(list(np.linspace(-25, 0, 24)) + [float("nan")] + list(np.linspace(0, 25, 24)))
                ),
                "ambient_temperature": 10,
                "soil_thermal_resistivity": 1.0,
                "soil_thermal_capacity": 2e6,
            },
        ),
        pd.DataFrame(
            index=pd.timedelta_range("0 days", "48 days", freq=pd.Timedelta("1 days")) + pd.Timestamp("2020-01-01"),
            data={
                "load_c1": pd.Series(
                    np.array(list(np.linspace(-25, 0, 24)) + ["not a number example"] + list(np.linspace(0, 25, 24)))
                ),
                "ambient_temperature": 10,
                "soil_thermal_resistivity": 1.0,
                "soil_thermal_capacity": 2e6,
            },
        ),
    ],
)
def test_validate_scenario(env: StaticEnvSoil, scenario: pd.DataFrame):
    """Tests whether invalid scenarios raise a schema error by plugging in cables and scenarios.

    Checks for:
    - a correct 'load_{cable_name} included in the scenario
    - temperature included in the scenario
    - soil thermal resistivity included in the scenario
    - soil thermal capacity included in the scenario
    - missing values (NaNs).
    """
    with pytest.raises(SchemaError):
        ModelFactory.create_model(static_env=env, scenario=cast(DataFrame[ScenarioSchemaSoil], scenario))


@pytest.mark.parametrize("temperature_dependent_electric_resistance", [True, False])
@pytest.mark.parametrize("soil_drying", [True, False])
@pytest.mark.parametrize("ac_current", [True, False])
@pytest.mark.parametrize("initial_state", [True, False])
def test_run(model, temperature_dependent_electric_resistance, soil_drying, ac_current, initial_state):
    """Tests whether we can go through the different options and get results but does not check output."""
    state = model.run().state if initial_state else None

    solution = model.run(
        initial_state=state,
        run_options={
            "temperature_dependent_electric_resistance": temperature_dependent_electric_resistance,
            "soil_drying": soil_drying,
            "ac_current": ac_current,
        },
    )
    assert solution is not None


def test_state_check_solution_consistency(single_core_cable_xlpe):
    """Test the check_solution_consistency validator in State class."""
    # Create test cable representation
    pos_cable = PosCable(
        circuit_name="test_circuit", cable_position=CablePosition.Single, cable=single_core_cable_xlpe, x=0.0, y=0.0
    )
    cable_key = pos_cable.name

    # Test 1: Matching keys should pass
    valid_full_solution = {cable_key: np.array([20.0])}
    valid_solution = {cable_key: np.array([15.0])}

    State(
        cable_representations=[pos_cable],
        full_solution=valid_full_solution,
        internal_heating_solution=valid_solution,
    )
    # Test 2: Mismatched keys should fail
    wrong_key = CableKey(circuit_name="wrong_circuit", cable_position=CablePosition.TrefoilLeft)
    invalid_solution = {wrong_key: np.array([15.0])}

    with pytest.raises(ValueError, match="Inconsistent keys between full_solution and solution"):
        State(
            cable_representations=[pos_cable],
            full_solution=valid_full_solution,
            internal_heating_solution=invalid_solution,
        )


def test_state_check_cable_representations_consistency(single_core_cable_xlpe):
    """Test the check_cable_representations_consistency validator in State class."""
    # Create test cable representations
    pos_cable_1 = PosCable(
        circuit_name="circuit_1", cable_position=CablePosition.TrefoilLeft, cable=single_core_cable_xlpe, x=0.0, y=0.0
    )
    pos_cable_2 = PosCable(
        circuit_name="circuit_1", cable_position=CablePosition.TrefoilRight, cable=single_core_cable_xlpe, x=1.0, y=0.0
    )

    cable_key_1 = pos_cable_1.name
    cable_key_2 = pos_cable_2.name

    # Test 1: Matching cable representations and solution keys should pass
    valid_full_solution = {cable_key_1: np.array([20.0]), cable_key_2: np.array([22.0])}
    valid_solution = {cable_key_1: np.array([15.0]), cable_key_2: np.array([17.0])}

    State(
        cable_representations=[pos_cable_1, pos_cable_2],
        full_solution=valid_full_solution,
        internal_heating_solution=valid_solution,
    )

    # Test 2: Missing cable in solution should fail
    incomplete_solution = {cable_key_1: np.array([15.0])}  # Missing cable_key_2
    incomplete_full_solution = {cable_key_1: np.array([20.0])}  # Missing cable_key_2

    with pytest.raises(ValueError, match="Keys in solution must match cable representations"):
        State(
            cable_representations=[pos_cable_1, pos_cable_2],
            full_solution=incomplete_full_solution,
            internal_heating_solution=incomplete_solution,
        )
