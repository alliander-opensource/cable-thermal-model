# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0


import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors
from pydantic_core import ValidationError

from cable_thermal_model.cable.cable_circuit import CableKey, CablePosition, PosCable
from cable_thermal_model.environment.static_env_soil import StaticEnvSoil
from cable_thermal_model.model.abstract_model import AbstractModel
from cable_thermal_model.model.model_factory import ModelFactory
from cable_thermal_model.model.schemas.state_schemas import State, StateSoil


def test_model_init_without_arguments():
    """Tests whether the construction fails if no arguments are supplied."""
    with pytest.raises(TypeError) as exc_info:
        # construct model without arguments, should fail as the methods are not defined.
        _ = AbstractModel()

    assert "abstract" in str(exc_info.value).lower()


@pytest.mark.parametrize(
    "new_scenario",
    [
        pd.DataFrame(
            index=pd.date_range("2020-01-01", "2020-01-03", freq="2h"),
            data={
                "load_c1": np.linspace(-25, 25, 25) + 100,
                "ambient_temperature": 10.0,
                "soil_thermal_resistivity": 1.0,
                "soil_thermal_capacity": 2e6,
            },
        ),
        pd.DataFrame(
            index=pd.date_range("2020-01-01", "2020-01-03", freq="1h"),
            data={
                "load_c1": np.linspace(-25, 25, 49) + 100 + 50 * np.sin(np.linspace(0, 4 * np.pi, 49)),
                "ambient_temperature": 10.0,
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
def test_run_accepts_different_scenarios(model, new_scenario):
    """Tests whether the same model instance can be executed with different scenarios."""
    first_result = model.run(new_scenario)
    second_result = model.run(new_scenario)

    pd.testing.assert_frame_equal(first_result.result, second_result.result)


@pytest.mark.parametrize(
    ("scenario", "exception", "error_msg"),
    [
        pytest.param(
            pd.DataFrame(
                index=pd.date_range("2020-01-01", "2020-01-03", freq="1h"),
                data={
                    "ambient_temperature": 10.0,
                    "soil_thermal_resistivity": 1.0,
                    "soil_thermal_capacity": 2e6,
                },
            ),
            SchemaError,
            "column 'load_c1' not in dataframe",
            id="missing_load_column",
        ),
        pytest.param(
            pd.DataFrame(
                index=pd.date_range("2020-01-01", "2020-01-03", freq="1h"),
                data={
                    "unknown_column": np.linspace(-25, 25, 49) + 100,
                    "load_c1": np.linspace(-25, 25, 49) + 100,
                    "ambient_temperature": 10.0,
                    "soil_thermal_resistivity": 1.0,
                    "soil_thermal_capacity": 2e6,
                },
            ),
            SchemaErrors,
            "column 'unknown_column' not in DataFrameSchema",
            id="unknown column in scenario",
        ),
        pytest.param(
            pd.DataFrame(
                index=pd.date_range("2020-01-01", "2020-01-03", freq="1h"),
                data={
                    "load_c1": np.linspace(-25, 25, 49) + 100,
                    "soil_thermal_resistivity": 1.0,
                    "soil_thermal_capacity": 2e6,
                },
            ),
            SchemaError,
            "column 'ambient_temperature' not in dataframe",
            id="missing_ambient_temperature",
        ),
        pytest.param(
            pd.DataFrame(
                index=pd.date_range("2020-01-01", "2020-01-03", freq="1h"),
                data={
                    "load_c1": np.linspace(-25, 25, 49) + 100,
                    "ambient_temperature": 10.0,
                    "soil_thermal_capacity": 2e6,
                },
            ),
            SchemaError,
            "",
            id="missing_soil_resistivity",
        ),
        pytest.param(
            pd.DataFrame(
                index=pd.date_range("2020-01-01", "2020-01-03", freq="1h"),
                data={
                    "load_c1": np.linspace(-25, 25, 49) + 100,
                    "ambient_temperature": 10.0,
                    "soil_thermal_resistivity": 1.0,
                },
            ),
            SchemaError,
            "",
            id="missing_soil_capacity",
        ),
        pytest.param(
            pd.DataFrame(
                index=pd.timedelta_range("0 days", "48 days", freq=pd.Timedelta("1 days")) + pd.Timestamp("2020-01-01"),
                data={
                    "load_c1": pd.Series(
                        np.array(list(np.linspace(-25, 0, 24)) + [None] + list(np.linspace(0, 25, 24)))
                    ),
                    "ambient_temperature": 10.0,
                    "soil_thermal_resistivity": 1.0,
                    "soil_thermal_capacity": 2e6,
                },
            ),
            SchemaError,
            "",
            id="none_in_load_series",
        ),
        pytest.param(
            pd.DataFrame(
                index=pd.timedelta_range("0 days", "48 days", freq=pd.Timedelta("1 days")) + pd.Timestamp("2020-01-01"),
                data={
                    "load_c1": pd.Series(
                        np.array(list(np.linspace(-25, 0, 24)) + [np.nan] + list(np.linspace(0, 25, 24)))
                    ),
                    "ambient_temperature": 10.0,
                    "soil_thermal_resistivity": 1.0,
                    "soil_thermal_capacity": 2e6,
                },
            ),
            SchemaError,
            "",
            id="nan_in_load_series",
        ),
        pytest.param(
            pd.DataFrame(
                index=pd.timedelta_range("0 days", "48 days", freq=pd.Timedelta("1 days")) + pd.Timestamp("2020-01-01"),
                data={
                    "load_c1": pd.Series(
                        np.array(list(np.linspace(-25, 0, 24)) + [float("nan")] + list(np.linspace(0, 25, 24)))
                    ),
                    "ambient_temperature": 10.0,
                    "soil_thermal_resistivity": 1.0,
                    "soil_thermal_capacity": 2e6,
                },
            ),
            SchemaError,
            "",
            id="float_nan_in_load_series",
        ),
        pytest.param(
            pd.DataFrame(
                index=pd.timedelta_range("0 days", "48 days", freq=pd.Timedelta("1 days")) + pd.Timestamp("2020-01-01"),
                data={
                    "load_c1": pd.Series(
                        np.array(
                            list(np.linspace(-25, 0, 24)) + ["not a number example"] + list(np.linspace(0, 25, 24))
                        )
                    ),
                    "ambient_temperature": 10.0,
                    "soil_thermal_resistivity": 1.0,
                    "soil_thermal_capacity": 2e6,
                },
            ),
            SchemaError,
            "",
            id="invalid_type_in_load_series",
        ),
    ],
)
def test_validate_scenario(
    single_circuit_env: StaticEnvSoil,
    scenario: pd.DataFrame,
    exception: type[Exception],
    error_msg: str,
):
    """Tests whether invalid scenarios raise the expected validation exception.

    Checks for:
    - a correct 'load_{cable_name} included in the scenario
    - temperature included in the scenario
    - soil thermal resistivity included in the scenario
    - soil thermal capacity included in the scenario
    - missing values (NaNs).
    """
    model = ModelFactory.create_model(static_env=single_circuit_env)

    if error_msg:
        with pytest.raises(exception, match=error_msg):
            model.run(scenario)
    else:
        with pytest.raises(exception):
            model.run(scenario)


@pytest.mark.parametrize("temperature_dependent_electric_resistance", [True, False])
@pytest.mark.parametrize("soil_drying", [True, False])
@pytest.mark.parametrize("ac_current", [True, False])
@pytest.mark.parametrize("initial_state", [True, False])
def test_run(
    model,
    scenario_constant,
    temperature_dependent_electric_resistance,
    soil_drying,
    ac_current,
    initial_state,
):
    """Tests whether we can go through the different options and get results but does not check output."""
    state = model.run(scenario_constant).state if initial_state else None

    solution = model.run(
        scenario_constant,
        initial_state=state,
        run_options={
            "temperature_dependent_electric_resistance": temperature_dependent_electric_resistance,
            "soil_drying": soil_drying,
            "ac_current": ac_current,
        },
    )

    assert solution is not None


def test_run_with_split_scenario(model, scenario_constant):
    """Tests that explicit run scenarios preserve chained results."""
    long_scenario = scenario_constant.copy()
    split_idx = len(long_scenario) // 2
    first_short_scenario = long_scenario.iloc[: split_idx + 1].copy()
    second_short_scenario = long_scenario.iloc[split_idx:].copy()

    long_output = model.__class__(model.static_env).run(long_scenario)

    reused_model = model.__class__(model.static_env)
    first_output = reused_model.run(first_short_scenario)
    second_output = reused_model.run(second_short_scenario, initial_state=first_output.state)

    reused_result = pd.concat([first_output.result.copy(), second_output.result.iloc[1:].copy()])

    pd.testing.assert_frame_equal(reused_result, long_output.result, rtol=1e-10, atol=1e-10)


def test_state_check_solution_consistency(single_core_cable_xlpe):
    """Test the check_solution_consistency validator in State class."""
    # Create test cable representation
    pos_cable = PosCable(
        circuit_name="test_circuit", cable_position=CablePosition.Single, cable=single_core_cable_xlpe, x=0.0, y=0.0
    )
    cable_key = pos_cable.key

    # Test 1: Matching keys should pass
    valid_full_solution = {cable_key: np.array([20.0])}
    valid_solution = {cable_key: np.array([15.0])}

    State(
        static_env_hash="dummy_fingerprint",
        temperature=valid_full_solution,
        self_heating_contribution=valid_solution,
        ambient_temperature=5.0,
    )
    # Test 2: Mismatched keys should fail
    wrong_key = CableKey(circuit_name="wrong_circuit", cable_position=CablePosition.TrefoilLeft)
    invalid_solution = {wrong_key: np.array([15.0])}

    with pytest.raises(ValidationError, match="Inconsistent keys between temperature and self_heating"):
        State(
            static_env_hash="dummy_fingerprint",
            temperature=valid_full_solution,
            self_heating_contribution=invalid_solution,
            ambient_temperature=5.0,
        )


def test_state_check_cable_representations_consistency(model):
    """Test initial-state validation against the model static environment."""
    cable_keys = list(model.static_env.get_cables().keys())
    env_hash = model.static_env.compute_hash()
    state_cls = model._state_class

    # Test 1: Matching static environment keys and fingerprint should pass
    valid_temperature = {key: np.array([20.0]) for key in cable_keys}
    valid_self_heating = {key: np.array([15.0]) for key in cable_keys}
    valid_state_kwargs = {
        "static_env_hash": env_hash,
        "temperature": valid_temperature,
        "self_heating_contribution": valid_self_heating,
        "ambient_temperature": 5.0,
    }
    if state_cls is StateSoil:
        valid_state_kwargs["mutual_heating_contribution"] = {key: np.array([10.0]) for key in cable_keys}
    valid_state = state_cls(**valid_state_kwargs)
    model._validate_initial_state(valid_state)

    # Test 2: Missing cable in state should fail
    incomplete_temperature = {cable_keys[0]: np.array([20.0])}
    incomplete_self_heating = {cable_keys[0]: np.array([15.0])}
    invalid_state_kwargs = {
        "static_env_hash": env_hash,
        "temperature": incomplete_temperature,
        "self_heating_contribution": incomplete_self_heating,
        "ambient_temperature": 5.0,
    }
    if state_cls is StateSoil:
        invalid_state_kwargs["mutual_heating_contribution"] = {cable_keys[0]: np.array([10.0])}
    invalid_state = state_cls(**invalid_state_kwargs)

    with pytest.raises(ValueError, match="Provided state cable keys do not match the used environment"):
        model._validate_initial_state(invalid_state)


def test_state_check_environment_hash_consistency(model):
    """Test that initial-state validation fails if environment hash does not match."""
    cable_keys = list(model.static_env.get_cables().keys())
    state_cls = model._state_class

    invalid_state_kwargs = {
        "static_env_hash": "different-environment-hash",
        "temperature": {key: np.array([20.0]) for key in cable_keys},
        "self_heating_contribution": {key: np.array([15.0]) for key in cable_keys},
        "ambient_temperature": 5.0,
    }
    if state_cls is StateSoil:
        invalid_state_kwargs["mutual_heating_contribution"] = {key: np.array([10.0]) for key in cable_keys}
    invalid_state = state_cls(**invalid_state_kwargs)

    with pytest.raises(ValueError, match="Provided state environment hash does not match the used environment"):
        model._validate_initial_state(invalid_state)


def test_model_str_representation(model):
    """Test concise model string for scenario-free model instances."""
    assert str(model) == "Model with 1 circuit environment"
