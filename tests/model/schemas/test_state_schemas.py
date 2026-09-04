# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0


import numpy as np
import pytest
from pydantic_core import ValidationError

from cable_thermal_model.cable.cable_circuit import CableKey, CablePosition
from cable_thermal_model.model.schemas.state_schemas import State, StateAir, StateSoil


@pytest.fixture()
def simple_state_soil():
    """Fixture for a valid StateSoil object."""
    cable_key = CableKey(circuit_name="circuit_1", cable_position=CablePosition.Single)
    return StateSoil(
        static_env_hash=123456789,
        temperature={cable_key: np.array([30.0])},
        self_heating_contribution={cable_key: np.array([15.0])},
        mutual_heating_contribution={cable_key: np.array([10.0])},
        ambient_temperature=5.0,
    )


def test_state_check_solution_consistency_passes():
    """State should accept matching temperature and self_heating keys."""
    cable_key = CableKey(circuit_name="circuit_1", cable_position=CablePosition.Single)

    state = State(
        static_env_hash=123456789,
        temperature={cable_key: np.array([20.0])},
        self_heating_contribution={cable_key: np.array([15.0])},
        ambient_temperature=5.0,
    )

    assert np.isclose(state.temperature[cable_key][0], 20.0)


def test_state_check_solution_consistency_raises_on_mismatch():
    """State should reject mismatched temperature and self_heating keys."""
    cable_key_temperature = CableKey(circuit_name="circuit_1", cable_position=CablePosition.Single)
    cable_key_self_heating = CableKey(circuit_name="circuit_2", cable_position=CablePosition.Single)

    temperature = {cable_key_temperature: np.array([20.0])}
    self_heating = {cable_key_self_heating: np.array([15.0])}

    with pytest.raises(ValidationError, match="Inconsistent keys between temperature and self_heating"):
        State(
            static_env_hash=123456789,
            temperature=temperature,
            self_heating_contribution=self_heating,
            ambient_temperature=5.0,
        )


def test_simple_state_soil(simple_state_soil):
    """StateSoil should accept matching mutual_heating keys."""
    state = StateSoil.model_validate(simple_state_soil)
    assert isinstance(state, StateSoil)


def test_statesoil_validate_mutual_heating_raises_on_mismatch(simple_state_soil):
    """StateSoil should reject mutual_heating keys that do not match temperature keys."""
    state_model_dump = simple_state_soil.model_dump()
    state_model_dump["mutual_heating_contribution"][
        CableKey(circuit_name="circuit_2", cable_position=CablePosition.Single)
    ] = np.array([10.0])

    with pytest.raises(ValidationError, match="CableKeys of mutual_heating_contribution should match"):
        StateSoil.model_validate(state_model_dump)


def test_stateair_validate_single_circuit_passes_and_rejects_multiple_circuits():
    """StateAir should allow one circuit and reject multiple circuits."""
    cable_key_single = CableKey(circuit_name="circuit_1", cable_position=CablePosition.Single)

    state = StateAir(
        static_env_hash=123456789,
        temperature={cable_key_single: np.array([20.0])},
        self_heating_contribution={cable_key_single: np.array([15.0])},
        ambient_temperature=5.0,
    )

    assert len(state.temperature) == 1

    cable_key_1 = CableKey(circuit_name="circuit_1", cable_position=CablePosition.TrefoilLeft)
    cable_key_2 = CableKey(circuit_name="circuit_2", cable_position=CablePosition.TrefoilRight)

    temperature = {cable_key_1: np.array([20.0]), cable_key_2: np.array([25.0])}
    self_heating = {cable_key_1: np.array([20.0]), cable_key_2: np.array([25.0])}

    with pytest.raises(ValidationError, match="StateAir should only contain one circuit"):
        StateAir(
            static_env_hash=123456789,
            temperature=temperature,
            self_heating_contribution=self_heating,
            ambient_temperature=0.0,
        )


def test_state_serialization_and_deserialization(simple_state_soil):
    """Test serialization and deserialization of State, StateSoil, and StateAir."""
    # Serialize to dict
    serialized_state_soil = simple_state_soil.model_dump_json()

    # Deserialize back to StateSoil
    deserialized_state_soil = StateSoil.model_validate_json(serialized_state_soil)

    assert deserialized_state_soil == simple_state_soil


def test_integer_in_state():
    """Test that integer values in State are serialized to float in JSON."""
    cable_key = CableKey(circuit_name="circuit_1", cable_position=CablePosition.Single)

    state = State(
        static_env_hash=123456789,
        temperature={cable_key: np.array([20])},  # Integer value
        self_heating_contribution={cable_key: np.array([15])},  # Integer value
        ambient_temperature=5,  # Integer value
    )

    serialized_state = state.model_dump_json()

    # Check that the serialized JSON contains float values
    assert ":[20.0]}" in serialized_state
    assert ":[15.0]}" in serialized_state
    assert ":5.0}" in serialized_state

    # An integer in the json should still result in the same State
    deserialized_state = State.model_validate_json(serialized_state.replace(".0", ""))
    assert state == deserialized_state


def test_string_in_state(simple_state_soil: StateSoil):
    """Test that string values in State raise a ValidationError."""
    state_model_dump = simple_state_soil.model_dump()

    with pytest.raises(ValidationError, match="could not convert string to float"):
        assert len(state_model_dump["temperature"]) == 1
        state_model_dump["temperature"] = {key: np.array(["not_a_float"]) for key in state_model_dump["temperature"]}
        StateSoil.model_validate(state_model_dump)

    state_model_json = simple_state_soil.model_dump_json()
    with pytest.raises(ValidationError, match="could not convert string to float"):
        state_model_json = state_model_json.replace("30.0", '"not_a_float"')
        StateSoil.model_validate_json(state_model_json)
