# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0


import numpy as np
import pytest
from pydantic_core import ValidationError

from cable_thermal_model.cable.cable_circuit import CableKey, CablePosition
from cable_thermal_model.model.schemas.state_schemas import State, StateAir, StateSoil


def test_state_check_solution_consistency_passes():
    """State should accept matching temperature and self_heating keys."""
    cable_key = CableKey(circuit_name="circuit_1", cable_position=CablePosition.Single)

    state = State(
        static_env_hash="dummy_fingerprint",
        temperature={cable_key: np.array([20.0])},
        self_heating_contribution={cable_key: np.array([15.0])},
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
            static_env_hash="dummy_fingerprint",
            temperature=temperature,
            self_heating_contribution=self_heating,
        )


def test_statesoil_validate_mutual_heating_passes():
    """StateSoil should accept matching mutual_heating keys."""
    cable_key = CableKey(circuit_name="circuit_1", cable_position=CablePosition.Single)

    state = StateSoil(
        static_env_hash="dummy_fingerprint",
        temperature={cable_key: np.array([20.0])},
        self_heating_contribution={cable_key: np.array([15.0])},
        mutual_heating_contribution={cable_key: np.array([10.0])},
    )

    assert np.isclose(state.mutual_heating_contribution[cable_key][0], 10.0)


def test_statesoil_validate_mutual_heating_raises_on_mismatch():
    """StateSoil should reject mutual_heating keys that do not match temperature keys."""
    cable_key_temperature = CableKey(circuit_name="circuit_1", cable_position=CablePosition.Single)
    cable_key_mutual_heating = CableKey(circuit_name="circuit_2", cable_position=CablePosition.Single)

    temperature = {cable_key_temperature: np.array([20.0])}
    self_heating = {cable_key_temperature: np.array([15.0])}
    mutual_heating = {cable_key_mutual_heating: np.array([10.0])}

    with pytest.raises(ValidationError, match="CableKeys of mutual_heating_contribution should match"):
        StateSoil(
            static_env_hash="dummy_fingerprint",
            temperature=temperature,
            self_heating_contribution=self_heating,
            mutual_heating_contribution=mutual_heating,
        )


def test_stateair_validate_single_circuit_passes_and_rejects_multiple_circuits():
    """StateAir should allow one circuit and reject multiple circuits."""
    cable_key_single = CableKey(circuit_name="circuit_1", cable_position=CablePosition.Single)

    state = StateAir(
        static_env_hash="dummy_fingerprint",
        temperature={cable_key_single: np.array([20.0])},
        self_heating_contribution={cable_key_single: np.array([15.0])},
    )

    assert len(state.temperature) == 1

    cable_key_1 = CableKey(circuit_name="circuit_1", cable_position=CablePosition.TrefoilLeft)
    cable_key_2 = CableKey(circuit_name="circuit_2", cable_position=CablePosition.TrefoilRight)

    temperature = {cable_key_1: np.array([20.0]), cable_key_2: np.array([25.0])}
    self_heating = {cable_key_1: np.array([20.0]), cable_key_2: np.array([25.0])}

    with pytest.raises(ValidationError, match="StateAir should only contain one circuit"):
        StateAir(
            static_env_hash="dummy_fingerprint",
            temperature=temperature,
            self_heating_contribution=self_heating,
        )
