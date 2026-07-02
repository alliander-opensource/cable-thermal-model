# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

import hashlib
from typing import TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from cable_thermal_model.cable.cable_circuit import CableKey
from cable_thermal_model.environment.static_env import StaticEnv

StateT = TypeVar("StateT", bound="State")


def build_environment_fingerprint(static_env: StaticEnv) -> str:
    """Build a deterministic environment fingerprint from positioned cable representations."""
    encoded_representations = []
    for cable in static_env.get_cables().values():
        key = cable.name
        encoded_representations.append(f"{key.circuit_name}|{key.cable_position.value}|{cable.cable_representation}")
    payload = "\n".join(sorted(encoded_representations)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class State(BaseModel):
    """Stores information about temperatures within cables at the final state.

    The final state is reached at the end of the simulation. In addition,
    the relevant cable representations and their properties are stored.

    Attributes:
        env_fingerprint: str
            Deterministic fingerprint of the static environment cable configuration.
        temperature: dict[CableKey, np.ndarray]:
            Combines the internal heating solution with the ambient
            temperature profile and, for a StateSoil object, the mutual
            heating solution.
        self_heating: dict[CableKey, np.ndarray]:
            The temperature delta profile as a result of self-heating due to the load.

    """

    env_fingerprint: str = Field(default_factory=str)
    temperature: dict[CableKey, np.ndarray] = Field(default_factory=dict)
    self_heating: dict[CableKey, np.ndarray] = Field(default_factory=dict)

    # Pydantic class configuration
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=False)

    @model_validator(mode="after")
    def check_solution_consistency(self):
        """Validate that temperature and self_heating share the same cable keys."""
        keys_temperature = set(self.temperature.keys())
        keys_solution = set(self.self_heating.keys())
        if keys_temperature != keys_solution:
            raise ValueError(
                f"Inconsistent keys between temperature and self_heating. Keys in temperature: {keys_temperature}, "
                f"keys in self_heating: {keys_solution}"
            )
        return self


class StateSoil(State):
    """Extends upon the base State class. Includes additional attribute mutual_heating and validation thereof.

    Attributes:
        mutual_heating: dict[CableKey, np.ndarray]
            A dictionary containing the temperature increase inside a cable
            due to mutual heating from other cables in the environment.
            This is stored as a dict with CableKey as key and an array of
            temperature increases per grid point as value.

    """

    mutual_heating: dict[CableKey, np.ndarray] = Field()

    @model_validator(mode="after")
    def validate_mutual_heating(self):
        """Validate that mutual_heating keys match cable keys."""
        found_keys = set(self.mutual_heating.keys())
        expected_keys = set(self.temperature.keys())
        if found_keys != expected_keys:
            raise ValueError(
                "CableKeys of mutual_heating should match with cable_keys of temperature."
                f"Found keys: {found_keys}, expected keys: {expected_keys}"
            )
        return self


class StateAir(State):
    """StateAir has no added attributes on top of State.

    However, we want to make sure there is only one circuit (check for a unique circuit_name).
    """

    @model_validator(mode="after")
    def validate_single_circuit(self):
        """Ensure that all cable keys in StateAir belong to the same circuit."""
        cable_keys = self.temperature.keys()
        circuit_names = {cable_key.circuit_name for cable_key in cable_keys}
        if len(circuit_names) > 1:
            raise ValueError(f"StateAir should only contain one circuit, but found multiple: {circuit_names}")
        return self
