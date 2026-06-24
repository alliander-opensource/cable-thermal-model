# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from typing import TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from cable_thermal_model.cable.cable_circuit import CableKey, PosCable

StateT = TypeVar("StateT", bound="State")


class State(BaseModel):
    """Stores information about temperatures within cables at the final state.

    The final state is reached at the end of the simulation. In addition,
    the relevant cable representations and their properties are stored.

    Attributes:
       cable_representations: list[PosCable]:
              List of cable representations with their properties and positions in the environment.
       full_solution: dict[CableKey, np.ndarray]:
            Combines the internal heating solution with the ambient
            temperature profile and, for a StateSoil object, the mutual
            heating solution.
       internal_heating_solution: dict[CableKey, np.ndarray]:
            The temperature delta profile as a result of internal heating due to the load.

    """

    cable_representations: list[PosCable] = Field(default_factory=list)
    full_solution: dict[CableKey, np.ndarray] = Field(default_factory=dict)
    internal_heating_solution: dict[CableKey, np.ndarray] = Field(default_factory=dict)

    # Pydantic class configuration
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @model_validator(mode="after")
    def check_solution_consistency(self):
        """Validate that full_solution and internal_heating_solution share the same cable keys."""
        keys_full_solution = set(self.full_solution.keys())
        keys_solution = set(self.internal_heating_solution.keys())
        if keys_full_solution != keys_solution:
            raise ValueError(
                f"Inconsistent keys between full_solution and solution. Keys in full_solution: {keys_full_solution}, "
                f"keys in solution: {keys_solution}"
            )
        return self

    @model_validator(mode="after")
    def check_cable_representations_consistency(self):
        """Validate that cable_representations and internal_heating_solution share the same cable keys."""
        keys_cables = {cable.name for cable in self.cable_representations}
        keys_solution = set(self.internal_heating_solution.keys())
        if keys_solution != keys_cables:
            raise ValueError(
                f"Keys in solution must match cable representations. "
                f"Keys in solution: {keys_solution}, "
                f"Cable keys from representations: {keys_cables}"
            )
        return self


class StateSoil(State):
    """Extends upon the base State class. Includes additional attribute mutual_heating_solutions and validation thereof.

    Attributes:
        mutual_heating_solutions: dict[CableKey, np.ndarray]
            A dictionary containing the temperature increase inside a cable
            due to mutual heating from other cables in the environment.
            This is stored as a dict with CableKey as key and an array of
            temperature increases per grid point as value.

    """

    mutual_heating_solutions: dict[CableKey, np.ndarray] = Field()

    @model_validator(mode="after")
    def validate_mutual_heating_solutions(self):
        """Validate that mutual_heating_solutions keys match the cable representation keys."""
        found_keys = set(self.mutual_heating_solutions.keys())
        expected_keys = {cable.name for cable in self.cable_representations}
        if found_keys != expected_keys:
            raise ValueError(
                "CableKeys of mutual_heating_solutions should match with "
                "the CableKeys in the cable representations. "
                f"Found keys: {found_keys}, expected keys: {expected_keys}"
            )
        return self


class StateAir(State):
    """StateAir has no added attributes on top of State.

    However, we want to make sure there is only one circuit (check for a unique circuit_name).
    """

    @model_validator(mode="after")
    def validate_single_circuit(self):
        """Ensure that all cable representations in StateAir belong to the same circuit."""
        circuit_names = {cable_key.circuit_name for cable_key in self.cable_representations}
        if len(circuit_names) > 1:
            raise ValueError(f"StateAir should only contain one circuit, but found multiple: {circuit_names}")
        return self
