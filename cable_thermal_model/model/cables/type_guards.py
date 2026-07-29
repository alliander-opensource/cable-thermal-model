# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0
from warnings import warn

from cable_thermal_model.model.cables.cable import Cable
from cable_thermal_model.model.cables.cable_trefoil_circuit_single_pipe import CableTrefoilCircuitSinglePipe


def require_implemented_cable(cable: Cable, hard_stop: bool = True) -> Cable:
    """Check if the cable is implemented.

    Args:
        cable (Cable): The cable to check.
        hard_stop (bool): If True, raise an error if the cable is not implemented. If False, warn the user.

    Returns:
        Cable: The original cable if it is implemented or hard_stop is False.

    Raises:
        NotImplementedError: If the cable is not implemented and hard_stop is True.
    """
    # Check if the cable only has the abstract methods implemented, which means it is not implemented.
    if type(cable) in [Cable, CableTrefoilCircuitSinglePipe]:
        if hard_stop:
            raise NotImplementedError(f"{type(cable).__name__} is not implemented")
        warn(f"{type(cable).__name__} is an abstract Cable class and not fully implemented!", UserWarning, stacklevel=2)

    return cable
