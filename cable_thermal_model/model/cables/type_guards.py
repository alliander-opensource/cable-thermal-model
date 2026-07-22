# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0
from warnings import warn

from cable_thermal_model.model.cables.cable import Cable, CableAir, CableSoil, CableTrefoilCircuitSinglePipe


def require_soil_cable(cable: Cable) -> CableSoil:
    """Check if the cable is a soil cable.

    Args:
        cable (Cable): The cable to check.

    Returns:
        CableSoil: The cable if it is a soil cable.

    Raises:
        TypeError: If the cable is not a soil cable.

    Note:
        Every cable class that is an air cable should inherit from CableSoil, so this check will return
        the CableSoil class for any soil cable class.
    """
    if not isinstance(cable, CableSoil):
        raise TypeError(f"{type(cable).__name__} does not support soil operations")
    return cable


def require_air_cable(cable: Cable) -> CableAir:
    """Check if the cable is an air cable.

    Args:
        cable (Cable): The cable to check.

    Returns:
        CableAir: The cable if it is an air cable.

    Raises:
        TypeError: If the cable is not an air cable.

    Note:
        Every cable class that is an air cable should inherit from CableAir, so this check will return
        the CableAir class for any air cable class.
    """
    if not isinstance(cable, CableAir):
        raise TypeError(f"{type(cable).__name__} does not support convection operations")
    return cable


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
