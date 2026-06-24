# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path


def data_path():
    """Return the path to the processed data directory."""
    return Path("data/processed")


def cache_path():
    """Return the path to the data cache directory."""
    return Path("data/cache")


def circuits_path():
    """Return the path to the circuit data directory."""
    # Get path to circuit data.
    path = Path.cwd()
    path_end = path.parts[-1]

    # Traverse path upwards to get to root folder
    while path_end != "cable-thermal-model":
        path = path.parent
        path_end = path.parts[-1]

    # When we are in the root directory.
    path_to_circuits = "data/circuits"
    return path / path_to_circuits
