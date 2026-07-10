# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent


def data_path():
    """Return the path to the processed data directory."""
    return DATA_DIR / "processed"


def cache_path():
    """Return the path to the data cache directory."""
    return DATA_DIR / "cache"


def circuits_path():
    """Return the path to the circuit data directory."""
    return DATA_DIR / "circuits"
