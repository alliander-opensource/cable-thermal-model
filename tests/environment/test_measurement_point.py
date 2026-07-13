# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

import logging

import pytest

from cable_thermal_model.environment.measurement_point import (
    MEASUREMENT_POINT_KEY_PREFIX,
    MeasurementPoint,
    MeasurementPointInputSchema,
    MeasurementPointRegistry,
)


def _build_input(x: float = 1.2349, y: float = -2.3451, ndigits: int = 3) -> MeasurementPointInputSchema:
    """Create valid measurement-point input for tests."""
    return MeasurementPointInputSchema(
        x=x,
        y=y,
        distances_to_cables={},
        distances_to_mirror_cables={},
        ndigits=ndigits,
    )


@pytest.mark.parametrize("y", [1.0, 0.0])
def test_non_negative_y_raises_validation_error(y: float) -> None:
    """MeasurementPointInputSchema should reject non-negative y values."""
    with pytest.raises(ValueError):
        MeasurementPointInputSchema(
            x=1.0,
            y=y,
            distances_to_cables={},
            distances_to_mirror_cables={},
        )


@pytest.mark.parametrize(
    "ndigits,expected_key",
    [
        (None, (MEASUREMENT_POINT_KEY_PREFIX, "x=1.235m", "y=-2.345m")),
        (3, (MEASUREMENT_POINT_KEY_PREFIX, "x=1.235m", "y=-2.345m")),
        (4, (MEASUREMENT_POINT_KEY_PREFIX, "x=1.2349m", "y=-2.3451m")),
        (5, (MEASUREMENT_POINT_KEY_PREFIX, "x=1.23490m", "y=-2.34510m")),
    ],
)
def test_create_key(ndigits, expected_key) -> None:
    """MeasurementPoint initialization should use the key-generation defaults."""
    if ndigits is not None:
        measurement_point = MeasurementPoint(_build_input(ndigits=ndigits))
    else:
        measurement_point = MeasurementPoint(_build_input())

    assert measurement_point.key == expected_key


def test_registry_adds_point_and_returns_key(caplog: pytest.LogCaptureFixture) -> None:
    """Adding a new point stores it and returns its key."""
    registry = MeasurementPointRegistry()
    input_data = _build_input()

    key = registry.add_measurement_point(input_data)

    assert key in registry.measurement_point_keys
    assert len(registry.points) == 1

    with caplog.at_level(logging.WARNING):
        duplicate_key = registry.add_measurement_point(input_data)

    assert len(registry.points) == 1
    assert duplicate_key in registry.measurement_point_keys
    assert "already exists" in caplog.text

    # Test that adding a point with the same rounded coordinates but different original coordinates
    # still results in a duplicate key
    new_input_data = _build_input(x=1.2353, y=-2.3446)
    with caplog.at_level(logging.WARNING):
        duplicate_key = registry.add_measurement_point(new_input_data)

    assert len(registry.points) == 1
    assert duplicate_key in registry.measurement_point_keys
    assert "already exists" in caplog.text

    # A new point with different rounded coordinates should be added successfully
    new_input_data = _build_input(x=1.236, y=-2.346)
    new_key = registry.add_measurement_point(new_input_data)

    assert key in registry.measurement_point_keys
    assert new_key in registry.measurement_point_keys
    assert len(registry.points) == 2
