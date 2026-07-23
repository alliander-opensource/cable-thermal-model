# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

import logging

import pytest

from cable_thermal_model.environment.measurement_point import (
    MEASUREMENT_POINT_KEY_PREFIX,
    MeasurementPoint,
    MeasurementPointRegistry,
)


@pytest.mark.parametrize("y", [1.0, 0.0])
def test_non_negative_y_raises_validation_error(y: float) -> None:
    """MeasurementPoint should reject non-negative y values."""
    with pytest.raises(ValueError):
        MeasurementPoint(x=1.0, y=y)


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
        measurement_point = MeasurementPoint(x=1.2349, y=-2.3451, ndigits=ndigits)
    else:
        measurement_point = MeasurementPoint(x=1.2349, y=-2.3451)

    assert measurement_point.key == expected_key


def test_registry_adds_point_and_returns_key(caplog: pytest.LogCaptureFixture) -> None:
    """Adding a new point stores it and returns its key."""
    x = 1.2349
    y = -2.3451

    registry = MeasurementPointRegistry()

    key = registry.add_measurement_point(x=x, y=y)

    assert key in registry.measurement_point_keys
    assert len(registry.points) == 1

    with caplog.at_level(logging.WARNING):
        duplicate_key = registry.add_measurement_point(x=x, y=y)

    assert len(registry.points) == 1
    assert duplicate_key in registry.measurement_point_keys
    assert "already exists" in caplog.text

    # Test that adding a point with the same rounded coordinates but different original coordinates
    # still results in a duplicate key
    with caplog.at_level(logging.WARNING):
        duplicate_key = registry.add_measurement_point(x=x + 1e-4, y=y + 1e-4)

    assert len(registry.points) == 1
    assert duplicate_key in registry.measurement_point_keys
    assert "already exists" in caplog.text

    # A new point with different rounded coordinates should be added successfully
    new_key = registry.add_measurement_point(x=x + 1e-3, y=y)

    assert key in registry.measurement_point_keys
    assert new_key in registry.measurement_point_keys
    assert len(registry.points) == 2
