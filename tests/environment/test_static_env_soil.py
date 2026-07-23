# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from cable_thermal_model.environment.static_env_soil import StaticEnvSoil


def test_add_measurement_point(single_circuit_env: StaticEnvSoil):
    """Test adding a measurement point to the model."""
    x, y = 1.0, -2.0
    key = single_circuit_env.add_measurement_point(x=x, y=y)

    # Check that the key is in the model's measurement points
    assert key in single_circuit_env._measurement_point_registry.measurement_point_keys

    # Check that the measurement point has the correct coordinates and ndigits
    measurement_point = next(
        (mp for mp in single_circuit_env._measurement_point_registry.points if mp.key == key), None
    )
    assert measurement_point is not None
    assert measurement_point.key == ("measurement_point", f"x={x:.3f}m", f"y={y:.3f}m")
