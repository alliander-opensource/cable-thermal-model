# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

import pytest

from cable_thermal_model.environment.static_env_air import StaticEnvAir
from cable_thermal_model.environment.static_env_soil import StaticEnvSoil
from cable_thermal_model.model.model_air import ModelAir
from cable_thermal_model.model.model_factory import ModelFactory
from cable_thermal_model.model.model_soil import ModelSoil


@pytest.mark.parametrize(
    "static_env,expected_model_type",
    [
        pytest.param(StaticEnvAir(), ModelAir, id="air-env"),
        pytest.param(StaticEnvSoil(), ModelSoil, id="soil-env"),
    ],
)
def test_create_model_returns_expected_type_for_supported_environments(
    static_env: StaticEnvAir | StaticEnvSoil,
    expected_model_type: type[ModelAir] | type[ModelSoil],
):
    """Supported environments should resolve to their corresponding model classes."""
    model = ModelFactory.create_model(static_env=static_env)

    assert isinstance(model, expected_model_type)
    assert model.static_env is static_env


def test_create_model_raises_for_unsupported_static_environment_type():
    """Unsupported environments should raise a clear ValueError."""
    unsupported_env = object()

    with pytest.raises(
        ValueError,
        match=("Unsupported static environment type: object\\. Expected StaticEnvAir or StaticEnvSoil\\."),
    ):
        ModelFactory.create_model(static_env=unsupported_env)  # type: ignore[arg-type]


def test_create_model_includes_custom_type_name_in_error_message():
    """The error should include the unsupported input type name for easier debugging."""

    class UnsupportedEnvironment:
        pass

    with pytest.raises(
        ValueError,
        match=(
            "Unsupported static environment type: UnsupportedEnvironment\\. Expected StaticEnvAir or StaticEnvSoil\\."
        ),
    ):
        ModelFactory.create_model(static_env=UnsupportedEnvironment())  # type: ignore[arg-type]
