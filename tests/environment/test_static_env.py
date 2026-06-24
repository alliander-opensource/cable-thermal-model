# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

import math

import pytest
from pydantic import ValidationError

from cable_thermal_model.cable.cable_circuit import CableKey, CablePosition, TrefoilCircuit
from cable_thermal_model.cable.enums.circuit_enums import BondingType, CircuitType
from cable_thermal_model.cable.schemas.cable_input_schemas import CableConstructionalInputSchema
from cable_thermal_model.cable.schemas.circuit_schemas import (
    CircuitConfigurationFromCableConstructionalInputSchema,
    CircuitInAirFromCableIdInputSchema,
    CircuitInAirFromCableInputSchema,
    CircuitInSoilFromCableConstructionalInputSchema,
    CircuitInSoilFromCableIdInputSchema,
    CircuitInSoilFromCableInputSchema,
)
from cable_thermal_model.cable.schemas.pipe_schemas import PipeInputSchema
from cable_thermal_model.environment.static_env import StaticEnv
from cable_thermal_model.environment.static_env_air import StaticEnvAir
from cable_thermal_model.environment.static_env_soil import StaticEnvSoil
from cable_thermal_model.model.cables.enum_classes_cable import PipeFillType
from cable_thermal_model.model.cables.fd_cable import FDCable, FDCableInAir, FDCableTrefoilCircuitInSinglePipeInAir


def test_cable_field_validation_cable_in_air(single_core_cable_xlpe: FDCable):
    with pytest.raises(ValidationError, match="Input should be an instance of FDCableInAir"):
        CircuitInAirFromCableInputSchema(
            circuit_name="c1",
            cable=single_core_cable_xlpe,
            circuit_type=CircuitType.Single,
            dist=0.3,
            bonding_type=BondingType.TwoSided,
        )


def test_basic_single_circuit(single_circuit_env: StaticEnvSoil):
    """Check if the correct number of cables and number of circuits is retrieved."""
    NUMBER_OF_CABLES_IN_TREFOIL_CIRCUIT = 3
    number_of_circuits = len(single_circuit_env.circuits)
    assert single_circuit_env.get_number_of_cables() == NUMBER_OF_CABLES_IN_TREFOIL_CIRCUIT
    assert number_of_circuits == 1


def test_basic_two_circuits(single_circuit_env: StaticEnvSoil, three_core_cable_xlpe: FDCable):
    """Check if the correct number of cables and number of circuits is retrieved."""
    single_circuit_env.add_circuit_from_cable(
        CircuitInSoilFromCableInputSchema(
            x=2.0, y=-1.0, circuit_name="c2", cable=three_core_cable_xlpe, circuit_type=CircuitType.Single
        )
    )
    number_of_circuits = len(single_circuit_env.circuits)
    assert single_circuit_env.get_number_of_cables() == (3 + 1)
    assert number_of_circuits == 2  # noqa: PLR2004


def test_string_representation(single_circuit_env: StaticEnvSoil, single_circuit_in_air_env: StaticEnvAir):
    """Check if the string representation of the environment contains the correct number of circuits and cables."""
    assert str(single_circuit_env) == "StaticEnvSoil with circuits: c1"
    assert str(single_circuit_in_air_env) == "StaticEnvAir with circuits: c1"


def test_add_existing_circuit_name_to_static_env(single_circuit_env: StaticEnvSoil, three_core_cable_xlpe: FDCable):
    """Check if adding a circuit with an existing name raises a ValueError."""
    with pytest.raises(ValueError, match="c1 already exists in environment. Circuit names should be unique."):
        single_circuit_env.add_circuit_from_cable(
            CircuitInSoilFromCableInputSchema(
                x=2.0, y=-1.0, circuit_name="c1", cable=three_core_cable_xlpe, circuit_type=CircuitType.Single
            )
        )


@pytest.mark.parametrize("env_fixture, expected", [("single_circuit_env", 3), ("env", 0)])
def test_staticenv_get_number_of_cables(env_fixture: str, expected: int, request: pytest.FixtureRequest):
    """Tests if the correct number of cables is returned."""
    env = request.getfixturevalue(env_fixture)
    assert env.get_number_of_cables() == expected


@pytest.mark.parametrize(
    "env_fixture, x, y, circuit_name, cable_fixture, circuit_type, expected",
    [
        ("env", 0, -1.15, "c1", "three_core_cable_pilc", CircuitType.Single, 1),
        ("env", 0, -1.15, "c1", "single_core_cable_xlpe", CircuitType.Trefoil, 3),
        ("env", 0, -1.15, "c1", "single_core_cable_xlpe", CircuitType.Linear, 3),
        # Test default value for circuit_type
        ("env", 0, -1.15, "c1", "single_core_cable_xlpe", None, 3),
    ],
)
def test_static_env_add_circuit_from_cable(
    env_fixture: str,
    x: float,
    y: float,
    circuit_name: str,
    cable_fixture: str,
    circuit_type: str,
    expected: int,
    request: pytest.FixtureRequest,
):
    """Tests that the correct number of cables is added to the environment given a Cable object."""
    env = request.getfixturevalue(env_fixture)
    cable = request.getfixturevalue(cable_fixture)
    assert env.get_number_of_cables() == 0
    env.add_circuit_from_cable(
        CircuitInSoilFromCableInputSchema(x=x, y=y, circuit_name=circuit_name, cable=cable, circuit_type=circuit_type)
    )
    assert env.get_number_of_cables() == expected


@pytest.mark.parametrize(
    "cable_fixture, circuit_type, expected_error_type, expected_error_message",
    [
        (
            "three_core_cable_pilc",
            CircuitType.Trefoil,
            ValueError,
            "Unrealistic combination of cable and circuit_type: cable has 3 conductors and circuit_type is 'trefoil'."
            " Did you mean circuit_type 'single' instead?",
        ),
        # Use this test to verify this edge case. Now it breaks the code due to single circuit not having a
        #   conductor distance
        (
            "single_core_cable_xlpe",
            CircuitType.Single,
            ValueError,
            "Unrealistic combination of cable and circuit_type: cable has 1 conductor and circuit_type 'single'."
            " Did you mean circuit_type 'trefoil' or 'linear' instead?",
        ),
    ],
)
def test_warn_for_unrealistic_circuits(
    env: StaticEnv,
    cable_fixture: str,
    circuit_type: CircuitType,
    expected_error_type,
    expected_error_message: str,
    request: pytest.FixtureRequest,
):
    """Tests if the correct error is raised when an unrealistic combination of cable and circuit_type is used."""
    cable = request.getfixturevalue(cable_fixture)
    with pytest.raises(expected_error_type) as exc_info:
        env.add_circuit_from_cable(
            CircuitInSoilFromCableInputSchema(x=0, y=0, circuit_name="c1", cable=cable, circuit_type=circuit_type)
        )
    assert str(exc_info.value) == expected_error_message


@pytest.mark.parametrize(
    "env_fixture, x, y, circuit_name, cable_id, circuit_type, expected",
    [
        ("env", 0, -0.15, "c1", "GPLK 10/10 kV 3x185 Al", CircuitType.Single, 1),
        ("env", 0, -0.15, "c1", "YMeKrvaslqwd 12/20kV 1x630 Alrm + as50", CircuitType.Trefoil, 3),
        ("env", 0, -0.15, "c1", "YMeKrvaslqwd 12/20kV 1x630 Alrm + as50", None, 3),
    ],
)
def test_staticenv_add_circuit_from_cable_id_number_of_cables(
    env_fixture: str,
    x: float,
    y: float,
    circuit_name: str,
    cable_id: str,
    circuit_type: CircuitType,
    expected: int,
    request: pytest.FixtureRequest,
):
    env = request.getfixturevalue(env_fixture)
    assert env.get_number_of_cables() == 0
    env.add_circuit_from_cable_id(
        CircuitInSoilFromCableIdInputSchema(
            x=x,
            y=y,
            circuit_name=circuit_name,
            cable_id=cable_id,
            circuit_type=circuit_type,
        )
    )
    assert env.get_number_of_cables() == expected


def test_staticenv_add_circuit_from_cable_constructional_information_number_of_cables(
    env: StaticEnvSoil, simple_screened_cable_constructional_information: CableConstructionalInputSchema
):
    assert env.get_number_of_cables() == 0
    env.add_circuit_from_cable_constructional_information(
        CircuitInSoilFromCableConstructionalInputSchema(
            x=0,
            y=-1.0,
            circuit_name="c1",
            cable_constructional_information=simple_screened_cable_constructional_information,
            multiple_configurations=[
                CircuitConfigurationFromCableConstructionalInputSchema(
                    circuit_type=CircuitType.Linear,
                    dist=0.2,
                    cable_constructional_information=simple_screened_cable_constructional_information,
                    length=100,
                ),
                CircuitConfigurationFromCableConstructionalInputSchema(
                    circuit_type=CircuitType.Trefoil,
                    cable_constructional_information=simple_screened_cable_constructional_information,
                    length=100,
                ),
            ],
        )
    )
    expected_number_of_cables = 3
    assert env.get_number_of_cables() == expected_number_of_cables


@pytest.mark.parametrize(
    "env_fixture, circuit_name, cable_id, circuit_type, expected",
    [
        ("env_air", "c1", "GPLK 10/10 kV 3x185 Al", CircuitType.Single, (0.21, 3.94, 0.60)),
        (
            "env_air",
            "c1",
            "YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
            CircuitType.Trefoil,
            (0.96, 1.25, 0.20),
        ),
        (
            "env_air",
            "c1",
            "YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
            CircuitType.Linear,
            (0.62, 1.95, 0.25),
        ),
    ],
)
def test_static_env_air_add_circuit_from_cable_id_number_of_cables(
    env_fixture: str,
    circuit_name: str,
    cable_id: str,
    circuit_type: CircuitType,
    expected: tuple[float, float, float],
    request: pytest.FixtureRequest,
):
    env = request.getfixturevalue(env_fixture)
    assert env.get_number_of_cables() == 0
    env.add_circuit_from_cable_id(
        CircuitInAirFromCableIdInputSchema(
            circuit_name=circuit_name,
            cable_id=cable_id,
            circuit_type=circuit_type,
        )
    )

    for cable in env.cables.values():
        convection_params = cable.cable.convection_params
        Z = convection_params.Z
        E = convection_params.E
        Cg = convection_params.Cg
        assert (Z, E, Cg) == expected

    with pytest.raises(ValueError):
        env.add_circuit_from_cable_id(
            CircuitInAirFromCableIdInputSchema(circuit_name=circuit_name, cable_id=cable_id, circuit_type=circuit_type)
        )
    with pytest.raises(ValueError):
        env.add_circuit_from_cable_id(
            CircuitInAirFromCableIdInputSchema(circuit_name=circuit_name, cable_id=cable_id, circuit_type=circuit_type)
        )


def test_staticenv_add_circuit_in_air_from_cable_id_trefoil_in_single_pipe():
    """Tests if adding a trefoil circuit in a single pipe in air is working correctly."""
    static_env = StaticEnvAir()
    static_env.add_circuit_from_cable_id(
        CircuitInAirFromCableIdInputSchema(
            circuit_name="c1",
            cable_id="YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
            pipe=PipeInputSchema(
                fill_type=PipeFillType.Air, outer_radius=0.1, sdr=11, trefoil_circuit_in_single_pipe=True
            ),
            circuit_type=CircuitType.Trefoil,
        )
    )

    # Verify that the circuit consists of a single cable and that the cable is of the correct type
    assert static_env.get_number_of_cables() == 1

    cable_key = CableKey(circuit_name="c1", cable_position=CablePosition.TrefoilCircuitInSinglePipe)
    cable = static_env.get_cable(cable_key).cable
    assert isinstance(cable, FDCableTrefoilCircuitInSinglePipeInAir)

    # Verify that the bonding type is inferred correctly
    assert static_env.circuits["c1"].bonding == BondingType.TwoSided

    # Check that conductor distance is set correctly
    assert cable.layer_metrics.conductor_distance == 2 * cable.layer_metrics.cable_radius


@pytest.mark.parametrize(
    "env_fixture, cable_fixture, expected",
    [
        ("env", "three_core_cable_pilc", CircuitType.Single),
        ("env", "single_core_cable_xlpe", CircuitType.Trefoil),
        ("env", "trefoil_in_single_pipe", CircuitType.Trefoil),
    ],
)
def test_staticenv_get_circuit_type(
    env_fixture: str, cable_fixture: str, expected: CircuitType, request: pytest.FixtureRequest
):
    """Tests retrieving the proper circuit_type from an environment."""
    env = request.getfixturevalue(env_fixture)
    cable: FDCable = request.getfixturevalue(cable_fixture)
    if cable.conductor.number_of_conductors == 1:
        with pytest.warns(UserWarning) as record:
            assert env.get_circuit_type(cable) == expected
            assert len(record) == 1
            assert record[0].message.args[0] == (
                '"trefoil" is assumed to be the circuit_type, since the cable has '
                'three conductors. If "linear" was meant as circuit_type, please '
                "specify."
            )
    else:
        assert env.get_circuit_type(cable) == expected


@pytest.mark.parametrize(
    "env_fixture, cable_key",
    [
        ("single_circuit_env", CableKey(circuit_name="c1", cable_position=CablePosition.TrefoilTop)),
        ("single_circuit_env", CableKey(circuit_name="c1", cable_position=CablePosition.TrefoilLeft)),
        ("single_circuit_env", CableKey(circuit_name="c1", cable_position=CablePosition.TrefoilRight)),
    ],
)
def test_staticenv_get_cable(env_fixture: str, cable_key: CableKey, request: pytest.FixtureRequest):
    """Tests if the cable is retrieved correctly."""
    env: StaticEnvSoil = request.getfixturevalue(env_fixture)
    actual_cable_name = env.get_cable(cable_key).name
    assert actual_cable_name == cable_key


def test_staticenv_from_file(env: StaticEnvSoil, request: pytest.FixtureRequest):
    """Tests if building cables in an environment from a file works accordingly."""
    file_location = "elst_four.csv"
    env._from_file(file_location)
    veld1 = "ELT2.24"
    veld2 = "ELT2.26"
    assert env.get_number_of_cables() == 6  # noqa: PLR2004
    assert env.circuits.keys() == {veld1, veld2}
    assert isinstance(env.circuits[veld1], TrefoilCircuit)
    assert math.isclose(env.circuits[veld1].x, 0, abs_tol=0)
    assert math.isclose(env.circuits[veld1].y, -1.15, abs_tol=0)
    assert isinstance(env.circuits[veld2], TrefoilCircuit)
    assert math.isclose(env.circuits[veld2].x, 0.23, abs_tol=0)
    assert math.isclose(env.circuits[veld2].y, -1.15, abs_tol=0)


@pytest.mark.parametrize(
    "circuit_type, dist, cable_fixture, clipped_to_wall, expected_parameters",
    [
        (
            CircuitType.Single,
            0.3,
            "three_core_cable_pilc_in_air",
            False,
            (0.21, 3.94, 0.60),
        ),
        (
            CircuitType.Linear,
            None,
            "single_core_cable_xlpe_in_air",
            False,
            (0.62, 1.95, 0.25),
        ),
        (
            CircuitType.Linear,
            10.0,  # Very large distance comparted to cable radius
            "single_core_cable_xlpe_in_air",
            False,
            (0.21, 3.94, 0.60),
        ),
        (
            CircuitType.Trefoil,
            None,
            "single_core_cable_xlpe_in_air",
            False,
            (0.96, 1.25, 0.20),
        ),
        (
            CircuitType.LinearVertical,
            None,
            "single_core_cable_xlpe_in_air",
            False,
            (1.61, 0.42, 0.20),
        ),
        (
            CircuitType.LinearVertical,
            10.0,  # Very large distance comparted to cable radius
            "single_core_cable_xlpe_in_air",
            False,
            (1.31, 2.00, 0.20),
        ),
        (
            CircuitType.Single,
            None,
            "three_core_cable_pilc_in_air",
            True,
            (1.69, 0.63, 0.25),
        ),
        (
            CircuitType.Trefoil,
            None,
            "single_core_cable_xlpe_in_air",
            True,
            (0.94, 0.79, 0.20),
        ),
    ],
)
def test_get_convection_parameters(
    env_air: StaticEnvAir,
    request: pytest.FixtureRequest,
    circuit_type: CircuitType,
    dist: float | None,
    cable_fixture: str,
    clipped_to_wall: bool,
    expected_parameters: tuple[float, float, float],
):
    """Tests if the correct convection parameters are set for different circuit types and clipping to wall."""
    cable = request.getfixturevalue(cable_fixture)
    assert isinstance(cable, FDCableInAir)
    Z, E, Cg = env_air._get_convection_parameters(
        circuit_type=circuit_type,
        dist=dist,
        cable=cable,
        clipped_to_wall=clipped_to_wall,
    )
    assert (Z, E, Cg) == expected_parameters


def test_set_convection_parameters(env_air: StaticEnvAir, single_core_cable_xlpe_in_air: FDCableInAir):
    """Tests if the convection parameters are set correctly when adding a circuit from a cable id."""
    env_air.add_circuit_from_cable_id(
        CircuitInAirFromCableIdInputSchema(
            circuit_name="c1",
            cable_id="YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
            circuit_type=CircuitType.Trefoil,
        )
    )
    cable = env_air.get_cable(CableKey(circuit_name="c1", cable_position=CablePosition.TrefoilTop)).cable
    assert isinstance(cable, FDCableInAir)
    assert cable.convection_params is not None
    assert (cable.convection_params.Z, cable.convection_params.E, cable.convection_params.Cg) == (0.96, 1.25, 0.20)


def test_set_convection_parameters_value_error(env_air: StaticEnvAir, single_core_cable_xlpe_in_air: FDCableInAir):
    """Tests if a ValueError is raised when no convection parameters are set for the given circuit type."""
    with pytest.raises(ValueError, match="No convection parameters set for None"):
        env_air._get_convection_parameters(
            circuit_type=None,
            dist=0.1,
            cable=single_core_cable_xlpe_in_air,
            clipped_to_wall=False,
        )

    with pytest.raises(ValueError, match="No convection parameters available for linear_vertical when clipped to wall"):
        env_air._get_convection_parameters(
            circuit_type=CircuitType.LinearVertical,
            dist=0.1,
            cable=single_core_cable_xlpe_in_air,
            clipped_to_wall=True,
        )
