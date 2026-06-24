# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

"""Testing Notes.

Things to test:
    - Very little actually happens in the source module. When scrapping the "methods" that only directly alter a single
       variable, or almost only call upon external methods that should be tested elsewhere, basically only the
       CableCircuit sub-classes bring anything to test, in the form of calculations.
    - This means that we should test input reliability if applicable.
       (if we want to prevent erroneous input from happening)
    - Also, we should test the calculations made in the CableCircuit sub-classes. Important to evaluate are boundary
       values for valid inputs, and if we want to validate input itself, of course invalid input as well.

Questions / TODO's:
    - What is the purpose of the [PosCable] namedtuple? It seems as if its sole purpose is to attach a location and
       optionally a name to a Cable object. Why not a [PositionedCable(Cable)] class extending a regular [Cable].

    - return_mirror_cable() seems to only mirror the y-coordinate. Why the y-coordinate and not the x-coordinate?
       (what is the y-coordinate relative to that it may be flipped, but not the x-coordinate?)
    - return_mirror_cable() already starts with a deep copy, meaning it already is a "PosCable", assuming the typehint
       is correct. We may as well just flip [pos_cable_mirror.y] and return that value, rather than making a third
       PosCable? That also emphasizes what was changed.

    - Horrible naming of [pos_cable_] variable, bound to be confused with [pos_cable]. To be renamed.
    - As with return_mirror_cable(), we've already deep copied a PosCable. We may as well replace the copy's [.cable]
       with the [.cable.add_soil_layers()] output directly, clarifying what we did.

    - The class CableCircuit itself has no declaration of "cables", but all three implementations carry the singular
       [Cable] type as its hinted content and implement it as a list of [Cable] objects.

    - CableCircuit sub-classes need to properly order methods, especially [initialize_screen_loss_functions()] for
       greater clarity.

    - The CircuitBuilder class has no real grounds to be a class. It's only contents are two methods, which hold the
       same return line, and at most one other. The CircuitType calling method used, also seems to be heavily
       influenced by C-related calls, rather than Pythonesque coding, suggesting better and more transparent options
       could be available?

    - Tests currently use "fd_objects.py" as a source of fixtures. However the import is currently done as a
       "from ... import *" import, which obfuscates what actually is being imported and generates a lot of linting
       issues. Ideally we'd import these files in a more official manner.

    - The SingleCable class method [initialize_screen_loss_functions()] ends with an if-branch statement for each cable
       type. However, earlier, a value [max_operating_temperature] is retrieved from a method
       [_get_operating_temperature()]. This method does nothing more than a single line of code per type, filling in a
       constant value based on type value. That singular value can be moved into the existing if-branch of
       [initialize_screen_loss_functions()] if the code for [initialize_screen_loss_functions()] gets re-organized.
"""

import re

import numpy as np
import pytest

from cable_thermal_model.cable.cable_builder import CableBuilder
from cable_thermal_model.cable.cable_circuit import (
    BondingType,
    CablePosition,
    CircuitBuilder,
    CircuitInitData,
    CircuitType,
    CircuitYReference,
    LinearCircuit,
    SingleCable,
    TrefoilCircuit,
    TrefoilCircuitInSinglePipe,
)
from cable_thermal_model.cable.schemas.circuit_schemas import CircuitInSoilFromCableIdInputSchema
from cable_thermal_model.cable.schemas.pipe_schemas import PipeInputSchema
from cable_thermal_model.environment.static_env_soil import StaticEnvSoil
from cable_thermal_model.model.cables.abstract_cable import WeightedScreenImpedance
from cable_thermal_model.model.cables.enum_classes_cable import CableScreenLossType, PipeFillType
from cable_thermal_model.model.cables.fd_cable import FDCable, FDCableTrefoilCircuitInSinglePipe

# CONSTANTS
_RADIAL_APPROXIMATION_RESOLUTION_IN_METERS = 1e-9


@pytest.mark.parametrize(
    "screen_temperature,conductor_temperature,conductor_distance,bonding_type,expected_lambda1",
    [
        (
            [71.2, 75.1, 71.2],
            [85.8, 90.0, 85.8],
            0.0,
            BondingType.OneSided,
            [0.0099, 0.0397, 0.0099],
        ),
        (
            [66.9, 71.6, 66.9],
            [85.0, 90.0, 85.0],
            0.2,
            BondingType.OneSided,
            [0.0007, 0.0028, 0.0007],
        ),
    ],
)
def test_screen_loss_function_linear_1x630(
    screen_temperature: list[float],
    conductor_temperature: list[float],
    conductor_distance: float | None,
    bonding_type: BondingType,
    expected_lambda1: list[float],
):
    """Compares steady state temperature with VCA for a circuit in flat formation."""
    test_cable = CableBuilder.build_cable_from_cable_id(
        cable_id="YMeKrvaslqwd 12/20kV 1x630 Alrm + as50", fd_cable_class=FDCable
    )
    test_circuit = LinearCircuit(
        CircuitInitData(
            x=0,
            y=-1,
            cable=test_cable,
            dist=conductor_distance,
            circuit_name="c1",
            bonding_type=bonding_type,
        )
    )
    test_circuit.initialize_screen_loss_functions()

    for circuit_cable, Ts, Tc, l1 in zip(
        test_circuit.cables, screen_temperature, conductor_temperature, expected_lambda1, strict=True
    ):
        ctm_lambda1 = circuit_cable.cable.get_cable_screen_loss_factor(Ts, Tc)

        assert np.isclose(ctm_lambda1, l1, atol=0.005)


@pytest.mark.parametrize(
    "screen_temperature,conductor_temperature,bonding_type,expected_lambda1",
    [
        (
            [75.2, 74.9, 75.2],
            [90.0, 89.6, 90.0],
            BondingType.OneSided,
            [0.0193, 0.0193, 0.0193],
        ),
    ],
)
def test_screen_loss_function_trefoil_1x630(
    screen_temperature: list[float],
    conductor_temperature: list[float],
    bonding_type: BondingType,
    expected_lambda1: list[float],
):
    """Compares steady state temperature with VCA for a circuits in trefoil formation."""
    test_cable = CableBuilder.build_cable_from_cable_id(
        cable_id="YMeKrvaslqwd 12/20kV 1x630 Alrm + as50", fd_cable_class=FDCable
    )
    circuit = TrefoilCircuit(CircuitInitData(x=0, y=-1, cable=test_cable, circuit_name="c1", bonding_type=bonding_type))
    circuit.initialize_screen_loss_functions()

    for circuit_cable, Ts, Tc, l1 in zip(
        circuit.cables, screen_temperature, conductor_temperature, expected_lambda1, strict=True
    ):
        ctm_lambda1 = circuit_cable.cable.get_cable_screen_loss_factor(Ts, Tc)
        assert np.isclose(ctm_lambda1, l1, rtol=0.02)


@pytest.mark.parametrize(
    "screen_temperature,conductor_temperature,conductor_distance,bonding_type,lambda1",
    [
        (
            [72.6, 75.6, 72.7],
            [81.9, 85.0, 82.0],
            None,
            BondingType.TwoSided,
            [0.1429, 0.0348, 0.1514],
        ),
        (
            [73.5, 75.5, 74.1],
            [82.9, 85.0, 83.6],
            0.2,
            BondingType.TwoSided,
            [0.4783, 0.2719, 0.5194],
        ),
        (
            [71.1, 74.8, 71.1],
            [81.2, 85.0, 81.2],
            None,
            BondingType.OneSided,
            [0.0041, 0.0165, 0.0041],
        ),
    ],
)
def test_screen_loss_function_linear(
    screen_temperature: list[float],
    conductor_temperature: list[float],
    conductor_distance: float | None,
    bonding_type: BondingType,
    lambda1: list[float],
):
    """Compares steady state temperature with VCA for a circuit in flat formation.

    The soil parameters used in this test are an ambient temperature of
    15 [degC] and a thermal resistivity of 0.75 [Km/W].
    """
    test_cable = CableBuilder.build_cable_from_cable_id(cable_id="OD 50kV 1x400Cu", fd_cable_class=FDCable)  # type: ignore
    circuit = LinearCircuit(
        CircuitInitData(
            x=0,
            y=-1,
            cable=test_cable,
            circuit_name="c1",
            dist=conductor_distance,
            bonding_type=bonding_type,
        )  # type: ignore
    )
    circuit.initialize_screen_loss_functions()

    for circuit_cable, Ts, Tc, l1 in zip(
        circuit.cables, screen_temperature, conductor_temperature, lambda1, strict=True
    ):
        ctm_lambda1 = circuit_cable.cable.get_cable_screen_loss_factor(Ts, Tc)
        assert np.isclose(ctm_lambda1, l1, rtol=0.03)


@pytest.mark.parametrize(
    "cable_id,pipe,screen_temperature,conductor_temperature,expected_lambda1",
    [
        ("OD 50kV 1x400Cu", None, 75.3, 85.0, 0.0602),
        ("OD 50kV 1x400Cu", PipeInputSchema(fill_type=PipeFillType.Air), 77.0, 85.0, 0.1785),
        ("YMeKrvaslqwd 12/20kV 1x630 Alrm + as50", None, 75.9, 90.0, 0.1089),
        (
            "YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
            PipeInputSchema(fill_type=PipeFillType.Air),
            79.4,
            90.0,
            0.3677,
        ),
    ],
)
def test_screen_loss_function_trefoil(
    cable_id: str,
    pipe: PipeInputSchema | None,
    screen_temperature: float,
    conductor_temperature: float,
    expected_lambda1: float,
):
    """Compares steady state temperature with VCA for a circuits in trefoil formation.

    VCA numbers are based on environment with 15 [degC] ambient temperature and thermal reisitivity of 0.75 [Km/W].
    """
    test_cable = CableBuilder.build_cable_from_cable_id(cable_id=cable_id, pipe=pipe, fd_cable_class=FDCable)
    circuit = TrefoilCircuit(CircuitInitData(x=0, y=-1, cable=test_cable, circuit_name="c1"))
    circuit.initialize_screen_loss_functions()
    circuit_cable = circuit.cables[0].cable

    Ts = screen_temperature
    Tc = conductor_temperature
    ctm_lambda1 = circuit_cable.get_cable_screen_loss_factor(Ts, Tc)
    assert np.isclose(ctm_lambda1, expected_lambda1, rtol=0.02)


def test_initialize_screen_loss_functions_with_erroneous_parameters():
    # Preparation:
    adjusted_single_core_cable = CableBuilder.build_cable_from_cable_id(
        cable_id="YMeKrvaslqwd 12/20kV 1x630 Alrm + as50", fd_cable_class=FDCable
    )

    fake_cable_type = 999
    expected_error_string = f"CableType: {fake_cable_type} not supported."

    adjusted_single_core_cable.cable_type = fake_cable_type  # type: ignore

    # Generation:
    test_single_cable = SingleCable(CircuitInitData(x=0, y=0, cable=adjusted_single_core_cable, circuit_name="c1"))

    # Evaluation:
    with pytest.raises(ValueError) as val_error:
        test_single_cable.initialize_screen_loss_functions()
    assert str(val_error.value) == expected_error_string


def test_cable_circuit_builder_basic_usage_using_cable_id():
    """Note: The from_cable() method is already covered via other tests."""
    # Preparation
    cable_id = "YMeKrvaslqwd 12/20kV 1x630 Alrm + as50"
    screen_temp = 85.2
    conductor_temp = 90.0

    # Generation
    test_circuit = CircuitBuilder.from_cable_id(
        x=0, y=0, circuit_type=CircuitType.Trefoil, cable_id=cable_id, circuit_name="Test circuit"
    )
    ctm_lambda1 = test_circuit.cables[0].cable.get_cable_screen_loss_factor(screen_temp, conductor_temp)

    # Evaluation
    # Verify Radii to confirm the building of the proper cable
    assert np.allclose(
        [layer_properties.outer_radius for layer_properties in test_circuit.cables[0].cable.layer_properties.values()],
        [0.01395, 0.0150425, 0.0205425, 0.021635, 0.022, 0.022785, 0.026],
        atol=_RADIAL_APPROXIMATION_RESOLUTION_IN_METERS,
    )
    # Verify initialize_screen_loss_functions
    assert np.isclose(ctm_lambda1, 0.107, rtol=0.02)


@pytest.mark.parametrize(
    "cable_id,circuit_type,y_ref,y_expected",
    [
        ("YMeKrvaslqwd 12/20kV 3x240 Alrm + as50", CircuitType.Single, CircuitYReference.Top, -1.0382),
        ("YMeKrvaslqwd 12/20kV 3x240 Alrm + as50", CircuitType.Single, CircuitYReference.Center, -1),
        ("YMeKrvaslqwd 12/20kV 3x240 Alrm + as50", CircuitType.Single, CircuitYReference.Bottom, -0.9618),
        (
            "YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
            CircuitType.Trefoil,
            CircuitYReference.Top,
            -1.0560222139978606,
        ),
        ("YMeKrvaslqwd 12/20kV 1x630 Alrm + as50", CircuitType.Trefoil, CircuitYReference.Center, -1),
        (
            "YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
            CircuitType.Trefoil,
            CircuitYReference.Bottom,
            -0.9849888930010697,
        ),
        ("YMeKrvaslqwd 12/20kV 1x630 Alrm + as50", CircuitType.Linear, CircuitYReference.Top, -1.026),
        ("YMeKrvaslqwd 12/20kV 1x630 Alrm + as50", CircuitType.Linear, CircuitYReference.Center, -1),
        ("YMeKrvaslqwd 12/20kV 1x630 Alrm + as50", CircuitType.Linear, CircuitYReference.Bottom, -0.974),
    ],
)
def test_cable_y_position(
    cable_id: str,
    circuit_type: CircuitType,
    y_ref: CircuitYReference,
    y_expected: float,
):
    # Generation
    test_circuit = CircuitBuilder.from_cable_id(
        x=0,
        y=-1.0,
        circuit_type=circuit_type,
        cable_id=cable_id,
        circuit_name="Test circuit",
        y_ref=y_ref,
    )
    # Evaluation
    # Verify Radii to confirm the building of the proper cable
    assert np.isclose(
        test_circuit.y,
        y_expected,
        atol=1e-4,
    )


@pytest.mark.parametrize(
    "sheath_currents,is_symmetric",
    [
        (np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]]), True),
        (np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]), False),
        (np.array([[1.0, -1.0, 0.0], [0.0, 1.0, 1.0]]), False),
        (np.array([[1.2, -1.2, 0.0], [0.0, 1.2, -1.2]]), True),
    ],
)
def test_symmetric_sheath_currents(sheath_currents: np.ndarray, is_symmetric: bool):
    is_symmetric_result = TrefoilCircuitInSinglePipe.symmetric_sheath_currents(sheath_currents)
    assert is_symmetric_result == is_symmetric


def test_symmetric_sheath_currents_invalid_input():
    invalid_sheath_currents = np.array([[1.0, -1.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match=re.escape("weighted_reactance_matrix must have shape (2, 3), got (2, 2)")):
        TrefoilCircuitInSinglePipe.symmetric_sheath_currents(invalid_sheath_currents)


@pytest.mark.parametrize(
    "bonding_type, expected_screen_loss_type, weighted_reactance_matrix",
    [
        (BondingType.NoBonding, CableScreenLossType.ReturnZero, None),
        (BondingType.CrossBonding, CableScreenLossType.CrossBondingOrOneSidedBondingTrefoil, None),
        (BondingType.OneSided, CableScreenLossType.CrossBondingOrOneSidedBondingTrefoil, None),
        (BondingType.TwoSided, CableScreenLossType.TwoSidedBondingTrefoil, None),
        (BondingType.TwoSided, CableScreenLossType.TwoSidedBondingCenter, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])),
    ],
)
def test_trefoil_in_single_pipe_screen_loss_function(
    single_core_cable_xlpe: FDCable,
    bonding_type: BondingType,
    expected_screen_loss_type: CableScreenLossType,
    weighted_reactance_matrix: np.ndarray | None,
):
    test_circuit = TrefoilCircuitInSinglePipe(
        CircuitInitData(
            x=0,
            y=0,
            cable=single_core_cable_xlpe,
            circuit_name="Test Trefoil in Single Pipe",
            bonding_type=bonding_type,
        )
    )
    # Set the weighted reactance matrix manually
    test_circuit.set_weighted_screen_impedance(
        WeightedScreenImpedance(
            weighted_reactance_matrix=weighted_reactance_matrix,
            weighted_resistance_factor=1.0,
        )
        if weighted_reactance_matrix is not None
        else None
    )

    test_circuit.initialize_screen_loss_functions()
    circuit_cable = test_circuit.cables[0].cable

    assert circuit_cable.layer_metrics.screen_loss_type == expected_screen_loss_type


def test_trefoil_in_single_pipe_screen_loss_function_non_symmetric_matrix(
    single_core_cable_xlpe: FDCable,
):
    test_circuit = TrefoilCircuitInSinglePipe(
        CircuitInitData(
            x=0,
            y=0,
            cable=single_core_cable_xlpe,
            circuit_name="Test Trefoil in Single Pipe Non-Symmetric",
            bonding_type=BondingType.TwoSided,
        )
    )
    # Set a non-symmetric weighted reactance matrix
    test_circuit.weighted_screen_impedance = WeightedScreenImpedance(
        weighted_reactance_matrix=np.array([[1.0, -1.0, 0.0], [0.0, 1.0, 1.0]]),
        weighted_resistance_factor=1.0,
    )

    with pytest.raises(
        NotImplementedError,
        match="Non-symmetric sheath currents are not supported for trefoil circuit in a single pipe.",
    ):
        test_circuit.initialize_screen_loss_functions()


def test_get_relative_screen_distances_trefoil_in_single_pipe(
    single_core_cable_xlpe: FDCable,
):
    test_circuit = TrefoilCircuitInSinglePipe(
        CircuitInitData(
            x=0,
            y=-1,
            cable=single_core_cable_xlpe,
            circuit_name="Test Trefoil in Single Pipe Relative Distances",
            bonding_type=BondingType.TwoSided,
        )
    )
    conductor_distance = test_circuit.cables[0].cable.layer_metrics.conductor_distance
    screen_radius = test_circuit.cables[0].cable.d / 2
    relative_distances = test_circuit.get_relative_screen_distances()
    expected_distances: np.ndarray = np.array(
        [
            [screen_radius, conductor_distance, conductor_distance],
            [conductor_distance, screen_radius, conductor_distance],
            [conductor_distance, conductor_distance, screen_radius],
        ]  # type: ignore
    )

    assert np.allclose(relative_distances, expected_distances)  # type: ignore


def test_get_relative_screen_distance_trefoil_compare_with_trefoil_in_single_pipe(
    single_core_cable_xlpe: FDCable,
):
    """Compare relative screen distances of TrefoilCircuit and TrefoilCircuitInSinglePipe.

    We expect the relative screen distances to be the same for both circuit types when using the same cable.
    """
    trefoil_circuit = TrefoilCircuit(
        CircuitInitData(x=0, y=-1, cable=single_core_cable_xlpe, circuit_name="Test Trefoil Relative Distances")
    )
    trefoil_in_single_pipe_circuit = TrefoilCircuitInSinglePipe(
        CircuitInitData(
            x=0, y=-1, cable=single_core_cable_xlpe, circuit_name="Test Trefoil in Single Pipe Relative Distances"
        )
    )

    trefoil_relative_distances = trefoil_circuit.get_relative_screen_distances()
    trefoil_in_single_pipe_relative_distances = trefoil_in_single_pipe_circuit.get_relative_screen_distances()

    assert np.allclose(trefoil_relative_distances, trefoil_in_single_pipe_relative_distances)


@pytest.mark.parametrize(
    "circuit_type, pipe, expected_result",
    [
        (CircuitType.Trefoil, PipeInputSchema(fill_type=PipeFillType.Air, trefoil_circuit_in_single_pipe=True), True),
        (CircuitType.Trefoil, PipeInputSchema(fill_type=PipeFillType.Air), False),
        (CircuitType.Single, PipeInputSchema(fill_type=PipeFillType.Air, trefoil_circuit_in_single_pipe=True), False),
        (CircuitType.Trefoil, None, False),
        (None, PipeInputSchema(fill_type=PipeFillType.Air, trefoil_circuit_in_single_pipe=True), True),
        (None, PipeInputSchema(fill_type=PipeFillType.Air), False),
    ],
)
def test_trefoil_in_single_pipe(circuit_type: CircuitType, pipe: PipeInputSchema, expected_result: bool):
    assert CircuitBuilder._is_trefoil_circuit_in_single_pipe(circuit_type=circuit_type, pipe=pipe) is expected_result


def test_non_trefoil_circuit_in_single_pipe_not_inmplemented_error():
    env = StaticEnvSoil()
    with pytest.raises(
        NotImplementedError,
        match="Three cables in one pipe not implemented for circuit type: linear",
    ):
        env.add_circuit_from_cable_id(
            circuit_input=CircuitInSoilFromCableIdInputSchema(
                x=0.0,
                y=-1.0,
                circuit_name="c1",
                cable_id="YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
                pipe=PipeInputSchema(
                    fill_type=PipeFillType.Water, outer_radius=0.2, sdr=11, trefoil_circuit_in_single_pipe=True
                ),
                circuit_type=CircuitType.Linear,
            )
        )

    with pytest.raises(
        NotImplementedError,
        match="Three cables in one pipe not implemented for circuit type: single",
    ):
        env.add_circuit_from_cable_id(
            circuit_input=CircuitInSoilFromCableIdInputSchema(
                x=0.0,
                y=-1.0,
                circuit_name="c1",
                cable_id="YMeKrvaslqwd 12/20kV 3x240 Alrm + as50",
                pipe=PipeInputSchema(
                    fill_type=PipeFillType.Water, outer_radius=0.2, sdr=11, trefoil_circuit_in_single_pipe=True
                ),
                circuit_type=CircuitType.Single,
            )
        )


def test_trefoil_circuit_in_single_pipe_initialization():
    cable = CableBuilder.build_cable_from_cable_id(
        cable_id="YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
        pipe=PipeInputSchema(fill_type=PipeFillType.Air, trefoil_circuit_in_single_pipe=True),
        fd_cable_class=FDCableTrefoilCircuitInSinglePipe,
    )
    test_circuit = TrefoilCircuitInSinglePipe(CircuitInitData(x=0, y=-1, cable=cable, circuit_name="c1"))

    # Check that bondingtype is inferred correctly
    assert test_circuit.bonding == BondingType.TwoSided

    # Check that conductor distance is set correctly
    circuit_cable = test_circuit.cables[0].cable
    assert circuit_cable.layer_metrics.conductor_distance == 2 * circuit_cable.layer_metrics.cable_radius

    # Check that the cable positions are set correctly
    assert test_circuit.cables[0].cable_position == CablePosition.TrefoilCircuitInSinglePipe


def test_trefoil_in_single_pipe_trefoil_circuit_error():
    with pytest.raises(
        ValueError,
        match=(
            "Three cables in one pipe configuration is not supported for "
            "TrefoilCircuit. Use TrefoilCircuitInSinglePipe instead."
        ),
    ):
        cable = CableBuilder.build_cable_from_cable_id(
            cable_id="YMeKrvaslqwd 12/20kV 3x240 Alrm + as50",
            fd_cable_class=FDCable,
            pipe=PipeInputSchema(
                fill_type=PipeFillType.Air, trefoil_circuit_in_single_pipe=True, inner_radius=0.1, outer_radius=0.2
            ),
        )
        TrefoilCircuit(CircuitInitData(x=0, y=-1, cable=cable, circuit_name="c1"))
