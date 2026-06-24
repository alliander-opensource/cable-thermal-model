# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

"""Testing Notes.

NOT TO TEST.

    Nothing.

TO TEST:
    All FDCable methods

Notes:
    FDCable's __init__() sets a lot of attributes as [None] objects. This heavily messed up IDE value evaluation and
     results in a lot of test and similar contexts expecting there to be a None value at these places. If these get
     initialized in the [finalize_init()] method anyway, we should probably not pre-initialise these as None. If
     initialization in __init__() is needed or desired regardless, then initialize by type as well, rather than only by
     value. (for example: [[-- radii_grid = None --]] becomes [[-- radii_grid: list[float] | None = None --]])
     This should solve a lot of the issues showing later in the file in the form of ".sum()" getting a warning because
     the evaluation it's "sum"-ing is considered boolean in nature.

    FDCable's finalize_init() has an odd name. If it is only used to finalize initialization, it should have a
     prefixing underscore, but if it is used in other circumstances as well, it shouldn't be called "finalize_init()".

    FDCable's finalize_init() uses a lot of confusing value representation. A lot of the "for ... in ..." clauses could
     probably be clarified.

    FDCable's get_redefined_cable() method uses a [cable_] variable. This should be replaced with a more Pythonesque
     name for clarity.

    FDCable's _add_outer_pipe() method uses [print()]. This should be replaced with either a logging or warning
     construction.

    FDCable's get_linear_system() method is way too long and complex. Split it into pieces.

"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from cable_thermal_model.environment.static_env_soil import StaticEnvSoil
from cable_thermal_model.model.cables.enum_classes_cable import CableLayer
from cable_thermal_model.model.cables.fd_cable import FDCable
from cable_thermal_model.validation.cable_analysis import CableAnalysis
from tests.conftest import test_cable_fixtures


@pytest.mark.parametrize(
    "fd_cable_fixture",
    test_cable_fixtures,
)
def test_fd_cable_get_layer_indices_radii(fd_cable_fixture: str, request):
    cable: FDCable = request.getfixturevalue(fd_cable_fixture)

    # Test conductor radii are compatible with computed indices for radii-grid
    conductor_start_index = (cable.radii_grid < cable.layer_properties[CableLayer.Conductor].inner_radius).sum()
    conductor_end_index = (cable.radii_grid < cable.layer_properties[CableLayer.Conductor].outer_radius).sum() - 1
    s, e = cable.get_layer_indices_for_layer(CableLayer.Conductor)
    assert conductor_start_index == s
    assert conductor_end_index == e
    # Test screen radii are compatible with computed indices for radii-grid
    screen_start_index = (cable.radii_grid < cable.layer_properties[CableLayer.Screen].inner_radius).sum()
    screen_end_index = (cable.radii_grid < cable.layer_properties[CableLayer.Screen].outer_radius).sum() - 1
    s, e = cable.get_layer_indices_for_layer(CableLayer.Screen)
    assert screen_start_index == s
    assert screen_end_index == e


@pytest.mark.parametrize(
    "fd_cable_fixture",
    test_cable_fixtures,
)
def test_fd_cable_get_layer_indices_material_properties(fd_cable_fixture: str, request):
    cable: FDCable = request.getfixturevalue(fd_cable_fixture)

    s, e = cable.get_layer_indices_for_layer(CableLayer.Conductor)
    # Test thermal resistance conductor is constant along slice from start to end indices
    assert np.isclose(np.abs(np.diff(cable.rho_grid[s:e])).sum(), 0, atol=1e-3)
    # Test electric resistance conductor is constant along slice from start to end indices
    assert np.isclose(np.abs(np.diff(cable.alpha_grid[s:e])).sum(), 0, atol=1e-3)
    s, e = cable.get_layer_indices_for_layer(CableLayer.Screen)
    # Test thermal resistance screen is constant along slice from start to end indices
    assert np.isclose(np.abs(np.diff(cable.rho_grid[s:e])).sum(), 0, atol=1e-3)
    # Test electric resistance screen is constant along slice from start to end indices
    assert np.isclose(np.abs(np.diff(cable.alpha_grid[s:e])).sum(), 0, atol=1e-3)


def test_cable_radius(single_core_cable_xlpe: FDCable, single_circuit_with_pipe_env: StaticEnvSoil):
    # Test that the cable radius equals the outer radius when no pipe is present
    assert np.isclose(
        single_core_cable_xlpe.layer_metrics.cable_radius, single_core_cable_xlpe.layer_metrics.outer_radius
    )

    # Test that the cable radius is less than the outer radius when a pipe is present
    cable_with_pipe = single_circuit_with_pipe_env.circuits["c1"].cables[0].cable
    assert cable_with_pipe.layer_metrics.cable_radius < cable_with_pipe.layer_metrics.outer_radius


@pytest.mark.parametrize(
    "fd_cable_fixture",
    test_cable_fixtures,
)
def test_construct_radii_grid(fd_cable_fixture: str, request):
    cable: FDCable = request.getfixturevalue(fd_cable_fixture)
    # Test that the radii grid starts at 0
    assert np.isclose(cable.radii_grid[0], 0)
    # Test that the radii grid ends at the outer radius of the outermost layer
    outermost_layer = max(cable.layer_properties.keys(), key=lambda layer: cable.layer_properties[layer].outer_radius)
    assert np.isclose(cable.radii_grid[-1], cable.layer_properties[outermost_layer].outer_radius)
    # Test that the radii grid has correct number of points
    expected_points = sum(cable.grid_counts.values())
    assert len(cable.radii_grid) == expected_points
    # Test that the radii grid is strictly increasing
    assert np.all(np.diff(cable.radii_grid) > 0)
    # Test that each layer boundary is exactly between two grid points
    cumulative_points = 0
    for layer in cable.layer_properties:
        layer_points = cable.grid_counts[layer]
        cumulative_points += layer_points
        if layer != outermost_layer:
            boundary_radius = cable.layer_properties[layer].outer_radius
            assert np.isclose(
                (cable.radii_grid[cumulative_points - 1] + cable.radii_grid[cumulative_points]) / 2, boundary_radius
            )


def test_get_heat_flow_value_error(single_core_cable_xlpe: FDCable):
    """Test that ValueError is raised when invalid index is provided to get_heat_flow()."""
    analysis = CableAnalysis(cable=single_core_cable_xlpe, solution=MagicMock())

    with pytest.raises(
        ValueError,
        match="The indices must lie within the finite-difference grid of the cable.",
    ):
        analysis.get_heat_flow(inner_index=-1)

    with pytest.raises(
        ValueError,
        match="The indices must lie within the finite-difference grid of the cable.",
    ):
        analysis.get_heat_flow(inner_index=100)


def test_calculate_inner_rhos(single_core_cable_xlpe: FDCable):
    """Test the _calculate_inter_rhos() method for calculating interstitial resistivity."""
    radii = np.array([0.01, 0.02, 0.03])
    inter_radii = np.array([0.015, 0.025])
    rhos = np.array([1.0, 2.0, 3.0])

    inter_rhos = single_core_cable_xlpe._calculate_inter_rhos(radii, inter_radii, rhos)
    assert rhos[0] < inter_rhos[0] < rhos[1], "First interstitial resistivity not between adjacent rhos"
    assert rhos[1] < inter_rhos[1] < rhos[2], "Second interstitial resistivity not between adjacent rhos"

    constant_rhos = single_core_cable_xlpe._calculate_inter_rhos(radii, inter_radii, np.array([2.0, 2.0, 2.0]))
    assert np.allclose(constant_rhos, 2.0)


def test_calculate_inner_rhos_inconsistent_length(single_core_cable_xlpe: FDCable):
    """Test that ValueError is raised for inconsistent input lengths in _calculate_inter_rhos()."""
    radii = np.array([0.01, 0.02])
    inter_radii = np.array([0.015, 0.025])
    rhos = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="The lengths of the input arrays are inconsistent!"):
        single_core_cable_xlpe._calculate_inter_rhos(radii, inter_radii, rhos)


def test_calculate_inner_rhos_constant_violation_near_zero(single_core_cable_xlpe: FDCable):
    """Test that ValueError is raised for inconsistent input lengths in _calculate_inter_rhos()."""
    radii = np.array([0.0, 0.01, 0.02])
    inter_radii = np.array([0.005, 0.015])
    rhos = np.array([1.0, 2.0, 3.0])

    with pytest.raises(
        ValueError,
        match=(
            "For the finite difference method, it is assumed that the "
            "resistivity between the first two grid points is constant. "
            "This assumption is violated."
        ),
    ):
        single_core_cable_xlpe._calculate_inter_rhos(radii, inter_radii, rhos)


def test_calculate_inner_rhos_non_increasing_radii(single_core_cable_xlpe: FDCable):
    """Test that ValueError is raised for non-increasing radii in _calculate_inter_rhos()."""
    radii = np.array([0.01, 0.02, 0.02])
    inter_radii = np.array([0.015, 0.025])
    rhos = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="The radii array must be strictly increasing!"):
        single_core_cable_xlpe._calculate_inter_rhos(radii, inter_radii, rhos)


def test_calculate_inner_rhos_negative_radii(single_core_cable_pilc: FDCable):
    """Test that ValueError is raised for non-increasing inter_radii in _calculate_inter_rhos()."""
    radii = np.array([-0.01, 0.02, 0.03])
    inter_radii = np.array([-0.015, 0.025])
    rhos = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="All radii must be non-negative!"):
        single_core_cable_pilc._calculate_inter_rhos(radii, inter_radii, rhos)


def test_calculate_inner_rho_alternating_values(single_core_cable_od: FDCable):
    """Test that _calculate_inter_rhos() handles alternating high and low resistivity values correctly."""
    radii = np.array([0.01, 0.02, 0.03, 0.04])
    inter_radii = np.array([0.005, 0.015, 0.035])
    rhos = np.array([1.0, 10.0, 1.0, 10.0])

    with pytest.raises(ValueError, match="All inter_radii values must be between the corresponding radii values!"):
        single_core_cable_od._calculate_inter_rhos(radii, inter_radii, rhos)


def test_get_cable_copy_with_added_soil_layer(three_core_cable_pilc: FDCable):
    """Test the get_cable_copy_with_added_soil_layer() method for adding one or multiple soil layers to a cable."""
    for layer in CableLayer.soil_layers():
        assert layer not in three_core_cable_pilc.layer_properties, (
            f"Soil layer '{layer}' already exists in original cable"
        )

    soil_radii = [0.5, 1.0, 5.0, 10.0]
    soil_rhos = [0.23, 2.5, 0.5, 0.5]
    soil_capacities = [2.4e6, 1e6, 3e6, 2e6]
    logarithmic_soil_gridpoint_density = 10

    fd_cables = [three_core_cable_pilc]
    for soil_radius, soil_rho, soil_capacity in zip(soil_radii, soil_rhos, soil_capacities, strict=True):
        new_cable = fd_cables[-1].get_cable_copy_with_added_soil_layer(
            soil_radius=soil_radius,
            soil_rho=soil_rho,
            soil_capacity=soil_capacity,
            logarithmic_soil_gridpoint_density=logarithmic_soil_gridpoint_density,
        )
        for layer in CableLayer.soil_layers()[: len(fd_cables)]:
            assert layer in new_cable.layer_properties, f"Soil layer '{layer}' not found in new cable after addition"

        for layer in CableLayer.soil_layers()[len(fd_cables) :]:
            assert layer not in new_cable.layer_properties, (
                f"Soil layer '{layer}' unexpectedly found in new cable after addition"
            )

        fd_cables.append(new_cable)

    for idx in range(1, len(fd_cables)):
        assert fd_cables[idx].layer_properties[CableLayer.soil_layers()[idx - 1]].outer_radius == soil_radii[idx - 1], (
            f"Outer radius of soil layer '{CableLayer.soil_layers()[idx - 1]}' "
            "does not match expected value in cable copy"
        )

    # Test that adding a soil layer does not modify the original cable
    assert CableLayer.SoilOne not in fd_cables[0].layer_properties

    # Test that the rho and capacity grids of the new cable are consistent with the original cable in the inner layers
    for idx in range(len(fd_cables)):
        rho_grid = fd_cables[idx].rho_grid
        capacity_grid = fd_cables[idx].capacity_grid
        for jdx in range(idx, len(fd_cables)):
            assert np.isclose(rho_grid, fd_cables[jdx].rho_grid[: len(rho_grid)]).all(), (
                f"Rho grid of cable copy with {idx} soil layers does not "
                f"match expected values in cable copy with {jdx} soil layers"
            )
            assert np.isclose(capacity_grid, fd_cables[jdx].capacity_grid[: len(capacity_grid)]).all(), (
                f"Capacity grid of cable copy with {idx} soil layers does "
                f"not match expected values in cable copy with {jdx} soil layers"
            )

    # Test that adding another soil layer raises a ValueError
    with pytest.raises(
        ValueError,
        match=(
            "The current cable already has the maximum amount of soil layers! "
            "This method cannot be used to add more soil layers!"
        ),
    ):
        fd_cables[-1].get_cable_copy_with_added_soil_layer(
            soil_radius=20.0,
            soil_rho=0.5,
            soil_capacity=2e6,
            logarithmic_soil_gridpoint_density=logarithmic_soil_gridpoint_density,
        )

    # Test that adding a soil layer with a radius smaller than the current outer radius raises a ValueError
    for soil_radius in [soil_radii[0], soil_radii[1]]:
        with pytest.raises(
            ValueError,
            match="The soil radius must be larger than the outer radius of the current outer layer!",
        ):
            fd_cables[2].get_cable_copy_with_added_soil_layer(
                soil_radius=soil_radius,
                soil_rho=0.5,
                soil_capacity=2e6,
                logarithmic_soil_gridpoint_density=logarithmic_soil_gridpoint_density,
            )


# TODO in refactor:

# Test cable redefine

# NOTE: Most of the above are currently picked up via other test. There should be a move towards having other tests
#       incidentally testing everything of an unrelated module.
