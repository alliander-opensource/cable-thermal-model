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
     initialized in the [finalize_init()] method anyway, we should probably not pre-initialize these as None. If
     initialization in __init__() is needed or desired regardless, then initialize by type as well, rather than only by
     value. (for example: [[-- radii_grid = None --]] becomes [[-- radii_grid: list[float] | None = None --]])
     This should solve a lot of the issues showing later in the file in the form of ".sum()" getting a warning because
     the evaluation it's "sum"-ing is considered boolean in nature.

    FDCable's get_redefined_cable() method uses a [cable_] variable. This should be replaced with a more Pythonesque
     name for clarity.

    FDCable's get_linear_system() method is way too long and complex. Split it into pieces.

"""

from unittest.mock import MagicMock

import numpy as np
import pytest

import cable_thermal_model.model.cables.cable as fd_cable_module
from cable_thermal_model.cable.cable_builder import CableBuilder
from cable_thermal_model.cable.schemas.pipe_schemas import PipeInputSchema
from cable_thermal_model.environment.static_env_soil import StaticEnvSoil
from cable_thermal_model.model.cables.cable import Cable, CableSoil
from cable_thermal_model.model.cables.enum_classes_cable import CableLayer, CableScreenLossType, PipeFillType
from cable_thermal_model.model.cables.pipe import Pipe
from cable_thermal_model.validation.cable_analysis import CableAnalysis
from tests.conftest import test_cable_fixtures


@pytest.mark.parametrize("load", [250.0, 500.0, 1000.0])
def test_get_heat_generation_conductor_and_screen(three_core_cable_pilc: FDCable, load: float):
    # Set the screen loss function.
    three_core_cable_pilc.layer_metrics.screen_loss_type = CableScreenLossType.TwoSidedBondingLinearCenter
    no_load_heat_generation_conductor, no_load_heat_generation_screen = (
        three_core_cable_pilc.get_heat_generation_conductor_and_screen(
            load=0.0,
            conductor_temperature=50.0,
            screen_temperature=40.0,
            temperature_dependent_electric_resistance=True,
            ac_current=True,
        )
    )

    assert np.isclose(no_load_heat_generation_conductor, 0.0)
    assert np.isclose(no_load_heat_generation_screen, 0.0)

    # Check that more heat is generated when incorporating AC effects.
    ac_heat_generation_conductor, ac_heat_generation_screen = (
        three_core_cable_pilc.get_heat_generation_conductor_and_screen(
            load=load,
            conductor_temperature=50.0,
            screen_temperature=40.0,
            temperature_dependent_electric_resistance=True,
            ac_current=True,
        )
    )

    dc_heat_generation_conductor, dc_heat_generation_screen = (
        three_core_cable_pilc.get_heat_generation_conductor_and_screen(
            load=load,
            conductor_temperature=50.0,
            screen_temperature=40.0,
            temperature_dependent_electric_resistance=True,
            ac_current=False,
        )
    )

    # Check that AC heat generation is higher than DC heat generation.
    assert ac_heat_generation_conductor > dc_heat_generation_conductor

    # Check that the heat generated in the screen is strictly positive in the AC case.
    assert ac_heat_generation_screen > 0.0

    # Check that no heat is generated in the screen in the DC case, where we set current_in_screen=False.
    assert np.isclose(dc_heat_generation_screen, 0.0)


def test_get_finite_difference_vector_for_state(three_core_cable_pilc: FDCable):
    three_core_cable_pilc.layer_metrics.screen_loss_type = CableScreenLossType.TwoSidedBondingLinearCenter

    conductor_temperature = 50.0
    screen_temperature = 40.0
    load = 500.0

    temperature_grid = np.full(three_core_cable_pilc.grid_size, 25.0)
    conductor_start_index, conductor_end_index = three_core_cable_pilc.get_layer_indices_for_layer(CableLayer.Conductor)
    screen_start_index, screen_end_index = three_core_cable_pilc.get_layer_indices_for_layer(CableLayer.Screen)
    temperature_grid[conductor_start_index : conductor_end_index + 1] = conductor_temperature
    temperature_grid[screen_start_index : screen_end_index + 1] = screen_temperature

    vector = three_core_cable_pilc.update_finite_difference_vector(
        vector=three_core_cable_pilc.get_finite_difference_vector(neglect_dielectric_loss=False),
        temperature_grid=temperature_grid,
        load=load,
        ac_current=True,
        temperature_dependent_electric_resistance=True,
    )

    expected_vector = three_core_cable_pilc.get_finite_difference_vector(neglect_dielectric_loss=False)
    heat_generation_conductor, heat_generation_screen = three_core_cable_pilc.get_heat_generation_conductor_and_screen(
        load=load,
        conductor_temperature=conductor_temperature,
        screen_temperature=screen_temperature,
        temperature_dependent_electric_resistance=True,
        ac_current=True,
    )
    expected_vector = three_core_cable_pilc._update_vector_with_heat_generation_for_layer(
        vector=expected_vector,
        heat_generation=heat_generation_screen,
        layer=CableLayer.Screen,
    )
    expected_vector = three_core_cable_pilc._update_vector_with_heat_generation_for_layer(
        vector=expected_vector,
        heat_generation=heat_generation_conductor,
        layer=CableLayer.Conductor,
    )

    assert np.allclose(vector, expected_vector)


@pytest.mark.parametrize(
    "fd_cable_fixture",
    test_cable_fixtures,
)
def test_fd_cable_get_layer_indices_radii(fd_cable_fixture: str, request):
    cable: Cable = request.getfixturevalue(fd_cable_fixture)

    # Test conductor radii are compatible with computed indices for radii-grid
    conductor_start_index = (cable._radii_grid < cable.layer_properties[CableLayer.Conductor].inner_radius).sum()
    conductor_end_index = (cable._radii_grid < cable.layer_properties[CableLayer.Conductor].outer_radius).sum() - 1
    s, e = cable.get_layer_indices_for_layer(CableLayer.Conductor)
    assert conductor_start_index == s
    assert conductor_end_index == e
    # Test screen radii are compatible with computed indices for radii-grid
    screen_start_index = (cable._radii_grid < cable.layer_properties[CableLayer.Screen].inner_radius).sum()
    screen_end_index = (cable._radii_grid < cable.layer_properties[CableLayer.Screen].outer_radius).sum() - 1
    s, e = cable.get_layer_indices_for_layer(CableLayer.Screen)
    assert screen_start_index == s
    assert screen_end_index == e


@pytest.mark.parametrize(
    "fd_cable_fixture",
    test_cable_fixtures,
)
def test_fd_cable_get_layer_indices_material_properties(fd_cable_fixture: str, request):
    cable: Cable = request.getfixturevalue(fd_cable_fixture)

    s, e = cable.get_layer_indices_for_layer(CableLayer.Conductor)
    # Test thermal resistance conductor is constant along slice from start to end indices
    assert np.isclose(np.abs(np.diff(cable._rho_grid[s : e + 1])).sum(), 0, atol=1e-3)
    s, e = cable.get_layer_indices_for_layer(CableLayer.Screen)
    # Test thermal resistance screen is constant along slice from start to end indices
    assert np.isclose(np.abs(np.diff(cable._rho_grid[s : e + 1])).sum(), 0, atol=1e-3)


def test_cable_radius(single_core_cable_xlpe: CableSoil, single_circuit_with_pipe_env: StaticEnvSoil):
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
    cable: Cable = request.getfixturevalue(fd_cable_fixture)
    # Test that the radii grid starts at 0
    assert np.isclose(cable._radii_grid[0], 0)
    # Test that the radii grid ends at the outer radius of the outermost layer
    outermost_layer = max(cable.layer_properties.keys(), key=lambda layer: cable.layer_properties[layer].outer_radius)
    assert np.isclose(cable._radii_grid[-1], cable.layer_properties[outermost_layer].outer_radius)
    # Test that the radii grid has correct number of points
    expected_points = sum(cable._grid_counts.values())
    assert len(cable._radii_grid) == expected_points
    # Test that the radii grid is strictly increasing
    assert np.all(np.diff(cable._radii_grid) > 0)
    # Test that each layer boundary is exactly between two grid points
    cumulative_points = 0
    for layer in cable.layer_properties:
        layer_points = cable._grid_counts[layer]
        cumulative_points += layer_points
        if layer != outermost_layer:
            boundary_radius = cable.layer_properties[layer].outer_radius
            assert np.isclose(
                (cable._radii_grid[cumulative_points - 1] + cable._radii_grid[cumulative_points]) / 2, boundary_radius
            )


def test_get_heat_flow_value_error(single_core_cable_xlpe: CableSoil):
    """Test that ValueError is raised when invalid index is provided to get_heat_flow()."""
    analysis = CableAnalysis(cable=single_core_cable_xlpe, solution=MagicMock())

    with pytest.raises(
        ValueError,
        match="The indices must lie within the finite difference grid of the cable.",
    ):
        analysis.get_heat_flow(inner_index=-1)

    with pytest.raises(
        ValueError,
        match="The indices must lie within the finite difference grid of the cable.",
    ):
        analysis.get_heat_flow(inner_index=100)


def test_calculate_inner_rhos(single_core_cable_xlpe: CableSoil):
    """Test the _calculate_inter_rhos() method for calculating interstitial resistivity."""
    radii = np.array([0.01, 0.02, 0.03])
    inter_radii = np.array([0.015, 0.025])
    rhos = np.array([1.0, 2.0, 3.0])

    inter_rhos = single_core_cable_xlpe._calculate_inter_rhos(radii, inter_radii, rhos)
    assert rhos[0] < inter_rhos[0] < rhos[1], "First interstitial resistivity not between adjacent rhos"
    assert rhos[1] < inter_rhos[1] < rhos[2], "Second interstitial resistivity not between adjacent rhos"

    constant_rhos = single_core_cable_xlpe._calculate_inter_rhos(radii, inter_radii, np.array([2.0, 2.0, 2.0]))
    assert np.allclose(constant_rhos, 2.0)


def test_calculate_inner_rhos_inconsistent_length(single_core_cable_xlpe: CableSoil):
    """Test that ValueError is raised for inconsistent input lengths in _calculate_inter_rhos()."""
    radii = np.array([0.01, 0.02])
    inter_radii = np.array([0.015, 0.025])
    rhos = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="The lengths of the input arrays are inconsistent!"):
        single_core_cable_xlpe._calculate_inter_rhos(radii, inter_radii, rhos)


def test_calculate_inner_rhos_constant_violation_near_zero(single_core_cable_xlpe: CableSoil):
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


def test_calculate_inner_rhos_non_increasing_radii(single_core_cable_xlpe: CableSoil):
    """Test that ValueError is raised for non-increasing radii in _calculate_inter_rhos()."""
    radii = np.array([0.01, 0.02, 0.02])
    inter_radii = np.array([0.015, 0.025])
    rhos = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="The radii array must be strictly increasing!"):
        single_core_cable_xlpe._calculate_inter_rhos(radii, inter_radii, rhos)


def test_calculate_inner_rhos_negative_radii(single_core_cable_pilc: CableSoil):
    """Test that ValueError is raised for non-increasing inter_radii in _calculate_inter_rhos()."""
    radii = np.array([-0.01, 0.02, 0.03])
    inter_radii = np.array([-0.015, 0.025])
    rhos = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="All radii must be non-negative!"):
        single_core_cable_pilc._calculate_inter_rhos(radii, inter_radii, rhos)


def test_calculate_inner_rho_alternating_values(single_core_cable_od: CableSoil):
    """Test that _calculate_inter_rhos() handles alternating high and low resistivity values correctly."""
    radii = np.array([0.01, 0.02, 0.03, 0.04])
    inter_radii = np.array([0.005, 0.015, 0.035])
    rhos = np.array([1.0, 10.0, 1.0, 10.0])

    with pytest.raises(ValueError, match="All inter_radii values must be between the corresponding radii values!"):
        single_core_cable_od._calculate_inter_rhos(radii, inter_radii, rhos)


def test_get_outer_boundary_coupling_coefficient_from_matrix(single_core_cable_xlpe: CableSoil):
    """Test that the matrix-based outer-boundary coupling matches the final upper diagonal term."""
    upper_diagonal, _, _ = single_core_cable_xlpe._get_finite_difference_matrix_diagonals()

    outer_boundary_coupling_coefficient = single_core_cable_xlpe.outer_boundary_coupling_coefficient

    assert np.isclose(outer_boundary_coupling_coefficient, upper_diagonal[-1])


def test_get_cable_copy_with_added_soil_layer(three_core_cable_pilc: CableSoil):
    """Test the from_cable_with_added_soil_layer() method for adding one or multiple soil layers to a cable."""
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
        new_cable = CableSoil.from_cable_with_added_soil_layer(
            cable=fd_cables[-1],
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
        rho_grid = fd_cables[idx]._rho_grid
        capacity_grid = fd_cables[idx]._capacity_grid
        for jdx in range(idx, len(fd_cables)):
            assert np.isclose(rho_grid, fd_cables[jdx]._rho_grid[: len(rho_grid)]).all(), (
                f"Rho grid of cable copy with {idx} soil layers does not "
                f"match expected values in cable copy with {jdx} soil layers"
            )
            assert np.isclose(capacity_grid, fd_cables[jdx]._capacity_grid[: len(capacity_grid)]).all(), (
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
        CableSoil.from_cable_with_added_soil_layer(
            cable=fd_cables[-1],
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
            CableSoil.from_cable_with_added_soil_layer(
                cable=fd_cables[2],
                soil_radius=soil_radius,
                soil_rho=0.5,
                soil_capacity=2e6,
                logarithmic_soil_gridpoint_density=logarithmic_soil_gridpoint_density,
            )


def test_fd_cable_init_invalid_grid_counts_type(single_core_cable_xlpe: CableSoil):
    with pytest.raises(TypeError, match="The grid_counts argument must be a dictionary of integers!"):
        CableSoil(
            conductor=single_core_cable_xlpe.conductor,
            layer_properties=single_core_cable_xlpe.layer_properties,
            layer_metrics=single_core_cable_xlpe.layer_metrics,
            cable_type=single_core_cable_xlpe.cable_type,
            grid_counts=[1, 2, 3],  # type: ignore[arg-type]
        )


def test_update_pipe_fill_resistivity_without_pipe_raises(single_core_cable_xlpe: CableSoil):
    with pytest.raises(ValueError, match="Pipe is not set. Cannot update pipe fill resistivity."):
        single_core_cable_xlpe.update_pipe_fill_resistivity(np.zeros(single_core_cable_xlpe.grid_size))


def test_update_pipe_fill_resistivity_without_inner_radius_raises(single_core_cable_xlpe: CableSoil):
    pipe = Pipe(
        pipe_input=PipeInputSchema(fill_type=PipeFillType.Air, inner_radius=0.06),
        outer_radius_cable=single_core_cable_xlpe.layer_metrics.outer_radius,
    )
    cable_with_pipe = single_core_cable_xlpe.get_cable_copy_with_pipe(pipe=pipe)
    assert cable_with_pipe.layer_metrics.pipe is not None
    cable_with_pipe.layer_metrics.pipe.inner_radius = None

    with pytest.raises(ValueError, match="Pipe inner radius is not set. Cannot update pipe fill resistivity."):
        cable_with_pipe.update_pipe_fill_resistivity(np.zeros(cable_with_pipe.grid_size))


def test_update_rho_grid(single_core_cable_xlpe: CableSoil):
    with pytest.raises(ValueError, match="The start_index exceeds the end_index. Cannot update the rho grid."):
        single_core_cable_xlpe._update_rho_grid(start_index=2, end_index=1, rho=1.0)

    # Ensure the cached diagonals are marked up-to-date before testing invalidation behavior.
    _ = single_core_cable_xlpe._banded_matrix
    assert not single_core_cable_xlpe._finite_difference_matrix_diagonals_outdated

    start_index, end_index = single_core_cable_xlpe.get_layer_indices_for_layer(CableLayer.Conductor)
    old_values = single_core_cable_xlpe._rho_grid.copy()
    rho = float(single_core_cable_xlpe._rho_grid[start_index])

    single_core_cable_xlpe._update_rho_grid(start_index=start_index, end_index=end_index, rho=rho * 1.005)
    assert np.array_equal(single_core_cable_xlpe._rho_grid, old_values)
    assert not single_core_cable_xlpe._finite_difference_matrix_diagonals_outdated

    single_core_cable_xlpe._update_rho_grid(start_index=start_index, end_index=end_index, rho=rho * 1.2)
    assert np.allclose(single_core_cable_xlpe._rho_grid[start_index : end_index + 1], rho * 1.2)
    assert single_core_cable_xlpe._finite_difference_matrix_diagonals_outdated


def test_update_capacity_grid(single_core_cable_xlpe: CableSoil):
    with pytest.raises(ValueError, match="The start_index exceeds the end_index. Cannot update the capacity grid."):
        single_core_cable_xlpe._update_capacity_grid(start_index=5, end_index=3, capacity=2.0e6)

    start_index, end_index = single_core_cable_xlpe.get_layer_indices_for_layer(CableLayer.Conductor)
    old_values = single_core_cable_xlpe._capacity_grid.copy()
    capacity = float(single_core_cable_xlpe._capacity_grid[start_index])

    single_core_cable_xlpe._update_capacity_grid(
        start_index=start_index, end_index=end_index, capacity=capacity * 1.005
    )
    assert np.array_equal(single_core_cable_xlpe._capacity_grid, old_values)

    single_core_cable_xlpe._update_capacity_grid(start_index=start_index, end_index=end_index, capacity=capacity * 1.2)
    assert np.allclose(single_core_cable_xlpe._capacity_grid[start_index : end_index + 1], capacity * 1.2)


def test_get_dry_soil_radius(three_core_cable_pilc: CableSoil):
    cable_with_soil = three_core_cable_pilc.get_cable_copy_with_added_soil_layer(
        soil_rho=0.9,
        soil_capacity=2.0e6,
        soil_radius=0.7,
        logarithmic_soil_gridpoint_density=8,
    )
    temperature_grid = np.full(cable_with_soil.grid_size, 20.0)

    assert cable_with_soil._get_dry_soil_radius(temperature_grid=temperature_grid, soil_drying=False) is None
    assert cable_with_soil._get_dry_soil_radius(temperature_grid=temperature_grid, soil_drying=True) is None

    temperature_grid[-1] = cable_with_soil._SOIL_DRYING_TEMPERATURE + 5
    dry_radius = cable_with_soil._get_dry_soil_radius(temperature_grid=temperature_grid, soil_drying=True)
    assert np.isclose(dry_radius, cable_with_soil._radii_grid[-1])


def test_update_soil_resistivity_with_dry_zone(three_core_cable_pilc: CableSoil):
    cable_with_soil = three_core_cable_pilc.get_cable_copy_with_added_soil_layer(
        soil_rho=0.7,
        soil_capacity=2.0e6,
        soil_radius=0.8,
        logarithmic_soil_gridpoint_density=8,
    )
    start_index = cable_with_soil._get_soil_grid_start_index()
    dry_soil_radius = (
        three_core_cable_pilc.layer_metrics.outer_radius
        + (
            cable_with_soil.layer_properties[CableLayer.SoilOne].outer_radius
            - three_core_cable_pilc.layer_metrics.outer_radius
        )
        / 2
    )

    cable_with_soil._update_soil_resistivity(soil_rho=0.9, dry_soil_radius=dry_soil_radius)
    end_index = (cable_with_soil._radii_grid <= dry_soil_radius).sum() - 1

    assert np.allclose(cable_with_soil._rho_grid[start_index : end_index + 1], 2.5)
    assert np.allclose(cable_with_soil._rho_grid[end_index + 1 :], 0.9)


def test_update_soil_resistivity_without_dry_zone(three_core_cable_pilc: CableSoil):
    cable_with_soil = three_core_cable_pilc.get_cable_copy_with_added_soil_layer(
        soil_rho=0.7,
        soil_capacity=2.0e6,
        soil_radius=0.8,
        logarithmic_soil_gridpoint_density=8,
    )
    start_index = cable_with_soil._get_soil_grid_start_index()
    dry_soil_radius = three_core_cable_pilc.layer_metrics.outer_radius

    cable_with_soil._update_soil_resistivity(soil_rho=0.9, dry_soil_radius=dry_soil_radius)

    assert np.allclose(cable_with_soil._rho_grid[start_index:], 0.9)


def test_update_soil_capacity_invalid_type_raises(three_core_cable_pilc: CableSoil):
    cable_with_soil = three_core_cable_pilc.get_cable_copy_with_added_soil_layer(
        soil_rho=0.9,
        soil_capacity=2.0e6,
        soil_radius=0.7,
        logarithmic_soil_gridpoint_density=8,
    )
    with pytest.raises(ValueError, match="The soil_c argument must be of type int or float!"):
        cable_with_soil._update_soil_capacity(soil_c="invalid")  # type: ignore[arg-type]


def test_get_cable_copy_with_pipe_error(single_core_cable_xlpe: CableSoil):
    pipe = Pipe(
        pipe_input=PipeInputSchema(fill_type=PipeFillType.Air, inner_radius=0.06),
        outer_radius_cable=single_core_cable_xlpe.layer_metrics.outer_radius,
    )

    cable_with_soil = single_core_cable_xlpe.get_cable_copy_with_added_soil_layer(
        soil_rho=1.0,
        soil_capacity=2.0e6,
        soil_radius=0.8,
        logarithmic_soil_gridpoint_density=8,
    )
    with pytest.raises(
        ValueError,
        match="Detected soil layers. The add_outer_pipe method is only intended for cable instances without soil.",
    ):
        cable_with_soil.get_cable_copy_with_pipe(pipe=pipe)

    cable_with_pipe = single_core_cable_xlpe.get_cable_copy_with_pipe(pipe=pipe)
    with pytest.raises(ValueError, match="Cannot add a pipe as the cable already has a pipe."):
        cable_with_pipe.get_cable_copy_with_pipe(pipe=pipe)


def test_get_mean_temperature_cable_layer_missing_layer_raises(single_core_cable_xlpe: CableSoil):
    with pytest.raises(ValueError, match="Layer Pipe is not present in the cable."):
        single_core_cable_xlpe._get_mean_temperature_cable_layer(
            temperature_grid=np.zeros(single_core_cable_xlpe.grid_size),
            layer=CableLayer.Pipe,
        )


def test_fd_cable_in_air_integrate_timestep_non_convergence_raises(
    single_core_cable_xlpe_in_air,
    monkeypatch,
):
    single_core_cable_xlpe_in_air.set_convection_parameters(Z=0.91, E=0.0, Cg=0.0)
    n = single_core_cable_xlpe_in_air.grid_size
    s = np.zeros(n)
    b = np.zeros(n - 1)

    call_count = {"n": 0}

    def _non_converging_solver(*args, **kwargs):
        call_count["n"] += 1
        out = np.zeros(n)
        out[-1] = float(call_count["n"])
        return out

    monkeypatch.setattr(fd_cable_module, "_MAX_ITERATIONS_PER_TIMESTEP", 3)
    monkeypatch.setattr(fd_cable_module.linalg, "solve_banded", _non_converging_solver)

    with pytest.raises(ValueError, match="Solution did not converge after 3 iterations"):
        single_core_cable_xlpe_in_air.integrate_timestep(s=s, b=b, time_step=1.0, internal_heating=True)


def test_trefoil_in_single_pipe_in_air_requires_convection_parameters():
    cable = CableBuilder.build_cable_from_cable_id(
        cable_id="YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
        fd_cable_class=FDCableTrefoilCircuitInSinglePipeInAir,
        pipe=PipeInputSchema(inner_radius=0.1, fill_type=PipeFillType.Air, trefoil_circuit_in_single_pipe=True),
    )
    s = np.zeros(cable.grid_size)
    b = np.zeros(cable.grid_size - 1)

    with pytest.raises(ValueError, match="Convection parameters have not been set for this cable in air!"):
        cable.integrate_timestep(s=s, b=b, time_step=1.0, internal_heating=True)


# TODO in refactor:

# Test cable redefine

# NOTE: Most of the above are currently picked up via other test. There should be a move towards having other tests
#       incidentally testing everything of an unrelated module.
