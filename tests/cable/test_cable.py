# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cable_thermal_model.cable.cable_builder import CableBuilder
from cable_thermal_model.cable.schemas.pipe_schemas import PipeInputSchema
from cable_thermal_model.model.cables.abstract_cable import CableConductorProperties, CableLayerMetrics
from cable_thermal_model.model.cables.cable import (
    Cable,
    CableAir,
    CableSoil,
    CableTrefoilCircuitSinglePipeInAir,
    CableTrefoilCircuitSinglePipeInSoil,
)
from cable_thermal_model.model.cables.enum_classes_cable import (
    CableConductorCount,
    CableConductorMaterial,
    CableConductorShape,
    CableConductorSurfaceType,
    CableInsulationMaterial,
    CableLayer,
    CableType,
    PipeFillType,
)
from cable_thermal_model.model.cables.pipe import Pipe
from tests.conftest import mock_load_cable_data_from_file

""" Testing Notes.

TO TEST:
    Building a cable always follows the following two steps:

        1. Build the cable
        2. Add a pipe if requested

    The first step can be subdivided as follows:

        - Generate a CableSpecParser object based on the number of conductors
          (cable_spec_parsers.py, so not tested here)

        - (FD WITH THREE CONDUCTORS ONLY:) Exchange the three cores with one equivalent core (tested here)

        - Collect material properties (tested here)
        - Collect cable properties (tested here)

        - (FD WITH THREE CONDUCTORS ONLY:) Replace the radii to match the generated equivalent core (tested here)

        - Generate a FDCable object depending on the request. (tested here)

NOTES:
    It is a bit unclear as to why the CableBuilder is a class. It is most likely due to the high dependency on data
     files that the class loads, but an official description would help explain why this was chosen over a separate
     [build_cable()] method.

    The [_build_fd_cable()] method may be refactored in the future. Re-evaluation of how cables are constructed may
    lead to cleaner code with less duplication. It may also lead to less clear code, so re-evaluation is key.

    A lot of dependencies of the created cable are clearly shown as having inconsistent naming of scientific parameters.
     Is [cable_spec_parser.cable_specs["t1"]] a temperature or a time? If a temperature, why isn't a capital "T" used?
     At other locations we find for example [cable_spec_parser.cable_specs["A_cond"]], which clearly

"""

# CONSTANTS
_RADIAL_APPROXIMATION_RESOLUTION_IN_METERS = 1e-9

LOSS_LAYERS = [CableLayer.Conductor, CableLayer.Insulation, CableLayer.Screen]


@pytest.mark.parametrize(
    (
        "cable_id, expected_layers, expected_radii, expected_rhos, "
        "expected_electric_rhos, expected_capacities, expected_alphas, "
        "expected_epsilon, expected_tan_delta"
    ),
    [
        (
            "YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
            [
                CableLayer.Conductor,
                CableLayer.ConductorScreen,
                CableLayer.Insulation,
                CableLayer.InsulationScreen,
                CableLayer.Screen,
                CableLayer.Sheath,
            ],
            np.array([0.01395, 0.0148, 0.02030, 0.02115, 0.0223, 0.0255]),
            np.array([0.004219409, 3.5, 3.5, 3.5, 0.002624672, 5.0, 3.5]),
            np.array([2.8261e-08, 0.0, 0.0, 0.0, 1.7241e-08, 0.0, 0.0]),
            np.array([2500000.0, 2400000.0, 2400000.0, 2400000.0, 3450000.0, 2000000.0, 2400000.0]),
            np.array([0.00403, 0.0, 0.0, 0.0, 0.00393, 0.0, 0.0]),
            np.array([np.nan, np.nan, 2.5, np.nan, np.nan, 4.0, 2.3]),
            np.array([np.nan, np.nan, 0.001, np.nan, np.nan, 0.05, 0.001]),
        ),
        (
            "GPLK 10/10 kV 3x185 Al",
            [
                CableLayer.Conductor,
                CableLayer.Insulation,
                CableLayer.Screen,
                CableLayer.Bedding,
                CableLayer.Armour,
                CableLayer.Sheath,
            ],
            np.array([0.02056318, 0.0259, 0.0285, 0.03075, 0.033, 0.036]),
            np.array([0.004219409, 6.0, 29.0e-03, 6.0, 13.3e-03, 6.0]),
            np.array([0.000000028261, 0, 21.4e-08, 0, 13.8e-08, 0]),
            np.array([2.50e06, 2.0e06, 1.45e06, 2.0e06, 3.8e06, 2.0e06]),
            np.array([4.03e-3, 0, 0.004, 0, 0.0045, 0]),
            np.array([np.nan, 4.0, np.nan, 4.0, np.nan, np.nan]),
            np.array([np.nan, 0.01, np.nan, 0.01, np.nan, np.nan]),
        ),
    ],
)
def test_cable_builder_generic_verify_valid_layer_properties(
    cable_id: str,
    expected_layers: list[CableLayer],
    expected_radii: np.ndarray,
    expected_rhos: np.ndarray,
    expected_electric_rhos: np.ndarray,
    expected_capacities: np.ndarray,
    expected_alphas: np.ndarray,
    expected_epsilon: np.ndarray,
    expected_tan_delta: np.ndarray,
):
    # Generation
    with patch(
        "cable_thermal_model.cable.cable_builder.CableBuilder._load_cable_data_from_file",
        side_effect=mock_load_cable_data_from_file,
    ):
        constructed_cable = CableBuilder.build_cable_from_cable_id(cable_id=cable_id, cable_class=CableSoil)
    layer_properties = constructed_cable.layer_properties

    # Evaluation
    assert np.array_equal(
        [layer_properties[layer].rho for layer in layer_properties],
        expected_rhos,
        equal_nan=True,
    )
    assert np.array_equal(
        [layer_properties[layer].electric_rho for layer in layer_properties],
        expected_electric_rhos,
        equal_nan=True,
    )
    assert np.array_equal(
        [layer_properties[layer].alpha for layer in layer_properties],
        expected_alphas,
        equal_nan=True,
    )
    assert np.array_equal(
        [layer_properties[layer].epsilon for layer in layer_properties],
        expected_epsilon,
        equal_nan=True,
    )
    assert np.array_equal(
        [layer_properties[layer].tan_delta for layer in layer_properties],
        expected_tan_delta,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "cable_id, expected_conductor_properties, expected_layer_metrics, expected_cable_type",
    [
        (
            "YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
            CableConductorProperties(
                number_of_conductors=CableConductorCount.One,
                shape=CableConductorShape.Round,
                material=CableConductorMaterial.Aluminium,
                surface_type=CableConductorSurfaceType.Solid,
            ),
            CableLayerMetrics(
                conductor_cross_section=0.000_63,
                conductor_virtual_cross_section=0.000_602_579_957_356_0768,
                screen_cross_section=5e-05,
                conductor_distance=None,
                armour_cross_section=None,
                diameter_over_stranded_conductors=None,
                conductor_radius_original=0.01395,
                conductor_equivalent_outer_diameter=0.0279,
                nominal_phase_voltage=12000,
                outer_radius=0.0255,
                cable_radius=0.0255,
                sector_radius=None,
                core_to_sector_distance=None,
                insulation_material=CableInsulationMaterial.XLPEUnfilled,
            ),
            CableType.XLPE,
        ),
        (
            "GPLK 10/10 kV 3x185 Al",
            CableConductorProperties(
                number_of_conductors=CableConductorCount.Three,
                shape=CableConductorShape.Sector,
                material=CableConductorMaterial.Aluminium,
                surface_type=CableConductorSurfaceType.Stranded,
            ),
            CableLayerMetrics(
                conductor_cross_section=0.000185,
                conductor_virtual_cross_section=0.0001723231707317073,
                screen_cross_section=0.000373975189,
                conductor_distance=0.0038,
                armour_cross_section=0.00023,
                diameter_over_stranded_conductors=0.041,
                conductor_radius_original=0.007673807981960538,
                conductor_equivalent_outer_diameter=0.036343405256116436,
                outer_radius=0.0335,
                cable_radius=0.0335,
                nominal_phase_voltage=10000,
                sector_radius=0.013291425312283247,
                core_to_sector_distance=0.0053085746877167515,
                insulation_material=CableInsulationMaterial.PaperMassImpregnated,
            ),
            CableType.PILC,
        ),
    ],
)
def test_cable_builder_generic_verify_valid_layer_metrics_and_conductor(
    cable_id: str,
    expected_conductor_properties: CableConductorProperties,
    expected_layer_metrics: CableLayerMetrics,
    expected_cable_type: CableType,
):
    # Generation
    with patch(
        "cable_thermal_model.cable.cable_builder.CableBuilder._load_cable_data_from_file",
        side_effect=mock_load_cable_data_from_file,
    ):
        constructed_cable = CableBuilder.build_cable_from_cable_id(cable_id=cable_id, cable_class=CableSoil)

    # Evaluation
    # TODO: Currently there is no direct comparison possible between layer_metrics attributes. Ideally the cable
    #        properties are to be refactored as a Dataclass with some comparison component built in.

    assert constructed_cable.conductor.number_of_conductors == expected_conductor_properties.number_of_conductors
    assert constructed_cable.cable_type == expected_cable_type

    # Conductor properties
    assert (
        pytest.approx(  # type: ignore[no-untyped-call]
            constructed_cable.layer_metrics.conductor_virtual_cross_section,
            _RADIAL_APPROXIMATION_RESOLUTION_IN_METERS,
        )
        == expected_layer_metrics.conductor_virtual_cross_section
    )
    assert np.isclose(
        constructed_cable.layer_metrics.conductor_cross_section,
        expected_layer_metrics.conductor_cross_section,
        atol=1e-9,
    )
    assert (
        constructed_cable.conductor.shape == expected_conductor_properties.shape
        and constructed_cable.conductor.surface_type == expected_conductor_properties.surface_type
        and constructed_cable.conductor.material == expected_conductor_properties.material
    )
    assert (
        pytest.approx(  # type: ignore[no-untyped-call]
            constructed_cable.layer_metrics.conductor_radius_original,
            _RADIAL_APPROXIMATION_RESOLUTION_IN_METERS,
        )
        == expected_layer_metrics.conductor_radius_original
    )
    assert (
        pytest.approx(  # type: ignore[no-untyped-call]
            constructed_cable.layer_metrics.conductor_distance,
            _RADIAL_APPROXIMATION_RESOLUTION_IN_METERS,
        )
        == expected_layer_metrics.conductor_distance
    )

    # ...
    if CableLayer.Armour in constructed_cable.layer_properties:
        assert constructed_cable.layer_metrics.armour_cross_section is not None
        assert (
            pytest.approx(constructed_cable.layer_metrics.armour_cross_section)  # type: ignore[no-untyped-call]
            == expected_layer_metrics.armour_cross_section
        )
    else:
        assert expected_layer_metrics.armour_cross_section is None
        assert constructed_cable.layer_metrics.armour_cross_section is None

    assert (
        pytest.approx(  # type: ignore[no-untyped-call]
            constructed_cable.layer_metrics.core_to_sector_distance,
            _RADIAL_APPROXIMATION_RESOLUTION_IN_METERS,
        )
        == expected_layer_metrics.core_to_sector_distance
    )
    assert (
        pytest.approx(  # type: ignore[no-untyped-call]
            constructed_cable.layer_metrics.conductor_equivalent_outer_diameter,
            _RADIAL_APPROXIMATION_RESOLUTION_IN_METERS,
        )
        == expected_layer_metrics.conductor_equivalent_outer_diameter
    )
    assert constructed_cable.layer_metrics.screen_cross_section is not None
    assert expected_layer_metrics.screen_cross_section is not None
    assert (
        pytest.approx(constructed_cable.layer_metrics.screen_cross_section)  # type: ignore[no-untyped-call]
        == expected_layer_metrics.screen_cross_section
    )
    assert (
        pytest.approx(  # type: ignore[no-untyped-call]
            constructed_cable.layer_metrics.sector_radius,
            _RADIAL_APPROXIMATION_RESOLUTION_IN_METERS,
        )
        == expected_layer_metrics.sector_radius
    )
    assert (
        pytest.approx(  # type: ignore[no-untyped-call]
            constructed_cable.layer_metrics.diameter_over_stranded_conductors,
            _RADIAL_APPROXIMATION_RESOLUTION_IN_METERS,
        )
        == expected_layer_metrics.diameter_over_stranded_conductors
    )
    assert constructed_cable.layer_metrics.nominal_phase_voltage == expected_layer_metrics.nominal_phase_voltage


def test_build_build_cable_trefoil_in_single_pipe_without_pipe_raises_value_error():
    """Test that building a FDCableTrefoilCircuitInSinglePipe without providing a pipe raises a ValueError."""
    with (
        patch(
            "cable_thermal_model.cable.cable_builder.CableBuilder._load_cable_data_from_file",
            side_effect=mock_load_cable_data_from_file,
        ),
        pytest.raises(
            ValueError,
            match="When using Cable class 'CableTrefoilCircuitSinglePipeInSoil', a pipe must be provided.",
        ),
    ):
        CableBuilder.build_cable_from_cable_id(
            cable_id="YMeKrvaslqwd 12/20kV 3x240 Alrm + as50",
            cable_class=CableTrefoilCircuitSinglePipeInSoil,
        )


def test_equivalent_single_core_creation():
    ...
    # fd_cable_with_equivalent_single_core
    # Don't forget scSL screen type!!


@pytest.mark.parametrize(
    "cable_outer_diameter_mm,fill_type,temp,expected_T4",
    [
        (51.0, PipeFillType.Air, 15, 0.6506),
        (72.0, PipeFillType.Water, 60, 0.0607),
    ],
)
def test_T4_pipe_fill(cable_outer_diameter_mm, temp, fill_type, expected_T4):
    pipe = Pipe(
        pipe_input=PipeInputSchema(
            inner_radius=0.045,
            fill_type=fill_type,
        ),
        outer_radius_cable=cable_outer_diameter_mm / 2e3,
    )
    ctm_T4 = pipe._get_lump_sum_resistivity_pipe_fill(temp)
    assert np.isclose(ctm_T4, expected_T4, rtol=0.01)


def test_integrate_timestep_internal_heating_warning():
    """Test that integrating a timestep for Cable without internal_heating raises a warning."""
    cable_in_air = CableBuilder.build_cable_from_cable_id(
        cable_id="YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
        cable_class=CableAir,
    )
    with pytest.raises(
        ValueError,
        match="Internal heating must be True for cables in air.",
    ):
        cable_in_air.integrate_timestep(
            s=MagicMock(),
            b=MagicMock(),
            time_step=MagicMock(),
            internal_heating=None,
        )
    with pytest.raises(
        ValueError,
        match="Internal heating must be True for cables in air.",
    ):
        cable_in_air.integrate_timestep(
            s=MagicMock(),
            b=MagicMock(),
            time_step=MagicMock(),
            internal_heating=False,
        )

    # We expect the same behavior for a trefoil circuit in a single pipe in air
    fd_cable_trefoil_in_single_pipe_in_air = CableBuilder.build_cable_from_cable_id(
        cable_id="YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
        cable_class=CableTrefoilCircuitSinglePipeInAir,
        pipe=PipeInputSchema(inner_radius=0.1, fill_type=PipeFillType.Air, trefoil_circuit_in_single_pipe=True),
    )
    with pytest.raises(
        ValueError,
        match="Internal heating must be True for cables in air.",
    ):
        fd_cable_trefoil_in_single_pipe_in_air.integrate_timestep(
            s=MagicMock(),
            b=MagicMock(),
            time_step=MagicMock(),
            internal_heating=None,
        )
        fd_cable_trefoil_in_single_pipe_in_air.integrate_timestep(
            s=MagicMock(),
            b=MagicMock(),
            time_step=MagicMock(),
            internal_heating=False,
        )


def test_integrate_timestep_internal_heating_value_error():
    """Test internal_heating validation for a trefoil circuit in a single pipe.

    Verify that integrating a timestep for
    CableTrefoilCircuitSinglePipeInSoil without internal_heating raises a
    ValueError.
    """
    fd_cable_trefoil_in_single_pipe = CableBuilder.build_cable_from_cable_id(
        cable_id="YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
        cable_class=CableTrefoilCircuitSinglePipeInSoil,
        pipe=PipeInputSchema(inner_radius=0.1, fill_type=PipeFillType.Air, trefoil_circuit_in_single_pipe=True),
    )
    with pytest.raises(
        ValueError,
        match="The internal_heating parameter must be provided for CableTrefoilCircuitSinglePipeInSoil.",
    ):
        fd_cable_trefoil_in_single_pipe.integrate_timestep(
            s=MagicMock(), b=MagicMock(), time_step=MagicMock(), internal_heating=None
        )


@pytest.mark.parametrize(
    "radii_grid",
    [
        np.array([0.0, 0.01, 0.02, 0.03]),
        np.array([0.01, 0.02, 0.03]),
        np.array([0.0, 0.01, 0.03, 0.02]),
    ],
)
def test_construct_surface_area_grid(radii_grid):
    """Test the construction of the surface area grid for a given radii grid."""
    if not np.isclose(radii_grid[0], 0.0):
        with pytest.raises(ValueError, match="The first value of the radii grid should be 0.0!"):
            Cable._construct_surface_area_grid(radii_grid)
    elif not np.all(np.diff(radii_grid) > 0):
        with pytest.raises(ValueError, match="The radii grid should be strictly increasing!"):
            Cable._construct_surface_area_grid(radii_grid)
    else:
        surface_area_grid = Cable._construct_surface_area_grid(radii_grid)
        inter_r = np.concatenate(([0.0], (radii_grid[:-1] + radii_grid[1:]) / 2))
        expected_surface_area_grid = np.pi * (inter_r[1:] ** 2 - inter_r[:-1] ** 2)
        assert np.allclose(surface_area_grid, expected_surface_area_grid)


def test_equivalent_conductor_radius_calculation():
    ...

    # TODO: Add parameters for the following situations:
    #       - cable_spec_parser.cable_specs[_SHEATH_CABLE_TYPE_STRING] is "scSL"
    # STATUS: Awaiting bugfix


# TODO: Add for the evaluation of still missing coverage on refactor
def test_get_material_properties_has_materials_missing(): ...


def test_determine_pipe_size_nonstandard_pipe_sizes(): ...


def test_determine_pipe_size_erroneous_pipe_sizes(): ...


def test_compute_conductor_distance_in_three_core_cables_with_screentype_scsl(): ...


def test_compute_iec_virtual_conductor_cross_section_where_cross_section_not_specified_in_iec_table(): ...
