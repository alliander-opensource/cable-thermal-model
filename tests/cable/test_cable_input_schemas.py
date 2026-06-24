# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from cable_thermal_model.cable.cable_builder import CableBuilder
from cable_thermal_model.cable.cable_spec_parsers import SingleCoreSpecParser, ThreeCoreSpecParser
from cable_thermal_model.cable.schemas.cable_input_schemas import CableConstructionalInputSchema
from cable_thermal_model.cable.schemas.cable_layer_input_schemas import (
    ConductorInputSchema,
    InsulationInputSchema,
    SheathInputSchema,
    ThreeCoreCableInsulationInputSchema,
)
from cable_thermal_model.model.cables.enum_classes_cable import (
    CableConductorCount,
    CableConductorMaterial,
    CableConductorShape,
    CableConductorSurfaceType,
    CableInsulationMaterial,
    CableLayer,
    CableSheathMaterial,
)
from tests.conftest import mock_load_cable_data_from_file


@pytest.mark.parametrize(
    (
        "conductor_dimensions, insulation_dimenstions, sheat_dimensions, "
        "number_of_conductors, conductor_shape, expected_error"
    ),
    [
        # Valid case: fully defined dimensions
        (
            {"inner_radius": 0.0, "thickness": 7.0, "outer_radius": 7.0},
            {"inner_radius": 7.0, "thickness": 1.0, "outer_radius": 8.0},
            {"inner_radius": 8.0, "thickness": 1.0, "outer_radius": 9.0},
            CableConductorCount.One,
            CableConductorShape.Round,
            None,
        ),
        # Invalid case: fully defined dimensions with non-zero conductor inner_radius
        (
            {"inner_radius": 5.0, "thickness": 2.0, "outer_radius": 7.0},
            {"inner_radius": 7.0, "thickness": 1.0, "outer_radius": 8.0},
            {"inner_radius": 8.0, "thickness": 1.0, "outer_radius": 9.0},
            CableConductorCount.One,
            CableConductorShape.Hollow,
            "Inconsistent dimensions",
        ),
        # Valid case: missing conductor dimensions, to be filled in based on insulation and sheath dimensions
        (
            {"inner_radius": None, "thickness": None, "outer_radius": None},
            {"inner_radius": 7.0, "thickness": None, "outer_radius": 8.0},
            {"inner_radius": None, "thickness": None, "outer_radius": 9.0},
            CableConductorCount.One,
            CableConductorShape.Round,
            None,
        ),
        # Valid case: one dimension provided per layer, to be filled in based on the other layer dimensions
        (
            {"inner_radius": None, "thickness": None, "outer_radius": 7.0},
            {"inner_radius": None, "thickness": 1.0, "outer_radius": None},
            {"inner_radius": None, "thickness": 1.0, "outer_radius": None},
            CableConductorCount.One,
            CableConductorShape.Round,
            None,
        ),
        # Invalid case: implicit non-zero conductor inner_radius
        (
            {"inner_radius": None, "thickness": 5.0, "outer_radius": 7.0},
            {"inner_radius": None, "thickness": None, "outer_radius": 8.0},
            {"inner_radius": None, "thickness": None, "outer_radius": 9.0},
            CableConductorCount.One,
            CableConductorShape.Hollow,
            "Inconsistent dimensions",
        ),
        # Invalid case: conflicting dimensions between layers
        (
            {"inner_radius": None, "thickness": None, "outer_radius": 6.0},
            {"inner_radius": 7.0, "thickness": None, "outer_radius": 8.0},
            {"inner_radius": None, "thickness": None, "outer_radius": 9.0},
            CableConductorCount.One,
            CableConductorShape.Round,
            "Inconsistent dimensions",
        ),
        # Invalid case: implicitly conflicting dimensions between layers
        (
            {"inner_radius": None, "thickness": None, "outer_radius": 6.0},
            {"inner_radius": None, "thickness": 1.0, "outer_radius": 8.0},
            {"inner_radius": None, "thickness": None, "outer_radius": 9.0},
            CableConductorCount.One,
            CableConductorShape.Round,
            "Inconsistent dimensions",
        ),
        # Invalid case: insufficient dimensions to build cable
        (
            {"inner_radius": None, "thickness": None, "outer_radius": None},
            {"inner_radius": None, "thickness": None, "outer_radius": 8.0},
            {"inner_radius": None, "thickness": None, "outer_radius": 9.0},
            CableConductorCount.One,
            CableConductorShape.Round,
            "Insufficient dimension information",
        ),
        # Valid case: conductor-insulation interface dimension should be determined by T1-calculation
        (
            {"inner_radius": None, "thickness": None, "outer_radius": None},
            {"inner_radius": None, "thickness": None, "outer_radius": 8.0},
            {"inner_radius": None, "thickness": None, "outer_radius": 9.0},
            CableConductorCount.Three,
            CableConductorShape.Round,
            None,
        ),
        # Invalid case: conflicting conductor-insulation interface determined by T1-calculation
        (
            {"inner_radius": None, "thickness": None, "outer_radius": None},
            {"inner_radius": 2.0, "thickness": None, "outer_radius": 8.0},
            {"inner_radius": None, "thickness": None, "outer_radius": 9.0},
            CableConductorCount.Three,
            CableConductorShape.Round,
            "Inconsistent dimensions: equivalent inner insulation radius conflicts",
        ),
    ],
)
def test_cable_constructional_input_schema_dimension_validation(
    conductor_dimensions,
    insulation_dimenstions,
    sheat_dimensions,
    number_of_conductors,
    conductor_shape,
    expected_error,
):
    """Test cross-layer dimension validation for complete cable schemas.

    Args:
        conductor_dimensions: Input dimension dict for conductor layer.
        insulation_dimenstions: Input dimension dict for insulation layer.
        sheat_dimensions: Input dimension dict for sheath layer.
        number_of_conductors: Number of conductors in the cable.
        conductor_shape: Shape of the cable conductor.
        expected_error: Expected error message pattern for invalid cases.

    """
    conductor_input = ConductorInputSchema(
        material=CableConductorMaterial.Copper,
        conducting_surface_area=1.0,
        shape=conductor_shape,
        surface_type=CableConductorSurfaceType.Solid,
        **conductor_dimensions,
    )
    insulation_input = InsulationInputSchema(
        material=CableInsulationMaterial.XLPEFilled,
        nominal_phase_voltage=10000,
        **insulation_dimenstions,
    )
    if number_of_conductors == CableConductorCount.Three:
        insulation_input = ThreeCoreCableInsulationInputSchema(
            diameter_over_stranded_conductors=12.0,
            single_conductor_insulation_thickness=1.0,
            **insulation_input.model_dump(),
        )

    sheath_input = SheathInputSchema(
        material=CableSheathMaterial.PVC,
        **sheat_dimensions,
    )

    def initialize_cable_constructional_input_schema():
        """Construct a minimal one-core cable input schema for this test.

        Returns:
            CableConstructionalInputSchema: Initialized schema instance.

        """
        return CableConstructionalInputSchema(
            number_of_conductors=number_of_conductors,
            conductor_input=conductor_input,
            insulation_input=insulation_input,
            sheath_input=sheath_input,
        )

    if expected_error:
        with pytest.raises(ValueError, match=expected_error):
            initialize_cable_constructional_input_schema()
    else:
        initialize_cable_constructional_input_schema()
        for layer in [conductor_input, insulation_input, sheath_input]:
            assert layer.outer_radius == pytest.approx(layer.inner_radius + layer.thickness)

        for layer, next_layer in [(conductor_input, insulation_input), (insulation_input, sheath_input)]:
            assert next_layer.inner_radius == pytest.approx(layer.outer_radius)


@pytest.mark.parametrize(
    "cable_id, expected_t1",
    [
        (
            "GPLK 10/10 kV 3x185 Al",
            0.7488,
        ),
        (
            "OD 50kV 3x120Cu",
            0.9397637795275591,
        ),
        (
            "YMeKrvaslqwd 12/20kV 3x240 Alrm + as50",
            0.6274000326251488,
        ),
    ],
)
def test_t1_computation(cable_id: str, expected_t1: float):
    """Test T1 computation for three-core cables."""
    with patch(
        "cable_thermal_model.cable.cable_builder.CableBuilder._load_cable_data_from_file",
        side_effect=mock_load_cable_data_from_file,
    ):
        cable_specs = CableBuilder._load_cable_data_from_file(Path("data/example_cables.csv"), cable_id=cable_id)
    cable_spec_parser = ThreeCoreSpecParser(cable_specs=cable_specs)
    cable_constructional_information = cable_spec_parser.get_cable_constructional_input()

    normalized_t1 = cable_constructional_information.compute_normalized_lumped_sum_thermal_resistance_insulation()
    assert normalized_t1 is not None
    insulation_thermal_resistivity = cast(
        float,
        CableBuilder.MATERIALS_DF.loc[
            cable_constructional_information.insulation_input.material.value, "thermal resistivity"
        ],
    )
    computed_t1 = normalized_t1 * insulation_thermal_resistivity

    # Evaluation
    assert computed_t1 == pytest.approx(expected_t1, rel=1e-5)


def test_t1_computation_single_core_cable():
    cable_id = "YMeKrvaslqwd 12/20kV 1x630 Alrm + as50"
    with patch(
        "cable_thermal_model.cable.cable_builder.CableBuilder._load_cable_data_from_file",
        side_effect=mock_load_cable_data_from_file,
    ):
        cable_specs = CableBuilder._load_cable_data_from_file(Path("data/example_cables.csv"), cable_id=cable_id)
    cable_spec_parser = SingleCoreSpecParser(cable_specs=cable_specs)

    cable_constructional_information = cable_spec_parser.get_cable_constructional_input()
    with pytest.raises(ValueError, match="Insulation input must be of type ThreeCoreCableInsulationInputSchema"):
        cable_constructional_information.compute_normalized_lumped_sum_thermal_resistance_insulation()


@pytest.mark.parametrize(
    "outer",
    [True, False],
)
def test_get_radii_recursion(outer):
    """Test that get_{outer/inner}_radii correctly calls get_and_validate_radii.

    This is only done when the pydantic scheme is not validated at first.
    """
    cable_constructional_information = CableConstructionalInputSchema.model_construct(
        number_of_conductors=CableConductorCount.One,
        conductor_input=ConductorInputSchema(
            material=CableConductorMaterial.Copper,
            conducting_surface_area=1.0,
            shape=CableConductorShape.Round,
            surface_type=CableConductorSurfaceType.Solid,
            inner_radius=None,
            thickness=None,
            outer_radius=1.0,
        ),
        insulation_input=InsulationInputSchema(
            material=CableInsulationMaterial.XLPEFilled,
            nominal_phase_voltage=10000,
            inner_radius=None,
            thickness=None,
            outer_radius=2.0,
        ),
        sheath_input=SheathInputSchema(
            material=CableSheathMaterial.PVC,
            inner_radius=None,
            thickness=1.0,
            outer_radius=None,
        ),
    )
    assert cable_constructional_information.sheath_input.outer_radius is None
    assert cable_constructional_information.sheath_input.inner_radius is None

    if outer:
        radii = cable_constructional_information.get_outer_radii()
    else:
        radii = cable_constructional_information.get_inner_radii()

    assert isinstance(radii, dict)
    assert all(isinstance(key, CableLayer) for key in radii)
    assert all(isinstance(value, float) for value in radii.values())

    assert cable_constructional_information.sheath_input.outer_radius == pytest.approx(3.0)


def test_EPR_cable_unrecognized_cable_type():
    cable_id = "test_cable_EPR"
    with patch(
        "cable_thermal_model.cable.cable_builder.CableBuilder._load_cable_data_from_file",
        side_effect=mock_load_cable_data_from_file,
    ):
        cable_specs = CableBuilder._load_cable_data_from_file(Path("data/example_cables.csv"), cable_id=cable_id)
    cable_spec_parser = ThreeCoreSpecParser(cable_specs=cable_specs)

    with pytest.raises(ValueError, match="imEPR is not recognized as XLPE, PILC or Oil Pressure."):
        cable_spec_parser.get_cable_constructional_input()


def test_get_or_compute_insulation_equivalent_radius_ratio():
    """Test equivalent insulation radius ratio computation for three-core cables."""
    cable_constructional_information = CableConstructionalInputSchema(
        number_of_conductors=CableConductorCount.Three,
        conductor_input=ConductorInputSchema(
            material=CableConductorMaterial.Copper,
            conducting_surface_area=1.0,
            shape=CableConductorShape.Round,
            surface_type=CableConductorSurfaceType.Solid,
            inner_radius=None,
            thickness=None,
            outer_radius=None,
        ),
        insulation_input=ThreeCoreCableInsulationInputSchema(
            material=CableInsulationMaterial.XLPEFilled,
            nominal_phase_voltage=10000,
            diameter_over_stranded_conductors=12.0,
            single_conductor_insulation_thickness=1.0,
            inner_radius=None,
            thickness=None,
            outer_radius=8.0,
        ),
        sheath_input=SheathInputSchema(
            material=CableSheathMaterial.PVC,
            inner_radius=None,
            thickness=1.0,
            outer_radius=None,
        ),
    )

    assert cable_constructional_information.insulation_input.insulation_equivalent_radius_ratio is not None

    # assert compute_normalized_lumped_sum_thermal_resistance_insulation is
    # not called again when value is requested again.
    with patch.object(
        CableConstructionalInputSchema,
        "compute_normalized_lumped_sum_thermal_resistance_insulation",
        side_effect=AssertionError("compute_normalized_lumped_sum_thermal_resistance_insulation should not be called"),
    ) as mock_callback:
        cable_constructional_information.get_or_compute_insulation_equivalent_radius_ratio()

    mock_callback.assert_not_called()
