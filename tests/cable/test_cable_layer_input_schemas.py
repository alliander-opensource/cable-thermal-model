# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from cable_thermal_model.cable.schemas.cable_layer_input_schemas import (
    AbstractLayerInputSchema,
    BeddingInputSchema,
    ConductorInputSchema,
    ThreeCoreCableInsulationInputSchema,
)
from cable_thermal_model.model.cables.enum_classes_cable import (
    CableBeddingMaterial,
    CableConductorMaterial,
    CableConductorShape,
    CableConductorSurfaceType,
    CableInsulationMaterial,
)


@pytest.mark.parametrize(
    "inner_radius, thickness, outer_radius, expected_error",
    [
        (5.0, 2.0, 7.0, None),  # Valid case
        (5.0, -1.0, 4.0, "Invalid layer thickness"),  # Negative thickness
        (5.0, 0.0, 5.0, "Invalid layer thickness"),  # Zero thickness
        (-1.0, 2.0, 1.0, "Invalid inner_radius"),  # Negative inner radius
        (5.0, 2.0, 6.0, "Inconsistent dimensions"),  # Inconsistent dimensions
        (5.0, None, 5.0, "must be greater than inner_radius"),  # Implicitly zero thickness
        (5.0, None, 4.0, "must be greater than inner_radius"),  # Implicitly negative thickness
        (5.0, None, 7.0, None),  # Valid case with missing thickness
        (None, 2.0, 7.0, None),  # Valid case with missing inner radius
        (5.0, 2.0, None, None),  # Valid case with missing outer radius
        (5.0, None, None, None),  # Valid case with only inner radius provided
        (None, None, None, None),  # Valid case with no dimensions provided
    ],
)
def test_AbstractLayerInputSchema_dimension_validation(inner_radius, thickness, outer_radius, expected_error):
    """Test geometric validation behavior of `_AbstractLayerInputSchema`.

    Args:
        inner_radius: Optional inner radius input.
        thickness: Optional thickness input.
        outer_radius: Optional outer radius input.
        expected_error: Expected exception type for invalid combinations.

    """
    if expected_error:
        with pytest.raises(ValueError, match=expected_error):
            AbstractLayerInputSchema(
                material=CableConductorMaterial.Copper,
                inner_radius=inner_radius,
                thickness=thickness,
                outer_radius=outer_radius,
            )
    else:
        layer = AbstractLayerInputSchema(
            material=CableConductorMaterial.Copper,
            inner_radius=inner_radius,
            thickness=thickness,
            outer_radius=outer_radius,
        )
        nonzero_input_count = sum(x is not None for x in [inner_radius, thickness, outer_radius])
        if nonzero_input_count >= 2:  # noqa: PLR2004
            assert layer.outer_radius == pytest.approx(layer.inner_radius + layer.thickness)
        else:
            assert layer.inner_radius == inner_radius
            assert layer.thickness == thickness
            assert layer.outer_radius == outer_radius


@pytest.mark.parametrize(
    "inner_radius, thickness, outer_radius, expected_error",
    [
        (5.0, 2.0, 7.0, None),  # Valid case
        (5.0, None, 7.0, None),  # Valid case
        (None, 2.0, 7.0, None),  # Valid case
        (5.0, 2.0, None, None),  # Valid case
        (
            5.0,
            None,
            None,
            "Cannot calculate surface area",
        ),  # Invalid case, cannot calculate surface area with only one dimension provided
    ],
)
def test_AbstractLayerInputSchema_surface_area_calculation(inner_radius, thickness, outer_radius, expected_error):
    """Test annular surface area calculation behavior for layer schema.

    Args:
        inner_radius: Optional inner radius input.
        thickness: Optional thickness input.
        outer_radius: Optional outer radius input.
        expected_error: Error message pattern for invalid input cases.

    """
    layer_input = AbstractLayerInputSchema(
        material=CableConductorMaterial.Copper,
        inner_radius=inner_radius,
        thickness=thickness,
        outer_radius=outer_radius,
    )

    if expected_error:
        with pytest.raises(ValueError, match=expected_error):
            layer_input.calculate_surface_area()
    else:
        expected_surface_area = np.pi * (layer_input.outer_radius**2 - layer_input.inner_radius**2)
        assert layer_input.calculate_surface_area() == pytest.approx(expected_surface_area)


def test_LayerInputSchema_None_material_validation():
    with pytest.raises(ValueError, match="None is not a valid material for a cable layer."):
        BeddingInputSchema(
            material=CableBeddingMaterial.NONE,
        )


def test_hollow_conductor_with_conductor_shape_round():
    with pytest.raises(
        ValueError, match="inner_radius is 5.0 but shape is ccRound, expected ccHollow for non-zero inner radius."
    ):
        ConductorInputSchema(
            material=CableConductorMaterial.Copper,
            conducting_surface_area=1.0,
            shape=CableConductorShape.Round,
            surface_type=CableConductorSurfaceType.Solid,
            inner_radius=5.0,
            thickness=None,
            outer_radius=None,
        )


def test_round_conductor_with_conductor_shape_hollow():
    with pytest.raises(ValueError, match="expected inner_radius > 0 for hollow conductor shape."):
        ConductorInputSchema(
            material=CableConductorMaterial.Copper,
            conducting_surface_area=1.0,
            shape=CableConductorShape.Hollow,
            surface_type=CableConductorSurfaceType.Solid,
            inner_radius=0.0,
            thickness=None,
            outer_radius=None,
        )


@pytest.mark.parametrize(
    "single_conductor_radius",
    [-1.0, 0.0],
)
def test_negative_single_conductor_radius(single_conductor_radius):
    with pytest.raises(
        ValueError, match=f"Invalid single_conductor_radius: {single_conductor_radius}, must be strictly positive."
    ):
        ConductorInputSchema(
            material=CableConductorMaterial.Copper,
            conducting_surface_area=1.0,
            shape=CableConductorShape.Round,
            surface_type=CableConductorSurfaceType.Solid,
            inner_radius=None,
            thickness=None,
            outer_radius=None,
            single_conductor_radius=single_conductor_radius,
        )


@pytest.mark.parametrize(
    "diameter_over_stranded_conductors",
    [-1.0, 0.0],
)
def test_negative_diameter_over_stranded_conductors(diameter_over_stranded_conductors):
    with pytest.raises(
        ValueError,
        match=(
            f"Invalid diameter_over_stranded_conductors: "
            f"{diameter_over_stranded_conductors}, must be strictly positive."
        ),
    ):
        ThreeCoreCableInsulationInputSchema(
            material=CableInsulationMaterial.XLPEFilled,
            nominal_phase_voltage=10000,
            diameter_over_stranded_conductors=diameter_over_stranded_conductors,
            single_conductor_insulation_thickness=1.0,
            inner_radius=None,
            thickness=None,
            outer_radius=None,
        )


@pytest.mark.parametrize(
    "single_conductor_insulation_thickness",
    [-1.0, 0.0],
)
def test_negative_single_conductor_insulation_thickness(single_conductor_insulation_thickness):
    with pytest.raises(
        ValueError,
        match=(
            f"Invalid single_conductor_insulation_thickness: "
            f"{single_conductor_insulation_thickness}, must be strictly positive."
        ),
    ):
        ThreeCoreCableInsulationInputSchema(
            material=CableInsulationMaterial.XLPEFilled,
            nominal_phase_voltage=10000,
            diameter_over_stranded_conductors=1.0,
            single_conductor_insulation_thickness=single_conductor_insulation_thickness,
            inner_radius=None,
            thickness=None,
            outer_radius=None,
        )
