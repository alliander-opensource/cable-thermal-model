# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from cable_thermal_model.cable.cable_builder import CableBuilder
from cable_thermal_model.cable.cable_spec_parsers import SingleCoreSpecParser, SpecParserFactory, ThreeCoreSpecParser
from cable_thermal_model.model.cables.abstract_cable import CableConductorType, CableType
from cable_thermal_model.model.cables.enum_classes_cable import (
    CableConductorMaterial,
    CableConductorShape,
    CableConductorSurfaceType,
    CableLayer,
)
from cable_thermal_model.model.cables.fd_cable import FDCable
from tests.conftest import mock_load_cable_data_from_file

"""
"""

_RADIAL_APPROXIMATION_RESOLUTION_IN_METERS = 1e-9


def test_single_core_sector_shaped_exception():
    with (
        patch(
            "cable_thermal_model.cable.cable_builder.CableBuilder._load_cable_data_from_file",
            side_effect=mock_load_cable_data_from_file,
        ),
        pytest.raises(ValueError, match="Single-core cables can not have sector shaped conductors."),
    ):
        CableBuilder.build_cable_from_cable_id(cable_id="sector_shaped_single_core", fd_cable_class=FDCable)


def test_circular_conductor_zero_diameter_exception():
    with (
        patch(
            "cable_thermal_model.cable.cable_builder.CableBuilder._load_cable_data_from_file",
            side_effect=mock_load_cable_data_from_file,
        ),
        pytest.raises(ValueError, match="Invalid dimensions: outer_radius"),
    ):
        CableBuilder.build_cable_from_cable_id(cable_id="circular_shaped_zero_diameter", fd_cable_class=FDCable)


def test_zero_conductor_area_exception():
    with (
        patch(
            "cable_thermal_model.cable.cable_builder.CableBuilder._load_cable_data_from_file",
            side_effect=mock_load_cable_data_from_file,
        ),
        pytest.raises(ValueError, match="Invalid conducting_surface_area: 0.0, must be strictly positive."),
    ):
        CableBuilder.build_cable_from_cable_id(cable_id="zero_conductor_area", fd_cable_class=FDCable)


def test_consistence_conductor_area_and_diameter_exception():
    with (
        patch(
            "cable_thermal_model.cable.cable_builder.CableBuilder._load_cable_data_from_file",
            side_effect=mock_load_cable_data_from_file,
        ),
        pytest.raises(
            ValueError,
            match="Invalid conducting_surface_area",
        ),
    ):
        CableBuilder.build_cable_from_cable_id(cable_id="inconsistent_conductor_diameter", fd_cable_class=FDCable)

    with (
        patch(
            "cable_thermal_model.cable.cable_builder.CableBuilder._load_cable_data_from_file",
            side_effect=mock_load_cable_data_from_file,
        ),
        pytest.raises(
            ValueError,
            match="Invalid conducting_surface_area",
        ),
    ):
        CableBuilder.build_cable_from_cable_id(
            cable_id="inconsistent_conductor_diameter_three_core_cable", fd_cable_class=FDCable
        )


def test_single_core_cable_with_armour_exception():
    with (
        patch(
            "cable_thermal_model.cable.cable_builder.CableBuilder._load_cable_data_from_file",
            side_effect=mock_load_cable_data_from_file,
        ),
        pytest.raises(
            NotImplementedError,
            match=re.escape(
                "Armour losses are not accounted for in the FD model. "
                "Therefore, armoured cables are not supported for single-core cables."
            ),
        ),
    ):
        CableBuilder.build_cable_from_cable_id(cable_id="armoured_single_core_cable", fd_cable_class=FDCable)


@pytest.fixture(scope="module")
def get_cable_specs_by_name() -> pd.DataFrame:
    """Fixture to load the cable specifications from the example_cables.csv file.

    and make it available for all tests in this module.
    """
    return pd.read_csv(Path("data/example_cables.csv")).set_index("Name")


# CableSpecParser
@pytest.mark.parametrize(
    "cable_name, expected_cable_type",
    [("YMeKrvaslqwd 12/20kV 1x630 Alrm + as50", CableType.XLPE)],
    # TODO: Add more cable diversity
)
def test_cable_spec_parser_get_cable_type(get_cable_specs_by_name, cable_name, expected_cable_type):
    # Preparation:
    cable_specs = get_cable_specs_by_name.loc[cable_name]
    test_cable_spec_parser = SingleCoreSpecParser(cable_specs=cable_specs)

    # Generation:
    cable_constructional_information = test_cable_spec_parser.get_cable_constructional_input()

    # Evaluation:
    assert cable_constructional_information.cable_type == expected_cable_type


@pytest.mark.parametrize(
    "cable_name, expected_conductor_type",
    [
        (
            "YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
            CableConductorType(
                material=CableConductorMaterial.Aluminium,
                shape=CableConductorShape.Round,
                surface_type=CableConductorSurfaceType.Solid,
            ),
        ),
    ],
    # TODO: Add more cable diversity
)
def test_cable_spec_parser_get_conductor_type(get_cable_specs_by_name, cable_name, expected_conductor_type):
    # Preparation:
    cable_specs = get_cable_specs_by_name.loc[cable_name]
    test_cable_spec_parser = SingleCoreSpecParser(cable_specs=cable_specs)

    # Generation:
    cable_constructional_information = test_cable_spec_parser.get_cable_constructional_input()
    conductor_information = cable_constructional_information.conductor_input

    # Evaluation:
    assert conductor_information.surface_type == expected_conductor_type.surface_type
    assert conductor_information.material == expected_conductor_type.material
    assert conductor_information.shape == expected_conductor_type.shape


# SingleCoreSpecParser
@pytest.mark.parametrize(
    "cable_name, expected_outer_radii",
    [
        (
            "YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
            [0.01395, 0.0150425, 0.0205425, 0.021635, 0.022, 0.022785, 0.026],
        ),
    ],
    # TODO: Add more cable diversity aimed at if-statement diversity
)
def test_single_core_cable_spec_parser_get_radii(get_cable_specs_by_name, cable_name, expected_outer_radii):
    # Preparation:
    cable_specs = get_cable_specs_by_name.loc[cable_name]
    test_cable_spec_parser = SingleCoreSpecParser(cable_specs=cable_specs)
    cable_constructional_information = test_cable_spec_parser.get_cable_constructional_input()

    # Generation:
    outer_radii = list(cable_constructional_information.get_outer_radii().values())

    # Evaluation:
    assert np.allclose(outer_radii, expected_outer_radii, atol=_RADIAL_APPROXIMATION_RESOLUTION_IN_METERS)


@pytest.mark.parametrize(
    "cable_name, expected_materials",
    [
        (
            "YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
            ["cmAl", "scrXLPE", "imXLPEUnfilled", "scrXLPE", "samCopper", "imRubber", "pcPE"],
        ),
        (
            "YMeKrvaslqwd 12/20kV 3x240 Alrm + as50",
            ["cmAl", "imXLPEUnfilled", "samCopper", "imRubber", "pcPE"],
        ),
    ],
    # TODO: Add more cable diversity aimed at if-statement diversity
)
def test_single_core_cable_spec_parser_get_materials(get_cable_specs_by_name, cable_name, expected_materials):
    # Preparation:
    cable_specs = get_cable_specs_by_name.loc[cable_name]
    test_cable_spec_parser = SpecParserFactory.get_spec_parser(cable_specs=cable_specs)

    # Generation:
    cable_constructional_information = test_cable_spec_parser.get_cable_constructional_input()
    materials = [layer_input.material.value for layer_input in cable_constructional_information.layers.values()]

    # Evaluation:
    assert materials == expected_materials


@pytest.mark.parametrize(
    "cable_name, expected_layer_names",
    [
        (
            "YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
            ["Conductor", "Conductor screen", "Insulation", "Insulation screen", "Screen", "Bedding", "Sheath"],
        ),
    ],
    # TODO: Add more cable diversity aimed at if-statement diversity
)
def test_single_core_cable_spec_parser_get_layer_names(get_cable_specs_by_name, cable_name, expected_layer_names):
    # Preparation:
    cable_specs = get_cable_specs_by_name.loc[cable_name]
    test_cable_spec_parser = SingleCoreSpecParser(cable_specs=cable_specs)
    cable_constructional_information = test_cable_spec_parser.get_cable_constructional_input()

    # Generation:
    layer_names = list(cable_constructional_information.layers.keys())

    # Evaluation:
    assert layer_names == expected_layer_names


# ThreeCoreSpecParser
@pytest.mark.parametrize(
    "cable_name, expected_outer_radii",
    [
        ("GPLK 10/10 kV 3x185 Al", [0.018171702628058218, 0.0236, 0.026, 0.02825, 0.030, 0.0335]),
    ],
    # TODO: Add more cable diversity aimed at if-statement diversity
)
def test_three_core_cable_spec_parser_get_radii(get_cable_specs_by_name, cable_name, expected_outer_radii):
    # Preparation:
    cable_specs = get_cable_specs_by_name.loc[cable_name]
    test_cable_spec_parser = ThreeCoreSpecParser(cable_specs=cable_specs)
    cable_constructional_information = test_cable_spec_parser.get_cable_constructional_input()

    # Generation:
    outer_radii = list(cable_constructional_information.get_outer_radii().values())

    # Evaluation:
    assert np.allclose(outer_radii, expected_outer_radii, atol=_RADIAL_APPROXIMATION_RESOLUTION_IN_METERS)


@pytest.mark.parametrize(
    "cable_name, expected_materials",
    [
        (
            "GPLK 10/10 kV 3x185 Al",
            ["cmAl", "imPaperMassImpregnated", "samLead", "imPaperMassImpregnated", "samSteel", "pcBitumenJute"],
        ),
    ],
    # TODO: Add more cable diversity aimed at if-statement diversity
)
def test_three_core_cable_spec_parser_get_materials(get_cable_specs_by_name, cable_name, expected_materials):
    # Preparation:
    cable_specs = get_cable_specs_by_name.loc[cable_name]
    test_cable_spec_parser = ThreeCoreSpecParser(cable_specs=cable_specs)
    cable_constructional_information = test_cable_spec_parser.get_cable_constructional_input()

    # Generation:
    materials = [layer_input.material for layer_input in cable_constructional_information.layers.values()]

    # Evaluation:
    assert materials == expected_materials


@pytest.mark.parametrize(
    "cable_name, expected_layer_names",
    [
        ("GPLK 10/10 kV 3x185 Al", ["Conductor", "Insulation", "Screen", "Bedding", "Armour", "Sheath"]),
    ],
    # TODO: Add more cable diversity aimed at if-statement diversity
)
def test_three_core_cable_spec_parser_get_layer_names(get_cable_specs_by_name, cable_name, expected_layer_names):
    # Preparation:
    cable_specs = get_cable_specs_by_name.loc[cable_name]
    test_cable_spec_parser = ThreeCoreSpecParser(cable_specs=cable_specs)
    cable_constructional_information = test_cable_spec_parser.get_cable_constructional_input()

    # Generation:
    layer_names = list(cable_constructional_information.layers.keys())

    # Evaluation:
    assert layer_names == expected_layer_names


def test_bedding_material_without_t2(get_cable_specs_by_name):
    cable_specs = get_cable_specs_by_name.loc["YMeKrvaslqwd 12/20kV 1x630 Alrm + as50"].copy()
    cable_specs["t_2"] = 0.0  # Setting bedding thickness to zero to trigger exception

    cable_constructional_information = SingleCoreSpecParser(cable_specs=cable_specs).get_cable_constructional_input()

    assert CableLayer.Bedding not in cable_constructional_information.layers


def test_scSL_screen_type_not_implemented_exception():

    with (
        patch(
            "cable_thermal_model.cable.cable_builder.CableBuilder._load_cable_data_from_file",
            side_effect=mock_load_cable_data_from_file,
        ),
        pytest.raises(
            NotImplementedError,
            match="Screen type scSL is not supported by the model.",
        ),
    ):
        CableBuilder.build_cable_from_cable_id(cable_id="screen_type_SL", fd_cable_class=FDCable)
