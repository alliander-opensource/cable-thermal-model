# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

"""Module responsible for building Cable objects based on given cable specifications."""

from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd

from cable_thermal_model.cable.cable_spec_parsers import SpecParserFactory
from cable_thermal_model.cable.schemas.cable_input_schemas import CableConstructionalInputSchema
from cable_thermal_model.cable.schemas.cable_layer_input_schemas import ThreeCoreCableInsulationInputSchema
from cable_thermal_model.cable.schemas.pipe_schemas import PipeInputSchema
from cable_thermal_model.model.cables.abstract_cable import (
    CableConductorProperties,
    CableLayerMetrics,
    CableLayerProperties,
)
from cable_thermal_model.model.cables.cable import (
    Cable,
    CableTrefoilCircuitSinglePipeInAir,
    CableTrefoilCircuitSinglePipeInSoil,
)
from cable_thermal_model.model.cables.enum_classes_cable import (
    CableConductorCount,
    CableConductorShape,
    CableConductorSurfaceType,
    CableLayer,
    CableScreenType,
    CableSheathMaterial,
)
from cable_thermal_model.model.cables.pipe import Pipe
from cable_thermal_model.utils.exceptions import MissingMaterialException

CableT = TypeVar("CableT", bound=Cable)


class CableBuilder:
    """Utility class responsible for constructing Cable objects based on various input specifications."""

    __PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
    _SHEATH_CABLE_TYPE_STRING: str = "Sheath/cable type"
    MATERIALS_DF: pd.DataFrame = pd.read_csv(__PROJECT_ROOT / "data" / "material_properties.csv", sep=";").set_index(
        "material"
    )
    IEC_TABLE: pd.DataFrame = pd.read_csv(
        __PROJECT_ROOT / "data" / "IEC_conductor_resistance_table.csv", index_col=[0, 1]
    )

    @classmethod
    def build_cable_from_cable_id(
        cls,
        cable_id: str,
        cable_class: type[CableT],
        grid_points_per_layer: int = 10,
        pipe: PipeInputSchema | None = None,
        cable_source_file_path: Path = __PROJECT_ROOT / "data" / "example_cables.csv",
    ) -> CableT:
        """Builds a new Cable instance based on a cable_id.

        Args:
            cable_id (str): The name of a cable mentioned in the file specified by `cable_source_file_path`.
            grid_points_per_layer (int): The number of points per cable layer.
            pipe (PipeInputSchema | None): A pipe instance to be added around the cable.
            cable_class (type[Cable]): The Cable class to instantiate.
            cable_source_file_path (Path): The path to the cable source file.
                Defaults to "data/example_cables.csv". The file can be either
                a CSV or an Excel file.

        Returns:
                TCable: A new Cable instance (based on a Cable instance).

        """
        if cable_class in [CableTrefoilCircuitSinglePipeInSoil, CableTrefoilCircuitSinglePipeInAir] and pipe is None:
            raise ValueError(f"When using Cable class '{cable_class.__name__}', a pipe must be provided.")

        # load the cable data from the specified file
        cable_specs = cls._load_cable_data_from_file(cable_source_file_path, cable_id)

        # build the cable based on the loaded data
        return cls.build_cable_from_cable_specs(
            cable_specs=cable_specs,
            cable_class=cable_class,
            grid_points_per_layer=grid_points_per_layer,
            pipe=pipe,
        )

    @classmethod
    def build_cable_from_cable_specs(
        cls,
        cable_specs: pd.Series,
        cable_class: type[CableT],
        grid_points_per_layer: int = 10,
        pipe: PipeInputSchema | None = None,
    ) -> CableT:
        """Builds a new Cable instance based on a given set of cable specifications.

        Args:
            cable_specs (pd.Series): A Pandas Series holding the cable specification.
            grid_points_per_layer (int | None): The number of points per layer to use in FD grids
            pipe (PipeInputSchema | None): A pipe instance to be added around the cable.
            cable_class (type[Cable]): The Cable class to instantiate.

        Returns:
                TCable: A new Cable instance (based on a Cable instance).

        """
        cable_spec_parser = SpecParserFactory.get_spec_parser(cable_specs)
        cable_constructional_input: CableConstructionalInputSchema = cable_spec_parser.get_cable_constructional_input()

        # build the cable based on the parsed specifications
        return cls.build_cable(
            cable_constructional_input=cable_constructional_input,
            cable_class=cable_class,
            grid_points_per_layer=grid_points_per_layer,
            pipe=pipe,
        )

    @classmethod
    def build_cable(
        cls,
        cable_constructional_input: CableConstructionalInputSchema,
        cable_class: type[CableT],
        grid_points_per_layer: int = 10,
        pipe: PipeInputSchema | None = None,
    ) -> CableT:
        """Build FD cable object.

        Args:
            cable_constructional_input (CableConstructionalInputSchema):
                A CableConstructionalInputSchema object holding the cable
                specification.
            grid_points_per_layer (int): The number of points per layer to use in FD grids
            cable_class (type[Cable]): The Cable class to instantiate.
            pipe (PipeInputSchema | None): A pipe instance to be added around the cable.

        Returns:
            CableT: A finite-difference cable object.

        """
        # retrieve the cable's conductor properties
        cable_conductor_properties: CableConductorProperties = cls._get_cable_conductor_properties(
            cable_constructional_input
        )

        # retrieve the cable's layer properties
        cable_layer_properties_by_layer: dict[CableLayer, CableLayerProperties] = cls._get_cable_layer_properties(
            cable_constructional_input
        )

        # retrieve the cable's layer metrics
        cable_layer_metrics: CableLayerMetrics = cls._get_cable_layer_metrics(cable_constructional_input)

        # instantiate the cable
        fd_cable = cable_class(
            conductor=cable_conductor_properties,
            layer_properties=cable_layer_properties_by_layer,
            layer_metrics=cable_layer_metrics,
            cable_type=cable_constructional_input.cable_type,
            grid_counts=dict.fromkeys(cable_layer_properties_by_layer.keys(), grid_points_per_layer),
        )

        # add a pipe around the cable if specified
        if pipe is not None:
            fd_cable = fd_cable.get_cable_copy_with_pipe(
                Pipe(
                    pipe_input=pipe,
                    outer_radius_cable=fd_cable.layer_metrics.cable_radius,
                )
            )

        return fd_cable

    @classmethod
    def _load_cable_data_from_file(cls, cable_source_file_path: Path, cable_id: str) -> pd.Series:
        """Load the cable data from the specified file.

        Args:
            cable_source_file_path (Path): The path to the cable source file,
                which is assumed to be in the 'data' directory. The file can
                be either a CSV or an Excel file.
            cable_id (str): The name of a cable mentioned in the file specified by `cable_source_file_path`.

        Returns:
            pd.Series: The chosen cable's data series.

        """
        match cable_source_file_path.suffix:
            case ".csv":
                cable_data_df = pd.read_csv(cable_source_file_path).set_index("Name")
            case ".xlsx" | ".xls":
                cable_data_df = pd.read_excel(cable_source_file_path).set_index("Name")  # type: ignore
            case _:
                raise ValueError(
                    f"Unsupported file format: {cable_source_file_path.suffix}. "
                    "Supported formats are .csv, .xlsx, and .xls."
                )

        if cable_id not in cable_data_df.index:
            raise ValueError(
                f"Cable ID '{cable_id}' not found in the file '{cable_source_file_path}'."
                f"Available cable IDs are: {cable_data_df.index.tolist()}"
            )

        if not cable_data_df.index.is_unique:
            raise ValueError(
                f"The index of the cable data file '{cable_source_file_path}' must be unique. "
                "Please ensure that the 'Name' column contains unique values."
            )
        return pd.Series(cable_data_df.loc[cable_id])

    @classmethod
    def _get_cable_conductor_properties(
        cls,
        cable_constructional_input: CableConstructionalInputSchema,
    ) -> CableConductorProperties:
        """The method to translate CableConstructionalInputSchema properties into CableConductorProperties.

        Values are directly derived from the CableConstructionalInputSchema object.

        Args:
            cable_constructional_input (CableConstructionalInputSchema):
                The CableConstructionalInputSchema object holding the
                properties used to generate the CableConductorProperties
                object.

        Returns:
            CableConductorProperties:
                A CableConductorProperties object containing the conductor
                properties of the given CableConstructionalInputSchema object.

        """
        return CableConductorProperties(
            number_of_conductors=cable_constructional_input.number_of_conductors,
            shape=cable_constructional_input.conductor_input.shape,
            material=cable_constructional_input.conductor_input.material,
            surface_type=cable_constructional_input.conductor_input.surface_type,
        )

    @classmethod
    def _get_cable_layer_properties(
        cls,
        cable_constructional_input: CableConstructionalInputSchema,
    ) -> dict[CableLayer, CableLayerProperties]:
        """The method to translate CableConstructionalInputSchema properties into a dictionary of CableLayerProperties.

        Values are extracted from the material_properties.csv file, using the
        materials in the CableConstructionalInputSchema object.
        The thermal resistivity of PVC changes from 5 to 6 mK/W when the voltage exceeds the 35kV-level.

        Args:
            cable_constructional_input (CableConstructionalInputSchema):
                A CableConstructionalInputSchema object holding the
                properties used to generate the CableLayerProperties objects.

        Returns:
            dict[CableLayer, CableLayerProperties]:
                A dictionary mapping each CableLayer to its
                CableLayerProperties, according to the given
                CableConstructionalInputSchema object.

        Raises:
            KeyError: If the material properties are missing for any of the
                materials needed for the CableConstructionalInputSchema object.

        """
        _pvc_voltage_threshold = 35_000
        materials_in_order = [layer.material for layer in cable_constructional_input.layers.values()]

        try:
            property_map = {
                "rho": "thermal resistivity",
                "electric_rho": "resistivity at twenty degrees",
                "capacity": "volumetric specific heat",
                "alpha": "temperature coefficient",
                "epsilon": "relative permittivity",
                "tan_delta": "dissipation factor",
            }
            properties = {
                key: cls.MATERIALS_DF[col].loc[materials_in_order].to_numpy() for key, col in property_map.items()
            }
            # Special PVC case
            if (
                cable_constructional_input.insulation_input.nominal_phase_voltage > _pvc_voltage_threshold
                and cable_constructional_input.sheath_input.material == CableSheathMaterial.PVC
            ):
                index = materials_in_order.index(CableSheathMaterial.PVC)
                properties["rho"][index] = cls.MATERIALS_DF["thermal resistivity"].loc["pcPVC-35kV"]

        except KeyError as key_error:
            missing_materials = [mat for mat in materials_in_order if mat not in cls.MATERIALS_DF.index]
            raise MissingMaterialException(
                f"Material properties missing the following materials {missing_materials}."
            ) from key_error

        inner_radii = cable_constructional_input.get_inner_radii()
        outer_radii = cable_constructional_input.get_outer_radii()

        return {
            cable_layer: CableLayerProperties(
                layer=cable_layer,
                inner_radius=inner_radii[cable_layer],
                outer_radius=outer_radii[cable_layer],
                rho=properties["rho"][idx],
                electric_rho=properties["electric_rho"][idx],
                capacity=properties["capacity"][idx],
                alpha=properties["alpha"][idx],
                epsilon=properties["epsilon"][idx],
                tan_delta=properties["tan_delta"][idx],
            )
            for idx, cable_layer in enumerate(cable_constructional_input.layers.keys())
        }

    @classmethod
    def _get_cable_layer_metrics(cls, cable_constructional_input: CableConstructionalInputSchema) -> CableLayerMetrics:
        """The method to translate CableConstructionalInputSchema properties into CableLayerMetrics.

        Values are extracted from the cable_constructional_information
        attribute in the CableConstructionalInputSchema object.

        Args:
            cable_constructional_input (CableConstructionalInputSchema): A CableConstructionalInputSchema object
                holding the properties used to generate the CableLayerMetrics object.

        Returns:
            CableLayerMetrics: A CableLayerMetrics object containing the size and certain calculated properties of
                a cable according to the given CableConstructionalInputSchema object.

        """
        conductor_cross_section = cable_constructional_input.conductor_input.conducting_surface_area
        outer_radii = cable_constructional_input.get_outer_radii()

        if cable_constructional_input.conductor_input.single_conductor_radius is not None:
            conductor_radius_original = cable_constructional_input.conductor_input.single_conductor_radius
        elif cable_constructional_input.number_of_conductors == CableConductorCount.One:
            conductor_radius_original = outer_radii[CableLayer.Conductor]
        else:
            conductor_radius_original = np.sqrt(conductor_cross_section / np.pi)

        conductor_virtual_cross_section = cls._compute_iec_virtual_conductor_cross_section(
            cable_constructional_input=cable_constructional_input
        )
        screen_cross_section = (
            cable_constructional_input.screen_input.conducting_surface_area
            if cable_constructional_input.screen_input is not None
            else None
        )
        armour_cross_section = (
            cable_constructional_input.armour_input.conducting_surface_area
            if cable_constructional_input.armour_input is not None
            else None
        )

        diameter_over_stranded_conductors = (
            cable_constructional_input.insulation_input.diameter_over_stranded_conductors
            if isinstance(cable_constructional_input.insulation_input, ThreeCoreCableInsulationInputSchema)
            else None
        )
        U_0 = cable_constructional_input.insulation_input.nominal_phase_voltage

        cable_radius = outer_radii[CableLayer.Sheath]

        # Preset default values for conditional values:
        conductor_centers = [(0.0, 0.0)]
        conductor_distance = None
        sector_radius = None
        core_to_sector_distance = None
        original_insulation_thickness = None

        # Set conditional values:
        if cable_constructional_input.number_of_conductors == CableConductorCount.Three:
            conductor_distance = cls._compute_conductor_distance_in_three_core_cables(
                cable_constructional_input=cable_constructional_input
            )

            if cable_constructional_input.conductor_input.shape == CableConductorShape.Sector:
                insulation_outer_radius = cable_constructional_input.get_outer_radii()[CableLayer.Insulation]
                insulation_input = cable_constructional_input.validate_three_core_cable_insulation()
                t1 = insulation_input.single_conductor_insulation_thickness
                A_c = cable_constructional_input.conductor_input.conducting_surface_area

                r_cc = insulation_outer_radius - t1  # step 3
                sector_radius = np.sqrt(3 * A_c / np.pi)  # step 4
                core_to_sector_distance = r_cc - sector_radius  # step 5

                original_insulation_thickness = t1

        return CableLayerMetrics(
            conductor_centers=conductor_centers,
            conductor_cross_section=conductor_cross_section,
            conductor_equivalent_outer_diameter=cable_constructional_input.get_outer_radii()[CableLayer.Conductor] * 2,
            conductor_radius_original=conductor_radius_original,
            conductor_virtual_cross_section=conductor_virtual_cross_section,
            conductor_distance=conductor_distance,
            screen_cross_section=screen_cross_section,
            armour_cross_section=armour_cross_section,
            diameter_over_stranded_conductors=diameter_over_stranded_conductors,
            nominal_phase_voltage=U_0,
            sector_radius=sector_radius,
            outer_radius=cable_radius,
            cable_radius=cable_radius,
            core_to_sector_distance=core_to_sector_distance,
            original_insulation_thickness=original_insulation_thickness,
            insulation_material=cable_constructional_input.insulation_input.material,
        )

    @classmethod
    def _compute_iec_virtual_conductor_cross_section(
        cls, cable_constructional_input: CableConstructionalInputSchema
    ) -> float:
        """Computes an IEC virtual conductor cross-section based on given cable specifications.

        The IEC implements a table of conductor resistances (table 1 in Norm 60228) which specifies conductor
        resistance (at T = 20 degrees) in Ohm/km, given the conductor cross-section.

        Notes:
            This table (again, table 1 in Norm 60228) does not follow the standard formula [ R = rho/A ]. As such, this
                method computes a virtual conductor cross-section that replicates the conductor resistance of the IEC
                table, when used in the formula [ R = rho/A ].

        Args:
            cable_constructional_input (CableConstructionalInputSchema):
                The cable specifications used to calculate the cross-section.

        Returns:
            float: The virtual conductor cross-section (in m2), which can be used
                to compute an IEC-compatible conductor resistance.

        """
        conductor_cross_section = (  # convert the surface area from m2 to mm2
            cable_constructional_input.conductor_input.conducting_surface_area * 1e6
        )

        iec_cross_sections = cls.IEC_TABLE.index.get_level_values("area")

        if np.sum(np.abs(iec_cross_sections - conductor_cross_section) < 1) == 0:
            raise ValueError(
                f"The conductor cross-section {conductor_cross_section} m2 is not present in the IEC conductor "
                f"resistance table. Available cross-sections are: {iec_cross_sections.unique().values} mm2."
            )

        # retrieve the conductor material
        conductor_material = cable_constructional_input.conductor_input.material

        # retrieve the surface type of the conductor
        if cable_constructional_input.conductor_input.surface_type in [
            CableConductorSurfaceType.Stranded,
            CableConductorSurfaceType.Milliken,
        ]:
            conductor_surface_type = "Stranded"
        else:
            conductor_surface_type = "Solid"

        # Compute the virtual conductor cross-section
        iec_resistance_value = cls.IEC_TABLE.loc[
            (int(conductor_cross_section), conductor_surface_type), conductor_material
        ]
        iec_resistance_float = float(
            iec_resistance_value.item() if hasattr(iec_resistance_value, "item") else iec_resistance_value  # type: ignore
        )

        if np.isnan(iec_resistance_float):
            raise ValueError(
                f"Conductor resistance for cross-section {conductor_cross_section} mm2, "
                f"surface type {conductor_surface_type}, and material {conductor_material} "
                "is not available in the IEC conductor resistance table."
            )
        resistivity = float(
            np.asarray(cls.MATERIALS_DF.loc[conductor_material, "resistivity at twenty degrees"])
        )  # in Ohm*mm2/m
        virtual_cross_section = resistivity / iec_resistance_float * 1e3  # convert from km to m
        return virtual_cross_section

    @classmethod
    def _compute_conductor_distance_in_three_core_cables(
        cls,
        cable_constructional_input: CableConstructionalInputSchema,
    ) -> float:
        """Compute the conductor distance for three core cables.

        Args:
            cable_constructional_input (CableConstructionalInputSchema): Cable specifications for a three core cable
                used to calculate the conductor distance between the three cores.

        Returns:
            float: The calculated conductor distance in meters, based on the supplied cable specs.

        """
        insulation_input = cable_constructional_input.validate_three_core_cable_insulation()
        conductor_shape = cable_constructional_input.conductor_input.shape

        if conductor_shape == CableConductorShape.Round:
            # The distance from the conductor center to core of the cable is equal to half the doga,
            # minus once the insulation thickness (t1),  and minus half the conductor radius.
            if cable_constructional_input.conductor_input.single_conductor_radius is None:
                single_conductor_radius = (
                    cable_constructional_input.compute_single_conductor_radius_from_conducting_surface_area()
                )
            else:
                single_conductor_radius = cable_constructional_input.conductor_input.single_conductor_radius
            dist_cond_to_core = (
                insulation_input.diameter_over_stranded_conductors / 2
                - insulation_input.single_conductor_insulation_thickness
                - single_conductor_radius
            )

            if (
                cable_constructional_input.screen_input is not None
                and cable_constructional_input.screen_input.screen_type != CableScreenType.Common
            ):
                raise NotImplementedError(
                    f"{cls._SHEATH_CABLE_TYPE_STRING} '{cable_constructional_input.screen_input.screen_type}' not "
                    "implemented."
                )

            cos30 = np.sqrt(3) / 2  # The angle of dist_cond_to_core with half_dist_cond_to_cond
            half_dist_cond_to_cond = cos30 * dist_cond_to_core  # Apply the cosine rule
            dist = 2 * half_dist_cond_to_cond  # Times two to obtain distance from conductor center to conductor center

        elif conductor_shape == CableConductorShape.Sector:
            insulation_outer_radius = cable_constructional_input.get_outer_radii()[CableLayer.Insulation]
            t1 = insulation_input.single_conductor_insulation_thickness
            diameter_over_stranded_conductors = insulation_input.diameter_over_stranded_conductors

            dist = 2 * t1 - (
                2 * insulation_outer_radius - diameter_over_stranded_conductors
            )  # Method used by Phase2Phase (verified by mail)
        else:
            raise NotImplementedError(f"Conductor shape {conductor_shape} not implemented.")

        return dist
