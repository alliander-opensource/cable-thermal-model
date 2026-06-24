# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from cable_thermal_model.cable.schemas.cable_input_schemas import (
    ArmourInputSchema,
    BeddingInputSchema,
    CableConstructionalInputSchema,
    ConductorInputSchema,
    ConductorScreenInputSchema,
    InsulationInputSchema,
    InsulationScreenInputSchema,
    InternalOilDuctInputSchema,
    ScreenInputSchema,
    SheathInputSchema,
    ThreeCoreCableInsulationInputSchema,
)
from cable_thermal_model.model.cables.enum_classes_cable import (
    CableArmourMaterial,
    CableBeddingMaterial,
    CableConductorCount,
    CableConductorInsulationScreenMaterial,
    CableConductorMaterial,
    CableConductorShape,
    CableConductorSurfaceType,
    CableInsulationMaterial,
    CableOilMaterial,
    CableScreenMaterial,
    CableScreenType,
    CableSheathMaterial,
)


class CableSpecParser(ABC):
    """Abstract base class for parsing cable specification rows into structured input schemas."""

    def __init__(self, cable_specs: pd.Series):
        """Initialize parser with raw cable specification row.

        Args:
            cable_specs: Cable specification row from the source cable table.

        """
        self.cable_specs = cable_specs

    @abstractmethod
    def get_internal_oil_duct_input(self) -> InternalOilDuctInputSchema | None:
        """Build internal oil duct schema.

        Returns:
            InternalOilDuctInputSchema | None: Oil duct schema when applicable,
                otherwise `None`.

        """
        pass

    @abstractmethod
    def get_conductor_input(self) -> ConductorInputSchema:
        """Build conductor schema for the current cable specs.

        Returns:
            ConductorInputSchema: Conductor layer schema.

        """
        pass

    @abstractmethod
    def get_conductor_screen_input(self) -> ConductorScreenInputSchema | None:
        """Build optional conductor screen schema.

        Returns:
            ConductorScreenInputSchema | None: Conductor screen schema when
                applicable, otherwise `None`.

        """
        pass

    @abstractmethod
    def get_insulation_input(self) -> InsulationInputSchema:
        """Build insulation schema for the current cable specs.

        Returns:
            InsulationInputSchema: Insulation layer schema.

        """
        pass

    @abstractmethod
    def get_insulation_screen_input(self) -> InsulationScreenInputSchema | None:
        """Build optional insulation screen schema.

        Returns:
            InsulationScreenInputSchema | None: Insulation screen schema when
                applicable, otherwise `None`.

        """
        pass

    def get_screen_input(self) -> ScreenInputSchema | None:
        """Build optional metallic screen input schema.

        Returns:
            ScreenInputSchema | None: Screen layer schema when present,
                otherwise `None`.

        """
        material = CableScreenMaterial(self.cable_specs["Sheath material"])
        if material == CableScreenMaterial.NONE.value:
            return None

        return ScreenInputSchema(
            material=material,
            inner_radius=None,
            thickness=float(self.cable_specs["t_she"]) / 1e3,  # convert to meters
            outer_radius=float(self.cable_specs["D_she_ext"]) / 2 / 1e3,  # convert to meters
            conducting_surface_area=float(self.cable_specs["A_she"]) / 1e6,  # convert to square meters
            screen_type=CableScreenType(self.cable_specs["Sheath/cable type"]),
        )

    def get_bedding_input(self) -> BeddingInputSchema | None:
        """Build optional bedding layer input schema.

        Returns:
            BeddingInputSchema | None: Bedding layer schema when present,
                otherwise `None`.

        """
        material = CableBeddingMaterial(self.cable_specs["Bedding material"])
        if material == CableBeddingMaterial.NONE.value:
            return None
        thickness = float(self.cable_specs["t_2"]) / 1e3  # convert to meters
        if np.isclose(thickness, 0):
            return None

        return BeddingInputSchema(
            material=material,
            inner_radius=None,
            thickness=thickness,
            outer_radius=None,
        )

    def get_armour_input(self) -> ArmourInputSchema | None:
        """Build optional armour layer input schema.

        Returns:
            ArmourInputSchema | None: Armour layer schema when present,
                otherwise `None`.

        """
        material = CableArmourMaterial(self.cable_specs["Arm material"])
        if material == CableArmourMaterial.NONE.value:
            return None

        conducting_surface_area = float(self.cable_specs["A_tape"]) / 1e6
        if conducting_surface_area <= 0:
            d_wire = float(self.cable_specs["d_wire"]) / 1e3  # convert to meters
            n_wireOrTape = float(self.cable_specs["n_wireOrTape"])
            conducting_surface_area = np.pi / 4 * d_wire**2 * n_wireOrTape

        return ArmourInputSchema(
            material=material,
            inner_radius=float(self.cable_specs["D_arm_int"]) / 2 / 1e3,  # convert to meters
            thickness=None,
            outer_radius=float(self.cable_specs["D_arm_ext"]) / 2 / 1e3,  # convert to meters
            conducting_surface_area=conducting_surface_area,  # convert to square meters
        )

    def get_sheath_input(self) -> SheathInputSchema:
        """Build outer sheath layer input schema.

        Returns:
            SheathInputSchema: Sheath layer schema.

        """
        material = CableSheathMaterial(self.cable_specs["Cable serving material"])

        return SheathInputSchema(
            material=material,
            inner_radius=None,
            thickness=None,
            outer_radius=float(self.cable_specs["D_ext"]) / 2 / 1e3,  # convert to meters
        )

    def get_cable_constructional_input(self) -> CableConstructionalInputSchema:
        """Compose full cable constructional schema from parsed fields.

        Returns:
            CableConstructionalInputSchema: Fully assembled constructional
                input schema.

        """
        return CableConstructionalInputSchema(
            number_of_conductors=CableConductorCount(self.cable_specs["Number of conductors"]),
            oil_duct_input=self.get_internal_oil_duct_input(),
            conductor_input=self.get_conductor_input(),
            conductor_screen_input=self.get_conductor_screen_input(),
            insulation_input=self.get_insulation_input(),
            insulation_screen_input=self.get_insulation_screen_input(),
            screen_input=self.get_screen_input(),
            bedding_input=self.get_bedding_input(),
            armour_input=self.get_armour_input(),
            sheath_input=self.get_sheath_input(),
        )


class SingleCoreSpecParser(CableSpecParser):
    """Parser for single-core cable specifications."""

    def get_internal_oil_duct_input(self) -> InternalOilDuctInputSchema | None:
        """Build optional internal oil duct schema for single-core cables.

        Returns:
            InternalOilDuctInputSchema | None: Oil duct schema when a duct is
                defined, otherwise `None`.

        """
        if self.cable_specs["d_int_cond"] > 0.0:
            return InternalOilDuctInputSchema(
                material=CableOilMaterial.Oil,
                inner_radius=None,
                thickness=None,
                outer_radius=float(self.cable_specs["d_int_cond"]) / 2 / 1e3,  # convert to meters
            )
        else:
            return None

    def get_conductor_input(self) -> ConductorInputSchema:
        """Build conductor input schema for a single-core cable.

        Returns:
            ConductorInputSchema: Conductor layer schema.

        """
        shape = CableConductorShape(self.cable_specs["Shape"])
        if shape == CableConductorShape.Round and self.cable_specs["d_int_cond"] > 0.0:
            shape = CableConductorShape.Hollow
        return ConductorInputSchema(
            material=CableConductorMaterial(self.cable_specs["Material"]),
            shape=shape,
            surface_type=CableConductorSurfaceType(self.cable_specs["Type"]),
            outer_radius=float(self.cable_specs["d_cond"]) / 2 / 1e3,  # convert to meters
            inner_radius=None,
            thickness=None,
            conducting_surface_area=float(self.cable_specs["A_cond"]) / 1e6,  # convert to square meters
        )

    def _compute_conductor_instulation_screen_thickness(self) -> float:
        """Compute conductor/insulation screen thickness from tabular fields.

        Assumes conductor screen thickness and insulation screen thickness
        are equal.

        Returns:
            float: Derived thickness in meters.

        """
        screen_inner_radius_mm = float(self.cable_specs["D_she_ext"]) / 2 - float(self.cable_specs["t_she"])
        conductor_outer_radius_mm = float(self.cable_specs["d_cond"]) / 2
        insulation_thickness_mm = float(self.cable_specs["t1"])
        return (
            (screen_inner_radius_mm - conductor_outer_radius_mm - insulation_thickness_mm) / 2
        ) / 1e3  # convert to meters

    def get_conductor_screen_input(self) -> ConductorScreenInputSchema | None:
        """Build optional conductor screen schema for single-core cables.

        Returns:
            ConductorScreenInputSchema | None: Screen schema when thickness is
                positive, otherwise `None`.

        """
        thickness = self._compute_conductor_instulation_screen_thickness()
        if thickness <= 0:
            return None

        material = CableConductorInsulationScreenMaterial(self.cable_specs["Conductor/isolation screen"])
        return ConductorScreenInputSchema(
            material=material,
            inner_radius=None,
            thickness=thickness,  # convert to meters
            outer_radius=None,
        )

    def get_insulation_input(self) -> InsulationInputSchema:
        """Build insulation input schema for a single-core cable.

        Returns:
            InsulationInputSchema: Insulation layer schema.

        """
        material = CableInsulationMaterial(self.cable_specs["Isolation material"])
        return InsulationInputSchema(
            material=material,
            nominal_phase_voltage=float(self.cable_specs["U_0"]),
            inner_radius=None,
            thickness=float(self.cable_specs["t1"]) / 1e3,  # convert to meters
            outer_radius=None,
        )

    def get_insulation_screen_input(self) -> InsulationScreenInputSchema | None:
        """Build optional insulation screen schema for single-core cables.

        Returns:
            InsulationScreenInputSchema | None: Screen schema when thickness is
                positive, otherwise `None`.

        """
        thickness = self._compute_conductor_instulation_screen_thickness()
        if thickness <= 0:
            return None

        material = CableConductorInsulationScreenMaterial(self.cable_specs["Conductor/isolation screen"])
        return InsulationScreenInputSchema(
            material=material,
            inner_radius=None,
            thickness=thickness,
            outer_radius=None,
        )


class ThreeCoreSpecParser(CableSpecParser):
    """Parser for three-core cable specifications."""

    def get_internal_oil_duct_input(self) -> InternalOilDuctInputSchema | None:
        """Return oil duct schema for three-core cables.

        Returns:
            InternalOilDuctInputSchema | None: Always `None`, because three-core
                cables are modeled without an internal oil duct layer.

        """
        return None

    def get_conductor_input(self) -> ConductorInputSchema:
        """Build conductor input schema for a three-core cable.

        Returns:
            ConductorInputSchema: Conductor layer schema.

        """
        shape = CableConductorShape(self.cable_specs["Shape"])

        return ConductorInputSchema(
            material=CableConductorMaterial(self.cable_specs["Material"]),
            shape=shape,
            surface_type=CableConductorSurfaceType(self.cable_specs["Type"]),
            outer_radius=None,
            inner_radius=None,
            thickness=None,
            conducting_surface_area=float(self.cable_specs["A_cond"]) / 1e6,  # convert to square meters
            single_conductor_radius=float(self.cable_specs["d_cond"]) / 2 / 1e3
            if shape == CableConductorShape.Round
            else None,  # convert to meters
        )

    def get_conductor_screen_input(self) -> ConductorScreenInputSchema | None:
        """Return conductor screen schema for three-core cables.

        Returns:
            ConductorScreenInputSchema | None: Always `None`; three-core
                conductor-screen handling is embedded in insulation schema.

        """
        return None

    def get_insulation_input(self) -> ThreeCoreCableInsulationInputSchema:
        """Build three-core specific insulation input schema.

        Returns:
            ThreeCoreCableInsulationInputSchema: Insulation layer schema with
                three-core specific fields.

        """
        material = CableInsulationMaterial(self.cable_specs["Isolation material"])

        if float(self.cable_specs["delta_1"]) > 0:
            conductor_screen_thickness = float(self.cable_specs["delta_1"]) / 1e3  # convert to meters
            conductor_screen_material = CableConductorInsulationScreenMaterial(
                self.cable_specs["Conductor/isolation screen"]
            )
        else:
            conductor_screen_thickness = None
            conductor_screen_material = None

        return ThreeCoreCableInsulationInputSchema(
            material=material,
            inner_radius=None,
            thickness=None,
            outer_radius=None,
            nominal_phase_voltage=float(self.cable_specs["U_0"]),
            diameter_over_stranded_conductors=float(self.cable_specs["Doga"]) / 1e3,  # convert to meters
            single_conductor_insulation_thickness=float(self.cable_specs["t1"]) / 1e3,  # convert to meters
            conductor_screen_material=conductor_screen_material,
            conductor_screen_thickness=conductor_screen_thickness,
        )

    def get_insulation_screen_input(self) -> InsulationScreenInputSchema | None:
        """Return insulation screen schema for three-core cables.

        Returns:
            InsulationScreenInputSchema | None: Always `None`; three-core
                equivalent modeling does not create a separate layer.

        """
        return None


class SpecParserFactory:
    """Factory that returns the appropriate CableSpecParser for cable specification."""

    @staticmethod
    def get_spec_parser(cable_specs: pd.Series) -> CableSpecParser:
        """Create the parser matching conductor count in cable specs.

        Args:
            cable_specs: Cable specification row from source table.

        Returns:
            CableSpecParser: Single-core or three-core parser instance.

        Raises:
            ValueError: If number of conductors is unsupported.

        """
        number_of_conductors = CableConductorCount(cable_specs["Number of conductors"])
        if number_of_conductors == CableConductorCount.One:
            return SingleCoreSpecParser(cable_specs=cable_specs)
        elif number_of_conductors == CableConductorCount.Three:
            return ThreeCoreSpecParser(cable_specs=cable_specs)
        else:
            raise ValueError(f"Unsupported number of conductors: {number_of_conductors}")
