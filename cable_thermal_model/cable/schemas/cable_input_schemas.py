# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from typing import Self

import numpy as np
from pydantic import BaseModel, Field, computed_field, model_validator

from cable_thermal_model.cable.schemas.cable_layer_input_schemas import (
    AbstractLayerInputSchema,
    ArmourInputSchema,
    BeddingInputSchema,
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
    CableConductorCount,
    CableConductorInsulationScreenMaterial,
    CableConductorShape,
    CableInsulationMaterial,
    CableLayer,
    CableScreenType,
    CableType,
)


class CableConstructionalInputSchema(BaseModel):
    """Input schema for complete cable constructional data.

    The schema validates layer-level and cross-layer geometric consistency for a
    full cable definition.
    """

    number_of_conductors: CableConductorCount = Field(..., description="Number of conductors in the cable.")

    oil_duct_input: InternalOilDuctInputSchema | None = Field(
        default=None,
        description="Optional internal oil duct layer definition.",
    )
    conductor_input: ConductorInputSchema = Field(..., description="Conductor layer definition.")
    conductor_screen_input: ConductorScreenInputSchema | None = Field(
        default=None,
        description="Optional conductor screen layer definition.",
    )
    insulation_input: InsulationInputSchema = Field(..., description="Insulation layer definition.")
    insulation_screen_input: InsulationScreenInputSchema | None = Field(
        default=None,
        description="Optional insulation screen layer definition.",
    )
    screen_input: ScreenInputSchema | None = Field(
        default=None, description="Optional metallic screen layer definition."
    )
    bedding_input: BeddingInputSchema | None = Field(default=None, description="Optional bedding layer definition.")
    armour_input: ArmourInputSchema | None = Field(default=None, description="Optional armour layer definition.")
    sheath_input: SheathInputSchema = Field(..., description="Outer sheath layer definition.")

    @computed_field  # type: ignore[misc]
    @property
    def cable_type(self) -> CableType:
        """Infer cable type from the insulation material.

        Returns:
            CableType: One of XLPE, PILC, or OilPressure.

        Raises:
            ValueError: If the insulation material cannot be mapped to a known
                cable type.

        """
        insulation_material = self.insulation_input.material
        if insulation_material in [
            CableInsulationMaterial.XLPEUnfilled,
            CableInsulationMaterial.XLPEFilled,
            CableInsulationMaterial.PPL,
        ]:
            return CableType.XLPE
        elif insulation_material == CableInsulationMaterial.PaperMassImpregnated:
            return CableType.PILC
        elif insulation_material in [
            CableInsulationMaterial.PaperOFSelfContained,
            CableInsulationMaterial.PaperOilPressure,
        ]:
            return CableType.OilPressure
        else:
            raise ValueError(f"{insulation_material} is not recognized as XLPE, PILC or Oil Pressure.")

    @property
    def layers(self) -> dict[CableLayer, AbstractLayerInputSchema]:
        """Return configured cable layers keyed by cable layer enum."""
        layers: dict[CableLayer, AbstractLayerInputSchema | None] = {
            CableLayer.InnerOilDuct: self.oil_duct_input,
            CableLayer.Conductor: self.conductor_input,
            CableLayer.ConductorScreen: self.conductor_screen_input,
            CableLayer.Insulation: self.insulation_input,
            CableLayer.InsulationScreen: self.insulation_screen_input,
            CableLayer.Screen: self.screen_input,
            CableLayer.Bedding: self.bedding_input,
            CableLayer.Armour: self.armour_input,
            CableLayer.Sheath: self.sheath_input,
        }
        return {layer_type: layer for layer_type, layer in layers.items() if layer is not None}

    def _set_and_validate_radius(self, radii: list[float | None], index: int, radius: float | None) -> bool:
        """Set and validate a radius value at an index in the radii list.

        Args:
            radii: Mutable list of resolved radii where missing values are `None`.
            index: The index of the radius to set.
            radius: The candidate radius value to set.

        Returns:
            bool: `True` if the radius value was newly set, `False` otherwise.

        Raises:
            ValueError: If a conflicting radius value is encountered.

        """
        if radius is None:
            return False
        current_radius = radii[index]
        if current_radius is None:
            radii[index] = radius
            return True
        if not np.isclose(current_radius, radius):
            raise ValueError(
                f"Inconsistent dimensions: radius ({radius}) conflicts with "
                f"previously determined radius ({current_radius})."
            )
        return False

    def get_and_validate_radii(self) -> list[float]:
        """Resolve and validate all layer interface radii.

        Returns:
            list[float]: Fully resolved list of interface radii including the
                conductor center radius at index 0.

        Raises:
            ValueError: If dimensions are inconsistent or insufficient.

        """
        radii = [0.0] + [None] * len(self.layers)

        made_progress = True
        while made_progress:
            made_progress = False
            for index, layer_input in enumerate(self.layers.values()):
                made_progress |= self._set_and_validate_radius(radii, index, layer_input.inner_radius)
                made_progress |= self._set_and_validate_radius(radii, index + 1, layer_input.outer_radius)
                layer_input.inner_radius = radii[index]
                layer_input.outer_radius = radii[index + 1]

                if (
                    isinstance(layer_input, ThreeCoreCableInsulationInputSchema)
                    and layer_input.outer_radius is not None
                    and layer_input.insulation_equivalent_radius_ratio is None
                ):
                    layer_input.insulation_equivalent_radius_ratio = (
                        self.get_or_compute_insulation_equivalent_radius_ratio()
                    )
                    made_progress |= layer_input.insulation_equivalent_radius_ratio is not None

        resolved_radii = [radius for radius in radii if radius is not None]
        if len(resolved_radii) != len(radii):
            raise ValueError("Insufficient dimension information: could not determine all layer radii.")

        return resolved_radii

    def get_outer_radii(self) -> dict[CableLayer, float]:
        """Get the outer radii of all cable layers.

        Returns a dictionary mapping each cable layer to its outer radius.
        If any outer radius is missing, this method first validates and derives
        all layer radii.

        Returns:
            dict[CableLayer, float]: Dictionary mapping cable layers to their
                outer radii in meters.

        """
        outer_radii = {
            layer: layer_input.outer_radius
            for layer, layer_input in self.layers.items()
            if layer_input.outer_radius is not None
        }
        if len(outer_radii) != len(self.layers):
            self.get_and_validate_radii()
            return self.get_outer_radii()

        return outer_radii

    def get_inner_radii(self) -> dict[CableLayer, float]:
        """Get the inner radii of all cable layers.

        Returns a dictionary mapping each cable layer to its inner radius.
        If any inner radius is missing, this method first validates and derives
        all layer radii.

        Returns:
            dict[CableLayer, float]: Dictionary mapping cable layers to their
                inner radii in meters.

        """
        inner_radii = {
            layer: layer_input.inner_radius
            for layer, layer_input in self.layers.items()
            if layer_input.inner_radius is not None
        }
        if len(inner_radii) != len(self.layers):
            self.get_and_validate_radii()
            return self.get_inner_radii()

        return inner_radii

    @model_validator(mode="after")
    def validate_cable_specs(self) -> Self:
        """Run global cable-level validations after model creation.

        Returns:
            CableConstructionalInputSchema: The validated model instance.

        Raises:
            ValueError: If number of conductors is unsupported.

        """
        if self.number_of_conductors == CableConductorCount.Three:
            self.validate_three_core_cable_specs()
        elif self.number_of_conductors == CableConductorCount.One:
            self.validate_single_core_cable_specs()
        else:
            raise ValueError(f"Unsupported number of conductors: {self.number_of_conductors.value}")

        self.get_and_validate_radii()
        return self

    def validate_three_core_cable_specs(self):
        """Validate and finalize constraints specific to three-core cables.

        Returns:
            CableConstructionalInputSchema: The validated model instance.

        Raises:
            NotImplementedError: If unsupported screen layers are provided for
                three-core cables.

        """
        self.validate_three_core_cable_insulation()

        if self.conductor_input.single_conductor_radius is None:
            self.conductor_input.single_conductor_radius = (
                self.compute_single_conductor_radius_from_conducting_surface_area()
            )

        if self.conductor_screen_input is not None or self.insulation_screen_input is not None:
            raise NotImplementedError(
                "Conductor / insulation screen layer is not supported for three core cables. For oil-filled cables, "
                "use the conductor_screen_material and conductor_screen_thickness "
                "fields in the insulation input schema "
                "to specify the conductor screen properties."
            )

        self.insulation_input.insulation_equivalent_radius_ratio = (
            self.get_or_compute_insulation_equivalent_radius_ratio()
        )

        return self

    def compute_single_conductor_radius_from_conducting_surface_area(self) -> float:
        """Compute single-conductor radius from conducting area.

        Returns:
            float: Radius in meters assuming a circular equivalent conductor.

        """
        return np.sqrt(self.conductor_input.conducting_surface_area / np.pi)

    def get_or_compute_insulation_equivalent_radius_ratio(self) -> float | None:
        """Compute equivalent insulation radius ratio for three-core cables.

        If insulation_equivalent_radius_ratio is already set, this method returns the existing value.

        Returns:
            float | None: Equivalent radius ratio, or `None` when required
                geometry is not yet available.

        """
        insulation_input = self.validate_three_core_cable_insulation()
        if insulation_input.insulation_equivalent_radius_ratio is not None:
            return insulation_input.insulation_equivalent_radius_ratio

        normalized_T1 = self.compute_normalized_lumped_sum_thermal_resistance_insulation()
        if normalized_T1 is None:
            return None

        return np.exp((normalized_T1 * 2 * np.pi) / 3)

    def validate_three_core_cable_insulation(self) -> ThreeCoreCableInsulationInputSchema:
        """Validate and return three-core insulation input schema.

        Returns:
            ThreeCoreCableInsulationInputSchema: Typed insulation input.

        Raises:
            ValueError: If insulation input is not the expected three-core
                schema.

        """
        if not isinstance(self.insulation_input, ThreeCoreCableInsulationInputSchema):
            raise ValueError(
                "Insulation input must be of type "
                "ThreeCoreCableInsulationInputSchema to validate three core "
                "cable insulation specifications."
            )
        return self.insulation_input

    def compute_normalized_lumped_sum_thermal_resistance_insulation(self) -> float | None:
        """Compute normalized lumped insulation thermal resistance for three-core cables.

        Dispatches to cable-type-specific equations for the normalized lumped
        thermal resistance $T_1$ between conductor and sheath/screen, following
        NEN-IEC 60287-2-1 and Anders (1997).

        Returns:
            float | None: Normalized $T_1$. Returns `None` when required
                insulation geometry is not yet available.

        Raises:
            NotImplementedError: If the cable configuration is not supported by
                the currently implemented equations.
            ValueError: If input data is invalid for the selected cable type.

        """
        insulation_input = self.validate_three_core_cable_insulation()

        if insulation_input.outer_radius is None:
            return None

        t1 = insulation_input.single_conductor_insulation_thickness
        dx = self.compute_single_conductor_radius_from_conducting_surface_area() * 2
        dc = (
            self.conductor_input.single_conductor_radius * 2
            if self.conductor_input.single_conductor_radius is not None
            else dx
        )

        if self.cable_type == CableType.OilPressure:
            return self.compute_normalized_lumped_sum_thermal_resistance_oil_pressure(
                insulation_input=insulation_input,
                t1=t1,
                dc=dc,
            )

        # SL-type XLPE cables
        if self.screen_input and self.screen_input.screen_type == CableScreenType.SL:
            return self.compute_normalized_lumped_sum_thermal_resistance_xlpe_sl(
                t1=t1,
                dc=dc,
            )

        # PILC and XLPE cables are assumed to be belted.
        # See section 4.1.2.2 in NEN-IEC 60287-2-1 (2015)
        da = insulation_input.outer_radius * 2  # Diameter over insulation
        r1 = (da / 2) - t1  # Radius of the circle circumscribing the conductors
        t_belt = (da - insulation_input.diameter_over_stranded_conductors) / 2  # Thickness of belt insulation
        t_cond = t1 - t_belt  # Thickness of conductor insulation
        t = 2 * t_cond  # Thickness of insulation between 2 conductors

        if self.cable_type == CableType.PILC:
            return self.compute_normalized_lumped_sum_thermal_resistance_pilc(
                dx=dx,
                da=da,
                r1=r1,
                t=t,
            )

        # Common screen or unscreened XLPE Cables
        if (
            self.cable_type == CableType.XLPE
            and (self.screen_input is None or self.screen_input.screen_type == CableScreenType.Common)
            and self.conductor_input.shape == CableConductorShape.Round
        ):
            return self.compute_normalized_lumped_sum_thermal_resistance_xlpe_common_or_unscreened(
                t1=t1,
                dc=dc,
                t=t,
            )

        # Raise a not implemented error if the cable type is not recognized
        raise NotImplementedError(
            f"Thermal resistance calculation for cable with sheath configuration "
            f"{self.screen_input.screen_type if self.screen_input is not None else None} "
            f"and conductor shape {self.conductor_input.shape} is not implemented."
        )

    def compute_normalized_lumped_sum_thermal_resistance_oil_pressure(
        self,
        insulation_input: ThreeCoreCableInsulationInputSchema,
        t1: float,
        dc: float,
    ) -> float:
        """Compute normalized $T_1$ for three-core oil-pressure cables.

        Args:
            insulation_input: Three-core insulation input including optional
                conductor screen data.
            t1: Single-conductor insulation thickness in meters.
            dc: Conductor diameter in meters.

        Returns:
            float: Normalized lumped thermal resistance for the oil-pressure
                branch.

        Raises:
            ValueError: If conductor screen material is not supported for
                oil-pressure cable equations.

        """
        delta = (
            insulation_input.conductor_screen_thickness
            if insulation_input.conductor_screen_thickness is not None
            else 0.0
        )
        ti = t1 + delta

        if insulation_input.conductor_screen_material == CableConductorInsulationScreenMaterial.MetalizedPaper:
            # Eqn. (9.9) in Anders for oil pressure cables with metalized paper screen.
            return 0.385 * ((2 * ti) / (dc + 2 * ti))

        if insulation_input.conductor_screen_material == CableConductorInsulationScreenMaterial.MetalTape:
            # NEN-IEC60287-2-1 (2015), par (4.1.2.4.2) for metal tape screens.
            # NOTE: Formula 9.10 in Anders is incorrect.
            return 0.35 * (0.923 - (dc / (dc + 2 * ti)))

        raise ValueError(
            f"Invalid conductor screen material for oil pressure cable: "
            f"{insulation_input.conductor_screen_material}. Expected either "
            "MetalizedPaper or MetalTape."
        )

    def compute_normalized_lumped_sum_thermal_resistance_xlpe_sl(self, t1: float, dc: float) -> float:
        """Compute normalized $T_1$ for SL-type XLPE cables.

        Args:
            t1: Single-conductor insulation thickness in meters.
            dc: Conductor diameter in meters.

        Returns:
            float: Normalized lumped thermal resistance for SL-type XLPE.

        """
        return 1 / (2 * np.pi) * np.log(1 + (2 * t1) / dc)

    def compute_normalized_lumped_sum_thermal_resistance_pilc(
        self,
        dx: float,
        da: float,
        r1: float,
        t: float,
    ) -> float:
        """Compute normalized $T_1$ for PILC cables.

        Args:
            dx: Equivalent diameter of one conductor in meters.
            da: Diameter over insulation in meters.
            r1: Radius of the circle circumscribing the conductors in meters.
            t: Insulation thickness between two conductors in meters.

        Returns:
            float: Normalized lumped thermal resistance for the PILC branch.

        """
        # PILC cables are assumed to have sector shaped conductors.
        # The T1 is calculated using section (4.1.2.2.6) in NEN-IEC 60287-2-1 (2015)
        F2 = 1 + ((3 * t) / (2 * np.pi * (dx + t) - t))
        G = 3 * F2 * np.log(da / (2 * r1))
        return G / (2 * np.pi)

    def compute_normalized_lumped_sum_thermal_resistance_xlpe_common_or_unscreened(
        self,
        t1: float,
        dc: float,
        t: float,
    ) -> float:
        """Compute normalized $T_1$ for common-screen or unscreened XLPE cables.

        Uses the quadratic approximation for the geometric factor in
        NEN-IEC 60287-2-1 (2015), section 5.3.

        Args:
            t1: Single-conductor insulation thickness in meters.
            dc: Conductor diameter in meters.
            t: Insulation thickness between two conductors in meters.

        Returns:
            float: Normalized lumped thermal resistance for the XLPE
                common/unscreened branch.

        """
        # Use the quadratic approximation method to determine the geometric
        # factor as described in section 5.3 of NEN-IEC60287-2-1 (2015).
        X = t1 / dc
        Y = (2 * t1 / t) - 1

        Z = (2 / np.sqrt(3)) * (1 + 2 * X / (1 + Y))

        alpha = 1 / (1 + 2 * X / (1 + Z)) ** 3
        beta = alpha * (Z - 3) / (Z + 3)

        M = np.log((1 - alpha * beta + ((1 - alpha**2) * (1 - beta**2)) ** 0.5) / (alpha - beta))

        a = 1.09414 - 0.0944045 * X + 0.0234464 * X**2
        b = 1.09605 - 0.0801857 * X + 0.0176917 * X**2
        c = 1.09831 - 0.0720631 * X + 0.0145909 * X**2

        G = M * (a + (-3 * a + 4 * b - c) * Y + (2 * a - 4 * b + 2 * c) * Y**2)

        # The T1 is then calculated using section (4.1.2.2.4) in NEN-IEC 60287-2-1 (2015)
        # NOTE: Here the thermal resistivity of the filler material is assumed
        # to be equal to the thermal resistivity of the insulation material.
        return G / (2 * np.pi)

    def validate_single_core_cable_specs(self):
        """Validate constraints specific to single-core cable definitions.

        Raises:
            ValueError: If single-conductor radius and conductor outer radius
                are inconsistent, or if sector conductors are provided.
            NotImplementedError: If armour is present on single-core cables.

        """
        self.validate_single_core_cable_insulation()

        if self.conductor_input.single_conductor_radius is not None and not np.isclose(
            self.conductor_input.single_conductor_radius, self.conductor_input.outer_radius
        ):
            raise ValueError("For single core cables, single_conductor_radius must equal the conductor outer_radius.")

        if self.conductor_input.shape == CableConductorShape.Sector:
            raise ValueError("Single-core cables can not have sector shaped conductors.")

        if self.armour_input is not None:
            raise NotImplementedError(
                "Armour losses are not accounted for in the FD model. "
                "Therefore, armoured cables are not supported for single-core cables."
            )

    def validate_single_core_cable_insulation(self) -> InsulationInputSchema:
        """Validate and return single-core insulation input schema.

        Returns:
            InsulationInputSchema: Typed insulation input.

        Raises:
            ValueError: If insulation input is not the expected single-core
                schema.

        """
        if not isinstance(self.insulation_input, InsulationInputSchema):
            raise ValueError(
                "Insulation input must be of type InsulationInputSchema to "
                "validate single core cable insulation specifications."
            )
        return self.insulation_input
