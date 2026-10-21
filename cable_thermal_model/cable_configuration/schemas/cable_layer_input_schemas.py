# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cable_thermal_model.model.cables.enum_classes_cable import (
    CableArmourMaterial,
    CableBeddingMaterial,
    CableConductorInsulationScreenMaterial,
    CableConductorMaterial,
    CableConductorShape,
    CableConductorSurfaceType,
    CableInsulationMaterial,
    CableOilMaterial,
    CableScreenMaterial,
    CableScreenType,
    CableSheathMaterial,
    Material,
)


class AbstractLayerInputSchema(BaseModel):
    """Base input schema for a generic cable layer.

    The schema stores geometric properties for a single radial cable layer and
    derives missing values when enough dimensional information is available.
    """

    model_config = ConfigDict(validate_assignment=True)

    material: Material = Field(..., description="Material type of the cable layer.")
    inner_radius: float | None = Field(default=None, description="Inner radius of the layer in meters.")
    thickness: float | None = Field(default=None, description="Layer thickness in meters.")
    outer_radius: float | None = Field(default=None, description="Outer radius of the layer in meters.")

    def calculate_surface_area(self) -> float:
        """Calculate the annular surface area of the layer.

        Returns:
            float: Layer annular area computed as pi * (r_o^2 - r_i^2).

        Raises:
            ValueError: If inner or outer radius is not defined.

        """
        if self.outer_radius is None or self.inner_radius is None:
            raise ValueError("Cannot calculate surface area: inner_radius and outer_radius must be defined.")
        return np.pi * (self.outer_radius**2 - self.inner_radius**2)

    @field_validator("thickness", mode="after")
    @classmethod
    def validate_thickness(cls, thickness: float | None = None) -> float | None:
        """Validate that thickness is strictly positive when provided.

        Args:
            thickness: Optional thickness value to validate.

        Returns:
            float | None: The validated thickness.

        Raises:
            ValueError: If thickness is zero or negative.

        """
        if thickness is not None and thickness <= 0:
            raise ValueError(f"Invalid layer thickness: {thickness}, must be strictly positive.")
        return thickness

    @model_validator(mode="after")
    def validate_inner_radius(self) -> Self:
        """Validate that the inner radius is non-negative.

        Returns:
            _AbstractLayerInputSchema: The validated model instance.

        Raises:
            ValueError: If inner_radius is negative.

        """
        inner_radius = self.inner_radius
        if inner_radius is not None and inner_radius < 0:
            raise ValueError(f"Invalid inner_radius: {inner_radius} must be non-negative.")
        return self

    @model_validator(mode="after")
    def validate_dimension_consistency(self) -> Self:
        """Validate and derive consistent radial dimensions.

        Returns:
            _AbstractLayerInputSchema: The validated model instance.

        Raises:
            ValueError: If provided dimensions are physically inconsistent.

        """
        inner_radius = self.inner_radius
        outer_radius = self.outer_radius
        thickness = self.thickness
        if inner_radius is not None and outer_radius is not None:
            if outer_radius <= inner_radius:
                raise ValueError(
                    f"Invalid dimensions: outer_radius ({outer_radius}) must "
                    f"be greater than inner_radius ({inner_radius})."
                )
            if thickness is not None:
                if not np.isclose(outer_radius, inner_radius + thickness):
                    raise ValueError(
                        f"Inconsistent dimensions: outer_radius ({outer_radius}) "
                        f"must equal inner_radius ({inner_radius}) + thickness "
                        f"({thickness})."
                    )
            else:
                self.thickness = outer_radius - inner_radius
        elif thickness is not None and inner_radius is not None:
            self.outer_radius = inner_radius + thickness
        elif thickness is not None and outer_radius is not None:
            self.inner_radius = outer_radius - thickness

        return self

    @field_validator("material", mode="after")
    @classmethod
    def validate_layer_material(cls, material: Material) -> Material:
        """Validate that a layer material is not explicitly set to `NONE`.

        Args:
            material: Layer material enum value.

        Returns:
            Material: The validated material.

        Raises:
            ValueError: If the material enum exposes a `NONE` value and that
                value is used.

        """
        material_type = type(material)
        if hasattr(material_type, "NONE") and material == material_type.NONE:
            raise ValueError(f"Invalid material: {material} is not a valid material for a cable layer.")
        return material


class ConductingLayerInputSchema(AbstractLayerInputSchema):
    """Input schema for conducting layers with optional conducting area."""

    conducting_surface_area: float = Field(
        description="Effective electrically conducting cross-sectional area of the layer in square meters.",
    )

    @model_validator(mode="after")
    def validate_conducting_surface_area(self) -> Self:
        """Validate that conducting surface area does not exceed layer area.

        Returns:
            _ConductingLayerInputSchema: The validated model instance.

        Raises:
            ValueError: If conducting_surface_area exceeds calculated area.

        """
        if self.conducting_surface_area <= 0:
            raise ValueError(
                f"Invalid conducting_surface_area: {self.conducting_surface_area}, must be strictly positive."
            )
        if self.outer_radius is not None and self.inner_radius is not None:
            calculated_surface_are = self.calculate_surface_area()
            if self.conducting_surface_area > calculated_surface_are * 1.15:  # allow 15% tolerance.
                raise ValueError(
                    f"Invalid conducting_surface_area: "
                    f"({self.conducting_surface_area}) exceeds the calculated "
                    f"layer surface area ({calculated_surface_are}). "
                    "Provided conducting surface area does not physically fit within the layer dimensions."
                )
        return self


class InternalOilDuctInputSchema(AbstractLayerInputSchema):
    """Input schema for internal oil duct layer properties."""

    material: CableOilMaterial = Field(..., description="Material of the oil in the internal oil duct.")


class ConductorInputSchema(ConductingLayerInputSchema):
    """Input schema for conductor layer properties."""

    shape: CableConductorShape = Field(..., description="Geometric shape of the conductor.")
    surface_type: CableConductorSurfaceType = Field(..., description="Conductor surface construction type.")
    single_conductor_radius: float | None = Field(
        default=None,
        description="Radius of a single conductor core in meters.",
    )
    material: CableConductorMaterial = Field(..., description="Conductor material.")

    @model_validator(mode="after")
    def validate_single_conductor_radius(self) -> Self:
        """Validate that conducting surface area does not exceed area calculated from single conductor radius.

        Returns:
            ConductorInputSchema: The validated model instance.

        Raises:
            ValueError: If single_conductor_radius is provided and
                conducting_surface_area exceeds the area calculated from it.

        """
        if self.single_conductor_radius is not None:
            if self.single_conductor_radius <= 0:
                raise ValueError(
                    f"Invalid single_conductor_radius: {self.single_conductor_radius}, must be strictly positive."
                )
            single_conductor_area = np.pi * self.single_conductor_radius**2
            if self.conducting_surface_area > single_conductor_area * 1.15:  # allow 15% tolerance.
                raise ValueError(
                    f"Invalid conducting_surface_area: "
                    f"{self.conducting_surface_area} exceeds the area "
                    f"calculated from single_conductor_radius "
                    f"({single_conductor_area}). "
                    "Provided conducting surface area does not physically fit "
                    "within the area defined by the single conductor radius."
                )

        return self

    @model_validator(mode="after")
    def validate_conductor_shape_and_radius(self) -> Self:
        """Validate that if inner_radius is not zero, the conductor shape is hollow.

        Returns:
            ConductorInputSchema: The validated model instance.

        Raises:
            ValueError: If inner_radius is greater than zero and shape is not hollow.

        """
        if self.inner_radius is not None and self.inner_radius > 0 and self.shape != CableConductorShape.Hollow:
            raise ValueError(
                f"Inconsistent conductor geometry: inner_radius is "
                f"{self.inner_radius} but shape is {self.shape}, expected "
                f"{CableConductorShape.Hollow} for non-zero inner radius."
            )

        if self.inner_radius is not None and self.inner_radius <= 0.0 and self.shape == CableConductorShape.Hollow:
            raise ValueError(
                f"Inconsistent conductor geometry: shape is {self.shape} but "
                f"inner_radius is {self.inner_radius}, expected inner_radius > "
                "0 for hollow conductor shape."
            )

        return self


class ConductorScreenInputSchema(AbstractLayerInputSchema):
    """Input schema for conductor screen layer properties."""

    material: CableConductorInsulationScreenMaterial = Field(..., description="Conductor screen material.")


class InsulationInputSchema(AbstractLayerInputSchema):
    """Input schema for insulation layer properties."""

    nominal_phase_voltage: float = Field(..., description="Nominal phase voltage in volts.")
    material: CableInsulationMaterial = Field(..., description="Insulation material.")


class ThreeCoreCableInsulationInputSchema(InsulationInputSchema):
    """Input schema for insulation layer properties specific to three core cables."""

    diameter_over_stranded_conductors: float = Field(
        description="Diameter over stranded conductors in meters for multi-core configurations.",
    )
    single_conductor_insulation_thickness: float = Field(
        description=(
            "Thickness of the insulation in meters for multi-core "
            "configurations. This distance includes thickness of belt "
            "insulation and thicknesses of (optional) conductor screen and "
            "insulation screen."
        ),
    )

    conductor_screen_material: CableConductorInsulationScreenMaterial | None = Field(
        default=None,
        description=(
            "Material of the conductor screen for three cores. Relevant for the T1 calculation of oil-filled cables."
        ),
    )
    conductor_screen_thickness: float | None = Field(
        default=None,
        description=(
            "Thickness of the conductor screen for three cores in meters. "
            "Relevant for the T1 calculation of oil-filled cables."
        ),
    )

    insulation_equivalent_radius_ratio: float | None = Field(
        default=None,
        description=(
            "Equivalent radius ratio for the insulation layer in three core "
            "cables, computed from the lumped sum thermal resistance (T1) "
            "between conductor and sheath/screen according to NEN-IEC "
            "60287-2-1."
        ),
    )

    @model_validator(mode="after")
    def compute_inner_radius(self) -> Self:
        """Compute or validate the equivalent inner insulation radius.

        When both `outer_radius` and `insulation_equivalent_radius_ratio` are
        available, this method derives `inner_radius` and validates consistency
        with a user-provided value.

        Returns:
            ThreeCoreCableInsulationInputSchema: The validated model instance.

        Raises:
            ValueError: If the derived inner radius conflicts with an already
                provided inner radius.

        """
        if self.outer_radius is not None and self.insulation_equivalent_radius_ratio is not None:
            current_inner_radius = self.inner_radius
            new_inner_radius = self.outer_radius / self.insulation_equivalent_radius_ratio
            if current_inner_radius is not None:
                if not np.isclose(current_inner_radius, new_inner_radius):
                    raise ValueError(
                        "Inconsistent dimensions: equivalent inner insulation "
                        "radius conflicts with value derived from outer_radius "
                        "and radius ratio derived from T1 calculation."
                    )
            else:
                self.inner_radius = new_inner_radius

        return self

    @field_validator("diameter_over_stranded_conductors", mode="after")
    @classmethod
    def validate_diameter_over_stranded_conductors(cls, diameter_over_stranded_conductors: float) -> float:
        """Validate the stranded-conductor diameter value.

        Args:
            diameter_over_stranded_conductors: Diameter over stranded
                conductors in meters.

        Returns:
            float: The validated diameter.

        Raises:
            ValueError: If the diameter is not strictly positive.

        """
        if diameter_over_stranded_conductors <= 0:
            raise ValueError(
                f"Invalid diameter_over_stranded_conductors: "
                f"{diameter_over_stranded_conductors}, must be strictly positive."
            )
        return diameter_over_stranded_conductors

    @field_validator("single_conductor_insulation_thickness", mode="after")
    @classmethod
    def validate_single_conductor_insulation_thickness(cls, single_conductor_insulation_thickness: float) -> float:
        """Validate the single-conductor insulation thickness.

        Args:
            single_conductor_insulation_thickness: Insulation thickness from
                conductor to screen in meters.

        Returns:
            float: The validated insulation thickness.

        Raises:
            ValueError: If thickness is not strictly positive.

        """
        if single_conductor_insulation_thickness <= 0:
            raise ValueError(
                f"Invalid single_conductor_insulation_thickness: "
                f"{single_conductor_insulation_thickness}, must be strictly positive."
            )
        return single_conductor_insulation_thickness


class InsulationScreenInputSchema(AbstractLayerInputSchema):
    """Input schema for insulation screen layer properties."""

    material: CableConductorInsulationScreenMaterial = Field(..., description="Insulation screen material.")


class ScreenInputSchema(ConductingLayerInputSchema):
    """Input schema for metallic screen layer properties."""

    screen_type: CableScreenType = Field(..., description="Type of metallic cable screen.")
    material: CableScreenMaterial = Field(..., description="Screen material.")

    @field_validator("screen_type", mode="after")
    @classmethod
    def validate_screen_type(cls, screen_type: CableScreenType) -> CableScreenType:
        """Validate supported screen types.

        Args:
            screen_type: Screen type enum value.

        Returns:
            CableScreenType: The validated screen type.

        Raises:
            NotImplementedError: If `scSL` is used, which is currently not
                supported by the model.

        """
        if screen_type == CableScreenType.SL:
            raise NotImplementedError(f"Screen type {CableScreenType.SL} is not supported by the model.")
        return screen_type


class BeddingInputSchema(AbstractLayerInputSchema):
    """Input schema for bedding layer properties."""

    material: CableBeddingMaterial = Field(..., description="Bedding material.")


class ArmourInputSchema(ConductingLayerInputSchema):
    """Input schema for armour layer properties."""

    material: CableArmourMaterial = Field(..., description="Armour material.")


class SheathInputSchema(AbstractLayerInputSchema):
    """Input schema for sheath layer properties."""

    material: CableSheathMaterial = Field(..., description="Outer sheath material.")
