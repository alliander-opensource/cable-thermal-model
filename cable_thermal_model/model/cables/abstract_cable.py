# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

import logging

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from cable_thermal_model.model.cables.enum_classes_cable import (
    CableConductorCount,
    CableConductorMaterial,
    CableConductorShape,
    CableConductorSurfaceType,
    CableInsulationMaterial,
    CableLayer,
    CableScreenLossType,
    CableType,
)
from cable_thermal_model.model.cables.pipe import Pipe
from cable_thermal_model.utils.exceptions import MissingAttributeException

ELECTRIC_RESISTANCE_REFERENCE_TEMPERATURE = 20.0  # degrees Celsius

logger = logging.getLogger(__name__)


class CableConductorProperties(BaseModel):
    """This dataclass is responsible for storing physical cable conductor properties.

    Relevant sizes, materials and other physical properties not addressed at a layer level will be stored in this
    dataclass.

    Attributes:
        number_of_conductors (CableConductorCount): A CableConductorCount object representing the number of
            conductors in the cable.
        shape (CableConductorShape): A CableConductorShape object representing the shape of the cable's conductors.
        material (CableConductorMaterial): A CableConductorMaterial object representing the material of the cable's
            conductors.
        surface_type (CableConductorSurfaceType): A CableConductorSurfaceType object representing the surface type
            of the cable's conductors.

    """

    # Count
    number_of_conductors: CableConductorCount

    # Build
    shape: CableConductorShape
    material: CableConductorMaterial
    surface_type: CableConductorSurfaceType

    # Pydantic class configuration
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)


class CableConductorType(BaseModel):
    """A backup dataclass for the original ConductorType class used for CableSpecParser.

    TODO: CableSpecParser should be updated to use the CableConductorProperties class instead of this one.

    """

    shape: CableConductorShape
    material: CableConductorMaterial
    surface_type: CableConductorSurfaceType


class WeightedScreenImpedance(BaseModel):
    """Weighted screen impedance data for multi-cable circuit loss calculations."""

    weighted_reactance_matrix: np.ndarray
    weighted_resistance_factor: float

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)


class CableLayerProperties(BaseModel):
    """A dataclass aimed at supplying cable layered properties.

    This dataclass holds the physical properties for a single cable layer.

    Attributes:
        layer (CableLayer): The type of cable layer this object represents.
        inner_radius (float): The inner radius of the cable layer (in meters).
        outer_radius (float): The outer radius of the cable layer (in meters).
        rho (float): The thermal resistivity value of the layer (in mK/W).
        capacity (float): The thermal capacity of the layer (in J/(m³K)).
        electric_rho (float): The electric resistivity value of the layer (in Ohm·m).
        alpha (float): The electrical resistivity temperature coefficient of the layer.
        epsilon (float): The relative permittivity value of the layer.
        tan_delta (float): The dissipation factor value of the layer.

    """

    layer: CableLayer = Field(title="Cable layer type")
    inner_radius: float = Field(title="Inner radius")
    outer_radius: float = Field(title="Outer radius")
    rho: float = Field(title="Thermal resistivity value")
    capacity: float = Field(title="Thermal capacity")
    electric_rho: float = Field(title="Electric resistivity value", default=0.0)
    alpha: float = Field(title="Electrical resistivity temperature coefficient", default=0.0)
    epsilon: float = Field(title="Relative permittivity value", default=0.0)
    tan_delta: float = Field(title="Dissipation factor value", default=0.0)
    # Pydantic class configuration
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)


class CableLayerMetrics(BaseModel):
    """The dataclass holding the remaining physical properties of a cable.

    After the CableConductorProperties and CableLayerProperties dataclasses, this dataclass holds the remaining
    physical properties remaining for a cable.

    Notes:
        It may be good idea to change the name and / or the contents of this class in the nearby future.
        As of now there is no proper identity for this class, outside of serving as a method of preventing a large
        number of AbstractCable class properties from having to be defined within that class itself.


    Attributes:
        conductor_centers (list[tuple[float,float]]): A list of (x,y) locations indicating the position of each
            conductor, relative to the cable's center.
        conductor_cross_section (float): A float value representing the conductor's cross-section in meters squared
            (m2).
        conductor_virtual_cross_section (float): A float value representing a virtual cross-section in meters squared
            (m2), which can be used to  compute an IEC compatible conductor resistance.
        conductor_equivalent_outer_diameter (float): An optional float value representing the outer diameter of an
            equivalent solid conductor with the same central duct as the original situation (in mm). Used to represent
            the conductor situation as a single solid conductor in situations with multiple conductors.
        conductor_distance (float): An optional float, applicable for situations with three conductors, where this
            value represents distance between the conductor centers.
        screen_cross_section (float | None): A float representing the cable screen's cross-section size.
        screen_loss_type (CableScreenLossType | None): A CableScreenLossType object representing the type of screen
            loss the cable has. This value is used to determine the proper method for calculating the screen loss.
        armour_cross_section (float | None): A float representing the cross-section size of the cable's armour.
        diameter_over_stranded_conductors (float): A float representing the cable's diagonal spanning the conductors'
            outermost borders.
        nominal_phase_voltage (float): A float representing the cable's phase voltage.
        sector_radius (float): An optional float representing cable sector radius.
        outer_radius (float): A float representing the cable's outer radius.
        core_to_sector_distance (float): An optional float representing the distance from the cable's core to its
            sector.
        original_insulation_thickness (float): An optional float representing the original t1 value of the cable.
        load_series (np.ndarray): An optional np.ndarray representing the cable's load series.
        pipe (Pipe): An optional Pipe object representing the cable's pipe, if there is one.

    """

    # conductor
    conductor_centers: list[tuple[float, float]] = [(0.0, 0.0)]
    conductor_cross_section: float
    conductor_equivalent_outer_diameter: float | None = None
    conductor_radius_original: float = Field(title="The original radius of the conductor")
    conductor_virtual_cross_section: float = Field(title="The virtual cross-section of the conductor")
    conductor_distance: float | None = None

    # Screen
    screen_cross_section: float | None = Field(title="The cross-section of the earth screen layer")
    screen_loss_type: CableScreenLossType | None = None

    # Armour
    armour_cross_section: float | None = Field(title="The cross-section of the armour layer")

    # Other
    diameter_over_stranded_conductors: float | None = Field(
        title="The diagonal spanning the insulation layers outermost borders"
    )
    nominal_phase_voltage: float = Field(title="The phase voltage of the cable")
    sector_radius: float | None = None
    outer_radius: float = Field(
        title="The outer radius of the cable object including pipe if present",
        description="If no pipe is present, outer_radius equals cable_radius",
    )
    cable_radius: float = Field(title="The radius of the cable (without pipe)")
    core_to_sector_distance: float | None = None
    original_insulation_thickness: float | None = None
    insulation_material: CableInsulationMaterial = Field(title="Cable insulation material")
    load_series: np.ndarray | None = None
    pipe: Pipe | None = None

    # Pydantic class configuration
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)


class CableConvectionParams(BaseModel):
    """Convection parameters used to model heat transfer in air-installed cables."""

    Z: float = Field()
    E: float = Field()
    Cg: float = Field()


class AbstractCable:
    """The Abstract base class for cables.

    This class represents all physical qualities of a single cable, outside of those specific to FD cables.
    Abstract cables will be extended by the FDCable class.

    Attributes:
        conductor (CableConductorProperties): A CableConductorProperties object containing the cable's intended
            conductor properties.
        layer_properties (CableLayerProperties): A CableLayerProperties object containing the cable's intended layer
            properties.
        layer_metrics (CableLayerMetrics): A CableLayerMetrics object containing the cable's intended layer metrics
            values.
        cable_type (CableType): A CableType object containing the cable's intended type.

    """

    # Constants
    _M_THRESHOLD = 0.1
    _X_P_THRESHOLD = 2.8

    def __init__(
        self,
        conductor: CableConductorProperties,
        layer_properties: dict[CableLayer, CableLayerProperties],
        layer_metrics: CableLayerMetrics,
        cable_type: CableType,
    ):
        """Initialize an abstract cable with its conductor, layer, and metric properties.

        Args:
            conductor (CableConductorProperties): Conductor properties of the cable.
            layer_properties (dict[CableLayer, CableLayerProperties]): Mapping of cable layers to their properties.
            layer_metrics (CableLayerMetrics): Geometric and calculated metrics for the cable layers.
            cable_type (CableType): The type of the cable.

        """
        if not isinstance(conductor, CableConductorProperties):
            raise TypeError(
                f"Expected [conductor] to be of type [CableConductorProperties], but got [{type(conductor)}]"
            )
        self.conductor = conductor

        if not isinstance(layer_properties, dict):
            raise TypeError(f"Expected [layer_properties] to be of type dict, but got [{type(layer_properties)}]")
        self.layer_properties = layer_properties

        if not isinstance(layer_metrics, CableLayerMetrics):
            raise TypeError(
                f"Expected [layer_metrics] to be of type [CableLayerMetrics], but got [{type(layer_metrics)}]"
            )
        self.layer_metrics = layer_metrics

        if not isinstance(cable_type, CableType):
            raise TypeError(f"Expected [cable_type] to be of type [CableType], but got [{type(cable_type)}]")
        self.cable_type = cable_type

        if self.conductor.number_of_conductors == CableConductorCount.Three:
            distance = self.layer_metrics.conductor_distance

            if not distance:
                raise MissingAttributeException(
                    "Trying to instantiate 3-core cable without specified conductor distance."
                )

            y_dif = distance * 3**0.5 / 2
            y_centerA = distance / 3**0.5
            y_centerBC = y_centerA - y_dif
            centerA = (0, y_centerA)
            centerB = (-distance / 2, y_centerBC)
            centerC = (distance / 2, y_centerBC)
            self.layer_metrics.conductor_centers = [centerA, centerB, centerC]

        self.weighted_screen_impedance: WeightedScreenImpedance | None = None

    @property
    def layers(self) -> list[CableLayer]:
        """A shorthand method for retrieving the list of layers of the cable."""
        return list(self.layer_properties.keys())

    @property
    def layer_count(self) -> int:
        """A shorthand method for retrieving the number of layers of the cable."""
        return len(self.layer_properties)

    @property
    def non_soil_layer_count(self) -> int:
        """A shorthand method for retrieving the number of layers of the cable, excluding the soil layer."""
        return len([layer for layer in self.layer_properties if layer not in CableLayer.soil_layers()])

    @property
    def omega(self) -> float:
        """Angular frequency at 50 Hz in rad/s."""
        return 2 * np.pi * 50

    @property
    def electric_rho_screen(self) -> float:
        """Electrical resistivity of the screen layer in Ohm·m."""
        return self.layer_properties[CableLayer.Screen].electric_rho

    @property
    def alpha_screen(self) -> float:
        """Temperature coefficient of electrical resistivity for the screen layer."""
        return self.layer_properties[CableLayer.Screen].alpha

    @property
    def c(self) -> float:
        """Distance between the conductors in a three-core cable."""
        return np.sqrt(
            self.layer_metrics.conductor_centers[0][0] ** 2 + self.layer_metrics.conductor_centers[0][1] ** 2
        )

    @property
    def d(self) -> float:
        """Average diameter of the earth screen layer."""
        return (
            self.layer_properties[CableLayer.Screen].outer_radius
            + self.layer_properties[CableLayer.Screen].inner_radius
        )

    @property
    def s(self) -> float:
        """Centre-to-centre distance between conductors in metres."""
        if self.layer_metrics.conductor_distance is None:
            raise ValueError("Conductor distance is not set.")
        return self.layer_metrics.conductor_distance

    @property
    def Ft(self) -> float:
        """Armour loss factor (1 when no armour is present)."""
        return self._get_armour_factor(self.d) if CableLayer.Armour in self.layers else 1

    @property
    def X(self) -> float:
        """Reactance factor for proximity effect calculations."""
        return 2 * self.omega * 1e-7 * np.log(2 * self.s / self.d)

    @property
    def Xm(self) -> float:
        """Mutual reactance factor used in screen loss calculations."""
        return 2 * self.omega * 10**-7 * np.log(2)

    def get_ac_resistance_conductor(self, Tc: float, s: float) -> float:
        """Computes the AC resistance of the conductor based on its temperature and distance to other conductors.

        Args:
            Tc (float): The conductor temperature in degrees Celsius.
            s (float): The distance between the conductors in meters.

        Returns:
            float: The AC resistance of the conductor in Ohm/m.

        """
        Rdc = self.get_dc_resistance_conductor(Tc)
        return self._get_ac_resistance_conductor_from_dc_resistance(Rdc, s)

    def _get_ac_resistance_conductor_from_dc_resistance(self, Rdc: float, s: float) -> float:
        """Computes the AC resistance of the conductor based on the DC resistance and conductor distance.

        Args:
            Rdc (float): The DC resistance of the conductor in Ohm/m at the operating temperature.
            s (float): The distance between the conductors in meters.

        Returns:
            float: The AC resistance of the conductor in Ohm/m.

        References:
            - IEC 60287-1-1:2023

        """
        y_s = self.get_skin_effect_factor(Rdc=Rdc)
        y_p = self.get_proximity_effect_factor(Rdc=Rdc, s=s)

        # According to guidance point 25 of Cigré TB 880, a factor of 1.5 should be applied to the skin and proximity
        # effect when armour is present, to account for the additional losses in the conductor caused by the armour.
        ac_resistance_factor = 1.5 if CableLayer.Armour in self.layers else 1.0
        return (1 + ac_resistance_factor * (y_s + y_p)) * Rdc

    def get_skin_effect_factor(self, Rdc: float) -> float:
        """Compute skin effect factor y_s from DC resistance.

        Args:
            Rdc (float): The DC resistance of the conductor in Ohm/m at the operating temperature.

        Returns:
            float: The skin effect factor y_s.

        References:
            - IEC 60287-1-1 - section 5.1.3

        """
        x_coefficient = self._calculate_x_coefficient(Rdc)
        k_s, _ = self._select_k_s_and_k_p()
        x_s = (x_coefficient * k_s) ** 0.5
        return type(self)._calculate_y_s_from_x_s(x_s=x_s)

    def get_proximity_effect_factor(self, Rdc: float, s: float) -> float:
        """Compute proximity effect factor y_p from DC resistance.

        Args:
            Rdc (float): The DC resistance of the conductor in Ohm/m at the operating temperature.
            s (float): The distance between the conductors in meters.

        Returns:
            float: The proximity effect factor y_p.

        References:
            - IEC 60287-1-1 - section 5.1.5

        """
        x_coefficient = self._calculate_x_coefficient(Rdc)
        _, k_p = self._select_k_s_and_k_p()

        conductor_diameter = 2 * self.layer_metrics.conductor_radius_original
        x_p = (x_coefficient * k_p) ** 0.5
        if x_p > self._X_P_THRESHOLD:
            logger.warning(
                f"The value for [x_p] is greater than [{self._X_P_THRESHOLD}]. Faults may occur when calculating [y_p]!"
            )

        if self.conductor.shape in [CableConductorShape.Round, CableConductorShape.Hollow]:
            return self._calculate_y_p_from_x_p(d=conductor_diameter, s=s, x_p=x_p, factor=1.0)

        if self.conductor.shape == CableConductorShape.Sector:
            return self._calculate_y_p_from_x_p(d=conductor_diameter, s=conductor_diameter + s, x_p=x_p, factor=2 / 3)

        raise ValueError(f"Conductor shape [{self.conductor.shape}] not recognized. Cannot calculate [y_p].")

    def _calculate_x_coefficient(self, Rdc: float) -> float:
        """Calculates the coefficient occurring in the x_s and x_p calculations.

        Args:
            Rdc (float): The DC resistance of the conductor in Ohm/m at the operating temperature.

        Returns:
            float: The calculated value for the x coefficient.

        References:
            - NEN-IEC 60287-1-1:2023 - section 5.1.3 and 5.1.4

        """
        return 4 * self.omega / Rdc * 1e-7

    @staticmethod
    def _calculate_y_p_from_x_p(d: float, s: float, x_p: float, factor: float) -> float:
        """A method for calculating y_p.

        Args:
            d (float): The diameter of the (round) conductor or (sector shaped) equivalent conductor in mm.
            s (float): The distance between the conductors' axes (round conductors) or the diameter plus thickness of
                insulation between conductors (sector shaped) in mm.
            x_p (float): The x coefficient for proximity effect, calculated according to the method from NEN-IEC
                60287-1-1:2023 - section 5.1.4.
            factor (float): A factor used to multiply the result of the y_p calculation. This factor is used to
                differentiate between the implementation of sections [5.1.5.1] (for round conductors) and [5.1.5.2]
                (for sector shaped conductors).

        Returns:
            float: The calculated value for the proximity effect factor y_p.

        References:
            - NEN-IEC 60287-1-1:2023 - sections 5.1.4 & 5.1.5

        """
        a = x_p**4 / (192 + 0.8 * x_p**4)
        y = d / s

        return factor * a * y**2 * (0.312 * y**2 + 1.18 / (a + 0.27))

    @staticmethod
    def _calculate_y_s_from_x_s(x_s: float) -> float:
        """Calculation of y_s.

        Args:
            x_s (float): Factor used for calculating y_s

        Returns:
            float: Skin effect factor y_s.

        References:
            - NEN-IEC 60287-1-1:2023 - [section 5.1.3]

        """
        if 0 < x_s <= 2.8:  # noqa: PLR2004
            y_s = x_s**4 / (192 + 0.8 * x_s**4)
        elif x_s <= 3.8:  # noqa: PLR2004
            y_s = -0.136 - 0.0177 * x_s + 0.0563 * x_s**2
        elif x_s > 3.8:  # noqa: PLR2004
            y_s = 0.354 * x_s - 0.733
        else:
            raise ValueError("x_s is smaller or equal to zero.")

        return y_s

    def _select_k_s_and_k_p_for_copper_cable(  # noqa: PLR0912
        self,
        surface_type: CableConductorSurfaceType,
        shape: CableConductorShape,
        insulation_material: CableInsulationMaterial,
    ) -> tuple[float, float]:
        """Select k_s and k_p for copper cables.

        Args:
            surface_type (CableConductorSurfaceType): The surface type of the cable's conductors.
            shape (CableConductorShape): The shape of the cable's conductors.
            insulation_material (CableInsulationMaterial): The material used for the insulation layer.

        Returns:
            k_s and k_p

        References:
            - NEN-IEC 60287-1-1:2023 - [table 2]

        """
        cable_contains_paper = (
            self.cable_type in [CableType.PILC, CableType.OilPressure]
            or insulation_material == CableInsulationMaterial.PPL
        )
        k_s, k_p = 0.0, 0.0
        if shape == CableConductorShape.Round:
            if surface_type == CableConductorSurfaceType.Solid:
                k_s = 1.0
                k_p = 1.0
            elif surface_type == CableConductorSurfaceType.Stranded:
                if cable_contains_paper:
                    k_s = 1.0
                    k_p = 0.8
                elif self.cable_type == CableType.XLPE:
                    k_s = 1.0
                    k_p = 1.0
            elif surface_type == CableConductorSurfaceType.Milliken:
                if cable_contains_paper:
                    k_s = 0.435
                    k_p = 0.37
                elif self.cable_type == CableType.XLPE:
                    # As the wire type of the conductor wires is not specified, the worst case of
                    # 'bare bi-directional wires' is assumed.
                    k_s = 0.8
                    k_p = 0.37
        elif shape == CableConductorShape.Hollow:
            if surface_type == CableConductorSurfaceType.Stranded:
                k_s = self._select_ks_for_hollow_helical_stranded()
                k_p = 0.8
            elif surface_type == CableConductorSurfaceType.Milliken:
                k_s = 0.435
                k_p = 0.37
                logger.warning(
                    "Hollow Milliken conductor detected. Source 'NEN-IEC 60287-1-1:2023' does not prescribe "
                    "values for k_s and k_p for this case. Default Milliken values k_s=0.435 and k_p=0.37 are "
                    "selected, but results may be wrong."
                )
        elif shape == CableConductorShape.Sector:
            if cable_contains_paper:
                k_s = 1.0
                k_p = 0.8
            elif self.cable_type == CableType.XLPE:
                k_s = 1.0
                k_p = 1.0
        return k_s, k_p

    def _select_k_s_and_k_p_for_aluminium_cable(
        self, surface_type: CableConductorSurfaceType, shape: CableConductorShape
    ) -> tuple[float, float]:
        """Select k_s and k_p for aluminium cables.

        Args:
            surface_type (CableConductorSurfaceType): The surface type of the cable's conductors.
            shape (CableConductorShape): The shape of the cable's conductors.

        Returns:
            k_s and k_p

        References:
            - NEN-IEC 60287-1-1:2023 - [table 2]

        """
        k_s, k_p = 0.0, 0.0
        if shape in [CableConductorShape.Round, CableConductorShape.Sector]:
            if surface_type == CableConductorSurfaceType.Solid:
                k_s = 1.0
                k_p = 1.0
            elif surface_type == CableConductorSurfaceType.Stranded:
                k_s = 1.0
                k_p = 0.8
            elif surface_type == CableConductorSurfaceType.Milliken:
                k_s = 0.25
                k_p = 0.15
        elif shape == CableConductorShape.Hollow:
            if surface_type == CableConductorSurfaceType.Stranded:
                k_s = self._select_ks_for_hollow_helical_stranded()
                k_p = 0.8
            elif surface_type == CableConductorSurfaceType.Milliken:
                k_s = 0.25
                k_p = 0.15
                logger.warning(
                    "Hollow Milliken conductor detected. Source 'NEN-IEC 60287-1-1:2023' does not prescribe "
                    "values for k_s and k_p for this case. Default Milliken values k_s=0.25 and k_p=0.15 are "
                    "selected, but results may be wrong."
                )
        return k_s, k_p

    def _select_k_s_and_k_p(self) -> tuple[float, float]:
        """Compute k_s and k_p numbers which are required to compute AC resistance. .

        Returns:
            k_s and k_p

        """
        conductor_material = self.conductor.material
        insulation_material = self.layer_metrics.insulation_material
        surface_type = self.conductor.surface_type
        shape = self.conductor.shape

        k_s, k_p = 0.0, 0.0

        if conductor_material == CableConductorMaterial.Copper:
            k_s, k_p = self._select_k_s_and_k_p_for_copper_cable(surface_type, shape, insulation_material)
        elif conductor_material == CableConductorMaterial.Aluminium:
            k_s, k_p = self._select_k_s_and_k_p_for_aluminium_cable(surface_type, shape)

        if (k_s, k_p) == (0.0, 0.0):
            logger.warning(
                f"Could not find skin and proximity effect coefficients k_s and k_p for the conductor type combination"
                f" of material {conductor_material}, shape {shape} and surface {surface_type}. "
                f"Default values k_s = 1 and k_p = 1 are chosen."
            )
            k_s, k_p = 1.0, 1.0

        return k_s, k_p

    def _select_ks_for_hollow_helical_stranded(self) -> float:
        """Select k_s for hollow helical stranded conductors.

        Notes:
            This case represents a 1-core OilFilled/OilPressure cable, which always has a hollow helical stranded
            core according to the norm.

        References:
            - NEN-IEC 60287-1-1:2023 - [table 2, comment a]

        """
        # outside diameter of equivalent solid conductor with the same central duct (mm):
        dc = self.layer_metrics.conductor_equivalent_outer_diameter
        if dc is None:
            raise ValueError(
                "Conductor equivalent outer diameter is not set. Cannot compute k_s for hollow helical stranded."
            )

        # inside diameter of the conductor or outside diameter of central duct (mm)
        di = 2 * self.layer_properties[CableLayer.InnerOilDuct].outer_radius

        # apply formula from NEN-IEC 60287-A1-2014  [table 2 comment a].
        k_s = ((dc - di) / (dc + di)) * ((dc + 2 * di) / (dc + di)) ** 2
        return k_s

    def get_dc_resistance_conductor(self, Tc: float) -> float:
        """Computes the DC resistance of the conductor at a given conductor temperature.

        Args:
            Tc (float): The conductor temperature in degrees Celsius.

        Returns:
            float: The DC resistance of the conductor in Ohm/m.

        """
        return (
            self.layer_properties[CableLayer.Conductor].electric_rho
            / self.layer_metrics.conductor_virtual_cross_section
            * (1 + self.layer_properties[CableLayer.Conductor].alpha * (Tc - ELECTRIC_RESISTANCE_REFERENCE_TEMPERATURE))
        )

    def get_heat_generation_conductor_and_screen(
        self,
        load: float,
        conductor_temperature: float,
        screen_temperature: float,
        temperature_dependent_electric_resistance: bool,
        ac_current: bool,
    ) -> tuple[float, float]:
        """Calculates the heat generation in the conductor based on the load and resistance.

        In W/m.

        Args:
            ac_current (bool): Whether to use AC resistance (skin/proximity effects) or only DC resistance.
            load (float): The load in Amperes.
            conductor_temperature (float): The temperature of the conductor in degrees Celsius.
            screen_temperature (float): The temperature of the screen in degrees Celsius.
            temperature_dependent_electric_resistance (bool): Whether to use temperature-dependent electric resistance.

        Raises:
            ValueError:
                If AC resistance is requested but conductor distance is not set.

        Returns:
            tuple[float, float]: A tuple containing the heat generation in the
                conductor and the screen respectively in W/m.

        """
        heat_generation_conductor = self.get_heat_generation_conductor(
            ac_current=ac_current,
            load=load,
            conductor_temperature=conductor_temperature,
            temperature_dependent_electric_resistance=temperature_dependent_electric_resistance,
        )

        heat_generation_screen = (
            self.get_heat_generation_screen(
                heat_generation_conductor=heat_generation_conductor,
                screen_temperature=screen_temperature,
                conductor_temperature=conductor_temperature,
                temperature_dependent_electric_resistance=temperature_dependent_electric_resistance,
            )
            if ac_current
            else 0.0
        )

        return heat_generation_conductor, heat_generation_screen

    def get_heat_generation_conductor(
        self,
        ac_current: bool,
        load: float,
        conductor_temperature: float,
        temperature_dependent_electric_resistance: bool,
    ) -> float:
        """Calculates the heat generation in the conductor based on the load and resistance.

        In W/m.

        Args:
            ac_current (bool): Whether to use AC resistance (skin/proximity effects) or only DC resistance.
            load (float): The load in Amperes.
            conductor_temperature (float): The temperature of the conductor in degrees Celsius.
            temperature_dependent_electric_resistance (bool): Whether to use temperature-dependent electric resistance.

        Raises:
            ValueError:
                If AC resistance is requested but conductor distance is not set.

        Returns:
            float: The heat generation in the conductor in W/m.

        """
        resistance_conductor = self.get_dc_resistance_conductor(
            Tc=(
                conductor_temperature
                if temperature_dependent_electric_resistance
                else ELECTRIC_RESISTANCE_REFERENCE_TEMPERATURE
            )
        )
        if ac_current:
            distance_conductor = self.layer_metrics.conductor_distance
            if distance_conductor is None:
                raise ValueError(
                    "Conductor distance has not been set, cannot compute AC resistance factor for conductor."
                )
            resistance_conductor = self._get_ac_resistance_conductor_from_dc_resistance(
                Rdc=resistance_conductor,
                s=distance_conductor,
            )

        return load**2 * resistance_conductor * self.conductor.number_of_conductors.value

    def get_heat_generation_screen(
        self,
        heat_generation_conductor: float,
        screen_temperature: float,
        conductor_temperature: float,
        temperature_dependent_electric_resistance: bool,
    ) -> float:
        """Calculates the heat generation in the screen based on the conductor heat generation and screen loss factor.

        In W/m.

        Args:
            heat_generation_conductor (float): The heat generation in the conductor in W/m.
            screen_temperature (float): The temperature of the screen in degrees Celsius.
            conductor_temperature (float): The temperature of the conductor in degrees Celsius.
            temperature_dependent_electric_resistance (bool): Whether to use temperature-dependent electric resistance.

        Returns:
            float: The heat generation in the screen in W/m.

        """
        return heat_generation_conductor * self.get_cable_screen_loss_factor(
            Ts=screen_temperature
            if temperature_dependent_electric_resistance
            else ELECTRIC_RESISTANCE_REFERENCE_TEMPERATURE,
            Tc=conductor_temperature
            if temperature_dependent_electric_resistance
            else ELECTRIC_RESISTANCE_REFERENCE_TEMPERATURE,
        )

    def _get_resistance_screen(self, Ts: float) -> float:
        """Computes the resistance for the screen, based on the screen temperature and known material properties.

        Args:
            Ts (float): The screen temperature in degrees Celsius.

        Returns:
            float: The resistance of the screen in Ohm/m.

        """
        if self.layer_metrics.screen_cross_section is None:
            raise ValueError("Screen cross-section is not set. Cannot compute screen resistance.")

        return (
            self.layer_properties[CableLayer.Screen].electric_rho
            / self.layer_metrics.screen_cross_section
            * (1 + self.layer_properties[CableLayer.Screen].alpha * (Ts - ELECTRIC_RESISTANCE_REFERENCE_TEMPERATURE))
        )

    def get_cable_screen_loss_factor(self, Ts: float, Tc: float) -> float:
        """Returns the screen loss factor for the cable.

        THe screen loss factor describes the ratio
        of heat generated in the screen relative to the conductor.

        The screen loss factor describes what proportion of the heat
        generation in the conductor will be generated in the screen layer.
        For example, if the factor is 0, no heat will be generated in the
        screen layer by inductance. If the factor is 0.5 and 100 W/m is
        generated in the conductor, then 50 W/m will be generated in the
        screen.

        Args:
            Ts (float): The screen temperature in degrees Celsius.
            Tc (float): The conductor temperature in degrees Celsius.

        Returns:
            float: Screen loss factor (ratio of loss in the screen relative to the conductor).

        """
        if self.layer_metrics.screen_loss_type is None:
            raise ValueError("Screen loss method is not set.")
        screen_loss_function = getattr(self, self.layer_metrics.screen_loss_type.value)
        return screen_loss_function(Tc, Ts)

    @staticmethod
    def _cable_screen_loss_method_return_zero(Tc: float, Ts: float) -> float:
        """The screen loss retrieval method used when no screen load calculations apply."""
        _ = (Tc, Ts)  # Fix for unused parameters
        return 0

    def _cable_screen_loss_method_round_or_oval_with_screen(self, Tc: float, Ts: float) -> float:
        """The screen loss retrieval method for round or oval-shaped cables with a screen.

        Notes:
            When no screen is present, the _cable_screen_loss_method_return_zero method is used, which is why there is
            no version of this method ending with ..._without_screen.

        References:
            - NEN-IEC 60287-1-1 (2007) - [section 2.3.8]

        """
        Rc = self.get_ac_resistance_conductor(Tc=Tc, s=self.s)
        Rs = self._get_resistance_screen(Ts)

        lambda1_eddy = (3.2 * self.omega**2 / (Rc * Rs)) * (2 * self.c / self.d) ** 2 * 10**-14 * self.Ft

        return lambda1_eddy

    def _cable_screen_loss_method_sector(self, Tc: float, Ts: float) -> float:
        """The screen loss retrieval method for sector-shaped cables.

        References:
            - NEN-IEC 60287-1-1 (2007) - [section 2.3.8]

        """
        if self.layer_metrics.original_insulation_thickness is None:
            raise ValueError("Original insulation thickness is not set.")
        if self.layer_metrics.diameter_over_stranded_conductors is None:
            raise ValueError("Diameter over stranded conductors is not set.")

        Rc = self.get_ac_resistance_conductor(Tc=Tc, s=self.s)
        Rs = self._get_resistance_screen(Ts)

        # inner radius of screen
        rsi = self.layer_properties[CableLayer.Screen].inner_radius
        # belt insulation thickness
        t3 = rsi - self.layer_metrics.diameter_over_stranded_conductors / 2
        # total insulation thickness
        t1 = self.layer_metrics.original_insulation_thickness
        # conductor insulation thickness
        t2 = t1 - t3
        # insulation thickness between the conductors
        t = 2 * t2
        # radius of the circle circumscribing the three conductors
        r1 = self.layer_metrics.diameter_over_stranded_conductors / 2 - t2

        lambda1_eddy = (
            0.94 * Rs / Rc * ((2 * r1 + t) / self.d) ** 2 * 1 / (1 + (Rs * 10**7 / self.omega) ** 2) * self.Ft
        )

        return lambda1_eddy

    def _cable_screen_loss_method_cross_bonding_or_one_sided_bonding_trefoil(self, Tc: float, Ts: float) -> float:
        """The screen loss retrieval method for cables with cross bonding or one-sided bonding. Trefoil version.

        References:
            - NEN-IEC 60287-1-1 (2023) - [section 5.3.7.1]

        """
        Rs = self._get_resistance_screen(Ts)

        m = self.omega / Rs * 1e-7

        lambda0 = 3 * m**2 / (1 + m**2) * (self.d / (2 * self.s)) ** 2

        Delta1, Delta2 = 0, 0
        if m > self._M_THRESHOLD:
            Delta1 = (1.14 * m**2.45 + 0.33) * (self.d / (2 * self.s)) ** (0.92 * m + 1.66)

        lambda1_eddy = self._get_lambda1_eddy(Ts, Tc, lambda0, (Delta1 + Delta2))

        return lambda1_eddy

    def _cable_screen_loss_method_two_sided_bonding_trefoil(self, Tc: float, Ts: float) -> float:
        """The screen loss retrieval method for cables with two-sided bonding. Trefoil version.

        Notes:
            Electrical resistances are computed at operating temperature.

        References:
            - NEN-IEC 60287-1-1 (2007) - [section 2.3.1] for screen loss calculation.
            - NEN-IEC 60287-1-1 (2023) for the lambda calculation.

        """
        Rc = self.get_ac_resistance_conductor(Tc=Tc, s=self.s)
        Rs = self._get_resistance_screen(Ts)

        lambda1 = Rs / Rc * 1 / (1 + (Rs / self.X) ** 2)

        # For Milliken conductors, the lambda_1'' term described in section 5.3.1 of the NEN-IEC 60287-1-1 (2023)
        # cannot be ignored and should be determined according to the method described in section 5.3.6.
        if self.conductor.surface_type == CableConductorSurfaceType.Milliken:
            lambda1_eddy = self._cable_screen_loss_method_cross_bonding_or_one_sided_bonding_trefoil(Tc=Tc, Ts=Ts)
            CM1 = Rs / self.X
            CF = type(self)._get_correction_factor_milliken(CM1=CM1, CN=CM1)

            lambda1 += CF * lambda1_eddy

        return lambda1

    def _cable_screen_loss_method_cross_bonding_or_one_sided_bonding_linear_leading(
        self, Tc: float, Ts: float
    ) -> float:
        """The screen loss retrieval method for cables with cross bonding or one-sided bonding.

        Linear version, conductor carrying the leading phase.

        Args:
            Tc (float): The conductor temperature in degrees Celsius.
            Ts (float): The screen temperature in degrees Celsius.

        Notes:
            Calculates eddy currents in the earthing sheath.

        References:
            - NEN-IEC 60287-1-1 (2023) - [section 5.3.7.1]

        """
        Rs = self._get_resistance_screen(Ts)

        m = self.omega / Rs * 1e-7
        Delta1, Delta2 = 0, 0
        if m > self._M_THRESHOLD:
            Delta1 = 4.7 * m**0.7 * (self.d / (2 * self.s)) ** (0.16 * m + 2)
            Delta2 = 21 * m**3.3 * (self.d / (2 * self.s)) ** (1.47 * m + 5.06)

        lambda0 = 1.5 * (m**2 / (1 + m**2)) * (self.d / (2 * self.s)) ** 2
        lambda1_eddy = self._get_lambda1_eddy(Ts, Tc, lambda0, (Delta1 + Delta2))

        return lambda1_eddy

    def _cable_screen_loss_method_cross_bonding_or_one_sided_bonding_linear_lagging(
        self, Tc: float, Ts: float
    ) -> float:
        """The screen loss retrieval method for cables with cross bonding or one-sided bonding.

        Linear version, conductor carrying the lagging phase.

        Notes:
            Calculates eddy currents in the earthing sheath.

        References:
            - NEN-IEC 60287-1-1 (2023) - [section 5.3.7.1]

        """
        Rs = self._get_resistance_screen(Ts)

        m = self.omega / Rs * 1e-7
        Delta1, Delta2 = 0, 0
        if m > self._M_THRESHOLD:
            Delta1 = -0.74 * (m + 2) * m**0.5 / (2 + (m - 0.3) ** 2) * (self.d / (2 * self.s)) ** (m + 1)
            Delta2 = 0.92 * m**3.7 * (self.d / (2 * self.s)) ** (m + 2)

        lambda0 = 1.5 * (m**2 / (1 + m**2)) * (self.d / (2 * self.s)) ** 2
        lambda1_eddy = self._get_lambda1_eddy(Ts, Tc, lambda0, (Delta1 + Delta2))

        return lambda1_eddy

    def _cable_screen_loss_method_cross_bonding_or_one_sided_bonding_linear_center(self, Tc: float, Ts: float) -> float:
        """The screen loss retrieval method for cables with cross bonding or one-sided bonding.

        Linear version, center conductor.

        Notes:
            Calculates eddy currents in the earthing screen.

        References:
            - NEN-IEC 60287-1-1 (2023) - [section 5.3.7.1]

        """
        Rs = self._get_resistance_screen(Ts)

        m = self.omega / Rs * 1e-7
        Delta1, Delta2 = 0, 0
        if m > self._M_THRESHOLD:
            Delta1 = 0.86 * m**3.08 * (self.d / (2 * self.s)) ** (1.4 * m + 0.7)

        lambda0 = 6 * (m**2 / (1 + m**2)) * (self.d / (2 * self.s)) ** 2
        lambda1_eddy = self._get_lambda1_eddy(Ts, Tc, lambda0, (Delta1 + Delta2))

        return lambda1_eddy

    def _cable_screen_loss_method_two_sided_bonding_linear_leading(self, Tc: float, Ts: float) -> float:
        """The screen loss retrieval method for cables with two-sided bonding.

        Linear version, conductor carrying the leading phase.

        Notes:
            Lambda1 gives sheath losses relative to conductor losses in W/m.
            Electrical resistances are computed at operating temperature.

        References:
            - NEN-IEC 60287-1-1 (2007) - [section 2.3.3]

        """
        Rc = self.get_ac_resistance_conductor(Tc=Tc, s=self.s)
        Rs = self._get_resistance_screen(Ts)

        P = self.X + self.Xm
        Q = self.X - self.Xm / 3

        lambda1 = (
            Rs
            / Rc
            * (
                0.75 * P**2 / (Rs**2 + P**2)
                + 0.25 * Q**2 / (Rs**2 + Q**2)
                - 2 * Rs * P * Q * self.Xm / (3**0.5 * (Rs**2 + P**2) * (Rs**2 + Q**2))
            )
        )

        # For Milliken conductors, the lambda_1'' term described in section 5.3.1 of the NEN-IEC 60287-1-1 (2023)
        # cannot be ignored and should be determined according to the method described in section 5.3.6.
        if self.conductor.surface_type == CableConductorSurfaceType.Milliken:
            lambda1_eddy = self._cable_screen_loss_method_cross_bonding_or_one_sided_bonding_linear_leading(
                Tc=Tc, Ts=Ts
            )

            CM1 = Rs / P
            CN = Rs / Q
            CF = type(self)._get_correction_factor_milliken(CM1=CM1, CN=CN)

            lambda1 += CF * lambda1_eddy

        return lambda1

    def _cable_screen_loss_method_two_sided_bonding_linear_lagging(self, Tc: float, Ts: float) -> float:
        """The screen loss retrieval method for cables with two-sided bonding.

        Linear version, conductor carrying the lagging phase.

        Notes:
            Lambda1 gives sheath losses relative to conductor losses in W/m.
            Electrical resistances are computed at operating temperature.

        References:
            - NEN-IEC 60287-1-1 (2007) - [section 2.3.3]

        """
        Rc = self.get_ac_resistance_conductor(Tc=Tc, s=self.s)
        Rs = self._get_resistance_screen(Ts)

        P = self.X + self.Xm
        Q = self.X - self.Xm / 3

        lambda1 = (
            Rs
            / Rc
            * (
                0.75 * P**2 / (Rs**2 + P**2)
                + 0.25 * Q**2 / (Rs**2 + Q**2)
                + 2 * Rs * P * Q * self.Xm / (3**0.5 * (Rs**2 + P**2) * (Rs**2 + Q**2))
            )
        )

        # For Milliken conductors, the lambda_1'' term described in section 5.3.1 of the NEN-IEC 60287-1-1 (2023)
        # cannot be ignored and should be determined according to the method described in section 5.3.6.
        if self.conductor.surface_type == CableConductorSurfaceType.Milliken:
            lambda1_eddy = (
                self._cable_screen_loss_method_cross_bonding_or_one_sided_bonding_linear_lagging(Tc=Tc, Ts=Ts) ** 2
            )

            CM1 = Rs / P
            CN = Rs / Q
            CF = type(self)._get_correction_factor_milliken(CM1=CM1, CN=CN)

            lambda1 += CF * lambda1_eddy

        return lambda1

    def _cable_screen_loss_method_two_sided_bonding_linear_center(self, Tc: float, Ts: float) -> float:
        """The screen loss retrieval method for cables with two-sided bonding.

        Linear version, center conductor.

        Notes:
            Lambda1 gives sheath losses relative to conductor losses in W/m.
            Electrical resistances are computed at operating temperature.

        References:
            - NEN-IEC 60287-1-1 (2007) - [section 2.3.3]

        """
        Rc = self.get_ac_resistance_conductor(Tc=Tc, s=self.s)
        Rs = self._get_resistance_screen(Ts)

        P = self.X + self.Xm
        Q = self.X - self.Xm / 3

        lambda1 = Rs / Rc * Q**2 / (Rs**2 + Q**2)

        # For Milliken conductors, the lambda_1'' term described in section 5.3.1 of the NEN-IEC 60287-1-1 (2023)
        # cannot be ignored and should be determined according to the method described in section 5.3.6.
        if self.conductor.surface_type == CableConductorSurfaceType.Milliken:
            lambda1_eddy = self._cable_screen_loss_method_cross_bonding_or_one_sided_bonding_linear_center(Tc=Tc, Ts=Ts)

            CM1 = Rs / P
            CN = Rs / Q
            CF = type(self)._get_correction_factor_milliken(CM1=CM1, CN=CN)

            lambda1 += CF * lambda1_eddy

        return lambda1

    def _cable_screen_loss_method_two_sided_bonding_leading(self, Tc: float, Ts: float) -> float:
        """The screen loss retrieval method for cables with two-sided bonding.

        Generalized version, leading phase.

        References:
            - NEN-IEC 60287-1-3

        """
        lambda1 = self._cable_screen_loss_method_two_sided_bonding(Tc=Tc, Ts=Ts, phase_idx=0)

        return lambda1

    def _cable_screen_loss_method_two_sided_bonding_center(self, Tc: float, Ts: float) -> float:
        """The screen loss retrieval method for cables with two-sided bonding.

        Generalized version, center phase.

        References:
            - NEN-IEC 60287-1-3

        """
        lambda1 = self._cable_screen_loss_method_two_sided_bonding(Tc=Tc, Ts=Ts, phase_idx=1)

        return lambda1

    def _cable_screen_loss_method_two_sided_bonding_lagging(self, Tc: float, Ts: float) -> float:
        """The screen loss retrieval method for cables with two-sided bonding.

        Generalized version, lagging phase.

        References:
            - NEN-IEC 60287-1-3

        """
        lambda1 = self._cable_screen_loss_method_two_sided_bonding(Tc=Tc, Ts=Ts, phase_idx=2)

        return lambda1

    def _cable_screen_loss_method_two_sided_bonding(self, Tc: float, Ts: float, phase_idx: int) -> float:
        """The screen loss retrieval method for cables with two-sided bonding. Generalized version.

        References:
            - NEN-IEC 60287-1-3

        """
        if self.conductor.surface_type == CableConductorSurfaceType.Milliken:
            raise NotImplementedError("Generalized screen loss not implemented for Milliken cables.")

        Rc = self.get_ac_resistance_conductor(Tc=Tc, s=self.s)
        Rs = self._get_resistance_screen(Ts)

        lambda1 = Rs / Rc * abs(self._calculate_relative_circulating_screen_currents(Rs=Rs)[phase_idx][0]) ** 2

        return lambda1

    def _calculate_relative_circulating_screen_currents(self, Rs: float):
        """Determines the ratio between the circulating sheath currents and the conductor current (|Is|/|Ic|).

        Uses numpy.linalg.solve to solve the linear system of equations Ax = b

        """
        if self.weighted_screen_impedance is None:
            raise ValueError("Screen impedance should be set.")
        X_matrix = self.weighted_screen_impedance.weighted_reactance_matrix
        weighted_screen_resistance_factor = self.weighted_screen_impedance.weighted_resistance_factor

        normalized_conductor_currents = np.array(
            [
                [1.0],
                [-1 / 2 - 1j / 2 * np.sqrt(3)],
                [-1 / 2 + 1j / 2 * np.sqrt(3)],
            ]
        )

        b = -np.matmul(
            np.vstack([1j * X_matrix, np.zeros((1, X_matrix.shape[1]), dtype=X_matrix.dtype)]),
            normalized_conductor_currents,
        )

        A = np.vstack([1j * X_matrix, np.ones((1, X_matrix.shape[1]), dtype=X_matrix.dtype)])
        for i in [0, 1]:
            A[i, i] += Rs * weighted_screen_resistance_factor
            A[i, i + 1] -= Rs * weighted_screen_resistance_factor

        return np.linalg.solve(A, b)

    def _cable_screen_loss_method_single_cable_oil_pressure_or_xlpe(self, Tc: float, Ts: float) -> float:
        """The screen loss retrieval method for single cables. Oil Pressure or XLPE cables version."""
        if CableLayer.Screen in self.layers:
            return self._cable_screen_loss_method_round_or_oval_with_screen(Tc=Tc, Ts=Ts)
        else:
            return type(self)._cable_screen_loss_method_return_zero(Tc=Tc, Ts=Ts)

    def _cable_screen_loss_method_single_cable_pilc(self, Tc: float, Ts: float) -> float:
        """The screen loss retrieval method for single cables. PILC version."""
        if CableLayer.Screen in self.layers:
            return self._cable_screen_loss_method_sector(Tc=Tc, Ts=Ts)
        else:
            return type(self)._cable_screen_loss_method_return_zero(Tc=Tc, Ts=Ts)

    def _get_lambda1_eddy(self, Ts: float, Tc: float, lambda0: float, delta_one_plus_delta_two: float) -> float:
        """Calculates the lambda1 eddy value for the screen loss calculation.

        References:
            - NEN-IEC 60287-1-1 (2023) - [section 5.3.7.1]

        """
        Rc = self.get_ac_resistance_conductor(Tc=Tc, s=self.s)
        Rs = self._get_resistance_screen(Ts)

        rhos = self.electric_rho_screen * (1 + self.alpha_screen * (Ts - ELECTRIC_RESISTANCE_REFERENCE_TEMPERATURE))
        Ds = 2e3 * self.layer_properties[CableLayer.Screen].outer_radius  # outer screen diameter in mm
        ts = 1e3 * (
            self.layer_properties[CableLayer.Screen].outer_radius
            - self.layer_properties[CableLayer.Screen].inner_radius
        )  # screen thickness in mm
        beta1 = (4 * np.pi * self.omega / (1e7 * rhos)) ** 0.5
        Cgs = 1 + (ts / Ds) ** 1.74 * (beta1 * Ds * 1e-3 - 1.6)

        return Rs / Rc * (Cgs * lambda0 * (1 + delta_one_plus_delta_two) + (beta1 * ts) ** 4 / 12e12)

    def _get_armour_factor(self, d: float) -> float:
        """Computes the factor for steel tape armour due to increased eddy-currents losses in the sheath.

        Args:
            d (float): The mean screen diameter in mm

        Returns:
            float: A factor multiplication value for steel-tape armoured cables.

        """
        if CableLayer.Armour not in self.layers:
            # No armour, no factor applicable
            return 1.0
        if self.layer_metrics.armour_cross_section is None:
            raise ValueError("Armour cross-section is not set. Cannot compute armour factor.")

        # The relative permeability of steel tape
        _MU = 300

        # The mean diameter of the armour
        da = (
            self.layer_properties[CableLayer.Armour].outer_radius
            + self.layer_properties[CableLayer.Armour].inner_radius
        )

        # The equivalent thickness of the armour
        delta = self.layer_metrics.armour_cross_section / (np.pi * da)

        # Return the multiplication factor for armoured cables
        return (1 + (d / da) ** 2 * 1 / (1 + (da / (_MU * delta)))) ** 2

    def get_dielectric_loss_for_cable(self) -> float:
        """Computes dielectric losses per conductor using permittivity and tan(delta) of the insulation material.

        This function checks first whether the dielectric loss can be neglected based on the cable configuration.
        If so, it returns 0. Otherwise, it computes the dielectric loss using the compute_dielectric_loss_per_conductor
        function.

        Returns:
            float: Dielectric losses per conductor in W/m.

        References:
            - NEN-IEC 60287-1-1 (2023) - [section 5.2]

        """
        if self.conductor.number_of_conductors == CableConductorCount.Three:
            # According to [section 5.2] in NEN-IEC 60287-1-1 (2023), the dielectric loss can be neglected for
            # unscreened multicore cables. Screened multicore cables are not supported by the model (yet).
            # Therefore, always return 0 when the number of conductors is 3. Also return 0 when the
            # neglect_dielectric_loss flag is set in the layer metrics.
            return 0.0

        dielectric_loss_per_conductor = self._compute_dielectric_loss_per_conductor()
        # The dielectric losses are multiplied by the amount of conductors in the cable.
        return dielectric_loss_per_conductor * self.conductor.number_of_conductors.value

    def _compute_dielectric_loss_per_conductor(self) -> float:
        """Computes the dielectric losses per conductor.

        Returns:
            float: Dielectric losses per conductor in W/m.

        References:
            - NEN-IEC 60287-1-1 (2023) - [section 5.2]

        """
        if self.conductor.number_of_conductors == CableConductorCount.Three:
            raise NotImplementedError("Dielectric loss calculation is not implemented for three-core cables.")

        epsilon_0 = 8.854e-12  # F/m, permittivity of the vacuum
        capacitance = (
            2
            * np.pi
            * epsilon_0
            * self.layer_properties[CableLayer.Insulation].epsilon
            / np.log(
                self.layer_properties[CableLayer.Insulation].outer_radius
                / self.layer_properties[CableLayer.Insulation].inner_radius
            )
        )

        tan_delta = self.layer_properties[CableLayer.Insulation].tan_delta

        # Calculating the total dielectric loss per conductor
        return self.omega * capacitance * self.layer_metrics.nominal_phase_voltage**2 * tan_delta

    @staticmethod
    def _get_correction_factor_milliken(CM1: float, CN: float) -> float:
        """Calculate the Eddy-current factor for reduced proximity effect in Milliken conductors.

        References:
            - NEN-IEC 60287-1-1 (2023) - [section 5.3.6]

        """
        return (4 * CM1**2 * CN**2 + (CM1 + CN) ** 2) / (4 * (CM1**2 + 1) * (CN**2 + 1))
