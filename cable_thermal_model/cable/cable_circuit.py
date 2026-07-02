# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import StrEnum

import numpy as np
from pydantic import BaseModel, ConfigDict, computed_field

from cable_thermal_model.cable.cable_builder import CableBuilder
from cable_thermal_model.cable.enums.circuit_enums import BondingType, CircuitType, CircuitYReference
from cable_thermal_model.cable.schemas.circuit_schemas import (
    CircuitFromCableInputSchema,
    CircuitInSoilFromCableInputSchema,
)
from cable_thermal_model.cable.schemas.pipe_schemas import PipeInputSchema
from cable_thermal_model.model.cables.abstract_cable import CableType, WeightedScreenImpedance
from cable_thermal_model.model.cables.cable import (
    Cable,
    CableSoil,
    CableTrefoilCircuitSinglePipeInAir,
    CableTrefoilCircuitSinglePipeInSoil,
)
from cable_thermal_model.model.cables.enum_classes_cable import CableLayer, CableScreenLossType
from cable_thermal_model.utils.str_utils import tab_lines


class CablePosition(StrEnum):
    """Enumeration of cable positions within a circuit."""

    Single = "single"
    LinearLeft = "linear_left"
    LinearCenter = "linear_center"
    LinearRight = "linear_right"
    LinearTop = "linear_top"
    LinearBottom = "linear_bottom"
    TrefoilTop = "trefoil_top"
    TrefoilLeft = "trefoil_left"
    TrefoilRight = "trefoil_right"
    TrefoilCircuitInSinglePipe = "trefoil_circuit_in_single_pipe"


class CableKey(BaseModel):
    """Immutable identifier for a cable within a named circuit."""

    model_config = ConfigDict(frozen=True)
    circuit_name: str
    cable_position: CablePosition

    def __hash__(self) -> int:
        """Return hash based on circuit name and cable position."""
        return hash((self.circuit_name, self.cable_position))


class PosCable(BaseModel):
    """A positioned cable within a circuit, combining cable data with spatial coordinates."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    circuit_name: str
    cable_position: CablePosition
    cable: Cable
    x: float
    y: float

    @computed_field  # type: ignore[misc]
    @property
    def name(self) -> CableKey:
        """Return the CableKey identifying this cable in the circuit."""
        return CableKey(circuit_name=self.circuit_name, cable_position=self.cable_position)

    @computed_field  # type: ignore[misc]
    @property
    def cable_info(self) -> str:
        """Return a compact string encoding the cable's physical properties."""
        return (
            f"{tuple([grid_count for grid_count in self.cable.grid_counts.values()])},"
            f"{tuple([layer_properties.outer_radius for layer_properties in self.cable.layer_properties.values()])},"
            f"{tuple([layer_properties.rho for layer_properties in self.cable.layer_properties.values()])},"
            f"{tuple([layer_properties.capacity for layer_properties in self.cable.layer_properties.values()])},"
            f"{tuple([layer_properties.electric_rho for layer_properties in self.cable.layer_properties.values()])},"
            f"{tuple([layer_properties.alpha for layer_properties in self.cable.layer_properties.values()])},"
            f"{tuple(self.cable.layers)},"
            f"{self.cable.layer_metrics.outer_radius},"
            f"{self.cable.layer_metrics.conductor_cross_section},"
            f"{self.cable.layer_metrics.screen_cross_section},"
            f"{self.cable.conductor.number_of_conductors.value},"
            f"{self.cable.layer_metrics.conductor_distance},"
            f"{self.cable.cable_type}"
        )

    @computed_field  # type: ignore[misc]
    @property
    def cable_representation(self) -> str:
        """Return a string representation of this positioned cable for serialization."""
        return f"Cable(cable=Cable({self.cable_info}, x={self.x}, y={self.y}, name={self.name}))"


class CircuitInitData(BaseModel):
    """Data class for initializing a cable circuit, encapsulating position, type, and bonding information."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    x: float
    y: float
    circuit_type: CircuitType | None = None
    cable: Cable
    circuit_name: str
    dist: float | None = None
    y_ref: CircuitYReference = CircuitYReference.Center
    bonding_type: BondingType | None = None

    @classmethod
    def from_schema(cls, schema: CircuitFromCableInputSchema[Cable]) -> "CircuitInitData":
        """Create CircuitInitData from CircuitFromCableInputSchema."""
        return cls(
            x=float(getattr(schema, "x", 0.0)),
            y=float(getattr(schema, "y", 0.0)),
            circuit_type=schema.circuit_type,
            cable=schema.cable,
            circuit_name=schema.circuit_name,
            dist=schema.dist,
            y_ref=CircuitYReference(getattr(schema, "y_ref", CircuitYReference.Center)),
            bonding_type=schema.bonding_type,
        )


def return_mirror_cable(pos_cable: PosCable) -> PosCable:
    """Return mirror cable based on given cable.

    A mirror cable is an exact copy of a given cable with the only difference being that
    the sign of the y-coordinate is switched.
    Mirror cables are essential in guaranteeing that the boundary conditions are satisfied.

    Args:
        pos_cable: Cable positioned in the static environment

    Returns:
        'mirrored' positioned cable

    """
    pos_cable_mirror = deepcopy(pos_cable)
    return PosCable(
        cable=pos_cable_mirror.cable,
        x=pos_cable_mirror.x,
        y=-pos_cable_mirror.y,
        circuit_name=pos_cable_mirror.circuit_name,
        cable_position=pos_cable_mirror.cable_position,
    )


def add_soil_layer(
    pos_cable: PosCable,
    soil_rho: float,
    soil_capacity: float,
    logarithmic_soil_gridpoint_density: float,
    soil_radius: float,
) -> PosCable:
    """Add soil layers to cable attribute of the given PosCable.

    Args:
        pos_cable:
            Positioned cable instance without any soil layers
        soil_rho:
            Thermal resistivity of the soil layer to add in Km/W
        soil_capacity:
            Thermal capacity of the soil layer to add in J/(m³K)
        logarithmic_soil_gridpoint_density:
            The density of grid points in the soil layer, this is used to compute the number of grid points in the
            soil layer based on its thickness. The density represents the number of grid points per factor 2 increase
            in soil layer thickness.
        soil_radius:
            The outer radius of the soil layer to add.

    Returns:
        New PosCable instance where the only difference is that the cable now has soil layers.

    """
    pos_cable_ = deepcopy(pos_cable)
    cable_in_soil = CableSoil.from_cable_with_added_soil_layer(
        cable=pos_cable_.cable,
        soil_rho=soil_rho,
        soil_capacity=soil_capacity,
        soil_radius=soil_radius,
        logarithmic_soil_gridpoint_density=logarithmic_soil_gridpoint_density,
    )
    return PosCable(
        cable=cable_in_soil,
        x=pos_cable_.x,
        y=pos_cable_.y,
        circuit_name=pos_cable_.circuit_name,
        cable_position=pos_cable_.cable_position,
    )


def remove_soil(
    pos_cable: PosCable,
) -> PosCable:
    """Remove soil layers from cable attribute of the given PosCable.

    Args:
        pos_cable: Positioned cable instance with soil layers

    Returns:
        New PosCable instance where the only difference is that the cable now has no soil layers.

    """
    pos_cable_ = deepcopy(pos_cable)
    return PosCable(
        cable=pos_cable_.cable.get_cable_copy_without_soil(),
        x=pos_cable_.x,
        y=pos_cable_.y,
        circuit_name=pos_cable_.circuit_name,
        cable_position=pos_cable_.cable_position,
    )


class CableCircuit(ABC):
    """Abstract base class for cable config."""

    def __init__(
        self,
        x: float,
        y: float,
        bonding: BondingType,
        circuit_name: str,
        dist: float | None = None,
    ):
        """Initialise the cable circuit with position, bonding type, and name."""
        self.x: float = x
        self.y: float = y
        self.circuit_name: str = circuit_name
        self.cables: list[PosCable] = []
        self.bonding: BondingType = bonding
        self.dist: float | None = dist

        self.n_phases: int = 3
        self.weighted_screen_impedance: WeightedScreenImpedance | None = None

    # Iterate over the cables in a circuit
    def __iter__(self):
        """Iterate over the positioned cables in this circuit."""
        return iter(self.cables)

    def __str__(self):
        """Generates a concise string representation of the cable circuit."""
        return f"{type(self).__name__} circuit at ({self.x}, {self.y})"

    def __repr__(self):
        """Generates an informative string representation of the cable circuit."""
        circuit_info = f"Type: {type(self).__name__}\nBondingType: {self.bonding}\nx: {self.x}\ny: {self.y}"
        return f"Circuit {self.circuit_name}\n" + tab_lines(circuit_info)

    @abstractmethod
    def initialize_screen_loss_functions(self) -> None:
        """Determines the screen loss functions for the cables in a configuration.

        The screen loss functions for the cables in a configuration are then added as attributes of
        the cable instances in the configuration. These functions depend on
        the temperature T and compute the factor that described the
        proportion of the heat generation (loss) in the screen relative to
        the heat generation in the conductor. For example, if the this
        factor is 0, no heat is generated in the screen. If the factor is
        0.5 and 400W/m is generated in the conductor, then 200W/m is
        generated in the screen.
        """
        raise NotImplementedError(
            f"Method initialize_screen_loss_functions not implemented for class based on CableCircuit: "
            f"{self.__class__.__name__}"
        )

    def get_relative_screen_distances(self) -> np.ndarray:
        """Function to put relative screen distances in a matrix.

        If the conductor is inside the screen, the screen radius is relevant.
        Otherwise, one should use the distances between the cable axes.
        Based on IEC 60287-1-3.
        """
        if len(self.cables) != self.n_phases:
            raise ValueError(f"Method can only be used for circuits with {self.n_phases} single-core cables.")

        d_matrix = np.zeros((self.n_phases, self.n_phases))
        for i in range(self.n_phases):
            for k in range(self.n_phases):
                if i == k:
                    d_matrix[i, k] = self.cables[i].cable.d / 2
                else:
                    d_matrix[i, k] = np.sqrt(
                        (self.cables[i].x - self.cables[k].x) ** 2 + (self.cables[i].y - self.cables[k].y) ** 2
                    )

        return d_matrix

    def get_relative_screen_reactances(self) -> np.ndarray:
        """Function to put relative screen reactances in a matrix.

        Based on IEC 60287-1-3.
        """
        d_matrix = self.get_relative_screen_distances()
        if d_matrix.shape != (self.n_phases, self.n_phases):
            raise ValueError(f"d_matrix must have shape ({self.n_phases}, {self.n_phases}), got {d_matrix.shape}")
        omega = self.cables[0].cable.omega

        X_matrix = np.zeros((self.n_phases - 1, self.n_phases))
        for i in range(self.n_phases - 1):
            for k in range(self.n_phases):
                X_matrix[i, k] = 2e-7 * omega * np.log(d_matrix[i + 1, k] / d_matrix[i, k])

        return X_matrix

    def initialize_pos_cables(
        self,
        cable: Cable,
        cable_centers: dict[CablePosition, tuple[float, float]],
    ) -> list[PosCable]:
        """Initialize the PosCable instances for the circuit based on the given cable and cable centers.

        Args:
            cable:          Cable instance to use in the circuit
            cable_centers:  Dictionary mapping CablePosition to (x, y) coordinates for each cable in the circuit

        Returns:
            List of PosCable instances representing the cables in the circuit.

        """
        return [
            PosCable(
                circuit_name=self.circuit_name,
                cable_position=cable_position,
                cable=deepcopy(cable),
                x=x,
                y=y,
            )
            for cable_position, (x, y) in cable_centers.items()
        ]

    def set_weighted_screen_impedance(self, weighted_screen_impedance: WeightedScreenImpedance | None):
        """Set the weighted screen impedance for the circuit.

        This is used in the calculation of the screen losses for two-sided
        bonding.

        Args:
            weighted_screen_impedance: WeightedScreenImpedance instance
                containing the weighted screen impedance values to use for
                the circuit.

        """
        self.weighted_screen_impedance = weighted_screen_impedance
        for cable in self.cables:
            cable.cable.weighted_screen_impedance = weighted_screen_impedance
        self.initialize_screen_loss_functions()

    def validate_no_screen_no_bonding(self, bonding_type: BondingType, cable: Cable):
        """Validate that if no screen is present, no bonding is applied."""
        if CableLayer.Screen not in cable.layers and bonding_type != BondingType.NoBonding:
            warnings.warn(
                f"Invalid configuration: bonding type {bonding_type} cannot "
                "be applied when no screen is present in the cable. "
                f" Bonding type set to {BondingType.NoBonding}.",
                stacklevel=2,
            )
            return BondingType.NoBonding

        return bonding_type


class SingleCable(CableCircuit):
    """A single cable circuit."""

    def __init__(self, circuit_init_data: CircuitInitData):
        """Initialise the single-cable circuit from CircuitInitData."""
        x = circuit_init_data.x
        y = circuit_init_data.y
        cable = circuit_init_data.cable
        circuit_name = circuit_init_data.circuit_name
        bonding_type = circuit_init_data.bonding_type or BondingType.NoBonding
        dist = circuit_init_data.dist
        y_ref = circuit_init_data.y_ref or CircuitYReference.Center

        if y_ref == CircuitYReference.Top:
            y_circuit_center = y - cable.layer_metrics.outer_radius
        elif y_ref == CircuitYReference.Center:
            y_circuit_center = y
        elif y_ref == CircuitYReference.Bottom:
            y_circuit_center = y + cable.layer_metrics.outer_radius
        else:
            raise ValueError(f"Invalid y_ref value: {y_ref}. Must be one of {CircuitYReference}.")

        bonding_type = self.validate_no_screen_no_bonding(bonding_type=bonding_type, cable=cable)

        super().__init__(x=x, y=y_circuit_center, bonding=bonding_type, circuit_name=circuit_name, dist=dist)

        self.cables = self.initialize_pos_cables(
            cable=cable,
            cable_centers=self._get_cable_centers(),
        )

    def _get_cable_centers(self) -> dict[CablePosition, tuple[float, float]]:
        return {CablePosition.Single: (self.x, self.y)}

    def initialize_screen_loss_functions(self):
        """Determines the screen loss functions for single cable configurations."""
        cable = self.cables[0].cable

        # Screen currents in case of round or oval conductors
        if cable.cable_type in (CableType.XLPE, CableType.OilPressure):
            if CableLayer.Screen in cable.layers:
                self.cables[0].cable.layer_metrics.screen_loss_type = CableScreenLossType.SingleCableOilPressureOrXLPE
            else:  # If no screen is present do not add screen currents.
                self.cables[0].cable.layer_metrics.screen_loss_type = CableScreenLossType.ReturnZero

        # Calculation of the screen currents in case of sector shaped conductors
        elif cable.cable_type == CableType.PILC:
            self.cables[0].cable.layer_metrics.screen_loss_type = CableScreenLossType.SingleCablePILC
        else:
            raise ValueError(f"CableType: {cable.cable_type} not supported.")


class LinearCircuit(CableCircuit):
    """A circuit of cables arranged in a linear (flat) formation."""

    def __init__(self, circuit_init_data: CircuitInitData):
        """Initialise the linear circuit from CircuitInitData."""
        x = circuit_init_data.x
        y = circuit_init_data.y
        circuit_type = circuit_init_data.circuit_type or CircuitType.Linear
        cable = circuit_init_data.cable
        circuit_name = circuit_init_data.circuit_name
        dist = circuit_init_data.dist
        bonding_type = circuit_init_data.bonding_type or BondingType.TwoSided
        y_ref = circuit_init_data.y_ref or CircuitYReference.Center

        if not cable.layer_metrics.outer_radius:
            raise ValueError("Cable layer metrics must include outer radius for LinearCircuit.")

        if circuit_init_data.circuit_type == CircuitType.LinearVertical:
            half_circuit_height = 3 * cable.layer_metrics.outer_radius
        else:
            half_circuit_height = cable.layer_metrics.outer_radius

        if y_ref == CircuitYReference.Top:
            y_circuit_center = y - half_circuit_height
        elif y_ref == CircuitYReference.Center:
            y_circuit_center = y
        elif y_ref == CircuitYReference.Bottom:
            y_circuit_center = y + half_circuit_height
        else:
            raise ValueError(f"Invalid y_ref value: {y_ref}. Must be one of {CircuitYReference}.")

        bonding_type = self.validate_no_screen_no_bonding(bonding_type=bonding_type, cable=cable)

        super().__init__(x=x, y=y_circuit_center, bonding=bonding_type, circuit_name=circuit_name, dist=dist)
        if self.dist is None:
            self.dist = cable.layer_metrics.outer_radius * 2

        if self.dist < cable.layer_metrics.outer_radius * 2:
            warnings.warn(
                f"LinearCircuit initialized with too small distance: ({dist}). "
                f"Conductor distance overwritten with cable diameter.",
                stacklevel=2,
            )
            self.dist = cable.layer_metrics.outer_radius * 2

        if cable.layer_metrics.conductor_distance is None:
            cable.layer_metrics.conductor_distance = self.dist

        if circuit_type == CircuitType.LinearVertical:
            cable_centers = {
                CablePosition.LinearTop: (self.x, self.y + self.dist),
                CablePosition.LinearCenter: (self.x, self.y),
                CablePosition.LinearBottom: (self.x, self.y - self.dist),
            }

        else:
            cable_centers = {
                CablePosition.LinearLeft: (self.x - self.dist, self.y),
                CablePosition.LinearCenter: (self.x, self.y),
                CablePosition.LinearRight: (self.x + self.dist, self.y),
            }

        self.cables = self.initialize_pos_cables(
            cable=cable,
            cable_centers=cable_centers,
        )

    def initialize_screen_loss_functions(self):
        """Determines the screen loss functions for the cables in a linear configuration."""
        if self.bonding == BondingType.NoBonding:
            self.cables[0].cable.layer_metrics.screen_loss_type = CableScreenLossType.ReturnZero
            self.cables[1].cable.layer_metrics.screen_loss_type = CableScreenLossType.ReturnZero
            self.cables[2].cable.layer_metrics.screen_loss_type = CableScreenLossType.ReturnZero

        elif self.bonding in (BondingType.CrossBonding, BondingType.OneSided):
            # Calculation of the eddy currents in the earthing sheath based on the
            # IEC 60287-1-1:2023 - par 5.3.7.1
            self.cables[
                0
            ].cable.layer_metrics.screen_loss_type = CableScreenLossType.CrossBondingOrOneSidedBondingLinearLeading
            self.cables[
                1
            ].cable.layer_metrics.screen_loss_type = CableScreenLossType.CrossBondingOrOneSidedBondingLinearCenter
            self.cables[
                2
            ].cable.layer_metrics.screen_loss_type = CableScreenLossType.CrossBondingOrOneSidedBondingLinearLagging

        elif self.bonding == BondingType.TwoSided:
            # Compute lambda1 based on NEN 60287-1-1. Lambda1 gives sheath losses relative to conductor losses in W/m.

            # Section 2.3.3 from NEN 60287-1-1. Note that the electrical resistances are computed at operating
            # temperature.
            if self.weighted_screen_impedance is not None:
                self.cables[0].cable.layer_metrics.screen_loss_type = CableScreenLossType.TwoSidedBondingLeading
                self.cables[1].cable.layer_metrics.screen_loss_type = CableScreenLossType.TwoSidedBondingCenter
                self.cables[2].cable.layer_metrics.screen_loss_type = CableScreenLossType.TwoSidedBondingLagging
            else:
                self.cables[0].cable.layer_metrics.screen_loss_type = CableScreenLossType.TwoSidedBondingLinearLeading
                self.cables[1].cable.layer_metrics.screen_loss_type = CableScreenLossType.TwoSidedBondingLinearCenter
                self.cables[2].cable.layer_metrics.screen_loss_type = CableScreenLossType.TwoSidedBondingLinearLagging


class TrefoilCircuit(CableCircuit):
    """A circuit of three cables arranged in a trefoil formation."""

    def __init__(
        self,
        circuit_init_data: CircuitInitData,
    ):
        """Initialise the trefoil circuit from CircuitInitData."""
        # Unpack the circuit initialization data
        x = circuit_init_data.x
        y = circuit_init_data.y
        cable = circuit_init_data.cable
        circuit_name = circuit_init_data.circuit_name
        bonding_type = circuit_init_data.bonding_type or BondingType.TwoSided
        y_ref = circuit_init_data.y_ref

        if not cable.layer_metrics.outer_radius:
            raise ValueError("Cable layer metrics must include outer radius for TrefoilCircuit.")

        outer_radius = cable.layer_metrics.outer_radius
        if y_ref == CircuitYReference.Top:
            y_circuit_center = y - (1 + 2 / np.sqrt(3)) * outer_radius
        elif y_ref == CircuitYReference.Center:
            y_circuit_center = y
        elif y_ref == CircuitYReference.Bottom:
            y_circuit_center = y + outer_radius / np.sqrt(3)
        else:
            raise ValueError(f"Invalid y_ref value: {y_ref}. Must be one of {CircuitYReference}.")

        if cable.layer_metrics.pipe and cable.layer_metrics.pipe.trefoil_circuit_in_single_pipe:
            raise ValueError(
                "Three cables in one pipe configuration is not supported for TrefoilCircuit. Use "
                "TrefoilCircuitInSinglePipe instead."
            )

        bonding_type = self.validate_no_screen_no_bonding(bonding_type=bonding_type, cable=cable)

        super().__init__(x=x, y=y_circuit_center, bonding=bonding_type, circuit_name=circuit_name, dist=None)
        if cable.layer_metrics.conductor_distance is None:
            cable.layer_metrics.conductor_distance = 2 * outer_radius

        cable_centers = {
            CablePosition.TrefoilTop: (self.x, self.y + 2 * outer_radius / np.sqrt(3)),
            CablePosition.TrefoilLeft: (self.x - outer_radius, self.y - outer_radius / np.sqrt(3)),
            CablePosition.TrefoilRight: (self.x + outer_radius, self.y - outer_radius / np.sqrt(3)),
        }

        self.cables = self.initialize_pos_cables(
            cable=cable,
            cable_centers=cable_centers,
        )

    def initialize_screen_loss_functions(self):
        """Determines the screen loss functions for the cables in a trefoil configuration."""
        if self.bonding == BondingType.NoBonding:
            self.cables[0].cable.layer_metrics.screen_loss_type = CableScreenLossType.ReturnZero
            self.cables[1].cable.layer_metrics.screen_loss_type = CableScreenLossType.ReturnZero
            self.cables[2].cable.layer_metrics.screen_loss_type = CableScreenLossType.ReturnZero

        elif self.bonding in (BondingType.CrossBonding, BondingType.OneSided):
            # Calculation of the eddy currents in the earthing sheath based on the
            # IEC 60287-1-1:2023 - par 5.3.7.1
            self.cables[
                0
            ].cable.layer_metrics.screen_loss_type = CableScreenLossType.CrossBondingOrOneSidedBondingTrefoil
            self.cables[
                1
            ].cable.layer_metrics.screen_loss_type = CableScreenLossType.CrossBondingOrOneSidedBondingTrefoil
            self.cables[
                2
            ].cable.layer_metrics.screen_loss_type = CableScreenLossType.CrossBondingOrOneSidedBondingTrefoil

        elif self.bonding == BondingType.TwoSided:
            # Compute lambda1 based on section 5.3 from NEN 60287-1-1 (2023).
            # Lambda1 gives sheath losses relative to conductor losses in W/m.

            # Note that the electrical resistances are computed at operating temperature.

            if self.weighted_screen_impedance is not None:
                self.cables[0].cable.layer_metrics.screen_loss_type = CableScreenLossType.TwoSidedBondingLeading
                self.cables[1].cable.layer_metrics.screen_loss_type = CableScreenLossType.TwoSidedBondingCenter
                self.cables[2].cable.layer_metrics.screen_loss_type = CableScreenLossType.TwoSidedBondingLagging

            else:
                self.cables[0].cable.layer_metrics.screen_loss_type = CableScreenLossType.TwoSidedBondingTrefoil
                self.cables[1].cable.layer_metrics.screen_loss_type = CableScreenLossType.TwoSidedBondingTrefoil
                self.cables[2].cable.layer_metrics.screen_loss_type = CableScreenLossType.TwoSidedBondingTrefoil


class TrefoilCircuitInSinglePipe(SingleCable):
    """Trefoil circuit with three cables in one pipe.

    This is modelled as a single equivalent cable with an extra heat source between the equivalent cable and the pipe.
    """

    def __init__(self, circuit_init_data: CircuitInitData):
        """Initialise the trefoil-in-single-pipe circuit from CircuitInitData."""
        # Initialize super, but with twosides bonding as default
        circuit_init_data.bonding_type = circuit_init_data.bonding_type or BondingType.TwoSided

        super().__init__(circuit_init_data)
        # Manually set the conductor distance
        cable = self.cables[0].cable
        if cable.layer_metrics.conductor_distance is None:
            cable.layer_metrics.conductor_distance = 2 * cable.layer_metrics.cable_radius

    def _get_cable_centers(self) -> dict[CablePosition, tuple[float, float]]:
        return {CablePosition.TrefoilCircuitInSinglePipe: (self.x, self.y)}

    def initialize_screen_loss_functions(self):
        """Determines the screen loss functions for the equivalent cable in a trefoil circuit in a single pipe.

        Raises:
            NotImplementedError: If the sheath currents are not symmetric.

        """
        cable = self.cables[0].cable

        if self.bonding == BondingType.NoBonding:
            cable.layer_metrics.screen_loss_type = CableScreenLossType.ReturnZero

        elif self.bonding in (BondingType.CrossBonding, BondingType.OneSided):
            cable.layer_metrics.screen_loss_type = CableScreenLossType.CrossBondingOrOneSidedBondingTrefoil

        elif self.bonding == BondingType.TwoSided:
            if self.weighted_screen_impedance is None:
                cable.layer_metrics.screen_loss_type = CableScreenLossType.TwoSidedBondingTrefoil
                return

            if not self.symmetric_sheath_currents(self.weighted_screen_impedance.weighted_reactance_matrix):
                raise NotImplementedError(
                    "Non-symmetric sheath currents are not supported for trefoil circuit in a single pipe."
                )

            cable.layer_metrics.screen_loss_type = CableScreenLossType.TwoSidedBondingCenter

    def get_relative_screen_distances(self) -> np.ndarray:
        """Returns the relative screen distances for a trefoil circuit with touching cables."""
        cable_diameter = self.cables[0].cable.layer_metrics.conductor_distance
        screen_radius = self.cables[0].cable.d / 2
        return np.array(
            [
                [screen_radius, cable_diameter, cable_diameter],  # type: ignore
                [cable_diameter, screen_radius, cable_diameter],  # type: ignore
                [cable_diameter, cable_diameter, screen_radius],  # type: ignore
            ]
        )

    @staticmethod
    def symmetric_sheath_currents(weighted_reactance_matrix: np.ndarray) -> bool:
        """Check if the sheath currents for three phases are symmetric.

        This is the case when the weighted reactance matrix has the following form:

            [ x, -x, 0 ]
            [ 0, x, -x ],

        for some real number x.

        Args:
            weighted_reactance_matrix: 2x3 weighted relative reactance matrix

        Returns:
            True if the sheath currents are symmetric, False otherwise.

        """
        if weighted_reactance_matrix.shape != (2, 3):
            raise ValueError(f"weighted_reactance_matrix must have shape (2, 3), got {weighted_reactance_matrix.shape}")

        x = weighted_reactance_matrix[0, 0]
        symmetric_matrix = np.array([[x, -x, 0], [0, x, -x]])  # type: ignore

        return np.allclose(weighted_reactance_matrix, symmetric_matrix)  # type: ignore


class CircuitBuilder:
    """Factory for constructing CableCircuit instances from various input sources."""

    @classmethod
    def from_cable_id(
        cls,
        x: float,
        y: float,
        circuit_type: CircuitType,
        cable_id: str,
        circuit_name: str,
        dist: float | None = None,
        pipe: PipeInputSchema | None = None,
        bonding_type: BondingType | None = None,
        y_ref: CircuitYReference = CircuitYReference.Center,
    ) -> CableCircuit:
        """Build circuit from cable id.

        First builds the cable instance using the CableBuilder and then builds the
        circuit using that cable instance.

        Args:
            x: Horizontal position of circuit in environment in meters
            y: Vertical position (depth) of circuit in environment in meters
            circuit_type: Type of circuit
            cable_id: Identifier of cable type
            circuit_name: Name of the circuit
            dist: Distance between cables, relevant for CircuitType.Linear
            pipe: Pipe object with pipe parameters, if None, no pipe is added to the cables
            bonding_type: Type of bonding used in the cable circuit
            y_ref: Reference of the circuit y position, either
                CircuitYReference.Center, CircuitYReference.Top or
                CircuitYReference.Bottom
                y_ref defines y: y is the distance from the ground surface level to y_ref

        Returns:
            A CableCircuit instance that can be added to the static environment.

        """
        cable = CableBuilder.build_cable_from_cable_id(
            cable_id=cable_id,
            cable_class=(
                CableTrefoilCircuitSinglePipeInSoil
                if cls._is_trefoil_circuit_in_single_pipe(circuit_type, pipe)
                else CableSoil
            ),
            pipe=pipe,
        )

        return cls.from_cable(
            CircuitInSoilFromCableInputSchema(
                x=x,
                y=y,
                circuit_type=circuit_type,
                cable=cable,
                circuit_name=circuit_name,
                dist=dist,
                bonding_type=bonding_type,
                y_ref=y_ref,
            )
        )

    @staticmethod
    def from_cable(
        circuit_input: CircuitFromCableInputSchema[Cable],
    ) -> CableCircuit:
        """Build circuit from cable.

        Useful to create circuits from cables that are not (yet) present in the
        cable database example_cables.csv.

        Args:
            circuit_input: CircuitFromCableInputSchema instance containing the necessary information to build the
                           circuit, including the cable instance.

        Returns:
            A CableCircuit instance that can be added to the static environment.

        """
        circuit_init_data = CircuitInitData.from_schema(
            circuit_input
        )  # Validate input schema and extract initialization data

        circuit: CableCircuit
        if circuit_input.circuit_type == CircuitType.Single:
            circuit = SingleCable(circuit_init_data)
        elif circuit_input.circuit_type in [CircuitType.Linear, CircuitType.LinearVertical]:
            circuit = LinearCircuit(circuit_init_data)
        elif circuit_input.circuit_type == CircuitType.Trefoil:
            if circuit_input.dist:
                if (
                    circuit_input.cable.layer_metrics.pipe
                    and not circuit_input.cable.layer_metrics.pipe.trefoil_circuit_in_single_pipe
                ):
                    touching = "cables"
                else:
                    touching = "pipes"
                raise NotImplementedError(
                    f"Cable distance is not supported for circuit type {circuit_input.circuit_type}. If touching "
                    f"{touching} are desired, dist should be set to {None}."
                )
            if isinstance(
                circuit_input.cable, CableTrefoilCircuitSinglePipeInSoil | CableTrefoilCircuitSinglePipeInAir
            ):
                circuit = TrefoilCircuitInSinglePipe(circuit_init_data)
            else:
                circuit = TrefoilCircuit(circuit_init_data)

        else:
            raise TypeError(f"Circuit type {circuit_input.circuit_type} not supported.")

        circuit.initialize_screen_loss_functions()

        return circuit

    @staticmethod
    def _is_trefoil_circuit_in_single_pipe(
        circuit_type: CircuitType | None,
        pipe: PipeInputSchema | None = None,
    ) -> bool:
        """Determine if the circuit is a trefoil circuit in a single pipe.

        Args:
            circuit_type:   Type of circuit.
            pipe:           PipeInputSchema instance or None.

        Returns:
            True if a pipe is present, the pipe contains three cables and the circuit type is trefoil or None.

        """
        return pipe is not None and pipe.trefoil_circuit_in_single_pipe and circuit_type in (None, CircuitType.Trefoil)
