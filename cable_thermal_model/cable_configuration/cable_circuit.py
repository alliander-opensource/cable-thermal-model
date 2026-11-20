# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import StrEnum
from typing import Generic

import numpy as np
from pydantic import BaseModel, ConfigDict, computed_field

from cable_thermal_model.cable_configuration.cable_builder import CableT
from cable_thermal_model.cable_configuration.enums.circuit_enums import BondingType, CircuitType, CircuitYReference
from cable_thermal_model.cable_configuration.schemas.circuit_schemas import (
    CircuitFromCableInputSchema,
)
from cable_thermal_model.cable_configuration.schemas.pipe_schemas import PipeInputSchema
from cable_thermal_model.model.cables.abstract_cable import CableType, WeightedScreenImpedance
from cable_thermal_model.model.cables.cable import (
    Cable,
)
from cable_thermal_model.model.cables.cable_trefoil_circuit_single_pipe import CableTrefoilCircuitSinglePipe
from cable_thermal_model.model.cables.enum_classes_cable import (
    CableLayer,
    CableScreenCirculatingLossType,
    CableScreenEddyLossType,
)
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


class PosCable(BaseModel, Generic[CableT]):
    """A positioned cable within a circuit, combining cable data with spatial coordinates."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    circuit_name: str
    cable_position: CablePosition
    cable: CableT
    x: float
    y: float

    @computed_field  # type: ignore[misc]
    @property
    def key(self) -> CableKey:
        """Return the CableKey identifying this cable in the circuit."""
        return CableKey(circuit_name=self.circuit_name, cable_position=self.cable_position)

    @computed_field  # type: ignore[misc]
    @property
    def cable_representation(self) -> str:
        """Return a string representation of this positioned cable for serialization."""
        return (
            f"PosCable(cable={type(self.cable).__name__}({self.cable.info}), x={self.x}, y={self.y}, "
            f"circuit_name={self.key.circuit_name}, cable_position={self.key.cable_position})"
        )

    def distance_to_point(self, x: float, y: float) -> float:
        """Return the distance from this cable center to a point in meters."""
        return float(np.hypot(self.x - x, self.y - y))

    def distance_to(self, other_cable: "PosCable") -> float:
        """Return the heart-to-heart distance to another positioned cable in meters."""
        return self.distance_to_point(other_cable.x, other_cable.y)


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


def return_mirror_cable(pos_cable: PosCable[CableT]) -> PosCable[CableT]:
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
        """Initialize the cable circuit with position, bonding type, and name."""
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
    def initialize_screen_loss_types(self) -> None:
        """Configure which screen-loss calculations the circuit's cables will use.

        Select circulating-current and eddy-current calculation types based on the
        circuit, bonding, and weighted screen impedance. Store them in each cable's
        layer metrics for use by its loss calculations. Called when the circuit is
        built and when set_weighted_screen_impedance updates the impedance.
        """
        raise NotImplementedError(
            f"Method initialize_screen_loss_types not implemented for class based on CableCircuit: "
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
    ) -> list[PosCable[Cable]]:
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
        self.initialize_screen_loss_types()

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
        """Initialize the single-cable circuit from CircuitInitData."""
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

    def initialize_screen_loss_types(self):
        """Set screen-loss calculation types for the single cable in this circuit.

        Raises:
            ValueError: If the cable type is unsupported.

        Limitations:
            - SL cables are not modeled.
                (IEC 60287-1-1:2023, 5.3.11 armoured / 5.3.2 unarmoured)

        References:
            - NEN-IEC 60287-1-1 (2023), section 5.3.9: screen eddy-current losses
              for three-core cables with a common sheath.
        """
        cable = self.cables[0].cable

        cable.layer_metrics.screen_circulating_loss_type = CableScreenCirculatingLossType.Zero
        if cable.cable_type in (CableType.XLPE, CableType.OilPressure):
            eddy_loss_type = CableScreenEddyLossType.RoundOrOvalWithScreen
        elif cable.cable_type == CableType.PILC:
            eddy_loss_type = CableScreenEddyLossType.Sector
        else:
            raise ValueError(f"CableType: {cable.cable_type} not supported.")
        cable.layer_metrics.screen_eddy_loss_type = (
            eddy_loss_type if CableLayer.Screen in cable.layers else CableScreenEddyLossType.Zero
        )


class LinearCircuit(CableCircuit):
    """A circuit of cables arranged in a linear (flat) formation."""

    def __init__(self, circuit_init_data: CircuitInitData):
        """Initialize the linear circuit from CircuitInitData."""
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

    def initialize_screen_loss_types(self):
        """Set circulating and eddy current calculation types for each cable in a linear configuration.

        Limitations:
            - Cross-bonded minor-section imbalance losses are not modeled
                (IEC 60287-1-1:2023, 5.3.7.2).
            - Generalized two-sided losses for Milliken conductors are not supported
                (IEC 60287-1-1:2023, 5.3.6; IEC 60287-1-3).

        References:
            - NEN-IEC 60287-1-1 (2023), section 5.3.7.1: eddy-current losses for
                single-point-bonded and cross-bonded cables.
            - NEN-IEC 60287-1-1 (2023), section 5.3.7.2: circulating-current losses for
                single-point-bonded and balanced cross-bonded cables.
            - NEN-IEC 60287-1-1 (2023), section 5.3.4: circulating-current losses for
                flat, untransposed cables bonded at both ends.
            - NEN-IEC 60287-1-1 (2023), section 5.3.6: Milliken conductor correction to
                two-sided eddy-current losses.
            - NEN-IEC 60287-1-3: generalized circulating-current losses.
        """
        for phase_idx, pos_cable in enumerate(self.cables):
            metrics = pos_cable.cable.layer_metrics
            metrics.screen_circulating_loss_type = CableScreenCirculatingLossType.Zero
            metrics.screen_eddy_loss_type = CableScreenEddyLossType.Zero
            if self.bonding in (BondingType.CrossBonding, BondingType.OneSided):
                metrics.screen_eddy_loss_type = (
                    CableScreenEddyLossType.LinearLeading,
                    CableScreenEddyLossType.LinearCenter,
                    CableScreenEddyLossType.LinearLagging,
                )[phase_idx]
            elif self.bonding == BondingType.TwoSided:
                if self.weighted_screen_impedance is not None:
                    metrics.screen_circulating_loss_type = (
                        CableScreenCirculatingLossType.GeneralizedLeading,
                        CableScreenCirculatingLossType.GeneralizedCenter,
                        CableScreenCirculatingLossType.GeneralizedLagging,
                    )[phase_idx]
                    metrics.screen_eddy_loss_type = CableScreenEddyLossType.GeneralizedTwoSided
                else:
                    metrics.screen_circulating_loss_type = (
                        CableScreenCirculatingLossType.LinearLeading,
                        CableScreenCirculatingLossType.LinearCenter,
                        CableScreenCirculatingLossType.LinearLagging,
                    )[phase_idx]
                    metrics.screen_eddy_loss_type = (
                        CableScreenEddyLossType.TwoSidedLinearLeading,
                        CableScreenEddyLossType.TwoSidedLinearCenter,
                        CableScreenEddyLossType.TwoSidedLinearLagging,
                    )[phase_idx]


class TrefoilCircuit(CableCircuit):
    """A circuit of three cables arranged in a trefoil formation."""

    def __init__(
        self,
        circuit_init_data: CircuitInitData,
    ):
        """Initialize the trefoil circuit from CircuitInitData."""
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

    def initialize_screen_loss_types(self):
        """Set circulating-current and eddy-current calculation types for each cable in a trefoil configuration.

        Limitations:
            - Cross-bonded minor-section imbalance losses are not modeled
                (IEC 60287-1-1:2023, 5.3.7.2).
            - Generalized two-sided losses for Milliken conductors are not supported
                (IEC 60287-1-1:2023, 5.3.6; IEC 60287-1-3).
        """
        for phase_idx, pos_cable in enumerate(self.cables):
            metrics = pos_cable.cable.layer_metrics
            metrics.screen_circulating_loss_type = CableScreenCirculatingLossType.Zero
            metrics.screen_eddy_loss_type = CableScreenEddyLossType.Zero
            if self.bonding in (BondingType.CrossBonding, BondingType.OneSided):
                metrics.screen_eddy_loss_type = CableScreenEddyLossType.Trefoil
            elif self.bonding == BondingType.TwoSided:
                if self.weighted_screen_impedance is not None:
                    metrics.screen_circulating_loss_type = (
                        CableScreenCirculatingLossType.GeneralizedLeading,
                        CableScreenCirculatingLossType.GeneralizedCenter,
                        CableScreenCirculatingLossType.GeneralizedLagging,
                    )[phase_idx]
                    metrics.screen_eddy_loss_type = CableScreenEddyLossType.GeneralizedTwoSided
                else:
                    metrics.screen_circulating_loss_type = CableScreenCirculatingLossType.Trefoil
                    metrics.screen_eddy_loss_type = CableScreenEddyLossType.TwoSidedTrefoil


class TrefoilCircuitInSinglePipe(SingleCable):
    """Trefoil circuit with three cables in one pipe.

    This is modeled as a single equivalent cable with an extra heat source between the equivalent cable and the pipe.
    """

    def __init__(self, circuit_init_data: CircuitInitData):
        """Initialize the trefoil-in-single-pipe circuit from CircuitInitData."""
        # Initialize super, but with two-sided bonding as default
        circuit_init_data.bonding_type = circuit_init_data.bonding_type or BondingType.TwoSided

        super().__init__(circuit_init_data)
        # Manually set the conductor distance
        cable = self.cables[0].cable
        if cable.layer_metrics.conductor_distance is None:
            cable.layer_metrics.conductor_distance = 2 * cable.layer_metrics.cable_radius

    def _get_cable_centers(self) -> dict[CablePosition, tuple[float, float]]:
        return {CablePosition.TrefoilCircuitInSinglePipe: (self.x, self.y)}

    def _validate_weighted_screen_impedance(self, weighted_screen_impedance: WeightedScreenImpedance | None) -> None:
        """Reject unsupported sheath currents before changing circuit or cable state."""
        if (
            self.bonding == BondingType.TwoSided
            and weighted_screen_impedance is not None
            and not self.symmetric_sheath_currents(weighted_screen_impedance.weighted_reactance_matrix)
        ):
            raise NotImplementedError(
                "Non-symmetric sheath currents are not supported for trefoil circuit in a single pipe."
            )

    def set_weighted_screen_impedance(self, weighted_screen_impedance: WeightedScreenImpedance | None):
        """Validate the proposed impedance before updating the circuit and cable loss cases."""
        self._validate_weighted_screen_impedance(weighted_screen_impedance)
        super().set_weighted_screen_impedance(weighted_screen_impedance)

    def initialize_screen_loss_types(self):
        """Set circulating-current and eddy-current loss types for the equivalent cable.

        Limitations:
            - Cross-bonded minor-section imbalance losses are not modeled
                (IEC 60287-1-1:2023, 5.3.7.2).

        Raises:
            ValueError: If two-sided bonding uses a weighted reactance matrix
                whose shape is not (2, 3).
            NotImplementedError: If two-sided bonding uses weighted screen
                impedance that produces non-symmetric sheath currents.
        """
        self._validate_weighted_screen_impedance(self.weighted_screen_impedance)
        circulating_loss_type = CableScreenCirculatingLossType.Zero
        eddy_loss_type = CableScreenEddyLossType.Zero
        if self.bonding in (BondingType.CrossBonding, BondingType.OneSided):
            eddy_loss_type = CableScreenEddyLossType.Trefoil

        elif self.bonding == BondingType.TwoSided:
            if self.weighted_screen_impedance is None:
                circulating_loss_type = CableScreenCirculatingLossType.Trefoil
                eddy_loss_type = CableScreenEddyLossType.TwoSidedTrefoil
            else:
                circulating_loss_type = CableScreenCirculatingLossType.GeneralizedCenter
                eddy_loss_type = CableScreenEddyLossType.GeneralizedTwoSided

        metrics = self.cables[0].cable.layer_metrics
        metrics.screen_circulating_loss_type = circulating_loss_type
        metrics.screen_eddy_loss_type = eddy_loss_type

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
            if isinstance(circuit_input.cable, CableTrefoilCircuitSinglePipe):
                circuit = TrefoilCircuitInSinglePipe(circuit_init_data)
            else:
                circuit = TrefoilCircuit(circuit_init_data)

        else:
            raise TypeError(f"Circuit type {circuit_input.circuit_type} not supported.")

        circuit.initialize_screen_loss_types()

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
