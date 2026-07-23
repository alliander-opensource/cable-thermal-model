# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0
import numpy as np
from scipy import linalg

from cable_thermal_model.model.cables.abstract_cable import (
    CableConductorProperties,
    CableConvectionParams,
    CableLayerMetrics,
    CableLayerProperties,
)
from cable_thermal_model.model.cables.cable import Cable
from cable_thermal_model.model.cables.enum_classes_cable import CableLayer, CableType


class CableAir(Cable):
    """Finite difference cable model with air discretization."""

    # Constants for the numerical integration process
    MAX_ITERATIONS_PER_TIMESTEP = 100
    MAX_ERROR_SHEATH = 0.001

    def __init__(
        self,
        conductor: CableConductorProperties,
        layer_properties: dict[CableLayer, CableLayerProperties],
        layer_metrics: CableLayerMetrics,
        cable_type: CableType,
        grid_counts: dict[CableLayer, int],
    ):
        """Initialize CableAir with convection parameters set to None until explicitly configured.

        Args:
            conductor (CableConductorProperties): Conductor properties of the cable.
            layer_properties (dict[CableLayer, CableLayerProperties]): Mapping of cable layers to their properties.
            layer_metrics (CableLayerMetrics): Geometric and calculated metrics for the cable layers.
            cable_type (CableType): The type of the cable.
            grid_counts (dict[CableLayer, int]): Number of grid points per cable layer.

        """
        self.convection_params: CableConvectionParams | None = None
        self.convection_coefficient: float | None = None
        super().__init__(conductor, layer_properties, layer_metrics, cable_type, grid_counts)

    def set_convection_parameters(self, Z: float, E: float, Cg: float):
        """Set the convection parameters used to compute the convection coefficient.

        Args:
            Z: Convection parameter Z.
            E: Convection parameter E.
            Cg: Convection parameter Cg.

        References:
            - NEN-IEC 60287-2-1 (2023) Section 4.2.1.

        """
        self.convection_params = CableConvectionParams(Z=Z, E=E, Cg=Cg)
        self.convection_coefficient = Z / (self.layer_metrics.outer_radius * 2) ** Cg + E

    def integrate_timestep(
        self,
        s: np.ndarray,
        b: np.ndarray,
        time_step: float,
        internal_heating: bool | None = True,
    ) -> np.ndarray:
        """Computes the temperature solution for the next time step.

        Computes the temperature solution at time step [t+1] given the solution at the
        current time step [t], the finite difference matrix, and the vector for [t].

        Args:
            s (np.ndarray): The solution of the heat equation [°C] at the
                previous timestep (t).
            b (np.ndarray): The finite difference vector [W/m³].
            time_step (float): The size of the time steps [s] in the linearized
                time grid.
            internal_heating (bool | None): A boolean representing whether
                internal heating is considered in this timestep. Must be None
                for this class.

        Returns:
            np.ndarray: The updated temperature solution at the new time step
                [t+1] for the cable.

        """
        if not internal_heating:
            raise ValueError("Internal heating must be True for cables in air.")

        temp_solution = s.copy()
        theta_N = temp_solution[-1]

        A_banded = self._banded_matrix
        A = np.zeros((A_banded.shape[0], A_banded.shape[1] + 1))

        A[:, :-1] = A_banded
        A[0, -1] = self.outer_boundary_coupling_coefficient
        A = -A * time_step
        A[1, :-1] += self._capacity_grid[:-1]
        A[2, -2] = 1

        b = b * time_step + self._capacity_grid[:-1] * s[:-1]

        b = np.append(b, 0.0)

        iteration = 0
        while True:
            iteration += 1

            A[1, -1] = -(1 + self._boundary_condition_coefficient * theta_N ** (1 / 4))

            temp_solution = linalg.solve_banded(l_and_u=(1, 1), ab=A, b=b)

            if abs(temp_solution[-1] - theta_N) <= self.MAX_ERROR_SHEATH:
                break
            elif iteration >= self.MAX_ITERATIONS_PER_TIMESTEP:
                raise ValueError(f"Solution did not converge after {self.MAX_ITERATIONS_PER_TIMESTEP} iterations")

            theta_N = temp_solution[-1]

        return temp_solution

    @property
    def _boundary_condition_coefficient(self) -> float:
        """This method calculates the coefficient for the boundary condition at the outer sheath in air.

        Returns:
            float: The boundary condition coefficient for the outer sheath in air.

        """
        if self.convection_coefficient is None:
            raise ValueError("Convection coefficient is not set. Please set convection parameters first.")

        r_N = self._radii_grid[-1]
        delta_min = self._grid_deltas[-1]
        r_N_min = r_N - 0.5 * delta_min

        return self.convection_coefficient * delta_min * self._rho_grid[-1] * r_N / r_N_min
