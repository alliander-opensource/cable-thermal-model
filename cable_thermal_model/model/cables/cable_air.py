# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0
import numpy as np

from cable_thermal_model.model.cables.abstract_cable import (
    CableConductorProperties,
    CableConvectionParams,
    CableLayerMetrics,
    CableLayerProperties,
)
from cable_thermal_model.model.cables.cable import Cable
from cable_thermal_model.model.cables.enum_classes_cable import CableLayer, CableType


class CableAir(Cable):
    """Finite difference cable model with air discretization.

    Attributes:
        _bottomright_index (tuple[int, int]): Index tuple (row, col) for accessing the bottom-right element
            of the banded matrix used in convection boundary condition updates.

    """

    _bottomright_index: tuple[int, int] = (1, -1)

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

    def _set_heating_vector(self) -> None:
        """Initialize the heating vector for the cable."""
        self._heating_vector = np.zeros(self._radii_grid.size)

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
        previous_solution: np.ndarray,
        time_step: float,
    ) -> np.ndarray:
        """Computes the temperature solution for the next time step.

        Computes the temperature solution at time step [t+1] given the solution at the
        current time step [t], the finite difference matrix, and the vector for [t].

        Args:
            previous_solution (np.ndarray): The solution of the heat equation [°C] at the
                previous timestep (t).
            time_step (float): The size of the time steps [s] in the linearized
                time grid.

        Returns:
            np.ndarray: The updated temperature solution at the new time step
                [t+1] for the cable.

        """
        A = self._processed_matrix(time_step=time_step)

        b = self._heating_vector * time_step + self._capacity_grid * previous_solution

        temp_solution = previous_solution.copy()
        theta_N = temp_solution[-1]

        iteration = 0
        while True:
            iteration += 1

            A[self._bottomright_index] += self._boundary_condition_coefficient * theta_N ** (1 / 4) * time_step
            temp_solution = self._solve_system(A=A, b=b)

            if abs(temp_solution[-1] - theta_N) <= self.MAX_ERROR_SHEATH:
                break
            elif iteration >= self.MAX_ITERATIONS_PER_TIMESTEP:
                raise ValueError(f"Solution did not converge after {self.MAX_ITERATIONS_PER_TIMESTEP} iterations")

            A[self._bottomright_index] -= self._boundary_condition_coefficient * theta_N ** (1 / 4) * time_step
            theta_N = temp_solution[-1]

        return temp_solution

    def _get_finite_difference_matrix_diagonals(self):
        """Build the three diagonals of the finite difference matrix.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the upper diagonal, base diagonal,
                and lower diagonal of the finite difference matrix.

        """
        upper_diagonal, base_diagonal, lower_diagonal = super()._get_finite_difference_matrix_diagonals()

        # Extend the diagonals to account for the boundary condition at the outer sheath in air
        common_factor_second_derivative = self._common_factors_second_derivative(
            radii=np.append(self._radii_grid[-2:], self._radii_grid[-1]),
            inter_radii=np.append(self._inter_radii[-1:], self._radii_grid[-1]),
        )[0]
        common_factor_first_derivative = self._common_factors_first_derivative(
            radii=self._radii_grid[-2:], inter_radii=self._inter_radii[-1:], rhos=self._rho_grid[-2:]
        )[0]

        new_element = common_factor_second_derivative * common_factor_first_derivative
        base_diagonal = np.append(base_diagonal, -new_element)
        lower_diagonal = np.append(lower_diagonal, new_element)

        return upper_diagonal, base_diagonal, lower_diagonal

    @property
    def _boundary_condition_coefficient(self) -> float:
        """This method calculates the coefficient for the boundary condition at the outer sheath in air.

        Returns:
            float: The boundary condition coefficient for the outer sheath in air.

        """
        if self.convection_coefficient is None:
            raise ValueError("Convection coefficient is not set. Please set convection parameters first.")

        common_factor_second_derivative = self._common_factors_second_derivative(
            radii=np.append(self._radii_grid[-2:], self._radii_grid[-1]),
            inter_radii=np.append(self._inter_radii[-1:], self._radii_grid[-1]),
        )[0]

        return common_factor_second_derivative * self._radii_grid[-1] * self.convection_coefficient
