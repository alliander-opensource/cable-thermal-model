# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0
import numpy as np
from scipy import sparse

from cable_thermal_model.model.cables.cable import Cable
from cable_thermal_model.model.cables.cable_air import CableAir
from cable_thermal_model.model.cables.cable_soil import CableSoil
from cable_thermal_model.model.cables.enum_classes_cable import CableLayer


class CableTrefoilCircuitSinglePipe(Cable):
    """Finite difference cable model for a trefoil circuit with a single pipe base class."""

    def _update_system_with_heat_source(self, A_sparse: sparse.lil_matrix) -> sparse.lil_matrix:
        """Add coefficients to the finite difference matrix.

        The added coefficients represent an internal heat source between the
        pipe and the equivalent cable representing the trefoil circuit. The
        amount of heat added equals twice the heat loss at the cable sheath,
        therefore representing the heat three cables in trefoil would
        generate together.

        Args:
            A_sparse (sparse.lil_matrix): The finite difference matrix
                [W/(°C*m³)] represented as a sparse lil matrix.

        Returns:
            sparse.lil_matrix:
                The updated finite difference matrix [W/(°C*m³)] represented as a sparse lil matrix.

        """
        # Determine the indices m (filling_heat_source_layer) and s (outer_sheath_index) where r_s<r_cable<r_{s+1}
        # and m (filling_heat_source_layer) is such that r_m^-<2*r_cable < r_m^+.
        # Since r_m^- and r_m^+ lie exactly between grid points, we can find
        # m by searching for the grid point closest to 2*r_cable.
        filling_heat_source_layer = int(np.abs(self._radii_grid - 2 * self.layer_metrics.cable_radius).argmin())
        _, outer_sheath_index = self.get_layer_indices_for_layer(CableLayer.Sheath)

        # Calculate the filling internal heating coefficient
        filling_internal_heating_coefficient = self._get_filling_internal_heating_coefficient(
            s=outer_sheath_index, m=int(filling_heat_source_layer)
        )

        # Add matrix entries at coordinates (m, s) and (m, s+1).
        # This indicates that the heat added at layer m depends on the temperature difference between layers s and s+1.
        A_sparse[filling_heat_source_layer, outer_sheath_index] = 2 * filling_internal_heating_coefficient
        A_sparse[filling_heat_source_layer, outer_sheath_index + 1] = -2 * filling_internal_heating_coefficient

        return A_sparse

    def _get_filling_internal_heating_coefficient(self, s, m) -> float:
        """This method calculates the internal heating coefficient for the filling material in the pipe.

        This coefficient represents the factor with which to multiply the heat generation at layer s, if one wants
        to add this as a heat source at layer m.

        Args:
            s (int): The index of the outer sheath layer.
            m (int): The index of the filling heat source layer.

        Returns:
            float: The internal heating coefficient for the filling material in
                the pipe.

        """
        # Calculate the thermal resistivity at the interstitial point between the grid points r_s and r_{s+1}
        r_s = self._radii_grid[s]
        inter_radius = np.array([r_s + 0.5 * self._grid_deltas[s]])
        inter_rho = self._calculate_inter_rhos(self._radii_grid[s : s + 2], inter_radius, self._rho_grid[s : s + 2])[0]
        return (
            2
            * self.layer_metrics.cable_radius
            / (
                inter_rho
                * self._radii_grid[m]
                * self._grid_deltas[s]
                * (self._radii_grid[m + 1] - self._radii_grid[m - 1])
            )
        )


class CableTrefoilCircuitSinglePipeInSoil(CableTrefoilCircuitSinglePipe, CableSoil):
    """Finite difference cable model for a trefoil circuit with a single pipe in soil."""

    def integrate_timestep(
        self,
        s: np.ndarray,
        b: np.ndarray,
        time_step: float,
        internal_heating: bool | None = None,
    ) -> np.ndarray:
        """This method solves the finite difference approximation to the heat equation using the implicit Euler method.

        We add a heat source between the pipe and the equivalent cable
        representing the trefoil circuit in the internal heating step. The
        amount of heat added equals twice the heat loss at the cable sheath,
        therefore representing the heat three cables in trefoil would
        generate together. Because we add a heat source between
        the pipe and the equivalent cable representing the trefoil circuit,
        the banded array is converted to a sparse matrix and adjusted
        appropriately before solving the linear system.

        Args:
            s (np.ndarray): The solution of the heat equation [°C] at the
                previous timestep (t).
            b (np.ndarray): The finite difference vector [W/m³].
            time_step (float): The size of the time steps [s] in the linearized
                time grid.
            internal_heating (bool): A boolean indicating whether internal
                heating between cables in the trefoil circuit is considered.

        Returns:
            np.ndarray: The solution [°C] to the heat equation at the next timestep (t+1) for all grid points except
                the final grid point, at which a boundary condition is enforced.

        """
        A_banded = self._banded_matrix
        if internal_heating is None:
            raise ValueError("The internal_heating parameter must be provided for CableTrefoilCircuitSinglePipeInSoil.")

        # Only add an extra heat source if internal heating is considered
        if not internal_heating:
            return super().integrate_timestep(s, b, time_step)

        # Convert the banded matrix to a sparse matrix
        # Use dia format for easy conversion and then convert to lil format to set individual elements
        A_sparse = sparse.dia_matrix((A_banded, [1, 0, -1]), shape=(A_banded.shape[1], A_banded.shape[1])).tolil()

        # Add coefficients to the matrix, representing adding an internal heat source
        # that depends on the heat that passes through the cable boundary
        A_sparse = self._update_system_with_heat_source(A_sparse)

        # Compute the other vectors that are required to solve the linear system
        capacity_vector = self._capacity_grid[:-1]
        capacity_diagonal_matrix = sparse.diags(diagonals=capacity_vector)

        return sparse.linalg.spsolve(
            capacity_diagonal_matrix - time_step * A_sparse, capacity_vector * s + time_step * b
        )


class CableTrefoilCircuitSinglePipeInAir(CableTrefoilCircuitSinglePipe, CableAir):
    """Finite difference cable model for a trefoil circuit with a single pipe in air."""

    def integrate_timestep(
        self,
        s: np.ndarray,
        b: np.ndarray,
        time_step: float,
        internal_heating: bool | None = True,
    ) -> np.ndarray:
        """This method solves the finite difference approximation to the heat equation using the implicit Euler method.

        We add a heat source between the pipe and the equivalent cable
        representing the trefoil circuit in the internal heating step. The
        amount of heat added equals twice the heat loss at the cable sheath,
        therefore representing the heat three cables in trefoil would
        generate together. Because we add a heat source between
        the pipe and the equivalent cable representing the trefoil circuit,
        the banded array is converted to a sparse matrix and adjusted
        appropriately before solving the linear system.

        Args:
            s (np.ndarray): The solution of the heat equation [°C] at the
                previous timestep (t).
            b (np.ndarray): The finite difference vector [W/m³].
            time_step (float): The size of the time steps [s] in the linearized
                time grid.
            internal_heating (bool | None): A boolean indicating whether
                internal heating between cables in the trefoil circuit is
                considered.

        Raises:
            ValueError:
                If the convection parameters have not been set for this cable in air.

        Returns:
            np.ndarray: The solution [°C] to the heat equation at the next timestep (t+1) for all grid points except
                the final grid point, at which a boundary condition is enforced.

        """
        if not internal_heating:
            raise ValueError("Internal heating must be True for cables in air.")

        if self.convection_coefficient is None:
            raise ValueError("Convection parameters have not been set for this cable in air!")

        temp_solution = s.copy()
        theta_N = temp_solution[-1]

        A_banded = self._banded_matrix
        A = np.zeros((A_banded.shape[0], A_banded.shape[1] + 1))

        A[:, :-1] = A_banded
        A[0, -1] = self.outer_boundary_coupling_coefficient
        A[2, -2] = 1

        # Convert the banded matrix to a sparse matrix
        # Use dia format for easy conversion and then convert to lil format to set individual elements
        A_sparse = sparse.dia_matrix((A, [1, 0, -1]), shape=(A.shape[1], A.shape[1])).tolil()

        # Add coefficients to the matrix, representing adding an internal heat source
        # that depends on the heat that passes through the cable boundary.
        A_sparse = self._update_system_with_heat_source(A_sparse)

        # Compute the other vectors that are required to solve the linear system
        capacity_vector = self._capacity_grid[:-1]
        capacity_vector = np.append(capacity_vector, 0.0)
        capacity_diagonal_matrix = sparse.diags(diagonals=capacity_vector)
        b = np.append(b, 0.0)

        iteration = 0
        while True:
            iteration += 1

            # Update the last diagonal element at each iteration
            A_sparse[-1, -1] = -(1 + self._boundary_condition_coefficient * theta_N ** (1 / 4))

            temp_solution = sparse.linalg.spsolve(
                capacity_diagonal_matrix - time_step * A_sparse, capacity_vector * s + time_step * b
            )

            if abs(temp_solution[-1] - theta_N) <= self.MAX_ERROR_SHEATH:
                break
            elif iteration >= self.MAX_ITERATIONS_PER_TIMESTEP:
                raise ValueError(f"Solution did not converge after {self.MAX_ITERATIONS_PER_TIMESTEP} iterations")

            theta_N = temp_solution[-1]

        return temp_solution
