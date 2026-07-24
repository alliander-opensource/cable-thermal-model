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
    """Finite difference cable model for a trefoil circuit with a single pipe base class.

    Attributes:
        _bottomright_index (tuple[int, int]): Index tuple (row, col) for accessing the bottom-right element
            of the sparse matrix used in convection boundary condition updates.

    """

    _bottomright_index: tuple[int, int] = (-1, -1)

    def _processed_matrix(self, time_step: float) -> np.ndarray | sparse.lil_matrix:

        ab = self._banded_matrix

        # Convert the banded matrix to a sparse matrix
        # Use dia format for easy conversion and then convert to lil format to set individual elements
        A_sparse = sparse.dia_matrix((ab, [1, 0, -1]), shape=(ab.shape[1], ab.shape[1])).tolil()

        # Add coefficients to the matrix, representing adding an internal heat source
        # that depends on the heat that passes through the cable boundary.
        A_sparse = self._update_system_with_heat_source(A_sparse)

        A_sparse = -A_sparse * time_step
        A_sparse += sparse.diags(diagonals=self._capacity_grid[: ab.shape[1]])

        return A_sparse

    @staticmethod
    def _solve_system(
        A: np.ndarray | sparse.lil_matrix,
        b: np.ndarray,
    ) -> np.ndarray:
        """Solve the linear system Ax = b using sparse solver.

        Args:
            A (np.ndarray | sparse.lil_matrix): The finite difference matrix (sparse format for trefoil).
            b (np.ndarray): The finite difference vector.

        Returns:
            np.ndarray: The solution vector x.

        Raises:
            TypeError: If A is not a sparse matrix.

        """
        if not sparse.issparse(A):
            raise TypeError(f"Expected sparse matrix, got {type(A).__name__}")

        # Convert to CSC format for optimal solver performance
        # (spsolve is fastest/most reliable with CSC/CSR inputs)
        return sparse.linalg.spsolve(A.tocsc(), b)

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

    def _get_filling_internal_heating_coefficient(self, s: int, m: int) -> float:
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
        inter_rho = self._calculate_inter_rhos(
            self._radii_grid[s : s + 2], self._inter_radii[s : s + 1], self._rho_grid[s : s + 2]
        )[0]
        return (
            2
            * self.layer_metrics.cable_radius
            / (
                inter_rho
                * self._radii_grid[m]
                * (self._radii_grid[s + 1] - self._radii_grid[s])
                * (self._radii_grid[m + 1] - self._radii_grid[m - 1])
            )
        )


class CableTrefoilCircuitSinglePipeInSoil(CableTrefoilCircuitSinglePipe, CableSoil):
    """Finite difference cable model for a trefoil circuit with a single pipe in soil."""


class CableTrefoilCircuitSinglePipeInAir(CableTrefoilCircuitSinglePipe, CableAir):
    """Finite difference cable model for a trefoil circuit with a single pipe in air."""
