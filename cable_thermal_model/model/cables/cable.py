# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0
from copy import deepcopy
from typing import Any, Self

import numpy as np
from scipy import linalg, sparse

from cable_thermal_model.model.cables.abstract_cable import (
    AbstractCable,
    CableConductorProperties,
    CableLayerMetrics,
    CableLayerProperties,
)
from cable_thermal_model.model.cables.enum_classes_cable import CableLayer, CableType
from cable_thermal_model.model.cables.pipe import Pipe


# noinspection PyPep8Naming
class Cable(AbstractCable):
    """Finite difference cable model that discretizes cable layers into a radial grid."""

    def __init__(
        self,
        conductor: CableConductorProperties,
        layer_properties: dict[CableLayer, CableLayerProperties],
        layer_metrics: CableLayerMetrics,
        cable_type: CableType,
        grid_counts: dict[CableLayer, int],
    ) -> None:
        """Initialize the Cable with conductor properties, layer data, and grid resolution.

        Args:
            conductor (CableConductorProperties): Conductor properties of the cable.
            layer_properties (dict[CableLayer, CableLayerProperties]): Mapping of cable layers to their properties.
            layer_metrics (CableLayerMetrics): Geometric and calculated metrics for the cable layers.
            cable_type (CableType): The type of the cable.
            grid_counts (dict[CableLayer, int]): Number of grid points per cable layer.

        """
        super().__init__(conductor, layer_properties, layer_metrics, cable_type)

        self._validate_grid_counts(grid_counts)

        self._grid_counts = grid_counts
        self._radii_grid = np.array([], dtype=float)
        self._inter_radii = np.array([], dtype=float)
        self._surface_area_grid = np.array([], dtype=float)
        self._capacity_grid = np.array([], dtype=float)
        self._rho_grid = np.array([], dtype=float)

        self._upper_diagonal = np.array([], dtype=float)
        self._base_diagonal = np.array([], dtype=float)
        self._lower_diagonal = np.array([], dtype=float)
        self._finite_difference_matrix_diagonals_outdated = True

        self._set_calculated_fields()

    @property
    def info(self) -> str:
        """Return a compact string encoding the cable's physical properties."""
        return (
            f"{tuple([layer_properties.outer_radius for layer_properties in self.layer_properties.values()])},"
            f"{tuple([layer_properties.rho for layer_properties in self.layer_properties.values()])},"
            f"{tuple([layer_properties.capacity for layer_properties in self.layer_properties.values()])},"
            f"{tuple([layer_properties.electric_rho for layer_properties in self.layer_properties.values()])},"
            f"{tuple([layer_properties.alpha for layer_properties in self.layer_properties.values()])},"
            f"{tuple(self.layers)},"
            f"{self.layer_metrics.outer_radius},"
            f"{self.layer_metrics.conductor_cross_section},"
            f"{self.layer_metrics.screen_cross_section},"
            f"{self.conductor.number_of_conductors.value},"
            f"{self.layer_metrics.conductor_distance},"
            f"{self.cable_type}"
        )

    @property
    def upper_diagonal_last_element(self) -> float:
        """Get the outer-boundary coupling coefficient from the finite difference matrix.

        The outer-boundary coupling coefficient is a value that represents the thermal interaction at the outer boundary
        of the cable. It is derived from the finite difference matrix and is used in the heat equation calculations.

        Returns:
            float: The outer-boundary coupling coefficient.

        """
        self._update_finite_difference_matrix_diagonals_if_needed()

        return float(self._upper_diagonal[-1])

    @property
    def grid_size(self) -> int:
        """Get the total number of grid points in the finite difference model.

        Returns:
            int: The total number of grid points.

        """
        return len(self._radii_grid)

    @property
    def _banded_matrix(self) -> np.ndarray:
        """Get the finite difference matrix in banded form.

        The finite difference matrix is central to the linearized heat equation. It is a matrix with one base
        diagonal and two "off" diagonals (one above and one below the base diagonal),
        and otherwise only zeros. We represent this matrix as a 3xN numpy array, where N
        is the length of the base diagonal.

        Notes:
            In the finite difference (FD) approximation, this single matrix combined with a vector control the
            linearized heat equation.

        Returns:
            np.ndarray: A 3xN numpy array representing the finite difference matrix in banded form.

        """
        self._update_finite_difference_matrix_diagonals_if_needed()

        matrix = np.zeros((3, len(self._base_diagonal)))

        matrix[0, 1:] = self._upper_diagonal[:-1]
        matrix[1, :] = self._base_diagonal
        matrix[2, :-1] = self._lower_diagonal

        return matrix

    def _get_redefined_cable(self, **kwargs) -> Self:
        """Get a new cable instance based on the current self, but with changed cable attributes.

        This method takes the parameters given in the **kwargs and tries to apply those to matching attributes in a
         copy made of the current self.

        Examples:
            An example where we create a cable, and then use this method to create a copy of the cable, but with the
            [rhos] and [capacities] attributes altered from their original values.
            >>> cable = Cable()
            >>> new_cable = cable._get_redefined_cable(rhos = (1,1,1), capacities = (5,5,5))

            (For other applications, please check out the 'add_soil' and 'add_outer_tube' methods.)

        Args:
            **kwargs:
                    Kwargs here is used to pass along cable parameters that would usually be configured using the
                    initializer. Recognized parameters will overwrite existing values, while other parameters will
                    be ignored.
                    (Some examples of parameters that could be changed in this way: 'rhos','radii','grid_counts')

        Returns:
            Self: A completely new Cable instance based on the cable the method was
                called from, but with changed cable properties based on the passed
                [**kwargs] parameters.

        Notes:
            There are two reasons this method should be re-evaluated in the future. First of all this method uses
            kwargs to pass along an unknown combination of parameters, which is only evaluated by parameter name.
            Secondly this method is found in the Cable class, but it is not specific to the Cable class.

        """
        new_cable = deepcopy(self)

        # Check all the items in the kwargs and apply them to the new cable if they are recognized as existing.
        for key, value in kwargs.items():
            if hasattr(new_cable, key):
                setattr(new_cable, key, value)

        # Recalculate the calculated fields of the new cable and reset the solution values
        new_cable._set_calculated_fields()

        return new_cable

    def _set_calculated_fields(self) -> None:
        """Initialize derived cable properties.

        The properties set in this function depend on the cable layers. When
        adding soil or pipe layers these need to be reset. This function can be used to do so.
        """
        self._radii_grid = self._construct_radii_grid()
        self._inter_radii = self._radii_grid[:-1] + 0.5 * np.diff(self._radii_grid)
        self._surface_area_grid = self._construct_surface_area_grid(self._radii_grid)

        capacity_grids = [
            np.full(self._grid_counts[layer], self.layer_properties[layer].capacity) for layer in self.layers
        ]
        self._capacity_grid = np.concatenate(capacity_grids)

        rho_grids = [np.full(self._grid_counts[layer], self.layer_properties[layer].rho) for layer in self.layers]
        self._rho_grid = np.concatenate(rho_grids)
        self._invalidate_finite_difference_matrix_diagonals()

    def _construct_radii_grid(self, maximal_boundary_distance: float = 0.000_1) -> np.ndarray:
        """Construct the radii grid for the cable based on the layer properties and grid counts.

        Args:
            maximal_boundary_distance (float): The maximal distance to use as a boundary distance between layers [m].
                Default is 0.1 mm.

        Returns:
            np.ndarray: A Numpy array representing the radii grid for the cable.

        """
        last_layer = self.layers[-1]

        boundary_distance = 0.0
        radii_grids: list[np.ndarray] = []
        for layer_idx, layer in enumerate(self.layers):
            start = self.layer_properties[layer].inner_radius + boundary_distance

            if layer == last_layer:
                end = self.layer_properties[layer].outer_radius

            else:
                next_layer = self.layers[layer_idx + 1]
                boundary_distance = min(
                    [
                        maximal_boundary_distance,
                        (self.layer_properties[layer].outer_radius - start) / (2 * (self._grid_counts[layer] - 0.5)),
                        (
                            self.layer_properties[next_layer].outer_radius
                            - self.layer_properties[next_layer].inner_radius
                        )
                        / (2 * self._grid_counts[next_layer]),
                    ]
                )
                end = self.layer_properties[layer].outer_radius - boundary_distance

            if layer not in CableLayer.soil_layers():
                radii_grids.append(np.linspace(start=start, stop=end, num=self._grid_counts[layer]))
            else:
                # For soil layers, we want to use a logarithmic grid to better
                # capture the temperature gradients close to the cable.
                radii_grids.append(
                    np.logspace(start=0.0, stop=np.log2(end / start), num=self._grid_counts[layer], base=2) * start
                )

        return np.concatenate(radii_grids)

    def integrate_timestep(
        self,
        previous_solution: np.ndarray,
        heating_vector: np.ndarray,
        time_step: float,
    ) -> np.ndarray:
        """Computes the temperature solution for the next time step.

        An abstract method that is implemented differently for different cable types, as the integration method may
        differ depending on the cable type.
        """
        raise NotImplementedError("This method should be implemented in child classes of Cable.")

    def update_pipe_fill_resistivity(self, temperature_grid: np.ndarray) -> None:
        """This method updates the (temperature dependent) thermal resistivity of the medium in the pipe of the cable.

        Args:
            temperature_grid (np.ndarray): The temperature grid for the cable, as calculated for a given timestep.

        """
        if self.layer_metrics.pipe is None:
            raise ValueError("Pipe is not set. Cannot update pipe fill resistivity.")
        if self.layer_metrics.pipe.inner_radius is None:
            raise ValueError("Pipe inner radius is not set. Cannot update pipe fill resistivity.")

        Tfill = self._get_mean_temperature_cable_layer(temperature_grid=temperature_grid, layer=CableLayer.PipeFill)

        new_pipe_fill_rho = self.layer_metrics.pipe.get_thermal_resistivity_pipe_fill(Tfill)
        pipe_fill_start_index, pipe_fill_end_index = self.get_layer_indices_for_layer(CableLayer.PipeFill)
        self._update_rho_grid(
            start_index=pipe_fill_start_index,
            end_index=pipe_fill_end_index,
            rho=new_pipe_fill_rho,
        )

    def get_layer_indices_for_layer(self, layer: CableLayer) -> tuple[int, int]:
        """This method fetches the inclusive start and end indices of the grid points for a given layer.

        Args:
            layer (CableLayer): A CableLayer object representing the layer for
                which the indices need to be fetched.

        Returns:
            tuple[int, int]: A tuple of integers representing the inclusive start and end
                indices of the grid points for the given layer, in that order.

        """
        layer_index = self.layers.index(layer)

        layer_start_index = sum([self._grid_counts[layer] for layer in self.layers[:layer_index]])
        layer_end_index = layer_start_index + self._grid_counts[layer] - 1
        return layer_start_index, layer_end_index

    @staticmethod
    def _construct_surface_area_grid(radii_grid: np.ndarray) -> np.ndarray:
        """Construct the surface area grid for the cable based on the radii grid.

        Args:
            radii_grid (np.ndarray):
                A Numpy array representing the radii grid for the cable.

        Returns:
            np.ndarray:
                A Numpy array representing the surface area grid for the cable.

        """
        # The radii_grid should start at 0.0 and be strictly increasing
        if not np.isclose(radii_grid[0], 0.0):
            raise ValueError("The first value of the radii grid should be 0.0!")
        if not np.all(np.diff(radii_grid) > 0):
            raise ValueError("The radii grid should be strictly increasing!")

        # Create a surface area grid of N-1 values
        surface_area_grid = np.zeros(radii_grid.size - 1)
        surface_area_grid[0] = np.pi * (radii_grid[1] / 2) ** 2

        surface_area_grid[1:] = np.pi * radii_grid[1:-1] * (radii_grid[2:] - radii_grid[0:-2])

        return surface_area_grid

    def _invalidate_finite_difference_matrix_diagonals(self) -> None:
        """Invalidate the cached finite difference matrix diagonals.

        This method marks the finite difference matrix diagonals as outdated, indicating that they need to be updated
        before the next use. This should be called when rho_grid changes, affecting the finite difference matrix.

        """
        self._finite_difference_matrix_diagonals_outdated = True

    def _get_finite_difference_matrix_diagonals(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the three diagonals of the finite difference matrix.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the upper diagonal, base diagonal,
                and lower diagonal of the finite difference matrix.

        """
        radii = self._radii_grid
        inter_radii = self._inter_radii

        common_factors_first_derivative = self._common_factors_first_derivative(radii, inter_radii, self._rho_grid)
        common_factors_second_derivative = self._common_factors_second_derivative(radii, inter_radii)

        upper_inter = common_factors_first_derivative[1:] * common_factors_second_derivative
        lower_diagonal = common_factors_first_derivative[:-1] * common_factors_second_derivative
        base_inter = -(upper_inter + lower_diagonal)

        boundary_value = 2 / (self._rho_grid[0] * inter_radii[0] * radii[1])
        upper_diagonal = np.append([boundary_value], upper_inter)
        base_diagonal = np.append([-boundary_value], base_inter)

        return upper_diagonal, base_diagonal, lower_diagonal

    def _common_factors_first_derivative(
        self,
        radii: np.ndarray,
        inter_radii: np.ndarray,
        rhos: np.ndarray,
    ) -> np.ndarray:
        """Calculate the common factor used in the finite difference matrix for the first derivative.

        Args:
            radii (np.ndarray): A Numpy array representing the radii grid for the cable.
            inter_radii (np.ndarray): A Numpy array representing the interstitial radii grid for the cable.
            rhos (np.ndarray): A Numpy array representing the resistivity values at the grid points.

        Returns:
            np.ndarray: The common finite difference factor for the first derivative.

        """
        inter_rhos = self._calculate_inter_rhos(radii=radii, inter_radii=inter_radii, rhos=rhos)
        radii_deltas = np.diff(radii)

        return inter_radii / (inter_rhos * radii_deltas)

    @staticmethod
    def _common_factors_second_derivative(radii: np.ndarray, inter_radii: np.ndarray) -> np.ndarray:
        """Calculate the common factor used in the finite difference matrix.

        Args:
            radii (np.ndarray): A Numpy array representing the radii grid for the cable.
            inter_radii (np.ndarray): A Numpy array representing the interstitial radii grid for the cable.

        Returns:
            np.ndarray: The common finite difference factor for the given radii grid.

        """
        return 1 / (radii[1:-1] * (inter_radii[1:] - inter_radii[:-1]))

    def _update_finite_difference_matrix_diagonals_if_needed(self) -> None:
        """This method updates the three diagonals of the finite difference matrix if they are outdated.

        The three diagonals share the same grid geometry and interstitial resistivity values.
        This method computes those shared terms and derives the three diagonals from them.
        """
        if not self._finite_difference_matrix_diagonals_outdated:
            return

        upper_diagonal, base_diagonal, lower_diagonal = self._get_finite_difference_matrix_diagonals()

        self._upper_diagonal = upper_diagonal
        self._base_diagonal = base_diagonal
        self._lower_diagonal = lower_diagonal
        self._finite_difference_matrix_diagonals_outdated = False

    def _calculate_inter_rhos(self, radii: np.ndarray, inter_radii: np.ndarray, rhos: np.ndarray) -> np.ndarray:
        """This method calculates the interstitial resistivity values between grid points.

        Args:
            radii (np.ndarray):
                A numpy array with the positions of N grid points to use.
            inter_radii (np.ndarray):
                A numpy array with the positions of N-1 interstitial grid points to use.
            rhos (np.ndarray):
                A numpy array with the resistivity values at the N grid points.

        Returns:
            np.ndarray:
                A numpy array representing the calculated interstitial resistivity values.

        """
        if not len(radii) == len(inter_radii) + 1 == len(rhos):
            raise ValueError("The lengths of the input arrays are inconsistent!")

        if not np.all(np.diff(radii) > 0):
            raise ValueError("The radii array must be strictly increasing!")

        if radii[0] < 0:
            raise ValueError("All radii must be non-negative!")

        if not np.all(inter_radii >= radii[:-1]) or not np.all(inter_radii <= radii[1:]):
            raise ValueError("All inter_radii values must be between the corresponding radii values!")

        if np.isclose(radii[0], 0.0):
            if not np.isclose(rhos[0], rhos[1]):
                raise ValueError(
                    "For the finite difference method, it is assumed that the "
                    "resistivity between the first two grid points is "
                    "constant. This assumption is violated."
                )

            inter_rhos = np.empty(len(inter_radii))
            inter_rhos[0] = rhos[0]

            inter_rhos[1:] = self._calculate_inter_rhos(radii[1:], inter_radii[1:], rhos[1:])
            return inter_rhos

        return (rhos[:-1] * np.log(inter_radii / radii[:-1]) + rhos[1:] * np.log(radii[1:] / inter_radii)) / np.log(
            radii[1:] / radii[:-1]
        )

    def _update_rho_grid(self, start_index: int, end_index: int, rho: float) -> None:
        """Update a slice of the rho-grid with a new value if significant change is detected.

        Args:
            start_index (int): The starting index of the slice to update (inclusive).
            end_index (int): The ending index of the slice to update (inclusive).
            rho (float): The new resistivity value to set for the specified slice.

        """
        if start_index > end_index:
            raise ValueError("The start_index exceeds the end_index. Cannot update the rho grid.")

        rho_slice = self._rho_grid[start_index : end_index + 1]
        if np.all(np.isclose(rho_slice, rho, rtol=1e-2)):
            return

        self._rho_grid[start_index : end_index + 1] = rho
        self._invalidate_finite_difference_matrix_diagonals()

    def _update_capacity_grid(self, start_index: int, end_index: int, capacity: float) -> None:
        """Update a slice of the capacity-grid with a new value if significant change is detected.

        Args:
            start_index (int): The starting index of the slice to update (inclusive).
            end_index (int): The ending index of the slice to update (inclusive).
            capacity (float): The new capacity value to set for the specified slice.

        """
        if start_index > end_index:
            raise ValueError("The start_index exceeds the end_index. Cannot update the capacity grid.")

        capacity_slice = self._capacity_grid[start_index : end_index + 1]
        if np.all(np.isclose(capacity_slice, capacity, rtol=1e-2)):
            return

        self._capacity_grid[start_index : end_index + 1] = capacity

    def _update_vector_with_heat_generation_for_layer(
        self, vector: np.ndarray, heat_generation: float, layer: CableLayer
    ) -> np.ndarray:
        """Update the vector with heat generation distributed over one cable layer.

        Args:
            vector (np.ndarray): The vector to update.
            heat_generation (float): The heat generation value in W/m.
            layer (CableLayer): The cable layer over which to distribute the heat generation.

        Returns:
            np.ndarray: The updated vector with heat generation distributed across the selected layer.

        """
        start_index, end_index = self.get_layer_indices_for_layer(layer)
        vector[start_index : end_index + 1] = (
            heat_generation / self._surface_area_grid[start_index : end_index + 1].sum()
        )
        return vector

    def _get_mean_temperature_cable_layer(self, temperature_grid: np.ndarray, layer: CableLayer) -> float:
        """Calculate the mean temperature for a cable layer.

        Args:
            temperature_grid (np.ndarray): The temperature grid for the cable, as calculated for a given timestep.
            layer (CableLayer): The cable layer for which the mean temperature needs to be calculated.

        Returns:
            float: The mean temperature for the given layer.
        """
        if layer not in self.layers:
            raise ValueError(f"Layer {layer} is not present in the cable.")

        layer_start, layer_end = self.get_layer_indices_for_layer(layer)
        return float((temperature_grid[layer_start] + temperature_grid[layer_end]) / 2.0)

    def get_heating_contribution_at_radius(self, radius: float, self_heating_contribution: np.ndarray) -> float:
        """Interpolate the self-heating contribution at a given radius.

        Args:
            radius (float): Radial distance at which to evaluate the self-heating contribution.
            self_heating_contribution (np.ndarray): Self-heating contribution state values for the cable.

        Returns:
            float: Interpolated temperature-rise contribution due to cable self-heating at the requested radius.

        """
        return float(np.interp(x=[radius], xp=self._radii_grid, fp=self_heating_contribution)[0])

    def get_cable_copy_with_pipe(self, pipe: Pipe) -> Self:
        """Get a new cable instance based on the current self, but with extra added layers that model a pipe.

        This method adds two layers:
         1. pipe_fill layer with an empiric resistance value
         2. PE layer for the pipe
        The resistivity of the pipe filling material is updated depending on the temperature.

        Args:
            pipe (Pipe): A pipe instance

        Returns:
            (Cable): A new Cable instance based on the Cable instance the method was called from, but with
                        the added pipe layers, as if the cable had an outer pipe added.

        """
        # Check whether there is already a soil layer present around the cable
        if self.layer_properties[self.layers[-1]].outer_radius != self.layer_metrics.outer_radius:
            raise ValueError(
                "Detected soil layers. "
                "The add_outer_pipe method is only intended for cable instances without soil layers."
            )

        if self.layer_metrics.pipe is not None:
            raise ValueError("Cannot add a pipe as the cable already has a pipe.")

        new_cable = deepcopy(self)

        # Create a new cable, using the get_redefined_cable() method, with the new values where the cable should be
        # altered to accommodate the pipe.
        grid_counts = new_cable._grid_counts
        layer_properties: list[tuple[CableLayer, float, float, float]] = [
            (CableLayer.PipeFill, pipe.inner_radius, pipe.get_thermal_resistivity_pipe_fill(), pipe.pipe_fill_cap),
            (CableLayer.Pipe, pipe.outer_radius, 3.5, 2.4e6),
        ]

        for layer, layer_outer_radius, rho, capacity in layer_properties:
            new_cable.layer_properties[layer] = CableLayerProperties(
                layer=layer,
                inner_radius=new_cable.layer_properties[new_cable.layers[-1]].outer_radius,
                outer_radius=layer_outer_radius,
                rho=rho,
                capacity=capacity,
            )
            new_cable.layers.append(layer)
            grid_counts[layer] = 10  # Default grid count for pipe layers

        new_cable.layer_metrics.pipe = pipe
        new_cable.layer_metrics.outer_radius = pipe.outer_radius

        return new_cable._get_redefined_cable(
            layer_properties=new_cable.layer_properties,
            layer_metrics=new_cable.layer_metrics,
            grid_counts=grid_counts,
        )

    def get_finite_difference_vector(self, neglect_dielectric_loss: bool = False) -> np.ndarray:
        """This method calculates and returns the finite difference vector.

        Args:
            neglect_dielectric_loss (bool): A boolean representing whether to
                neglect the dielectric losses in the calculation of the vector.
                Default is False.

        Returns:
            np.ndarray: A Numpy array representing the finite difference vector [W/m³].
        """
        vector = np.zeros(self._radii_grid.size - 1)

        if not neglect_dielectric_loss:
            dielectric_loss = self.get_dielectric_loss_for_cable()
            vector = self._update_vector_with_heat_generation_for_layer(
                vector=vector,
                heat_generation=dielectric_loss,
                layer=CableLayer.Insulation,
            )

        return vector

    def update_finite_difference_vector(
        self,
        vector: np.ndarray,
        temperature_grid: np.ndarray,
        load: float,
        ac_current: bool,
        temperature_dependent_electric_resistance: bool,
    ) -> np.ndarray:
        """Build the finite difference vector for a specific thermal state and circuit load.

        Args:
            vector (np.ndarray): The finite difference vector to be updated.
            temperature_grid (np.ndarray): The current temperature grid for the cable.
            load (float): The electrical load in amperes.
            ac_current (bool): Whether AC conductor losses are included.
            temperature_dependent_electric_resistance (bool): Whether resistance depends on temperature.

        Returns:
            np.ndarray: A finite difference vector for the given state and load.
        """
        conductor_temperature = self._get_mean_temperature_cable_layer(
            temperature_grid=temperature_grid, layer=CableLayer.Conductor
        )

        if CableLayer.Screen in self.layers and ac_current:
            screen_temperature = self._get_mean_temperature_cable_layer(
                temperature_grid=temperature_grid, layer=CableLayer.Screen
            )

            heat_generation_conductor, heat_generation_screen = self.get_heat_generation_conductor_and_screen(
                ac_current=ac_current,
                load=load,
                conductor_temperature=conductor_temperature,
                screen_temperature=screen_temperature,
                temperature_dependent_electric_resistance=temperature_dependent_electric_resistance,
            )

            vector = self._update_vector_with_heat_generation_for_layer(
                vector=vector,
                heat_generation=heat_generation_screen,
                layer=CableLayer.Screen,
            )
        else:
            heat_generation_conductor = self.get_heat_generation_conductor(
                ac_current=ac_current,
                load=load,
                conductor_temperature=conductor_temperature,
                temperature_dependent_electric_resistance=temperature_dependent_electric_resistance,
            )

        return self._update_vector_with_heat_generation_for_layer(
            vector=vector,
            heat_generation=heat_generation_conductor,
            layer=CableLayer.Conductor,
        )

    @staticmethod
    def _validate_grid_counts(grid_counts: Any) -> None:
        if not isinstance(grid_counts, dict):
            raise TypeError("The grid_counts argument must be a dictionary of integers!")

        if not all(isinstance(key, CableLayer) for key in grid_counts):
            raise TypeError("The grid_counts argument must be a dictionary with CableLayer keys!")

        if not all(isinstance(value, (int, np.integer)) for value in grid_counts.values()):
            raise TypeError("The grid_counts argument must be a dictionary of integers!")

    def _processed_matrix(self, time_step: float) -> np.ndarray | sparse.lil_matrix:
        """Process the finite difference matrix for the implicit Euler method.

        Args:
            time_step (float): The size of the time step [s] in the linearized time grid.

        Returns:
            np.ndarray | sparse.lil_matrix: The processed finite difference matrix ready for solving the linear system.

        """
        ab = self._banded_matrix

        ab = -ab * time_step
        ab[1, :] += self._capacity_grid[: ab.shape[1]]

        return ab

    @staticmethod
    def _solve_system(
        A: np.ndarray | sparse.lil_matrix,
        b: np.ndarray,
    ) -> np.ndarray:
        """Solve the linear system Ax = b.

        Args:
            A (np.ndarray | sparse.lil_matrix): The finite difference matrix.
            b (np.ndarray): The finite difference vector.

        Returns:
            np.ndarray: The solution vector x.

        Raises:
            TypeError: If A is not a banded matrix (3xN numpy array).

        """
        if not isinstance(A, np.ndarray):
            raise TypeError(f"Expected banded matrix (np.ndarray), got {type(A).__name__}")
        if A.shape[0] != 3:
            raise ValueError(f"Expected banded matrix with shape (3, N), got {A.shape}")

        return linalg.solve_banded(l_and_u=(1, 1), ab=A, b=b)
