# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from copy import deepcopy
from typing import Self

import numpy as np
from scipy import linalg, sparse

from cable_thermal_model.model.cables.abstract_cable import (
    AbstractCable,
    CableConductorProperties,
    CableConvectionParams,
    CableLayerMetrics,
    CableLayerProperties,
)
from cable_thermal_model.model.cables.enum_classes_cable import CableLayer, CableType
from cable_thermal_model.model.cables.pipe import Pipe


class FDCable(AbstractCable):
    """Finite-difference cable model that discretizes cable layers into a radial grid."""

    def __init__(
        self,
        conductor: CableConductorProperties,
        layer_properties: dict[CableLayer, CableLayerProperties],
        layer_metrics: CableLayerMetrics,
        cable_type: CableType,
        grid_counts: dict[CableLayer, int],
    ) -> None:
        """Initialize the FDCable with conductor properties, layer data, and grid resolution.

        Args:
            conductor (CableConductorProperties): Conductor properties of the cable.
            layer_properties (dict[CableLayer, CableLayerProperties]): Mapping of cable layers to their properties.
            layer_metrics (CableLayerMetrics): Geometric and calculated metrics for the cable layers.
            cable_type (CableType): The type of the cable.
            grid_counts (dict[CableLayer, int]): Number of grid points per cable layer.

        """
        super().__init__(conductor, layer_properties, layer_metrics, cable_type)

        if not isinstance(grid_counts, dict) or not all(isinstance(v, (int, np.integer)) for v in grid_counts.values()):
            raise TypeError("The grid_counts argument must be a dictionary of integers!")

        self.grid_counts = grid_counts
        self.radii_grid: np.ndarray = np.array([])
        self.grid_deltas: np.ndarray = np.array([])
        self.layer_name_grid: np.ndarray = np.array([])
        self.rho_grid: np.ndarray = np.array([])
        self.capacity_grid: np.ndarray = np.array([])
        self.alpha_grid: np.ndarray = np.array([])
        self.surface_area_grid: np.ndarray = np.array([])

        self.set_calculated_fields()

    def set_calculated_fields(self) -> None:
        """Initialize derived cable properties.

        The properties set in this function depend on the cable layers. When
        adding soil or pipe layers these need to be reset. This function can be used to do so.
        """
        self.radii_grid = self.construct_radii_grid()
        self.grid_deltas = np.diff(self.radii_grid)
        self.surface_area_grid = FDCable.construct_surface_area_grid(self.radii_grid)

        rho_grids = [np.full(self.grid_counts[layer], self.layer_properties[layer].rho) for layer in self.layers]
        self.rho_grid = np.concatenate(rho_grids)

        capacity_grids = [
            np.full(self.grid_counts[layer], self.layer_properties[layer].capacity) for layer in self.layers
        ]
        self.capacity_grid = np.concatenate(capacity_grids)

        alpha_grids = [np.full(self.grid_counts[layer], self.layer_properties[layer].alpha) for layer in self.layers]
        self.alpha_grid = np.concatenate(alpha_grids)

    def construct_radii_grid(self, maximal_boundary_distance: float = 0.000_1) -> np.ndarray:
        """Construct the radii grid for the cable based on the layer properties and grid counts.

        Args:
            maximal_boundary_distance (float): The maximal distance to use as a
                boundary distance between layers [m]. Default is 0.1 mm.

        Returns:
            np.ndarray: A Numpy array representing the radii grid for the cable.

        """
        last_layer = self.layers[-1]

        boundary_distance = 0.0
        radii_grids = []
        for layer_idx, layer in enumerate(self.layers):
            start = self.layer_properties[layer].inner_radius + boundary_distance

            if layer == last_layer:
                end = self.layer_properties[layer].outer_radius

            else:
                next_layer = self.layers[layer_idx + 1]
                boundary_distance = min(
                    [
                        maximal_boundary_distance,
                        (self.layer_properties[layer].outer_radius - start) / (2 * (self.grid_counts[layer] - 0.5)),
                        (
                            self.layer_properties[next_layer].outer_radius
                            - self.layer_properties[next_layer].inner_radius
                        )
                        / (2 * self.grid_counts[next_layer]),
                    ]
                )
                end = self.layer_properties[layer].outer_radius - boundary_distance

            if layer not in CableLayer.soil_layers():
                radii_grids.append(np.linspace(start=start, stop=end, num=self.grid_counts[layer]))
            else:
                # For soil layers, we want to use a logarithmic grid to better
                # capture the temperature gradients close to the cable.
                radii_grids.append(
                    np.logspace(start=0.0, stop=np.log2(end / start), num=self.grid_counts[layer], base=2) * start
                )

        return np.concatenate(radii_grids)

    @staticmethod
    def construct_surface_area_grid(radii_grid: np.ndarray) -> np.ndarray:
        """Construct the surface area grid for the cable based on the radii grid.

        Args:
            radii_grid (np.ndarray): A Numpy array representing the radii grid
                for the cable.

        Returns:
            np.ndarray: A Numpy array representing the surface area grid for the cable.

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

    def update_vector_with_heat_generation(
        self, vector: np.ndarray, heat_generation: float, start_index: int, end_index: int
    ) -> np.ndarray:
        """Updates the given vector with the given heat generation value.

        The heat generation is distributed over the grid points between the given start and end indices.

        Args:
            vector (np.ndarray): The vector to update.
            heat_generation (float): The heat generation value in W/m to distribute over the grid points.
            start_index (int): The start index of the grid points to update.
            end_index (int): The end index of the grid points to update.

        Returns:
            np.ndarray: The updated vector with the heat generation distributed over the specified grid points.

        """
        vector[start_index : end_index + 1] = (
            heat_generation / self.surface_area_grid[start_index : end_index + 1].sum()
        )
        return vector

    def get_redefined_cable(self, **kwargs) -> Self:
        """Get a new cable instance based on the current self, but with changed cable attributes.

        This method takes the parameters given in the **kwargs and tries to apply those to matching attributes in a
         copy made of the current self.

        Examples:
            An example where we create a cable, and then use this method to create a copy of the cable, but with the
            [rhos] and [capacities] attributes altered from their original values.
            >>> cable = Cable()
            >>> new_cable = cable.get_redefined_cable(rhos = (1,1,1), capacities = (5,5,5))

            (For other applications, please check out the 'add_soil' and 'add_outer_tube' methods.)

        Args:
            **kwargs:
                    Kwargs here is used to pass along cable parameters that would usually be configured using the
                    initializer. Recognized parameters will overwrite existing values, while other parameters will
                    be ignored.
                    (Some examples of parameters that could be changed in this way: 'rhos','radii','grid_counts')

        Returns:
            Self: A completely new cable instance based on the cable the method was
                called from, but with changed cable properties based on the passed
                [**kwargs] parameters.

        Notes:
            There are two reasons this method should be re-evaluated in the future. First of all this method uses
            kwargs to pass along an unknown combination of parameters, which is only evaluated by parameter name.
            Secondly this method is found in the FDCable class, but it is not specific to the FDCable class.

        """
        new_cable = deepcopy(self)

        # Check all the items in the kwargs and apply them to the new cable if they are recognized as existing.
        for key, value in kwargs.items():
            if hasattr(new_cable, key):
                setattr(new_cable, key, value)

        # Recalculate the calculated fields of the new cable and reset the solution values
        new_cable.set_calculated_fields()

        return new_cable

    def get_cable_copy_with_added_soil_layer(
        self, soil_rho: float, soil_capacity: float, soil_radius: float, logarithmic_soil_gridpoint_density: float
    ) -> Self:
        """This method creates a copy of the current cable object this was run from, but with an extra added soil layer.

        Args:
            soil_rho (float):
                    The thermal resistivity of the soil layer to add.
            soil_capacity (float):
                    The thermal capacity of the soil layer to add.
            soil_radius (float):
                    The radius of the soil layer to add.
            logarithmic_soil_gridpoint_density (float):
                    The density of grid points in the soil layer, this is used
                    to compute the number of grid points in the soil layer
                    based on its thickness. The density represents the number
                    of grid points per factor 2 increase in soil layer
                    thickness.

        Returns:
            Self: A completely new cable instance based on the cable the method was
                called from, but with the added soil layers.

        """
        new_cable = deepcopy(self)
        outer_layer = new_cable.layers[-1]
        current_outer_radius = new_cable.layer_properties[outer_layer].outer_radius
        if soil_radius <= current_outer_radius:
            raise ValueError("The soil radius must be larger than the outer radius of the current outer layer!")

        soil_layers = CableLayer.soil_layers()

        if outer_layer in soil_layers:
            if outer_layer == soil_layers[-1]:
                raise ValueError(
                    "The current cable already has the maximum amount of soil layers! "
                    "This method cannot be used to add more soil layers!"
                )
            new_layer = soil_layers[soil_layers.index(outer_layer) + 1]
        else:
            new_layer = soil_layers[0]

        grid_counts = new_cable.grid_counts
        new_cable.layer_properties[new_layer] = CableLayerProperties(
            layer=new_layer,
            inner_radius=current_outer_radius,
            outer_radius=soil_radius,
            rho=soil_rho,
            capacity=soil_capacity,
        )
        new_cable.layers.append(new_layer)

        radius_factor = soil_radius / current_outer_radius
        grid_counts[new_layer] = max(2, int(logarithmic_soil_gridpoint_density * np.log2(radius_factor)))

        return new_cable.get_redefined_cable(layer_properties=new_cable.layer_properties, grid_counts=grid_counts)

    def get_cable_copy_without_soil(self) -> Self:
        """This method returns a new FDCable object with the soil layer removed."""
        if CableLayer.SoilOne not in self.layers:
            raise ValueError("No soil layers detected!")

        non_soil_layers = [layer for layer in self.layers if layer not in CableLayer.soil_layers()]
        grid_count_for_cable_without_soil = {
            layer: grid_count for layer, grid_count in self.grid_counts.items() if layer in non_soil_layers
        }

        new_layer_properties = {layer: self.layer_properties[layer] for layer in non_soil_layers}

        return self.get_redefined_cable(
            layer_properties=new_layer_properties, grid_counts=grid_count_for_cable_without_soil
        )

    def get_layer_indices_for_layer(self, layer: CableLayer) -> tuple[int, int]:
        """This method fetches the start and end indices of the grid points for a given layer.

        Args:
            layer (CableLayer): A CableLayer object representing the layer for
                which the indices need to be fetched.

        Returns:
            tuple[int, int]: A tuple of integers representing the start and end
                indices of the grid points for the given layer, in that order.

        """
        layer_index = self.layers.index(layer)

        layer_start_index = sum([self.grid_counts[layer] for layer in self.layers[:layer_index]])
        layer_end_index = layer_start_index + self.grid_counts[layer] - 1
        return layer_start_index, layer_end_index

    def _get_finite_differences_matrix_upper_diagonal(self) -> np.ndarray:
        """This method returns the upper diagonal component for the finite-differences matrix.

        This method computes the upper "off" diagonal of the finite-differences matrix. For more information on the
        finite-differences matrix, please check the method [get_finite_differences_matrix()].

        Returns:
            np.ndarray: A Numpy array representing the upper "off" diagonal of
                the finite-differences matrix, as calculated for the current cable.

        """
        radii = self.radii_grid
        delta_plus = self.grid_deltas[1:]
        delta_minus = self.grid_deltas[:-1]
        inter_radii = radii[:-1] + 0.5 * self.grid_deltas

        # Calculate interstitial resistivity values
        inter_rhos = self._calculate_inter_rhos(radii, inter_radii, self.rho_grid)
        rho_plus = inter_rhos[1:]

        r = radii[1:-1]
        r_plus = r + 0.5 * delta_plus

        upper_diagonal = r_plus / (rho_plus * r * 0.5 * (delta_plus + delta_minus) * delta_plus)
        upper_diagonal = np.append([4 / (self.rho_grid[0] * delta_minus[0] ** 2)], upper_diagonal)

        return upper_diagonal

    def _get_finite_differences_matrix_base_diagonal(self) -> np.ndarray:
        """This method returns the base diagonal component for the finite-differences matrix.

        This method computes the base diagonal of the finite-differences matrix. For more information on the
        finite-differences matrix, please check the method [get_finite_differences_matrix()].

        Returns:
            np.ndarray: A Numpy array representing the base diagonal of the
                finite-differences matrix, as calculated for the current cable.

        """
        radii = self.radii_grid
        delta_plus = self.grid_deltas[1:]
        delta_minus = self.grid_deltas[:-1]
        inter_radii = radii[:-1] + 0.5 * self.grid_deltas

        # Calculate interstitial resistivity values
        inter_rhos = self._calculate_inter_rhos(radii, inter_radii, self.rho_grid)
        rho_plus = inter_rhos[1:]
        rho_minus = inter_rhos[:-1]

        r = radii[1:-1]
        r_plus = r + 0.5 * delta_plus
        r_minus = r - 0.5 * delta_minus

        base_diagonal = (-1 / (r * 0.5 * (delta_plus + delta_minus))) * (
            r_plus / (rho_plus * delta_plus) + r_minus / (rho_minus * delta_minus)
        )

        # Exception situation for the grid point at r=0:
        base_diagonal = np.append([-4 / (self.rho_grid[0] * delta_minus[0] ** 2)], base_diagonal)

        return base_diagonal

    def _get_finite_differences_matrix_lower_diagonal(self) -> np.ndarray:
        """This method returns the lower "off" diagonal component for the finite-differences matrix.

        This method computes the lower "off" diagonal of the finite-differences matrix. For more information on the
        finite-differences matrix, please check the method [get_finite_differences_matrix()].

        Returns:
            np.ndarray: A Numpy array representing the lower "off" diagonal of
                the finite-differences matrix, as calculated for the current cable.

        """
        radii = self.radii_grid
        delta_plus = self.grid_deltas[1:]
        delta_minus = self.grid_deltas[:-1]
        inter_radii = radii[:-1] + 0.5 * self.grid_deltas

        rho_minus = self._calculate_inter_rhos(radii, inter_radii, self.rho_grid)[:-1]

        r = radii[1:-1]
        r_minus = r - 0.5 * delta_minus

        lower_diagonal = r_minus / (rho_minus * r * 0.5 * (delta_plus + delta_minus) * delta_minus)

        return lower_diagonal

    def _calculate_inter_rhos(self, radii: np.ndarray, inter_radii: np.ndarray, rhos: np.ndarray) -> np.ndarray:
        """This method calculates the interstitial resistivity values between grid points.

        Args:
            radii (np.ndarray): A numpy array with the positions of N grid
                points to use.
            inter_radii (np.ndarray): A numpy array with the positions of N-1
                interstitial grid points to use.
            rhos (np.ndarray): A numpy array with the resistivity values at the
                N grid points.

        Returns:
            np.ndarray: A numpy array representing the calculated interstitial
                resistivity values.

        """
        if not (len(radii) == len(inter_radii) + 1 == len(rhos)):
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

    def get_finite_differences_matrix(self) -> np.ndarray:
        """Calculates and returns the finite-differences matrix.

        The finite-differences matrix is central to the linearized heat equation. It is a matrix with one base
        diagonal and two "off" diagonals (one above and
        one below the base diagonal), and otherwise only zeros. We represent this matrix as a 3xN numpy array, where N
        is the length of the base diagonal.

        Notes:
            In the finite differences (FD) approximation, this single matrix combined with a vector control the
            linearized heat equation.

        Returns:
            np.ndarray: The (3xN) matrix representing the finite-differences matrix
                [W/(°C*m³)].

        """
        upper_diagonal = self._get_finite_differences_matrix_upper_diagonal()
        base_diagonal = self._get_finite_differences_matrix_base_diagonal()
        lower_diagonal = self._get_finite_differences_matrix_lower_diagonal()

        matrix = np.zeros((3, len(base_diagonal)))
        matrix[0, 1:] = upper_diagonal[:-1]
        matrix[1, :] = base_diagonal
        matrix[2, :-1] = lower_diagonal

        return matrix

    def get_linear_system(
        self,
        neglect_dielectric_loss: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """This method retrieves the two elements that control the linearized heat equation.

        These are:
            - The finite-differences matrix, which contains the linearized interaction terms between grid points defined
              by material properties.
            - The vector, which contains the energy that would be released and internally generated heat terms. In this
              step, only the time-independent dielectric losses are added.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                A tuple of two Numpy arrays, representing the finite-differences matrix and the vector,
                respectively.

        """
        vector = np.zeros(self.radii_grid.size - 1)

        if not neglect_dielectric_loss:
            # Account for dielectric losses
            insulation_start_index, insulation_end_index = self.get_layer_indices_for_layer(CableLayer.Insulation)
            Wd = self.get_dielectric_loss_for_cable()  # Dielectric loss in W/m

            # The generated heat is added to the loss vector
            vector = self.update_vector_with_heat_generation(
                vector=vector, heat_generation=Wd, start_index=insulation_start_index, end_index=insulation_end_index
            )

        matrix = self.get_finite_differences_matrix()
        return matrix, vector

    def integrate_timestep(
        self,
        s: np.ndarray,
        A_banded: np.ndarray,
        b: np.ndarray,
        time_step: float,
        internal_heating: bool | None = None,
    ) -> np.ndarray:
        """This method solves the finite-difference approximation to the heat equation using the imiplicit Euler method.

        For optimization purposes, the method uses the scipy.linalg.solve_banded method to solve the linear system.
        This means the the three diagonals of finite-differences matrix A are instead stored in a (3, N) array, where
        N is the length of the diagonal.

        Args:
            s (np.ndarray): The solution of the heat equation [°C] at the
                previous timestep (t).
            A_banded (np.ndarray): The finite-differences matrix [W/(°C*m³)]
                represented as a banded matrix.
            b (np.ndarray): The finite-differences vector [W/m³].
            time_step (float): The size of the time steps [s] in the linearized
                time grid.
            internal_heating (bool | None): A boolean representing whether
                internal heating is considered in this timestep.
                This implementation of the method does not use this parameter, but some child classes do.

        Returns:
            np.ndarray: The solution [°C] to the heat equation at the next timestep (t+1) for all grid points except
                the final grid point, at which a boundary condition is enforced.

        """
        number_of_non_zero_diagonals = (1, 1)  # one upper and one lower diagonal

        A = A_banded * -time_step
        A[1, :] += self.capacity_grid[:-1]

        b = self.capacity_grid[:-1] * s + time_step * b

        return linalg.solve_banded(l_and_u=number_of_non_zero_diagonals, ab=A, b=b)

    def update_soil_resistivity(self, soil_rho: float, dry_soil_radius: float | None = None):
        """This method updates the soil resistivity values around a cable.

        This is meant to represent the IEC dried-out soil model. The soil will consist of an inner part of dried-out
        soil around the cable, and then a secondary part of standard soil
        The inner part has predefined thermal resistivity, which is defined in NPR Norm 3626.

        Notes:
            We do not update the number of layers of the cable, so the rho-grid may consist of a part that corresponds
            to a single layer, yet has multiple distinct values.

        Args:
            soil_rho (float): An optional float representing the thermal
                resistivity of the soil that is not dried out.
            dry_soil_radius (float | None): A float representing the radius of
                the dried-out soil around the cable.

        """
        start_index = (self.radii_grid <= self.layer_metrics.outer_radius).sum()
        self.rho_grid[start_index:] = soil_rho

        # Also update the layer properties to keep them consistent
        for layer in CableLayer.soil_layers():
            if layer in self.layers:
                self.layer_properties[layer].rho = soil_rho

        if dry_soil_radius is not None:
            dry_soil_rho = 2.5  # mK/W, value taken from NPR3626
            end_index = max((self.radii_grid <= dry_soil_radius).sum(), start_index)

            # Assign the resistivity values
            self.rho_grid[start_index:end_index] = dry_soil_rho

            for layer in CableLayer.soil_layers():
                # Only update the layers that are fully within the dry soil radius
                if (layer in self.layers) and (self.layer_properties[layer].outer_radius <= dry_soil_radius):
                    self.layer_properties[layer].rho = dry_soil_rho

    def update_pipe_resistivity(self, Tfill: float) -> bool:
        """This method updates the (temperature dependent) thermal resistivity of the medium in the pipe of the cable.

        Args:
            Tfill (float): The mean temperature of the medium within the pipe in
                degree Celsius.

        """
        if self.layer_metrics.pipe is None:
            raise ValueError("Pipe is not set. Cannot update pipe resistivity.")
        if self.layer_metrics.pipe.inner_radius is None:
            raise ValueError("Pipe inner radius is not set. Cannot update pipe resistivity.")

        old_pipe_fill_rho = self.layer_properties[CableLayer.PipeFill].rho
        new_pipe_fill_rho = self.layer_metrics.pipe.get_thermal_resistivity_pipe_fill(Tfill)
        if not np.isclose(old_pipe_fill_rho, new_pipe_fill_rho, rtol=1e-2):
            pipe_fill_start_index, pipe_fill_end_index = self.get_layer_indices_for_layer(CableLayer.PipeFill)
            self.rho_grid[pipe_fill_start_index : pipe_fill_end_index + 1] = new_pipe_fill_rho
            self.layer_properties[CableLayer.PipeFill].rho = new_pipe_fill_rho
            return True
        else:
            return False

    def update_soil_capacity(self, soil_c: float):
        """This method updates the soil capacity values around a cable.

        If multiple soil layers are present, it sets them all (the entire soil).

        Args:
            soil_c (float): A float representing the thermal capacity of the
                (entire) soil.

        """
        if not isinstance(soil_c, (int, float, np.integer, np.floating)):
            raise ValueError("The soil_c argument must be of type int or float!")

        start_index = (self.radii_grid <= self.layer_metrics.outer_radius).sum()
        self.capacity_grid[start_index:] = soil_c

    def get_cable_copy_with_pipe(self, pipe: Pipe) -> Self:
        """Get a new cable instance based on the current self, but with extra added layers that model a pipe.

        This method adds two layers:
         1. pipe_fill layer with an empiric resistance value
         2. PE layer for the pipe
        The resistivity of the pipe filling material is updated depending on the temperature.

        Args:
            pipe (Pipe): A pipe instance

        Returns:
            Self: A new cable instance based on this instance, but with added
                pipe layers as if the cable had an outer pipe.

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
        grid_counts = new_cable.grid_counts
        for layer, layer_outer_radius, rho, capacity in [
            (CableLayer.PipeFill, pipe.inner_radius, pipe.get_thermal_resistivity_pipe_fill(), pipe.pipe_fill_cap),
            (CableLayer.Pipe, pipe.outer_radius, 3.5, 2.4e6),
        ]:
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

        return new_cable.get_redefined_cable(
            layer_properties=new_cable.layer_properties,
            layer_metrics=new_cable.layer_metrics,
            grid_counts=grid_counts,
        )


class FDCableTrefoilCircuitInSinglePipe(FDCable):
    """Class that represents a finite-difference cable trefoil circuit that lies in a single pipe."""

    def integrate_timestep(
        self,
        s: np.ndarray,
        A_banded: np.ndarray,
        b: np.ndarray,
        time_step: float,
        internal_heating: bool | None = None,
    ) -> np.ndarray:
        """This method solves the finite-difference approximation to the heat equation using the implicit Euler method.

        We add a heat source between the pipe and the equivalent cable
        representing the trefoil circuit in the internal heating step. The
        amount of heat added equals twice the heat loss at the cable sheath,
        therefore representing the heat three cables in trefoil would
        generate together. Because we add an additional heat source between
        the pipe and the equivalent cable representing the trefoil circuit,
        the banded array is converted to a sparse matrix and adjusted
        appropriately before solving the linear system.

        Args:
            s (np.ndarray): The solution of the heat equation [°C] at the
                previous timestep (t).
            A_banded (np.ndarray): The finite-differences matrix [W/(°C*m³)]
                represented as a banded matrix.
            b (np.ndarray): The finite-differences vector [W/m³].
            time_step (float): The size of the time steps [s] in the linearized
                time grid.
            internal_heating (bool): A boolean indicating whether internal
                heating between cables in the trefoil circuit is considered.

        Returns:
            np.ndarray: The solution [°C] to the heat equation at the next timestep (t+1) for all grid points except
                the final grid point, at which a boundary condition is enforced.

        """
        if internal_heating is None:
            raise ValueError("The internal_heating parameter must be provided for FDCableTrefoilCircuitInSinglePipe.")

        # Only add an extra heat source if internal heating is considered
        if not internal_heating:
            return super().integrate_timestep(s, A_banded, b, time_step)

        # Convert the banded matrix to a sparse matrix
        # Use dia format for easy conversion and then convert to lil format to set individual elements
        A_sparse = sparse.dia_matrix((A_banded, [1, 0, -1]), shape=(A_banded.shape[1], A_banded.shape[1])).tolil()

        # Add coefficients to the matrix, representing adding an internal heat source
        # that depends on the heat that passes through the cable boundary
        A_sparse = self._update_system_with_heat_source(A_sparse)

        # Compute the other vectors that are required to solve the linear system
        capacity_vector = self.capacity_grid[:-1]
        capacity_diagonal_matrix = sparse.diags(diagonals=capacity_vector)

        return sparse.linalg.spsolve(
            capacity_diagonal_matrix - time_step * A_sparse, capacity_vector * s + time_step * b
        )

    def _update_system_with_heat_source(self, A_sparse: sparse.lil_matrix) -> sparse.lil_matrix:
        """Add coefficients to the finite-differences matrix.

        The added coefficients represent an internal heat source between the
        pipe and the equivalent cable representing the trefoil circuit. The
        amount of heat added equals twice the heat loss at the cable sheath,
        therefore representing the heat three cables in trefoil would
        generate together.

        Args:
            A_sparse (sparse.lil_matrix): The finite-differences matrix
                [W/(°C*m³)] represented as a sparse lil matrix.

        Returns:
            sparse.lil_matrix:
                The updated finite-differences matrix [W/(°C*m³)] represented as a sparse lil matrix.

        """
        # Determine the indices m (filling_heat_source_layer) and s (outer_sheath_index) where r_s<r_cable<r_{s+1}
        # and m (filling_heat_source_layer) is such that r_m^-<2*r_cable < r_m^+.
        # Since r_m^- and r_m^+ lie exactly between grid points, we can find
        # m by searching for the grid point closest to 2*r_cable.
        filling_heat_source_layer = int(np.abs(self.radii_grid - 2 * self.layer_metrics.cable_radius).argmin())
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
        r_s = self.radii_grid[s]
        inter_radius = np.array([r_s + 0.5 * self.grid_deltas[s]])
        inter_rho = self._calculate_inter_rhos(self.radii_grid[s : s + 2], inter_radius, self.rho_grid[s : s + 2])[0]
        return (
            2
            * self.layer_metrics.cable_radius
            / (inter_rho * self.radii_grid[m] * self.grid_deltas[s] * (self.radii_grid[m + 1] - self.radii_grid[m - 1]))
        )


_MAX_ITERATIONS_PER_TIMESTEP = 100
_MAX_ERROR_SHEATH = 0.001


class FDCableInAir(FDCable):
    """Class that represents a finite-difference cable installed in air.

    This class inherits from FDCable, and only differs in the convection parameters used for the cable.
    """

    def __init__(
        self,
        conductor: CableConductorProperties,
        layer_properties: dict[CableLayer, CableLayerProperties],
        layer_metrics: CableLayerMetrics,
        cable_type: CableType,
        grid_counts: dict[CableLayer, int],
    ):
        """Initialize FDCableInAir with convection parameters set to None until explicitly configured.

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
        A_banded: np.ndarray,
        b: np.ndarray,
        time_step: float,
        internal_heating: bool | None = True,
    ) -> np.ndarray:
        """Computes the temperature solution for the next time step.

        Computes the temperature solution at time step [t+1] given the solution at the
        current time step [t], the finite-difference matrix, and the vector for [t].

        Args:
            s (np.ndarray): The solution of the heat equation [°C] at the
                previous timestep (t).
            A_banded (np.ndarray): The finite-differences matrix [W/(°C*m³)]
                represented as a banded matrix.
            b (np.ndarray): The finite-differences vector [W/m³].
            time_step (float): The size of the time steps [s] in the linearized
                time grid.
            internal_heating (bool | None): A boolean representing whether
                internal heating is considered in this timestep. Must be None
                for this class.

        Returns:
            np.ndarray: The updated temperature solution at the new time step
                [t+1] for the cable.

        """
        if internal_heating is not True:
            raise ValueError("Internal heating must be True for cables in air.")

        temp_solution = s.copy()
        theta_N = temp_solution[-1]

        A = np.zeros((A_banded.shape[0], A_banded.shape[1] + 1))

        A[:, :-1] = A_banded
        A[0, -1] = self._get_finite_differences_matrix_upper_diagonal()[-1]
        A = -A * time_step
        A[1, :-1] += self.capacity_grid[:-1]
        A[2, -2] = 1

        b = b * time_step + self.capacity_grid[:-1] * s[:-1]

        b = np.append(b, 0.0)

        iteration = 0
        while True:
            iteration += 1

            A[1, -1] = -(1 + self._boundary_condition_coefficient * theta_N ** (1 / 4))

            temp_solution = linalg.solve_banded(l_and_u=(1, 1), ab=A, b=b)

            if abs(temp_solution[-1] - theta_N) <= _MAX_ERROR_SHEATH:
                break
            elif iteration >= _MAX_ITERATIONS_PER_TIMESTEP:
                raise ValueError(f"Solution did not converge after {_MAX_ITERATIONS_PER_TIMESTEP} iterations")

            theta_N = temp_solution[-1]

        return temp_solution

    @property
    def _boundary_condition_coefficient(self) -> float:
        """This method calculates the coefficient for the boundary condition at the outer sheath in air.

        Returns:
            float: The boundary condition coefficient for the outer sheath in air.

        """
        r_N = self.radii_grid[-1]
        delta_min = self.grid_deltas[-1]
        r_N_min = r_N - 0.5 * delta_min

        return self.convection_coefficient * delta_min * self.rho_grid[-1] * r_N / r_N_min


class FDCableTrefoilCircuitInSinglePipeInAir(FDCableTrefoilCircuitInSinglePipe, FDCableInAir):
    """Class that represents a finite-difference cable trefoil circuit that lies in a single pipe in air."""

    def integrate_timestep(
        self,
        s: np.ndarray,
        A_banded: np.ndarray,
        b: np.ndarray,
        time_step: float,
        internal_heating: bool | None = True,
    ) -> np.ndarray:
        """This method solves the finite-difference approximation to the heat equation using the implicit Euler method.

        We add a heat source between the pipe and the equivalent cable
        representing the trefoil circuit in the internal heating step. The
        amount of heat added equals twice the heat loss at the cable sheath,
        therefore representing the heat three cables in trefoil would
        generate together. Because we add an additional heat source between
        the pipe and the equivalent cable representing the trefoil circuit,
        the banded array is converted to a sparse matrix and adjusted
        appropriately before solving the linear system.

        Args:
            s (np.ndarray): The solution of the heat equation [°C] at the
                previous timestep (t).
            A_banded (np.ndarray): The finite-differences matrix [W/(°C*m³)]
                represented as a banded matrix.
            b (np.ndarray): The finite-differences vector [W/m³].
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
        if internal_heating is not True:
            raise ValueError("Internal heating must be True for cables in air.")

        if self.convection_coefficient is None:
            raise ValueError("Convection parameters have not been set for this cable in air!")

        temp_solution = s.copy()
        theta_N = temp_solution[-1]

        A = np.zeros((A_banded.shape[0], A_banded.shape[1] + 1))

        A[:, :-1] = A_banded
        A[0, -1] = self._get_finite_differences_matrix_upper_diagonal()[-1]
        A[2, -2] = 1

        # Convert the banded matrix to a sparse matrix
        # Use dia format for easy conversion and then convert to lil format to set individual elements
        A_sparse = sparse.dia_matrix((A, [1, 0, -1]), shape=(A.shape[1], A.shape[1])).tolil()

        # Add coefficients to the matrix, representing adding an internal heat source
        # that depends on the heat that passes through the cable boundary.
        A_sparse = self._update_system_with_heat_source(A_sparse)

        # Compute the other vectors that are required to solve the linear system
        capacity_vector = self.capacity_grid[:-1]
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

            if abs(temp_solution[-1] - theta_N) <= _MAX_ERROR_SHEATH:
                break
            elif iteration >= _MAX_ITERATIONS_PER_TIMESTEP:
                raise ValueError(f"Solution did not converge after {_MAX_ITERATIONS_PER_TIMESTEP} iterations")

            theta_N = temp_solution[-1]

        return temp_solution
