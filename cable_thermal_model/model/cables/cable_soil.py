# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0
from copy import deepcopy
from typing import Self

import numpy as np
from scipy import linalg

from cable_thermal_model.model.cables.abstract_cable import CableLayerProperties
from cable_thermal_model.model.cables.cable import Cable
from cable_thermal_model.model.cables.enum_classes_cable import CableLayer


class CableSoil(Cable):
    """Finite difference cable model with soil discretization."""

    _SOIL_DRYING_TEMPERATURE = 30

    def integrate_timestep(
        self,
        s: np.ndarray,
        b: np.ndarray,
        time_step: float,
        internal_heating: bool | None = None,
    ) -> np.ndarray:
        """This method solves the finite difference approximation to the heat equation using the implicit Euler method.

        For optimization purposes, the method uses the scipy.linalg.solve_banded method to solve the linear system.
        This means the three diagonals of finite difference matrix A are instead stored in a (3, N) array, where
        N is the length of the diagonal.

        Args:
            s (np.ndarray): The solution of the heat equation [°C] at the
                previous timestep (t).
            b (np.ndarray): The finite difference vector [W/m³].
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

        A = self._banded_matrix * -time_step
        A[1, :] += self._capacity_grid[:-1]

        b = self._capacity_grid[:-1] * s + time_step * b

        return linalg.solve_banded(l_and_u=number_of_non_zero_diagonals, ab=A, b=b)

    def update_soil_properties(
        self, soil_rho: float, soil_c: float, temperature_grid: np.ndarray, soil_drying: bool = False
    ):
        """This method updates the soil properties around a cable.

        Args:
            soil_rho (float): The thermal resistivity of the soil that is not dried out
            soil_c (float): The thermal capacity of the soil that is not dried out
            temperature_grid (np.ndarray): The temperature grid for the cable, as calculated for a given timestep.
            soil_drying (bool): Whether the scenario takes soil drying into account.

        """
        dry_soil_radius = self._get_dry_soil_radius(temperature_grid=temperature_grid, soil_drying=soil_drying)

        self._update_soil_resistivity(
            soil_rho=soil_rho,
            dry_soil_radius=dry_soil_radius,
        )

        self._update_soil_capacity(soil_c=soil_c)

    def _update_soil_capacity(self, soil_c: float):
        """This method updates the soil capacity values around a cable.

        If multiple soil layers are present, it sets them all (the entire soil).

        Args:
            soil_c (float): A float representing the thermal capacity of the
                (entire) soil.

        """
        if not isinstance(soil_c, (int, float, np.integer, np.floating)):
            raise ValueError("The soil_c argument must be of type int or float!")

        start_index = self._get_soil_grid_start_index()
        self._update_capacity_grid(start_index=start_index, end_index=self.grid_size - 1, capacity=soil_c)

    def _get_soil_grid_start_index(self) -> int:
        """Return the first grid index that belongs to soil around the cable."""
        return int((self._radii_grid <= self.layer_metrics.outer_radius).sum())

    def _update_soil_resistivity(self, soil_rho: float, dry_soil_radius: float | None = None) -> None:
        """This method updates the soil resistivity values around a cable.

        This is meant to represent the IEC dried-out soil model. The soil will consist of an inner part of dried-out
        soil around the cable, and then a secondary part of standard soil
        The inner part has predefined thermal resistivity, which is defined in NPR Norm 3626.

        Notes:
            We do not update the number of layers of the cable, so the rho-grid may consist of a part that corresponds
            to a single layer, yet has multiple distinct values.

        Args:
            soil_rho (float):
                An optional float representing the thermal resistivity of the soil that is not dried out.
            dry_soil_radius (float | None):
                A float representing the radius of the dried-out soil around the cable.

        """
        start_index = self._get_soil_grid_start_index()
        self._update_rho_grid(
            start_index=start_index,
            end_index=self.grid_size - 1,
            rho=soil_rho,
        )

        if dry_soil_radius is not None:
            dry_soil_rho = 2.5  # mK/W, value taken from NPR3626
            end_index = int((self._radii_grid <= dry_soil_radius).sum()) - 1
            if end_index > start_index:
                self._update_rho_grid(
                    start_index=start_index,
                    end_index=end_index,
                    rho=dry_soil_rho,
                )

    def _get_dry_soil_radius(self, temperature_grid: np.ndarray, soil_drying: bool) -> float | None:
        """Return the radius of dried-out soil based on temperature and scenario settings."""
        if not soil_drying:
            return None

        dry_idxs = np.nonzero(temperature_grid >= self._SOIL_DRYING_TEMPERATURE)[0]
        if dry_idxs.size == 0:
            return None

        return float(self._radii_grid[dry_idxs[-1]])

    def get_cable_copy_without_soil(self) -> Self:
        """This method returns a new CableSoil object with the soil layer removed."""
        if CableLayer.SoilOne not in self.layers:
            raise ValueError("No soil layers detected!")

        non_soil_layers = [layer for layer in self.layers if layer not in CableLayer.soil_layers()]
        grid_count_for_cable_without_soil = {
            layer: grid_count for layer, grid_count in self._grid_counts.items() if layer in non_soil_layers
        }

        new_layer_properties = {layer: self.layer_properties[layer] for layer in non_soil_layers}

        return self._get_redefined_cable(
            layer_properties=new_layer_properties, grid_counts=grid_count_for_cable_without_soil
        )

    @classmethod
    def from_cable_with_added_soil_layer(
        cls,
        cable: Cable,
        soil_rho: float,
        soil_capacity: float,
        soil_radius: float,
        logarithmic_soil_gridpoint_density: float,
    ) -> Self:
        """This method creates a copy of the current cable object this was run from, but with an extra added soil layer.

        Args:
            cable (Cable):
                    The cable object to create a CableSoil instance from.
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
            CableSoil:
                    A completely new CableSoil instance based on the Cable object the method was called from, but with
                    the added soil layers.

        """
        # copy source data so we don't mutate the original cable
        layer_properties = deepcopy(cable.layer_properties)
        grid_counts = deepcopy(cable._grid_counts)

        outer_layer = cable.layers[-1]
        current_outer_radius = layer_properties[outer_layer].outer_radius
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

        layer_properties[new_layer] = CableLayerProperties(
            layer=new_layer,
            inner_radius=current_outer_radius,
            outer_radius=soil_radius,
            rho=soil_rho,
            capacity=soil_capacity,
        )

        radius_factor = soil_radius / current_outer_radius
        grid_counts[new_layer] = max(2, int(logarithmic_soil_gridpoint_density * np.log2(radius_factor)))
        new_cable_soil_cable = cls(
            conductor=deepcopy(cable.conductor),
            layer_properties=layer_properties,
            layer_metrics=deepcopy(cable.layer_metrics),
            cable_type=cable.cable_type,
            grid_counts=grid_counts,
        )
        new_cable_soil_cable.weighted_screen_impedance = cable.weighted_screen_impedance
        return new_cable_soil_cable
