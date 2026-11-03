# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0
from copy import deepcopy
from typing import Self

import numpy as np
from pydantic import BaseModel

from cable_thermal_model.model.cables.abstract_cable import CableLayerProperties
from cable_thermal_model.model.cables.cable import Cable
from cable_thermal_model.model.cables.enum_classes_cable import CableLayer


class SoilDryingProperties(BaseModel):
    critical_soil_drying_temperature: float
    dry_soil_rho: float
    dry_soil_c: float


class CableSoil(Cable):
    """Finite difference cable model with soil discretization."""

    DRY_SOIL_RADIUS = 5.0

    def _set_heating_vector(self) -> None:
        """Initialize the heating vector for the cable."""
        self._heating_vector = np.zeros(self._radii_grid.size - 1)

    def integrate_timestep(
        self,
        previous_solution: np.ndarray,
        time_step: float,
        solution_at_boundary: float,
    ) -> np.ndarray:
        """This method solves the finite difference approximation to the heat equation using the implicit Euler method.

        For optimization purposes, the method uses the scipy.linalg.solve_banded method to solve the linear system.
        This means the three diagonals of finite difference matrix A are instead stored in a (3, N) array, where
        N is the length of the diagonal.

        Args:
            previous_solution (np.ndarray): The solution of the heat equation [°C] at the
                previous timestep (t).
            time_step (float): The size of the time steps [s] in the linearized
                time grid.
            solution_at_boundary (float): The solution at the boundary grid point [°C] used as a boundary condition.

        Returns:
            np.ndarray: The solution [°C] to the heat equation at the next timestep (t+1) for all grid points except
                the final grid point, at which a boundary condition is enforced.

        """
        A = self._get_processed_matrix(time_step=time_step)
        b = self._heating_vector.copy()
        b[-1] += self._upper_diagonal_last_element * solution_at_boundary
        b = self._capacity_grid[:-1] * previous_solution[:-1] + time_step * b

        return np.append(self._solve_system(A=A, b=b), solution_at_boundary)

    def update_soil_properties(
        self,
        soil_rho: float,
        soil_c: float,
        temperature_grid: np.ndarray,
        soil_drying: None | SoilDryingProperties = None,
    ) -> None:
        """This method updates the soil properties around a cable.

        Args:
            soil_rho (float): The thermal resistivity of the soil that is not dried out
            soil_c (float): The thermal capacity of the soil that is not dried out
            temperature_grid (np.ndarray): The temperature grid for the cable, as calculated for a given timestep.
            soil_drying (None | SoilDryingProperties): Whether the scenario takes soil drying into account, if provided, contains the properties of the dried-out soil.

        """
        self._apply_soil_drying(
            soil_rho=soil_rho, soil_c=soil_c, temperature_grid=temperature_grid, soil_drying=soil_drying
        )

    def _get_soil_grid_start_index(self) -> int:
        """Return the first grid index that belongs to soil around the cable."""
        return int((self._radii_grid <= self.layer_metrics.outer_radius).sum())

    def _apply_soil_drying(
        self,
        soil_rho: float,
        soil_c: float,
        temperature_grid: np.ndarray,
        soil_drying: None | SoilDryingProperties = None,
    ) -> None:
        """This method updates the soil resistivity values around a cable.

        This is meant to represent the IEC dried-out soil model. The soil will consist of an inner part of dried-out
        soil around the cable, and then a secondary part of standard soil
        The inner part has predefined thermal resistivity, which is defined in NPR Norm 3626.

        Notes:
            We do not update the number of layers of the cable, so the rho-grid may consist of a part that corresponds
            to a single layer, yet has multiple distinct values.

        Args:
            soil_rho (float): The thermal resistivity of the soil that is not dried out
            soil_c (float): The thermal capacity of the soil that is not dried out
            temperature_grid (np.ndarray): The temperature grid for the cable, as calculated for a given timestep.
            soil_drying (None | SoilDryingProperties):
                An instance of the SoilDrying class containing the parameters for soil drying. If None, no soil drying is applied.

        """
        if soil_drying is None:
            return

        critical_soil_drying_temperature = soil_drying.critical_soil_drying_temperature
        dry_soil_radius = self._get_dry_soil_radius(
            temperature_grid=temperature_grid, critical_soil_drying_temperature=critical_soil_drying_temperature
        )

        if abs(dry_soil_radius - self.DRY_SOIL_RADIUS) < 0.1:
            return

        self.DRY_SOIL_RADIUS = dry_soil_radius

        if CableLayer.WetSoil in self.layers:
            if CableLayer.DrySoil in self.layers:
                self.layer_properties[CableLayer.DrySoil].outer_radius = dry_soil_radius
                self.layer_properties[CableLayer.WetSoil].inner_radius = dry_soil_radius
            else:
                if dry_soil_radius > self.layer_properties[CableLayer.WetSoil].inner_radius:
                    self.layer_properties[CableLayer.DrySoil] = CableLayerProperties(
                        layer=CableLayer.DrySoil,
                        inner_radius=self.layer_properties[CableLayer.WetSoil].inner_radius,
                        outer_radius=dry_soil_radius,
                        rho=soil_drying.dry_soil_rho,
                        capacity=soil_drying.dry_soil_c,
                    )
                    self.layer_properties[CableLayer.WetSoil].inner_radius = dry_soil_radius
                    self.layer_properties = {
                        layer: self.layer_properties[layer]
                        for layer in sorted(self.layer_properties, key=lambda l: 1 if l == CableLayer.WetSoil else 0)
                    }
                if dry_soil_radius >= self.layer_properties[CableLayer.WetSoil].outer_radius:
                    del self.layer_properties[CableLayer.WetSoil]
        elif CableLayer.DrySoil in self.layers:
            if dry_soil_radius < self.layer_properties[CableLayer.DrySoil].outer_radius:
                self.layer_properties[CableLayer.WetSoil] = CableLayerProperties(
                    layer=CableLayer.WetSoil,
                    inner_radius=dry_soil_radius,
                    outer_radius=self.layer_properties[CableLayer.DrySoil].outer_radius,
                    rho=soil_rho,
                    capacity=soil_c,
                )
                self.layer_properties[CableLayer.DrySoil].outer_radius = dry_soil_radius
            if dry_soil_radius <= self.layer_properties[CableLayer.DrySoil].inner_radius:
                del self.layer_properties[CableLayer.DrySoil]
        else:
            raise ValueError("No soil layers detected!")

        if CableLayer.DrySoil in self.layer_properties and (
            self.layer_properties[CableLayer.DrySoil].outer_radius
            - self.layer_properties[CableLayer.DrySoil].inner_radius
            < 1e-3
        ):
            self.layer_properties[CableLayer.WetSoil].inner_radius = self.layer_properties[
                CableLayer.DrySoil
            ].inner_radius
            del self.layer_properties[CableLayer.DrySoil]
        elif CableLayer.WetSoil in self.layer_properties and (
            self.layer_properties[CableLayer.WetSoil].outer_radius
            - self.layer_properties[CableLayer.WetSoil].inner_radius
            < 1e-3
        ):
            self.layer_properties[CableLayer.DrySoil].outer_radius = self.layer_properties[
                CableLayer.WetSoil
            ].outer_radius
            del self.layer_properties[CableLayer.WetSoil]

        for cable_layer in CableLayer.soil_layers():
            if cable_layer in self.layer_properties:
                radius_factor = (
                    self.layer_properties[cable_layer].outer_radius / self.layer_properties[cable_layer].inner_radius
                )
                self._grid_counts[cable_layer] = int(max(2, int(20 * np.log2(radius_factor))))

        self._set_calculated_fields()

    def _get_dry_soil_radius(self, temperature_grid: np.ndarray, critical_soil_drying_temperature: float) -> float:
        """Return the dried-out soil radius from the current temperature grid.

        When soil drying is enabled, all grid points with temperature >= ``critical_soil_drying_temperature``
        are considered dried out and the radius of the outermost such grid point is returned, if any.

        Args:
            temperature_grid (np.ndarray): The temperature grid for the cable, as calculated for a given timestep.
            critical_soil_drying_temperature (float): The critical temperature at which soil is considered dried out.

        Returns:
            float | None: Radius of dried-out soil, or None if drying is disabled or absent.
        """
        soil_start_index = self._get_soil_grid_start_index()
        soil_temperature_grid = temperature_grid[soil_start_index:]
        if soil_temperature_grid[0] < critical_soil_drying_temperature:
            return self.layer_metrics.outer_radius

        soil_radii_grid = self._radii_grid[soil_start_index:]
        return np.interp(
            critical_soil_drying_temperature,
            soil_temperature_grid[::-1],
            soil_radii_grid[::-1],
        )

    def get_cable_copy_without_soil(self) -> Self:
        """This method returns a new CableSoil object with the soil layer removed."""
        for layer in self.layers:
            if layer in CableLayer.soil_layers():
                break
        else:
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
        """Create a fresh copy of the current cable object this was run from, but with an extra added soil layer.

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
        new_cable_soil_cable = cls.__new__(cls)
        new_cable_soil_cable.__dict__.update(deepcopy(cable.__dict__))

        outer_layer = new_cable_soil_cable.layers[-1]
        current_outer_radius = new_cable_soil_cable.layer_properties[outer_layer].outer_radius
        if soil_radius <= current_outer_radius:
            raise ValueError("The soil radius must be larger than the outer radius of the current outer layer!")

        new_cable_soil_cable.layer_properties[CableLayer.WetSoil] = CableLayerProperties(
            layer=CableLayer.WetSoil,
            inner_radius=current_outer_radius,
            outer_radius=soil_radius,
            rho=soil_rho,
            capacity=soil_capacity,
        )

        radius_factor = soil_radius / current_outer_radius
        new_cable_soil_cable._grid_counts[CableLayer.WetSoil] = int(
            max(2, int(logarithmic_soil_gridpoint_density * np.log2(radius_factor)))
        )
        new_cable_soil_cable._set_calculated_fields()
        return new_cable_soil_cable
