# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from dataclasses import dataclass

import numpy as np

from cable_thermal_model import CableLayer
from cable_thermal_model.model.cables.fd_cable import FDCable


@dataclass(frozen=True)
class CableAnalysis:
    """Post-processing and validation helpers for one cable solution state."""

    cable: FDCable  # The finite-difference representation of the cable.
    solution: np.ndarray  # The grid of temperatures for the cable.

    @staticmethod
    def get_thermal_resistance(rho: float, r_inner: float, r_outer: float) -> float:
        """Calculate the thermal resistance of a concentrical layer."""
        return np.log(r_outer / r_inner) / (2 * np.pi) * rho

    @staticmethod
    def get_temperature_correction(
        reference_radius: float,
        neighbor_radius: float,
        heat_flow: float,
        rho: float,
    ) -> float:
        """Get the boundary-temperature correction from a neighboring grid point."""
        return (reference_radius - neighbor_radius) / (3 * reference_radius + neighbor_radius) * heat_flow * rho / np.pi

    def get_mean_temperature_cable_layer(self, layer: CableLayer) -> float | None:
        """Calculate the mean temperature for a cable layer, ``None`` if absent."""
        if layer not in self.cable.layers:
            return None

        layer_start, layer_end = self.cable.get_layer_indices_for_layer(layer)
        return float((self.solution[layer_start] + self.solution[layer_end]) / 2.0)

    def get_boundary_temperatures_for_layer(self, layer: CableLayer) -> tuple[float, float]:
        """Calculate inner and outer boundary temperatures for a cable layer."""
        if layer not in self.cable.layers:
            raise ValueError(f"Layer {layer} is not present in the cable, cannot calculate boundary temperatures.")
        start_index, end_index = self.cable.get_layer_indices_for_layer(layer)
        outer_neighbor_index = end_index + 1 if end_index != len(self.solution) - 1 else end_index
        required_index = max(start_index, outer_neighbor_index)
        if required_index >= len(self.solution):
            raise ValueError("The solution grid is too short for the requested layer boundaries.")

        heat_flow = self.get_heat_flow_cable_layer(layer=layer)

        radii = self.cable.radii_grid
        rho = self.cable.layer_properties[layer].rho

        if start_index == 0:
            inner_temperature = self.solution[start_index]
        else:
            inner_temperature = self.solution[start_index] + self.get_temperature_correction(
                reference_radius=radii[start_index],
                neighbor_radius=radii[start_index - 1],
                heat_flow=heat_flow,
                rho=rho,
            )

        if end_index == len(self.solution) - 1:
            outer_temperature = self.solution[end_index]
        else:
            outer_temperature = self.solution[end_index] - self.get_temperature_correction(
                reference_radius=radii[end_index],
                neighbor_radius=radii[end_index + 1],
                heat_flow=heat_flow,
                rho=rho,
            )

        return inner_temperature, outer_temperature

    def get_heat_flow(self, inner_index: int) -> float:
        """Calculate heat flow per unit length between grid points at inner_index and inner_index + 1."""
        outer_index = inner_index + 1

        if inner_index < 0 or outer_index >= len(self.cable.radii_grid):
            raise ValueError("The indices must lie within the finite-difference grid of the cable.")

        if np.asarray(self.solution).ndim != 1 or outer_index >= len(self.solution):
            raise ValueError("The solution array must be one-dimensional and include values for both grid points.")

        delta_radius = self.cable.grid_deltas[inner_index]
        inter_radius = self.cable.radii_grid[inner_index] + 0.5 * delta_radius

        # Calculate the interstitial resistivity value between the two grid points
        inter_rho = self.cable._calculate_inter_rhos(
            radii=self.cable.radii_grid[inner_index : outer_index + 1],
            inter_radii=np.array([inter_radius]),
            rhos=self.cable.rho_grid[inner_index : outer_index + 1],
        )[0]

        # Calculate the temperature gradient between the two grid points
        temperature_gradient = (self.solution[outer_index] - self.solution[inner_index]) / delta_radius

        # Calculate the heat flow using Fourier's law of heat conduction
        return -2 * np.pi * inter_radius * temperature_gradient / inter_rho

    def get_heat_flow_at_radius(self, r: float) -> float:
        """Calculate approximate heat flow per unit length for a given radius.

        If radius lies between two grid points r[i] and r[i+1], heat flow between these two points is returned.
        If radius equals a grid point r[i], heat flow between r[i-1] and r[i] is returned.
        """
        if r <= self.cable.radii_grid[0] or r > self.cable.radii_grid[-1]:
            raise ValueError("The radius must lie within the finite-difference grid of the cable.")

        i = np.searchsorted(self.cable.radii_grid, r) - 1

        return self.get_heat_flow(inner_index=int(i))

    def get_heat_flow_cable_layer(self, layer: CableLayer) -> float:
        """Calculate the heat flow per meter for a non-heat-generating cable layer."""
        start_index, end_index = self.cable.get_layer_indices_for_layer(layer)
        if end_index <= start_index:
            raise ValueError("The outer index should be larger than the inner index.")
        if layer == CableLayer.Conductor:
            raise ValueError("The conductor layer generates heat, cannot calculate heat flow.")

        heat_flow_start = self.get_heat_flow(inner_index=start_index)
        heat_flow_end = self.get_heat_flow(inner_index=end_index - 1)

        if not np.isclose(heat_flow_start, heat_flow_end, rtol=1e-5):
            raise ValueError(
                "The heat flow should be the same across both grid points of the layer, but got "
                f"{heat_flow_start} and {heat_flow_end} for layer {layer}."
            )

        return heat_flow_start

    def get_heat_loss_cable_layer(self, layer: CableLayer) -> float:
        """Calculate the net heat loss (W/m) for one layer in the cable."""
        layers = self.cable.layers
        if layer not in layers:
            return 0.0

        layer_index = layers.index(layer)
        heat_in = 0.0 if layer_index == 0 else self.get_heat_flow_cable_layer(layer=layers[layer_index - 1])

        heat_out = self.get_heat_flow_cable_layer(layer=layers[layer_index + 1])
        return heat_out - heat_in

    def get_thermal_resistance_cable_layer(self, layer: CableLayer) -> float:
        """Calculate the thermal resistance for a cable layer, 0 if absent."""
        if layer not in self.cable.layers:
            return 0.0

        layer_properties = self.cable.layer_properties[layer]
        return self.get_thermal_resistance(
            rho=layer_properties.rho,
            r_inner=layer_properties.inner_radius,
            r_outer=layer_properties.outer_radius,
        )

    def get_thermal_resistance_external_medium(self, ambient_temperature: float) -> float:
        """Calculate thermal resistance between cable sheath surface and ambient."""
        _, surface_temperature = self.get_boundary_temperatures_for_layer(layer=CableLayer.Sheath)
        heat_flow = self.get_heat_flow_cable_layer(layer=CableLayer.Sheath)
        return (surface_temperature - ambient_temperature) / heat_flow
