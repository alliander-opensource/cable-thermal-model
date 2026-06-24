# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path

import numpy as np
import pandas as pd

from cable_thermal_model.cable.schemas.pipe_schemas import PipeInputSchema


class Pipe:
    """Represents a pipe surrounding a cable, including its filling material and geometry."""

    def __init__(
        self,
        pipe_input: PipeInputSchema,
        outer_radius_cable: float,
    ):
        """Initialize the Pipe from input schema and the outer radius of the enclosed cable.

        Args:
            pipe_input: Schema containing pipe configuration (SDR, fill type, radii).
            outer_radius_cable: Outer radius of the cable that the pipe surrounds, in meters.

        """
        path = Path(__file__).parent.parent.parent.parent.resolve()
        self.outer_radius_cable = outer_radius_cable
        self.sdr = pipe_input.sdr
        self.fill_mat_df = (
            pd.read_csv(path / "data" / "pipe_filling_material_properties.csv", sep=";")
            .set_index("material")
            .loc[pipe_input.fill_type]
        )

        self.pipe_fill_cap = float(self.fill_mat_df["volumetric specific heat"])  # type: ignore[arg-type]
        self.trefoil_circuit_in_single_pipe = pipe_input.trefoil_circuit_in_single_pipe

        self.radius_factor = 2 / np.sqrt(3) + 1 if self.trefoil_circuit_in_single_pipe else 1
        self.inner_radius, self.outer_radius = self.determine_pipe_size(
            inner_radius=pipe_input.inner_radius,
            outer_radius=pipe_input.outer_radius,
        )

    def determine_pipe_size(
        self,
        inner_radius: float | None = None,
        outer_radius: float | None = None,
    ) -> tuple[float, float]:
        """Determine the correct pipe size for a given cable.

        Notes:
            The missing radius size values are calculated using the following rules:
            - If only the inner ór outer radius is specified:
                The SDR value is used to compute the other radius.
                (SDR or Standard Dimensional Ratio is defined as the ratio of outer diameter to the pipe thickness.)
            - If neither is specified:
                The first conventional size that leaves enough space for the cable is chosen according to the S7002.

            If the method is called with both inner_radius and outer_radius already prefilled, this method only checks
             if the inner radius is smaller than the outer radius, raising a ValueError if not the case.

        Returns:
            tuple[float, float]: A tuple indicating, in order, the inner and outer radius for the pipe.

        """
        if inner_radius is None:
            if outer_radius is None:
                inner_radius, outer_radius = self.choose_default_pipe_size()
            else:
                inner_radius = outer_radius - 2 * outer_radius / self.sdr

        if outer_radius is None:
            outer_radius = inner_radius * self.sdr / (self.sdr - 2)

        if inner_radius <= self.outer_radius_cable * self.radius_factor:
            if self.trefoil_circuit_in_single_pipe:
                raise ValueError(
                    f"Cable circuit does not fit into pipe with given radius. "
                    f"Radius of the circumcircle of the three cables is "
                    f"{self.outer_radius_cable * self.radius_factor}, "
                    f"inner radius of pipe is {inner_radius}."
                )
            else:
                raise ValueError(
                    f"Cable does not fit into pipe with given radius. "
                    f"Outer radius of cable is {self.outer_radius_cable}, "
                    f"inner radius of pipe is {inner_radius}."
                )

        if inner_radius > outer_radius:
            raise ValueError(
                f"Specified inner radius of the pipe is larger than the outer radius of the pipe. "
                f"Outer radius of cable is {outer_radius}, "
                f"inner radius of pipe is {inner_radius}."
            )

        return inner_radius, outer_radius

    def choose_default_pipe_size(
        self,
    ) -> tuple[float, float]:
        """Choose a default pipe size for a given cable.

        The first conventional size that leaves enough space for the cable is chosen.

        Returns:
            tuple[float, float]: A tuple indicating, in order, the
                inner and outer radius for the pipe.

        """
        default_outer_radii = [0.055, 0.0625, 0.08, 0.100]
        default_inner_radii = [r - 2 * r / self.sdr for r in default_outer_radii]

        # leave at least 30% wiggle room for the cable inside the pipe
        wiggle_room = 1.30
        minimum_inner_radius = self.outer_radius_cable * self.radius_factor * wiggle_room

        for inner_radius, outer_radius in zip(default_inner_radii, default_outer_radii, strict=True):
            if inner_radius >= minimum_inner_radius:
                self.inner_radius = inner_radius
                self.outer_radius = outer_radius
                return inner_radius, outer_radius

        raise ValueError("Cable does not fit in any default pipe. Please specify custom pipe size. ")

    def _get_lump_sum_resistivity_pipe_fill(
        self,
        T: float = 20,
    ):
        """Computes the lump sum resistivity of the medium inside the pipe via an inverse linear model.

        The model uses temperature-dependent resistivity of the medium inside a pipe.

        Args:
            T: temperature of the medium in the pipe in degrees Celsius.
                Room temperature (~20°C) is used as a default.

        Returns:
            float: The lump sum resistivity of the pipe filling material.

        References:
            - IEC 60287-2-1 - section 4.2.7.2

        """
        De = self.outer_radius_cable * self.radius_factor * 2e3

        U = self.fill_mat_df["U"]
        V = self.fill_mat_df["V"]
        Y = self.fill_mat_df["Y"]
        T4 = U / (1 + 0.1 * (V + Y * T) * De)

        return T4

    def get_thermal_resistivity_pipe_fill(
        self,
        T: float = 20,
    ) -> float:
        """This method computes the thermal resistivity by extracting it from the lump sum resistivity.

        Args:
            T: temperature of the medium in the pipe in degrees Celsius

        Returns:
            float: The thermal resistivity (Km/W) of the pipe filling material.

        """
        T4 = self._get_lump_sum_resistivity_pipe_fill(T)
        equivalent_radius_factor = 2 ** (2 / 3) if self.trefoil_circuit_in_single_pipe else 1
        pipe_fill_rho = (
            T4 * 2 * np.pi / np.log(self.inner_radius / (self.outer_radius_cable * equivalent_radius_factor))
        )
        return pipe_fill_rho
