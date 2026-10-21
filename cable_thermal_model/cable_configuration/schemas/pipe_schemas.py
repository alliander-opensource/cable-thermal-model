# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from pydantic import BaseModel, Field

from cable_thermal_model.model.cables.enum_classes_cable import PipeFillType


class PipeInputSchema(BaseModel):
    """Input schema defining pipe geometry and fill material."""

    inner_radius: float | None = Field(default=None, description="Inner radius of the pipe in meters")
    outer_radius: float | None = Field(default=None, description="Outer radius of the pipe in meters")
    fill_type: PipeFillType = Field(
        description="Type of material filling the pipe",
        examples=[PipeFillType.Air, PipeFillType.Water],
    )
    sdr: float = Field(default=11.0, description="Standard Dimension Ratio of the pipe")
    trefoil_circuit_in_single_pipe: bool = Field(default=False, description="Whether three cables are in one pipe")
