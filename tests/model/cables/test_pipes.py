# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from cable_thermal_model.cable.schemas.pipe_schemas import PipeInputSchema
from cable_thermal_model.model.cables.enum_classes_cable import PipeFillType
from cable_thermal_model.model.cables.pipe import Pipe


@pytest.mark.parametrize(
    "outer_radius_cable,trefoil_circuit_in_single_pipe,expected_error",
    [
        (0.01, True, None),
        (0.025, False, None),
        (0.045, False, "Cable does not fit"),
        (0.025, True, "Cable circuit does not fit"),
        (0.05, False, "Cable does not fit"),
    ],
)
def test_nonfitting_pipe(outer_radius_cable, trefoil_circuit_in_single_pipe, expected_error):
    pipe_input = PipeInputSchema(
        outer_radius=0.055,
        fill_type=PipeFillType.Water,
        trefoil_circuit_in_single_pipe=trefoil_circuit_in_single_pipe,
    )
    if expected_error is None:
        pipe = Pipe(
            pipe_input=pipe_input,
            outer_radius_cable=outer_radius_cable,
        )
        if trefoil_circuit_in_single_pipe:
            assert np.isclose(pipe.radius_factor, 2.15, atol=0.01)
        else:
            assert np.isclose(pipe.radius_factor, 1.0, atol=0.01)

        assert pipe.inner_radius > outer_radius_cable * pipe.radius_factor
    else:
        with pytest.raises(ValueError) as excinfo:
            Pipe(pipe_input=pipe_input, outer_radius_cable=outer_radius_cable)
        assert expected_error in str(excinfo.value)
