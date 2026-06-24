# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for CIGRE TB880 case 0 variants.

Current status:
- Dielectric loss should not be neglected in software.
- t3 and t4 values are significantly different from reference.
- No analysis yet for cables in ductbancks, air and unfilled troughs.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cable_thermal_model import (
    BondingType,
    CablePosition,
    CircuitType,
    PipeFillType,
    PipeInputSchema,
    StaticEnvSoil,
)
from cable_thermal_model.cable.cable_circuit import CableKey
from cable_thermal_model.cable.schemas.circuit_schemas import CircuitInSoilFromCableIdInputSchema
from cable_thermal_model.validation.iec_60287_parameter_extractor import (
    IEC60287CableParameters,
    extract_iec_60287_parameters,
)

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

CABLE_ID = "XLPE 76/132kV 1x630 mm2 Cu"
BURIAL_DEPTH = 1.0
FORMATION = CircuitType.Trefoil
SOIL_RHO = 1.0
SOIL_CAPACITY = 2e6
AMBIENT_T = 20.0

CABLE_KEY = CableKey(circuit_name="A", cable_position=CablePosition.TrefoilTop)
DEFAULT_RTOL = 0.005  # Relative tolerance for most assertions, overwritten where needed

# ----------------------------------------------------------------------
# Fixture: parameter builder
# ----------------------------------------------------------------------


@pytest.fixture
def build_parameters() -> Callable:  # type: ignore[no-untyped-def]
    """Return a helper that builds IEC 60287 parameters."""

    def _builder(
        *,
        bonding_type: BondingType,
        ampacity: float,
        conductor_spacing: float | None = None,
        pipe: PipeInputSchema | None = None,
    ) -> pd.DataFrame:
        env = StaticEnvSoil()
        env.add_circuit_from_cable_id(
            CircuitInSoilFromCableIdInputSchema(
                x=0,
                y=-BURIAL_DEPTH,
                circuit_name="A",
                cable_id=CABLE_ID,
                circuit_type=FORMATION,
                bonding_type=bonding_type,
                dist=conductor_spacing,
                pipe=pipe,
                cable_source_file_path=Path("data/cable_specs_TB880.csv"),
            )
        )

        return extract_iec_60287_parameters(
            static_env=env,
            circuit_ratings={"A": ampacity},
            soil_thermal_resistivity=SOIL_RHO,
            soil_thermal_capacity=SOIL_CAPACITY,
            ambient_temperature=AMBIENT_T,
        )

    return _builder


# ----------------------------------------------------------------------
# Test case dataclass
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TestCase:
    """Definition of a test case, including inputs and expected outputs."""

    name: str
    bonding_type: BondingType
    pipe: PipeInputSchema | None
    expected: IEC60287CableParameters
    rtol: dict[str, float] | None = None


# ----------------------------------------------------------------------
# Test cases using your existing dataclass
# ----------------------------------------------------------------------

TEST_CASES = [
    TestCase(
        name="double-bonded",
        bonding_type=BondingType.TwoSided,
        pipe=None,
        expected=IEC60287CableParameters(
            ampacity=821.7763,
            conductor_temperature=90.0,
            screen_temperature=78.7130,
            armour_temperature=None,
            surface_temperature=75.68,
            dc_resistance_conductor_at_20=0.0283,
            skin_effect_factor=0.06012412684,
            proximity_effect_factor=0.03510006481,
            ac_resistance_conductor=0.03952152638,
            dc_resistance_screen=0.20641,
            t1=0.4198714890,
            t2=0.0,
            t3=0.08671937479,
            t4=1.594692892,
            conductor_loss=26.6895,
            dielectric_loss=0.3851382172,
            screen_loss=7.8442,
            armour_loss=0.0,
            total_loss=34.9188,
            screen_loss_factor=0.29390,
            armour_loss_factor=0.0,
        ),
        rtol={
            # Overrides based on original test tolerances
            "t3": 0.4,
            "t4": 0.05,
            "conductor_temperature": 0.02,
            "screen_temperature": 0.02,
            "surface_temperature": 0.04,
            "skin_effect_factor": 0.01,
            "proximity_effect_factor": 0.01,
            "screen_loss_factor": 0.01,
            "total_loss": 0.01,
        },
    ),
    TestCase(
        name="single-bonded",
        bonding_type=BondingType.OneSided,
        pipe=None,
        expected=IEC60287CableParameters(
            ampacity=886.1752854720,
            conductor_temperature=90.0,
            screen_temperature=76.8878,
            armour_temperature=None,
            surface_temperature=73.95,
            dc_resistance_conductor_at_20=0.0283,
            skin_effect_factor=0.06012412684,
            proximity_effect_factor=0.03510006481,
            ac_resistance_conductor=0.03952152638,
            dc_resistance_screen=0.20518,
            t1=0.4198714890,
            t2=0.0,
            t3=0.08671937479,
            t4=1.594692892,
            conductor_loss=31.0365,
            dielectric_loss=0.3851382172,
            screen_loss=2.4117,
            armour_loss=0.0,
            total_loss=33.8333,
            screen_loss_factor=0.07770,
            armour_loss_factor=0.0,
        ),
        rtol={
            "t3": 0.4,
            "t4": 0.05,
            "conductor_temperature": 0.02,
            "screen_temperature": 0.02,
            "surface_temperature": 0.04,
            "skin_effect_factor": 0.01,
            "proximity_effect_factor": 0.01,
            "screen_loss_factor": 0.01,
            "total_loss": 0.01,
        },
    ),
    TestCase(
        name="ducts",
        bonding_type=BondingType.TwoSided,
        pipe=PipeInputSchema(
            inner_radius=0.0597,
            outer_radius=0.07,
            fill_type=PipeFillType.Air,
            sdr=13.6,
            trefoil_circuit_in_single_pipe=False,
        ),
        expected=IEC60287CableParameters(
            ampacity=682.8145,
            conductor_temperature=90.0,
            screen_temperature=82.3590,
            armour_temperature=None,
            surface_temperature=80.55,
            dc_resistance_conductor_at_20=0.0283,
            skin_effect_factor=0.06012412684,
            proximity_effect_factor=0.01010775634,
            ac_resistance_conductor=0.0386196707,
            dc_resistance_screen=0.20886,
            t1=0.419871488892,
            t2=0.0,
            t3=0.05419960924,
            t4=1.8121,
            conductor_loss=18.0059,
            dielectric_loss=0.385138217161,
            screen_loss=15.0224,
            armour_loss=0.0,
            total_loss=33.4134,
            screen_loss_factor=0.83431,
            armour_loss_factor=0.0,
        ),
        rtol={
            "t4": 0.01,
            "conductor_temperature": 0.02,
            "screen_temperature": 0.02,
            "surface_temperature": 0.02,
            "skin_effect_factor": 0.01,
            "proximity_effect_factor": 0.01,
            "total_loss": 0.02,
            "screen_loss_factor": 0.02,
        },
    ),
]


# ----------------------------------------------------------------------
# Unified test
# ----------------------------------------------------------------------


@pytest.mark.parametrize("case", TEST_CASES, ids=lambda c: c.name)
def test_case_0_variants(case: TestCase, build_parameters: Callable) -> None:
    """Validate IEC 60287 reference values using typed dataclasses."""
    params = build_parameters(
        bonding_type=case.bonding_type,
        ampacity=case.expected.ampacity,
        pipe=case.pipe,
    )

    actual = params[CABLE_KEY]
    expected = case.expected

    # Compare each field numerically
    for field in expected.__dataclass_fields__:
        expected_value = getattr(expected, field)
        actual_value = getattr(actual, field)

        if expected_value is None:
            assert actual_value is None or np.isnan(actual_value)
            continue

        # Get field-specific tolerance or default
        tol = case.rtol.get(field, DEFAULT_RTOL) if case.rtol else DEFAULT_RTOL

        assert np.isclose(actual_value, expected_value, rtol=tol), (
            f"Mismatch in {field}: expected {expected_value}, got {actual_value}"
        )
