# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cable_thermal_model import CableLayer
from cable_thermal_model.validation.cable_analysis import CableAnalysis


class TestCableAnalysisStatics:
    def test_get_thermal_resistance(self) -> None:
        result = CableAnalysis.get_thermal_resistance(rho=2.5, r_inner=0.01, r_outer=0.02)
        expected = np.log(2.0) / (2 * np.pi) * 2.5
        assert result == pytest.approx(expected)

    def test_get_temperature_correction(self) -> None:
        result = CableAnalysis.get_temperature_correction(
            reference_radius=0.03,
            neighbor_radius=0.02,
            heat_flow=5.0,
            rho=2.0,
        )
        expected = (0.03 - 0.02) / (3 * 0.03 + 0.02) * 5.0 * 2.0 / np.pi
        assert result == pytest.approx(expected)


class TestCableAnalysisMeanTemperature:
    def test_returns_none_when_layer_absent(self) -> None:
        cable = Mock()
        cable.layers = [CableLayer.Conductor, CableLayer.Insulation]
        analysis = CableAnalysis(cable=cable, solution=np.array([90.0, 80.0, 70.0]))

        assert analysis.get_mean_temperature_cable_layer(CableLayer.Sheath) is None
        cable.get_layer_indices_for_layer.assert_not_called()

    def test_returns_average_of_boundary_points(self) -> None:
        cable = Mock()
        cable.layers = [CableLayer.Conductor, CableLayer.Sheath]
        cable.get_layer_indices_for_layer.return_value = (1, 3)
        analysis = CableAnalysis(cable=cable, solution=np.array([92.0, 84.0, 79.0, 74.0]))

        assert analysis.get_mean_temperature_cable_layer(CableLayer.Sheath) == pytest.approx(79.0)


class TestCableAnalysisHeatFlowAtRadius:
    def test_rejects_radius_at_or_below_first_grid_point(self) -> None:
        cable = Mock()
        cable.radii_grid = np.array([0.0, 0.01, 0.02])
        analysis = CableAnalysis(cable=cable, solution=np.array([80.0, 70.0, 60.0]))

        with pytest.raises(ValueError, match="must lie within"):
            analysis.get_heat_flow_at_radius(0.0)

    def test_rejects_radius_above_last_grid_point(self) -> None:
        cable = Mock()
        cable.radii_grid = np.array([0.0, 0.01, 0.02])
        analysis = CableAnalysis(cable=cable, solution=np.array([80.0, 70.0, 60.0]))

        with pytest.raises(ValueError, match="must lie within"):
            analysis.get_heat_flow_at_radius(0.03)

    def test_radius_between_grid_points_uses_enclosing_inner_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cable = Mock()
        cable.radii_grid = np.array([0.0, 0.01, 0.02, 0.03])
        analysis = CableAnalysis(cable=cable, solution=np.array([90.0, 80.0, 70.0, 60.0]))

        captured: dict[str, int] = {}

        def fake_get_heat_flow(self: CableAnalysis, inner_index: int) -> float:
            captured["inner_index"] = inner_index
            return 12.3

        monkeypatch.setattr(CableAnalysis, "get_heat_flow", fake_get_heat_flow)

        result = analysis.get_heat_flow_at_radius(0.015)

        assert result == pytest.approx(12.3)
        assert captured["inner_index"] == 1

    def test_radius_equal_to_grid_point_uses_left_interval(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cable = Mock()
        cable.radii_grid = np.array([0.0, 0.01, 0.02, 0.03])
        analysis = CableAnalysis(cable=cable, solution=np.array([90.0, 80.0, 70.0, 60.0]))

        captured: dict[str, int] = {}

        def fake_get_heat_flow(self: CableAnalysis, inner_index: int) -> float:
            captured["inner_index"] = inner_index
            return 6.7

        monkeypatch.setattr(CableAnalysis, "get_heat_flow", fake_get_heat_flow)

        result = analysis.get_heat_flow_at_radius(0.02)

        assert result == pytest.approx(6.7)
        assert captured["inner_index"] == 1

    def test_integration_radius_between_grid_points_matches_interval_heat_flow(self, single_core_cable_xlpe) -> None:
        radii_grid = single_core_cable_xlpe.radii_grid
        inner_index = 10
        radius = 0.5 * (radii_grid[inner_index] + radii_grid[inner_index + 1])
        solution = np.linspace(90.0, 40.0, len(radii_grid))
        analysis = CableAnalysis(cable=single_core_cable_xlpe, solution=solution)

        assert analysis.get_heat_flow_at_radius(radius) == pytest.approx(analysis.get_heat_flow(inner_index))

    def test_integration_radius_equal_to_grid_point_uses_left_interval(self, single_core_cable_xlpe) -> None:
        radii_grid = single_core_cable_xlpe.radii_grid
        grid_index = 10
        radius = radii_grid[grid_index]
        solution = np.linspace(90.0, 40.0, len(radii_grid))
        analysis = CableAnalysis(cable=single_core_cable_xlpe, solution=solution)

        assert analysis.get_heat_flow_at_radius(radius) == pytest.approx(analysis.get_heat_flow(grid_index - 1))


class TestCableAnalysisBoundaryTemperatures:
    def test_rejects_missing_layer(self) -> None:
        cable = Mock()
        cable.layers = [CableLayer.Conductor]
        analysis = CableAnalysis(cable=cable, solution=np.array([90.0, 80.0, 70.0]))

        with pytest.raises(ValueError, match="not present in the cable"):
            analysis.get_boundary_temperatures_for_layer(CableLayer.Sheath)

    def test_rejects_short_solution_for_requested_layer(self) -> None:
        cable = Mock()
        cable.layers = [CableLayer.Sheath]
        cable.layer_properties = {CableLayer.Sheath: SimpleNamespace(rho=2.0)}
        cable.radii_grid = np.array([0.01, 0.02, 0.03, 0.04])
        cable.get_layer_indices_for_layer.return_value = (1, 3)
        analysis = CableAnalysis(cable=cable, solution=np.array([80.0, 70.0]))

        with pytest.raises(ValueError, match="too short"):
            analysis.get_boundary_temperatures_for_layer(CableLayer.Sheath)


class TestCableAnalysisHeatFlowLayer:
    def test_rejects_conductor_layer(self) -> None:
        cable = Mock()
        cable.get_layer_indices_for_layer.return_value = (1, 2)
        analysis = CableAnalysis(cable=cable, solution=np.array([90.0, 80.0]))

        with pytest.raises(ValueError, match="conductor layer generates heat"):
            analysis.get_heat_flow_cable_layer(CableLayer.Conductor)

    def test_rejects_non_increasing_indices(self) -> None:
        cable = Mock()
        cable.get_layer_indices_for_layer.return_value = (3, 3)
        analysis = CableAnalysis(cable=cable, solution=np.array([95.0, 90.0, 85.0, 80.0]))

        with pytest.raises(ValueError, match="outer index should be larger"):
            analysis.get_heat_flow_cable_layer(CableLayer.Sheath)


class TestCableAnalysisHeatLoss:
    def test_returns_zero_for_absent_layer(self) -> None:
        cable = Mock()
        cable.layers = [CableLayer.Conductor, CableLayer.Sheath]
        analysis = CableAnalysis(cable=cable, solution=np.array([90.0, 80.0]))

        assert analysis.get_heat_loss_cable_layer(CableLayer.Armour) == 0.0


class TestCableAnalysisThermalResistance:
    def test_layer_resistance_zero_when_absent(self) -> None:
        cable = Mock()
        cable.layers = [CableLayer.Conductor]
        analysis = CableAnalysis(cable=cable, solution=np.array([]))

        assert analysis.get_thermal_resistance_cable_layer(CableLayer.Sheath) == 0.0

    def test_layer_resistance_when_present(self) -> None:
        cable = Mock()
        cable.layers = [CableLayer.Conductor, CableLayer.Sheath]
        cable.layer_properties = {CableLayer.Sheath: SimpleNamespace(rho=3.0, inner_radius=0.02, outer_radius=0.03)}
        analysis = CableAnalysis(cable=cable, solution=np.array([]))

        expected = CableAnalysis.get_thermal_resistance(3.0, 0.02, 0.03)
        assert analysis.get_thermal_resistance_cable_layer(CableLayer.Sheath) == pytest.approx(expected)
