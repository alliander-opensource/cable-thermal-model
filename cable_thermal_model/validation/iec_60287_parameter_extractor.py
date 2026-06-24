# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from pandera.typing import DataFrame

from cable_thermal_model import ModelFactory, StaticEnvSoil
from cable_thermal_model.cable.cable_circuit import CableKey
from cable_thermal_model.model.cables.enum_classes_cable import CableLayer
from cable_thermal_model.model.cables.fd_cable import FDCable
from cable_thermal_model.model.schemas import StateSoil
from cable_thermal_model.model.schemas.model_input_schemas import ScenarioSchemaSoil
from cable_thermal_model.model.schemas.model_output_schemas import ModelOutputSchema
from cable_thermal_model.validation.cable_analysis import CableAnalysis


@dataclass(frozen=True)
class IEC60287CableParameters:
    """Reference IEC 60287 parameters for a single cable validation case."""

    ampacity: float
    conductor_temperature: float
    screen_temperature: float
    armour_temperature: float | None
    surface_temperature: float
    dc_resistance_conductor_at_20: float
    skin_effect_factor: float
    proximity_effect_factor: float
    ac_resistance_conductor: float
    dc_resistance_screen: float
    t1: float
    t2: float
    t3: float
    t4: float
    conductor_loss: float
    dielectric_loss: float
    screen_loss: float
    armour_loss: float
    total_loss: float
    screen_loss_factor: float
    armour_loss_factor: float

    units = {
        "ampacity": "A",
        "conductor_temperature": "°C",
        "screen_temperature": "°C",
        "armour_temperature": "°C",
        "surface_temperature": "°C",
        "dc_resistance_conductor_at_20": "Ω/km",
        "skin_effect_factor": "",
        "proximity_effect_factor": "",
        "ac_resistance_conductor": "Ω/km",
        "dc_resistance_screen": "Ω/km",
        "t1": "K.m/W",
        "t2": "K.m/W",
        "t3": "K.m/W",
        "t4": "K.m/W",
        "conductor_loss": "W/m",
        "dielectric_loss": "W/m",
        "screen_loss": "W/m",
        "armour_loss": "W/m",
        "total_loss": "W/m",
        "screen_loss_factor": "",
        "armour_loss_factor": "",
    }

    @classmethod
    def from_dict(cls, d: dict):
        """Create an IEC60287CableParameters instance from a dictionary, ensuring all required keys are present."""
        return cls(**d)


@dataclass(frozen=True)
class _CableContext:
    """Context object containing all relevant information for a single cable."""

    cable_key: CableKey
    cable: FDCable
    analysis: CableAnalysis
    conductor_temperature: float
    screen_temperature: float
    armour_temperature: float | None
    surface_temperature: float
    ampacity: float
    ambient_temperature: float


def _get_cable_context(
    cable: FDCable,
    cable_key: CableKey,
    scenario: DataFrame[ScenarioSchemaSoil],
    model_output: ModelOutputSchema[StateSoil],
) -> _CableContext:
    load_column = f"load_{cable_key.circuit_name}"
    if load_column not in scenario.columns:
        raise KeyError(f"Missing load column '{load_column}' in scenario.")

    analysis = CableAnalysis(cable=cable, solution=model_output.state.full_solution[cable_key])

    screen_temperature = analysis.get_mean_temperature_cable_layer(layer=CableLayer.Screen)
    if screen_temperature is None:
        raise ValueError("Screen temperature is required and cannot be null.")

    armour_temperature = analysis.get_mean_temperature_cable_layer(layer=CableLayer.Armour)

    _, surface_temperature = analysis.get_boundary_temperatures_for_layer(layer=CableLayer.Sheath)

    return _CableContext(
        cable_key=cable_key,
        cable=cable,
        analysis=analysis,
        conductor_temperature=model_output.result[cable_key.circuit_name][cable_key.cable_position][
            CableLayer.Conductor
        ].iloc[-1],
        screen_temperature=screen_temperature,
        armour_temperature=armour_temperature,
        surface_temperature=surface_temperature,
        ampacity=float(scenario[load_column].iloc[0]),
        ambient_temperature=float(scenario["ambient_temperature"].iloc[0]),
    )


def _assert_loss_matches_heat_flow(
    loss: float,
    layer: CableLayer,
    context: _CableContext,
) -> None:
    """Assert that the loss for a layer matches the heat flow through that layer.

    Args:
        loss: The calculated heat loss for the layer (W/m).
        layer: The CableLayer for which to check the loss.
        context: The _CableContext containing cable and solution information.

    Raises:
        AssertionError: If the loss does not match the heat flow within set tolerance.

    """
    if not np.isclose(
        loss,
        context.analysis.get_heat_loss_cable_layer(layer=layer),
        rtol=0.01,
        atol=0.001,
    ):
        raise AssertionError(f"{layer.name} loss calculation is not consistent with temperatures!")


def _extract_ampacity_and_temperatures(context: _CableContext) -> dict[str, float | None]:
    """Extract ampacity and relevant temperatures for the cable.

    Args:
        context: The _CableContext containing cable and solution information.

    Returns:
        A dictionary with keys 'ampacity', 'conductor_temperature',
        'screen_temperature', 'armour_temperature', and 'surface_temperature'.

    """
    return {
        "ampacity": context.ampacity,
        "conductor_temperature": context.conductor_temperature,
        "screen_temperature": context.screen_temperature,
        "armour_temperature": context.armour_temperature,
        "surface_temperature": context.surface_temperature,
    }


def _extract_electrical_resistances(context: _CableContext) -> dict[str, float]:
    """Extract electrical resistances and skin and proximity effect factors.

    Args:
        context: The _CableContext containing cable and solution information.

    Returns:
        A dictionary with keys 'dc_resistance_conductor_at_20', 'skin_effect_factor',
        'proximity_effect_factor', 'ac_resistance_conductor', and 'dc_resistance_screen'
        with resistances in Ω/km.

    """
    cable = context.cable
    conductor_temperature = context.conductor_temperature

    r_dc_conductor_at_20 = cable.get_dc_resistance_conductor(20.0)
    r_dc_conductor = cable.get_dc_resistance_conductor(conductor_temperature)

    y_s = cable.get_skin_effect_factor(Rdc=r_dc_conductor)
    y_p = cable.get_proximity_effect_factor(Rdc=r_dc_conductor, s=cable.s)
    r_ac_conductor = cable._get_ac_resistance_conductor_from_dc_resistance(Rdc=r_dc_conductor, s=cable.s)
    r_dc_screen = cable._get_resistance_screen(context.screen_temperature)

    return {
        "dc_resistance_conductor_at_20": r_dc_conductor_at_20 * 1e3,
        "skin_effect_factor": y_s,
        "proximity_effect_factor": y_p,
        "ac_resistance_conductor": r_ac_conductor * 1e3,
        "dc_resistance_screen": r_dc_screen * 1e3,
    }


def _extract_thermal_resistances(context: _CableContext) -> dict[str, float]:
    """Extract thermal resistances for T1-T4 according to IEC 60287 definitions.

    Args:
        context: The _CableContext containing cable and solution information.

    Returns:
        A dictionary with keys 't1', 't2', 't3', 't4'.

    """
    analysis = context.analysis

    t1_layers = [
        CableLayer.ConductorScreen,
        CableLayer.Insulation,
        CableLayer.InsulationScreen,
    ]
    t1 = sum(analysis.get_thermal_resistance_cable_layer(layer) for layer in t1_layers)

    t2 = analysis.get_thermal_resistance_cable_layer(CableLayer.Bedding)
    t3 = analysis.get_thermal_resistance_cable_layer(CableLayer.Sheath)

    t4 = analysis.get_thermal_resistance_external_medium(ambient_temperature=context.ambient_temperature)

    return {
        "t1": t1,
        "t2": t2,
        "t3": t3,
        "t4": t4,
    }


def _extract_losses(context: _CableContext) -> dict[str, float]:
    """Extract heat losses for the cable layers and screen and armour loss factors.

    Args:
        context: The _CableContext containing cable and solution information.

    Returns:
        A dictionary with keys 'conductor_loss', 'dielectric_loss', 'screen_loss',
        'armour_loss', 'total_loss', 'screen_loss_factor', and 'armour_loss_factor'.

    """
    cable = context.cable
    screen_loss_factor = cable.get_cable_screen_loss_factor(
        Ts=context.screen_temperature,
        Tc=context.conductor_temperature,
    )
    armour_loss_factor = cable.Ft - 1.0

    w_c = cable.get_heat_generation_conductor(
        ac_current=True,
        load=context.ampacity,
        conductor_temperature=context.conductor_temperature,
        temperature_dependent_electric_resistance=True,
    )
    w_d = cable.get_dielectric_loss_for_cable()
    w_s = w_c * screen_loss_factor
    w_a = w_c * armour_loss_factor

    _assert_loss_matches_heat_flow(w_c, CableLayer.Conductor, context)
    _assert_loss_matches_heat_flow(w_d, CableLayer.Insulation, context)
    _assert_loss_matches_heat_flow(w_s, CableLayer.Screen, context)
    _assert_loss_matches_heat_flow(w_a, CableLayer.Armour, context)

    return {
        "conductor_loss": w_c,
        "dielectric_loss": w_d,
        "screen_loss": w_s,
        "armour_loss": w_a,
        "total_loss": w_c + w_d + w_s + w_a,
        "screen_loss_factor": screen_loss_factor,
        "armour_loss_factor": armour_loss_factor,
    }


def _extract_parameters_for_cable(context: _CableContext) -> pd.Series:
    """Extract all relevant IEC parameters for a single cable and return as a pd.Series.

    Args:
        context: The _CableContext containing cable and solution information.

    Returns:
        A Series with all extracted parameters for the cable, indexed by parameter name.

    """
    params = {
        **_extract_ampacity_and_temperatures(context),
        **_extract_electrical_resistances(context),
        **_extract_thermal_resistances(context),
        **_extract_losses(context),
    }

    parameters = IEC60287CableParameters.from_dict(params)

    return pd.Series(asdict(parameters), name=context.cable_key)


def build_scenario(
    circuit_ratings: dict[str, float],
    soil_thermal_resistivity: float,
    soil_thermal_capacity: float,
    ambient_temperature: float,
) -> DataFrame[ScenarioSchemaSoil]:
    """Build a static scenario DataFrame used for IEC parameter extraction.

    Args:
        circuit_ratings: Mapping of circuit names to their load ratings (A).
        soil_thermal_resistivity: Thermal resistivity of the surrounding soil (K*m/W).
        soil_thermal_capacity: Volumetric heat capacity of the soil (J/(m^3*K)).
        ambient_temperature: Ambient temperature of the soil (C).

    Returns:
        A DataFrame with columns for ambient temperature, soil properties, and load.

    """
    scenario = pd.DataFrame(
        data={
            "ambient_temperature": ambient_temperature,
            "soil_thermal_resistivity": soil_thermal_resistivity,
            "soil_thermal_capacity": soil_thermal_capacity,
        },
        index=pd.timedelta_range(start="0D", end="30000D", periods=20),
    )

    for circuit_name, rating in circuit_ratings.items():
        scenario[f"load_{circuit_name}"] = rating

    return ScenarioSchemaSoil.validate(scenario)


def extract_iec_60287_parameters(
    static_env: StaticEnvSoil,
    circuit_ratings: dict[str, float],
    soil_thermal_resistivity: float,
    soil_thermal_capacity: float,
    ambient_temperature: float,
) -> pd.DataFrame:
    """Compute IEC validation parameters and return a compact DataFrame.

    Args:
        static_env: The static environment to solve.
        circuit_ratings: Mapping of circuit names to their load ratings (A).
        soil_thermal_resistivity: Thermal resistivity of the surrounding soil (K*m/W).
        soil_thermal_capacity: Volumetric heat capacity of the soil (J/(m^3*K)).
        ambient_temperature: Ambient temperature of the soil (°C).

    Returns:
        A DataFrame with the relevant IEC parameters for all cables in static_env.

    """
    scenario = build_scenario(
        circuit_ratings=circuit_ratings,
        soil_thermal_resistivity=soil_thermal_resistivity,
        soil_thermal_capacity=soil_thermal_capacity,
        ambient_temperature=ambient_temperature,
    )

    model = ModelFactory.create_model(static_env, scenario)
    model_output = model.run()

    parameters = pd.DataFrame()
    for cable_key, pos_cable in model.cables.items():
        context = _get_cable_context(
            cable=pos_cable.cable,
            cable_key=cable_key,
            scenario=scenario,
            model_output=model_output,
        )

        results = _extract_parameters_for_cable(context)
        parameters = pd.concat([parameters, results], axis=1)

    return parameters


def compare_parameters_to_reference(
    extracted_parameters: pd.Series,
    reference_parameters: IEC60287CableParameters,
) -> pd.DataFrame:
    """Compare extracted parameters to reference values and compute differences.

    Args:
        extracted_parameters: A Series containing the extracted parameters for a cable.
        reference_parameters: An IEC60287CableParameters object with reference values.

    Returns:
        A DataFrame comparing extracted and reference parameters, including differences.

    """
    reference_series = pd.Series(asdict(reference_parameters))

    comparison = pd.DataFrame({"Extracted": extracted_parameters, "Reference": reference_series})
    comparison["Diff"] = comparison["Extracted"] - comparison["Reference"]
    comparison["Rel Diff (%)"] = round(comparison["Diff"] / comparison["Reference"].replace(0, np.nan) * 100, 2)
    comparison.insert(0, "Unit", pd.Series(IEC60287CableParameters.units))

    return comparison
