# SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandera.typing import DataFrame

from cable_thermal_model.cable.cable_builder import CableBuilder
from cable_thermal_model.cable.cable_circuit import (
    BondingType,
    CableKey,
    CablePosition,
    CircuitBuilder,
    CircuitType,
)
from cable_thermal_model.cable.schemas.cable_input_schemas import (
    CableConstructionalInputSchema,
    CableScreenType,
    ConductorInputSchema,
    InsulationInputSchema,
    ScreenInputSchema,
    SheathInputSchema,
)
from cable_thermal_model.cable.schemas.circuit_schemas import (
    CircuitConfigurationFromCableId,
    CircuitInAirFromCableIdInputSchema,
    CircuitInSoilFromCableIdInputSchema,
    CircuitInSoilFromCableInputSchema,
)
from cable_thermal_model.cable.schemas.pipe_schemas import PipeInputSchema
from cable_thermal_model.environment.measurement_point import MeasurementPointKey
from cable_thermal_model.environment.static_env_air import StaticEnvAir
from cable_thermal_model.environment.static_env_soil import StaticEnvSoil
from cable_thermal_model.model.cables.cable_air import CableAir
from cable_thermal_model.model.cables.cable_soil import CableSoil
from cable_thermal_model.model.cables.cable_trefoil_circuit_single_pipe import CableTrefoilCircuitSinglePipeInSoil
from cable_thermal_model.model.cables.enum_classes_cable import (
    CableConductorCount,
    CableConductorMaterial,
    CableConductorShape,
    CableConductorSurfaceType,
    CableInsulationMaterial,
    CableScreenMaterial,
    CableSheathMaterial,
    PipeFillType,
)
from cable_thermal_model.model.model import Model
from cable_thermal_model.model.model_factory import ModelFactory
from cable_thermal_model.model.model_soil import ModelSoil
from cable_thermal_model.model.schemas.model_input_schemas import ScenarioSchemaSoil

# Models

_DEFAULT_TEST_CABLE = "YMeKrvaslqwd 12/20kV 1x630 Alrm + as50"


@pytest.fixture(scope="function")
def model(single_circuit_env: StaticEnvSoil, scenario_constant: DataFrame[ScenarioSchemaSoil]) -> Model:
    return ModelFactory.create_model(static_env=single_circuit_env, scenario=scenario_constant)


@pytest.fixture(scope="function")
def model_single_config(  # type: ignore
    single_circuit_single_config_env: StaticEnvSoil, scenario_constant: DataFrame[ScenarioSchemaSoil]
) -> Model:
    return ModelFactory.create_model(static_env=single_circuit_single_config_env, scenario=scenario_constant)


@pytest.fixture(scope="function")
def model_multiple_configs(
    single_circuit_multiple_configs_env: StaticEnvSoil, scenario_constant: DataFrame[ScenarioSchemaSoil]
) -> Model:
    return ModelFactory.create_model(static_env=single_circuit_multiple_configs_env, scenario=scenario_constant)


@pytest.fixture(scope="function")
def model_with_pipe(
    single_circuit_with_pipe_env: StaticEnvSoil, scenario_constant: DataFrame[ScenarioSchemaSoil]
) -> Model:
    return ModelFactory.create_model(static_env=single_circuit_with_pipe_env, scenario=scenario_constant)


@pytest.fixture(scope="function")
def model_dynamic_soil(
    single_circuit_env: StaticEnvSoil, scenario_dynamic_soil_prop: DataFrame[ScenarioSchemaSoil]
) -> Model:
    return ModelFactory.create_model(static_env=single_circuit_env, scenario=scenario_dynamic_soil_prop)


@pytest.fixture(scope="function")
def model_with_measurement_points(
    single_circuit_env: StaticEnvSoil, scenario_constant: DataFrame[ScenarioSchemaSoil]
) -> tuple[Model, MeasurementPointKey, MeasurementPointKey]:
    """Create a model with measurement points added to the environment."""
    # Add measurement points to the environment
    key1 = single_circuit_env.add_measurement_point(x=0.1, y=-1.0)
    key2 = single_circuit_env.add_measurement_point(x=0.3, y=-1.0)

    return (
        ModelFactory.create_model(static_env=single_circuit_env, scenario=scenario_constant),
        key1,
        key2,
    )


# Environments


@pytest.fixture(scope="function")
def env() -> StaticEnvSoil:
    return StaticEnvSoil()


@pytest.fixture(scope="function")
def env_air() -> StaticEnvAir:
    return StaticEnvAir()


@pytest.fixture(scope="function")
def single_circuit_env() -> StaticEnvSoil:
    env = StaticEnvSoil()
    env.add_circuit_from_cable_id(
        CircuitInSoilFromCableIdInputSchema(
            x=0,
            y=-0.8,
            circuit_name="c1",
            cable_id=_DEFAULT_TEST_CABLE,
            circuit_type=CircuitType.Trefoil,
        )
    )
    return env


@pytest.fixture(scope="function")
def single_circuit_in_air_env() -> StaticEnvAir:
    env = StaticEnvAir()
    env.add_circuit_from_cable_id(
        CircuitInAirFromCableIdInputSchema(
            circuit_name="c1",
            cable_id=_DEFAULT_TEST_CABLE,
            circuit_type=CircuitType.Trefoil,
        )
    )
    return env


@pytest.fixture(scope="function")
def single_circuit_with_pipe_env() -> StaticEnvSoil:
    env = StaticEnvSoil()
    env.add_circuit_from_cable_id(
        CircuitInSoilFromCableIdInputSchema(
            x=0,
            y=-0.8,
            circuit_name="c1",
            cable_id=_DEFAULT_TEST_CABLE,
            circuit_type=CircuitType.Trefoil,
            pipe=PipeInputSchema(fill_type=PipeFillType.Air),
        )
    )
    return env


@pytest.fixture(scope="function")
def single_circuit_single_config_env() -> StaticEnvSoil:
    env = StaticEnvSoil()
    env.add_circuit_from_cable_id(
        CircuitInSoilFromCableIdInputSchema(
            x=0,
            y=-0.8,
            circuit_name="c1",
            cable_id=_DEFAULT_TEST_CABLE,
            circuit_type=CircuitType.Trefoil,
            bonding_type=BondingType.TwoSided,
            multiple_configurations=[
                CircuitConfigurationFromCableId(
                    circuit_type=CircuitType.Trefoil,
                    length=1000,
                    cable_id=_DEFAULT_TEST_CABLE,
                ),
            ],
        )
    )
    return env


@pytest.fixture(scope="function")
def single_circuit_multiple_configs_env() -> StaticEnvSoil:
    env = StaticEnvSoil()
    env.add_circuit_from_cable_id(
        CircuitInSoilFromCableIdInputSchema(
            x=0,
            y=-0.8,
            circuit_name="c1",
            cable_id=_DEFAULT_TEST_CABLE,
            circuit_type=CircuitType.Trefoil,
            bonding_type=BondingType.TwoSided,
            multiple_configurations=[
                CircuitConfigurationFromCableId(
                    circuit_type=CircuitType.Trefoil,
                    length=1000,
                    cable_id=_DEFAULT_TEST_CABLE,
                ),
                CircuitConfigurationFromCableId(
                    circuit_type=CircuitType.Trefoil,
                    pipe=PipeInputSchema(fill_type=PipeFillType.Air),
                    length=10000,
                    cable_id=_DEFAULT_TEST_CABLE,
                ),
            ],
        )
    )
    return env


@pytest.fixture(scope="function")
def two_circuit_with_pipe_env(env: StaticEnvSoil) -> StaticEnvSoil:
    # initialize environment with two circuits
    for i in range(2):
        env.add_circuit_from_cable_id(
            CircuitInSoilFromCableIdInputSchema(
                x=i,
                y=-0.8,
                circuit_name=f"c{i}",
                cable_id=_DEFAULT_TEST_CABLE,
                circuit_type=CircuitType.Trefoil,
                pipe=PipeInputSchema(fill_type=PipeFillType.Air),
            )
        )
    return env


@pytest.fixture(scope="function")
def two_circuit_env(env: StaticEnvSoil) -> StaticEnvSoil:
    # initialize environment with two circuits
    for i in range(2):
        env.add_circuit_from_cable_id(
            CircuitInSoilFromCableIdInputSchema(
                x=i,
                y=-0.8,
                circuit_name=f"c{i}",
                cable_id=_DEFAULT_TEST_CABLE,
                circuit_type=CircuitType.Trefoil,
            )
        )
    return env


# Cables

test_cable_fixtures = [
    "single_core_cable_xlpe",
    "single_core_cable_od",
    "single_core_cable_pilc",
    "three_core_cable_xlpe",
    "three_core_cable_od",
    "three_core_cable_pilc",
]


@pytest.fixture()
def single_core_cable_xlpe() -> CableSoil:
    name = _DEFAULT_TEST_CABLE
    return CableBuilder.build_cable_from_cable_id(cable_id=name, cable_class=CableSoil)


@pytest.fixture()
def three_core_cable_xlpe() -> CableSoil:
    name = "YMeKrvaslqwd 12/20kV 3x240 Alrm + as50"
    return CableBuilder.build_cable_from_cable_id(cable_id=name, cable_class=CableSoil)


@pytest.fixture()
def single_core_cable_pilc() -> CableSoil:
    name = "VPLK 8/10 kV 1x240 Cu"
    return CableBuilder.build_cable_from_cable_id(cable_id=name, cable_class=CableSoil)


@pytest.fixture()
def three_core_cable_pilc() -> CableSoil:
    name = "GPLK 10/10 kV 3x185 Al"
    return CableBuilder.build_cable_from_cable_id(cable_id=name, cable_class=CableSoil)


@pytest.fixture()
def single_core_cable_od() -> CableSoil:
    name = "OD 50kV 1x400Cu"
    return CableBuilder.build_cable_from_cable_id(cable_id=name, cable_class=CableSoil)


@pytest.fixture()
def three_core_cable_od() -> CableSoil:
    name = "OD 50kV 3x120Cu"
    return CableBuilder.build_cable_from_cable_id(cable_id=name, cable_class=CableSoil)


@pytest.fixture()
def three_core_cable_pilc_in_air() -> CableAir:
    name = "GPLK 10/10 kV 3x185 Al"
    return CableBuilder.build_cable_from_cable_id(cable_id=name, cable_class=CableAir)


@pytest.fixture()
def single_core_cable_xlpe_in_air() -> CableAir:
    return CableBuilder.build_cable_from_cable_id(cable_id=_DEFAULT_TEST_CABLE, cable_class=CableAir)


@pytest.fixture()
def simple_cable_constructional_input() -> CableConstructionalInputSchema:
    return CableConstructionalInputSchema(
        number_of_conductors=CableConductorCount.One,
        conductor_input=ConductorInputSchema(
            material=CableConductorMaterial.Copper,
            inner_radius=0.0,
            outer_radius=0.01395,
            conducting_surface_area=6.30e-4,
            shape=CableConductorShape.Round,
            surface_type=CableConductorSurfaceType.Stranded,
        ),
        insulation_input=InsulationInputSchema(
            material=CableInsulationMaterial.XLPEUnfilled,
            thickness=0.010,
            nominal_phase_voltage=12000.0,
        ),
        sheath_input=SheathInputSchema(
            material=CableSheathMaterial.PE,
            thickness=0.005,
        ),
    )


@pytest.fixture()
def simple_screened_cable_constructional_information() -> CableConstructionalInputSchema:
    return CableConstructionalInputSchema(
        number_of_conductors=CableConductorCount.One,
        conductor_input=ConductorInputSchema(
            material=CableConductorMaterial.Copper,
            inner_radius=0.0,
            outer_radius=0.01395,
            conducting_surface_area=6.30e-4,
            shape=CableConductorShape.Round,
            surface_type=CableConductorSurfaceType.Stranded,
        ),
        insulation_input=InsulationInputSchema(
            material=CableInsulationMaterial.XLPEUnfilled,
            thickness=0.010,
            nominal_phase_voltage=12000.0,
        ),
        screen_input=ScreenInputSchema(
            material=CableScreenMaterial.Copper,
            thickness=0.005,
            conducting_surface_area=35e-6,
            screen_type=CableScreenType.Separate,
        ),
        sheath_input=SheathInputSchema(
            material=CableSheathMaterial.PE,
            thickness=0.005,
        ),
    )


@pytest.fixture()
def simple_cable(simple_cable_constructional_input: CableConstructionalInputSchema) -> CableSoil:
    return CableBuilder.build_cable(simple_cable_constructional_input, cable_class=CableSoil)


# Circuits


@pytest.fixture()
def trefoil_in_single_pipe() -> CableTrefoilCircuitSinglePipeInSoil:
    return CableBuilder.build_cable_from_cable_id(
        cable_id=_DEFAULT_TEST_CABLE,
        cable_class=CableTrefoilCircuitSinglePipeInSoil,
        pipe=PipeInputSchema(fill_type=PipeFillType.Air, trefoil_circuit_in_single_pipe=True),
    )


# Instances
@pytest.fixture(scope="module")
def circuit_builder():
    cb = CircuitBuilder()
    return cb


# Data
@pytest.fixture(scope="module")
def max_absolute_temperature_error():
    """Return the maximum absolute temperature error, which is used in tests that calculate cable temperatures.

    The B5901 states that temperatures should be accurate within 2.0 degrees Celsius with and without pipes.

    Remark: all temperature tests in this model are evaluated by comparing the model results to results from Vision
    Cable Analysis. When this model is put into practice, then probably more errors will occur. These errors emerge from
    uncertainty in the input data, such as thermal resistivity of soil or the current of the scenario. When this actual
    practical validation data becomes available, the maximum absolute temperature error and test method in general
    should be reviewed once more.
    """
    return 2.0


@pytest.fixture(scope="module")
def index() -> pd.TimedeltaIndex:
    return pd.timedelta_range(start="0 days", end="7 days", freq="1h")


@pytest.fixture(scope="module")
def placeholder_temperatures(index: pd.TimedeltaIndex) -> pd.DataFrame:
    return pd.DataFrame(
        index=index, columns=pd.MultiIndex.from_tuples([("cable", "Conductor")]), data=np.random.rand(len(index), 1)
    )


@pytest.fixture(scope="function")
def load_series_constant() -> pd.Series:
    trend = np.linspace(-25, 25, 49)
    return pd.Series(
        trend + 100,
        index=pd.date_range("2020-01-01", "2020-01-03", freq="1h"),
    )


@pytest.fixture(scope="function")
def load_series_constant_load() -> pd.Series:
    trend = 120 * np.ones(49)
    return pd.Series(
        trend,
        index=pd.date_range("2020-01-01", "2020-01-03", freq="1h"),
    )


@pytest.fixture(scope="function")
def dynamic_soil_capacity_series() -> pd.Series:
    trend = 2e6 + 1e6 * np.sin(np.linspace(0, 4 * np.pi, 49))
    return pd.Series(
        trend,
        index=pd.date_range("2020-01-01", "2020-01-03", freq="1h"),
    )


@pytest.fixture(scope="function")
def dynamic_soil_resistivitiy_series() -> pd.Series:
    trend = 0.5 + 0.25 * np.sin(np.linspace(0, 4 * np.pi, 49))
    return pd.Series(
        trend,
        index=pd.date_range("2020-01-01", "2020-01-03", freq="1h"),
    )


@pytest.fixture(scope="function")
def frequency() -> str:
    return "1h"


@pytest.fixture(scope="function")
def load_series_dynamic(frequency) -> pd.Series:
    trend = np.linspace(-25, 25, 49)
    return pd.Series(
        trend + 100 + 50 * np.sin(np.linspace(0, 4 * np.pi, 49)),
        index=pd.date_range("2020-01-01", "2020-01-03", freq=frequency),
    )


@pytest.fixture(scope="function")
def scenario_dynamic(load_series_dynamic, frequency) -> pd.DataFrame:
    scenario_dynamic = pd.DataFrame(
        data={"load_c1": load_series_dynamic, "ambient_temperature": 10},
        index=load_series_dynamic.index,
    )
    scenario_dynamic.index = pd.date_range(
        "2021-01-01",
        periods=len(scenario_dynamic.index),
        freq=frequency,
    )
    return scenario_dynamic


@pytest.fixture(scope="function")
def scenario_dynamic_soil_prop(
    load_series_constant, dynamic_soil_resistivitiy_series, dynamic_soil_capacity_series
) -> DataFrame[ScenarioSchemaSoil]:
    scenario_dynamic = pd.DataFrame(
        data={
            "load_c1": load_series_constant,
            "ambient_temperature": 10,
            "soil_thermal_capacity": dynamic_soil_capacity_series,
            "soil_thermal_resistivity": dynamic_soil_resistivitiy_series,
        },
        index=load_series_constant.index,
    )
    return ScenarioSchemaSoil.validate(scenario_dynamic)


@pytest.fixture(scope="function")
def scenario_constant(load_series_constant) -> DataFrame[ScenarioSchemaSoil]:
    return ScenarioSchemaSoil.validate(
        pd.DataFrame(
            data={
                "load_c1": load_series_constant,
                "ambient_temperature": 10,
                "soil_thermal_resistivity": 0.75,
                "soil_thermal_capacity": 2e6,
            },
            index=load_series_constant.index,
        )
    )


@pytest.fixture(scope="function")
def scenario_constant_multi(load_series_constant) -> DataFrame[ScenarioSchemaSoil]:
    return ScenarioSchemaSoil.validate(
        pd.DataFrame(
            data={
                "load_c0": load_series_constant,
                "load_c1": load_series_constant,
                "ambient_temperature": 10,
                "soil_thermal_resistivity": 0.75,
                "soil_thermal_capacity": 2e6,
            },
            index=load_series_constant.index,
        )
    )


@pytest.fixture(scope="function")
def scenario_steady_state() -> DataFrame[ScenarioSchemaSoil]:
    return ScenarioSchemaSoil.validate(
        pd.DataFrame(
            data={
                "load_c1": 0,
                "ambient_temperature": 10,
                "soil_thermal_resistivity": 0.75,
                "soil_thermal_capacity": 2e6,
            },
            index=pd.timedelta_range(start="0D", end="30000D", periods=5),
        )
    )


@pytest.fixture(scope="function")
def b5901_scenario_steady_state(scenario_steady_state: DataFrame[ScenarioSchemaSoil]) -> DataFrame[ScenarioSchemaSoil]:
    scenario_steady_state["ambient_temperature"] = 15
    return scenario_steady_state


@pytest.fixture(scope="function")
def scenario_non_uniform() -> pd.DataFrame:
    non_uniform_index = pd.timedelta_range(start="0 min", end="15 min", freq="5 min").append(
        pd.timedelta_range(start="20 min", end="40 min", freq="10 min")
    )
    return pd.DataFrame(
        data={
            "load_c1": 300,
            "ambient_temperature": 10,
            "soil_thermal_resistivity": 0.75,
            "soil_thermal_capacity": 2e6,
        },
        index=non_uniform_index,
    )


# Test data (e.g. VCA computations)


def vca_pipe_results():
    # Read the CSV file
    vca_pipe_result_df = pd.read_csv(
        Path(__file__).parent.parent.resolve() / "test_results/test_pipe_model/vca_pipe_results.csv"
    ).to_dict(orient="records")

    return [
        (
            test_case["cable_type"],
            float(test_case["pipe_outer_radius"]) * 1e-3,  # Convert to meters
            float(test_case["sdr"]),
            PipeFillType(test_case["pipe_fill_type"]),
            float(test_case["current"]),
            float(test_case["conductor_temperature"]),
            float(test_case["pipe_temperature"]),
        )
        for test_case in vca_pipe_result_df
    ]


@pytest.fixture(scope="module")
def elst_five_static_env() -> StaticEnvSoil:
    static_env = StaticEnvSoil()
    static_env._from_file("elst_five.csv")
    return static_env


@pytest.fixture(scope="module")
def TB880_case_10_fd_cable() -> CableSoil:
    return CableBuilder.build_cable_from_cable_id(
        cable_id="PILC 8/10 kV 3x 95 Al",
        cable_class=CableSoil,
        cable_source_file_path=Path("data/cable_specs_TB880.csv"),
    )


@pytest.fixture(scope="module")
def TB880_case_10_model(TB880_case_10_fd_cable: CableSoil) -> ModelSoil:
    I_rating = 165.7415608133

    static_env = StaticEnvSoil()
    static_env.add_circuit_from_cable(
        CircuitInSoilFromCableInputSchema(
            x=0,
            y=-1.0,
            circuit_name="TB880_case_10",
            cable=TB880_case_10_fd_cable,
            circuit_type=CircuitType.Single,
            bonding_type=BondingType.TwoSided,
        )
    )

    scenario = pd.DataFrame(
        data={
            "load_TB880_case_10": I_rating,
            "ambient_temperature": 15,
            "soil_thermal_resistivity": 1.0,
            "soil_thermal_capacity": 2e6,
        },
        index=pd.timedelta_range(start="0D", end="30000D", periods=100),
    )
    return ModelSoil(static_env, ScenarioSchemaSoil.validate(scenario))


@pytest.fixture(scope="module")
def TB880_case_10_steady_state_full_solution(TB880_case_10_model: ModelSoil) -> np.ndarray:
    return TB880_case_10_model.run().state.temperature[
        CableKey(circuit_name="TB880_case_10", cable_position=CablePosition.Single)
    ]


# Helper functions
def mock_load_cable_data_from_file(cable_source_file_path: Path, cable_id: str) -> pd.Series | pd.DataFrame:
    _ = cable_source_file_path  # not used in this test function, but required for the signature
    # Combine test and example cables, then return the row for cable_id
    example_cables_df = pd.read_csv(Path("data/example_cables.csv"))
    test_cables_df = pd.read_csv(Path("tests/data/test_cables.csv"))
    combined_df = pd.concat([test_cables_df, example_cables_df], ignore_index=True).set_index("Name")
    return combined_df.loc[cable_id]
