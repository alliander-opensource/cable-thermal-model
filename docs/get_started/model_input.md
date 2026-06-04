<!--
SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project

SPDX-License-Identifier: MPL-2.0
-->

# Model Input

This page describes the input parameters required to run Cable Thermal Model simulations. Understanding these parameters is essential for setting up accurate thermal calculations.

## Overview

The Cable Thermal Model requires two main types of input:

1. **Static Environment**: Time-invariant information about cables, circuits, and geometry
2. **Scenario**: Time-dependent parameters such as loads and ambient conditions

## Static Environment

The static environment defines the physical configuration of your cable system. Choose between `StaticEnvSoil` for underground cables or `StaticEnvAir` for cables in air.

### Circuits

Circuits are the fundamental building blocks of the static environment. Each circuit represents one or more cables with a specific configuration.

#### Circuit Input Parameters

##### Common Parameters (All Circuits)

| Parameter | Type | Unit | Description | Default |
|-----------|------|------|-------------|---------|
| `circuit_name` | `str` | - | Unique identifier for the circuit | Required |
| `circuit_type` | `CircuitType` | - | Configuration of cables (Single, Linear, Trefoil, etc.) | `CircuitType.Single` |
| `bonding_type` | `BondingType` | - | Bonding configuration (NoBonding, OneSided, TwoSided) | `BondingType.NoBonding` |
| `dist` | `float` | m | Distance between cables in the circuit | Circuit-type dependent |
| `pipe` | `PipeInputSchema` | - | Optional pipe containing the circuit | `None` |

##### Circuit Type Options

The `CircuitType` enum defines the cable arrangement:

- **`Single`**: Single cable configuration
- **`Linear`**: Multiple cables in a horizontal line
- **`Trefoil`**: Three cables in triangular formation (most compact)
- **`TrefoilLeft`**: Left cable in trefoil configuration
- **`TrefoilRight`**: Right cable in trefoil configuration
- **`TrefoilTop`**: Top cable in trefoil configuration

##### Bonding Type Options

The `BondingType` enum specifies sheath bonding:

- **`NoBonding`**: No electrical bonding (highest induced currents)
- **`OneSided`**: Single-point bonding
- **`TwoSided`**: Both-ends bonding (lowest induced currents)

#### Cables in Soil - Additional Parameters

| Parameter | Type | Unit | Description | Default |
|-----------|------|------|-------------|---------|
| `x` | `float` | m | Horizontal position of circuit center | Required |
| `y` | `float` | m | Vertical position (depth) of circuit center | Required |
| `y_ref` | `CircuitYReference` | - | Reference point for y-position | `CircuitYReference.Center` |

##### Circuit Y-Reference Options

- **`Center`**: Y-position refers to the circuit center
- **`Top`**: Y-position refers to the top of the circuit
- **`Bottom`**: Y-position refers to the bottom of the circuit

#### Cables in Air - Additional Parameters

| Parameter | Type | Unit | Description | Default |
|-----------|------|------|-------------|---------|
| `clipped_to_wall` | `bool` | - | Whether circuit is mounted on a wall | `False` |

### Cable Specification

Cables can be specified in three ways:

#### 1. From Cable Database (Cable ID)

Reference a cable from a CSV file containing cable specifications:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `cable_id` | `str` | Identifier matching an entry in the cable database | Required |
| `cable_source_file_path` | `Path` | Path to CSV file with cable specifications | `data/example_cables.csv` |

**Example:**

```python
from cable_thermal_model.cable.schemas.circuit_schemas import CircuitInSoilFromCableIdInputSchema

circuit = CircuitInSoilFromCableIdInputSchema(
    circuit_name="circuit_1",
    cable_id="GPLK 10/10 kV 3x185 Al",
    cable_source_file_path="data/example_cables.csv",
    x=0.0,
    y=-1.0
)
```

#### 2. From Constructional Details

Provide detailed cable construction parameters directly:

See the [build cable example](../examples/build_cable_example.ipynb) for a complete demonstration of constructional input.

#### 3. From Cable Object

Pass a pre-built `FDCable` object:

```python
from cable_thermal_model.cable.schemas.circuit_schemas import CircuitInSoilFromCableInputSchema

circuit = CircuitInSoilFromCableInputSchema(
    circuit_name="circuit_1",
    cable=my_cable_object,
    x=0.0,
    y=-1.0
)
```

### Pipe Configuration

When cables are installed in pipes, use `PipeInputSchema` to specify pipe properties:

| Parameter | Type | Unit | Description | Default |
|-----------|------|------|-------------|---------|
| `outer_radius` | `float` | m | Outer radius of the pipe | Required |
| `inner_radius` | `float` | m | Inner radius of the pipe | Calculated from SDR |
| `sdr` | `float` | - | Standard Dimension Ratio (outer diameter / wall thickness) | 11.0 |
| `fill_type` | `PipeFillType` | - | Material filling the pipe | Required |
| `trefoil_circuit_in_single_pipe` | `bool` | - | Whether three cables form a trefoil inside one pipe | `False` |

##### Pipe Fill Type Options

- **`Air`**: Air-filled pipe
- **`Water`**: Water-filled pipe
- **`Sand`**: Sand-filled pipe
- **`Bentonite`**: Bentonite-filled pipe
- **`ControlledLowStrengthMaterial`**: CLSM-filled pipe

**Example:**

```python
from cable_thermal_model import PipeInputSchema, PipeFillType

pipe = PipeInputSchema(
    outer_radius=0.08,  # 160 mm outer diameter
    fill_type=PipeFillType.Water,
    sdr=11.0
)
```

## Scenario (Time-Dependent Input)

Scenarios are defined using pandas DataFrames with a `DatetimeIndex`. Each row represents a timestep in the simulation.

### Required Columns for Soil Environments

| Column Name | Type | Unit | Description |
|-------------|------|------|-------------|
| `load_{circuit_name}` | `float` | A | Current load for each circuit (one column per circuit) |
| `ambient_temperature` | `float` | °C | Temperature of the surrounding soil |
| `soil_thermal_resistivity` | `float` | K·m/W | Thermal resistivity of the soil |
| `soil_thermal_capacity` | `float` | J/(m³·K) | Volumetric heat capacity of the soil |

**Example:**

```python
import pandas as pd
from datetime import datetime

scenario = pd.DataFrame({
    'load_circuit_1': [200, 250, 300],
    'load_circuit_2': [150, 175, 200],
    'ambient_temperature': [15, 16, 17],
    'soil_thermal_resistivity': [0.75, 0.75, 0.75],
    'soil_thermal_capacity': [2e6, 2e6, 2e6]
}, index=pd.date_range(start=datetime(2026, 1, 1), periods=3, freq='1h'))
```

### Required Columns for Air Environments

| Column Name | Type | Unit | Description |
|-------------|------|------|-------------|
| `load_{circuit_name}` | `float` | A | Current load for each circuit (one column per circuit) |
| `ambient_temperature` | `float` | °C | Temperature of the surrounding air |

**Example:**

```python
scenario = pd.DataFrame({
    'load_circuit_1': [200, 250, 300],
    'ambient_temperature': [20, 22, 24]
}, index=pd.date_range(start=datetime(2026, 1, 1), periods=3, freq='1h'))
```

### Scenario Validation

You can validate your scenario before running the model:

```python
from cable_thermal_model.model.schemas import ScenarioSchemaSoil, ScenarioSchemaAir

# For soil environments
validated_scenario = ScenarioSchemaSoil(scenario)

# For air environments
validated_scenario = ScenarioSchemaAir(scenario)
```

## Complete Example

Here's a minimal complete example for a cable in soil:

```python
from datetime import datetime
import pandas as pd
from cable_thermal_model import (
    StaticEnvSoil,
    ModelFactory,
    CircuitType,
    BondingType
)
from cable_thermal_model.cable.schemas.circuit_schemas import CircuitInSoilFromCableIdInputSchema

# 1. Create static environment
static_env = StaticEnvSoil()

# 2. Add a circuit
circuit = CircuitInSoilFromCableIdInputSchema(
    circuit_name="main_circuit",
    cable_id="GPLK 10/10 kV 3x185 Al",
    cable_source_file_path="data/example_cables.csv",
    x=0.0,
    y=-1.0,
    circuit_type=CircuitType.Trefoil,
    bonding_type=BondingType.TwoSided
)
static_env.add_circuit_from_cable_id(circuit)

# 3. Create scenario
scenario = pd.DataFrame({
    'load_main_circuit': [200, 250, 300],
    'ambient_temperature': [15, 16, 17],
    'soil_thermal_resistivity': [0.75, 0.75, 0.75],
    'soil_thermal_capacity': [2e6, 2e6, 2e6]
}, index=pd.date_range(start=datetime(2026, 1, 1), periods=3, freq='1h'))

# 4. Run the model
model = ModelFactory.create_model(static_env=static_env, scenario=scenario)
solution = model.run()
temperature_result = solution.result
```

## Typical Parameter Values

### Soil Properties

| Property | Typical Range | Unit | Notes |
|----------|---------------|------|-------|
| Thermal resistivity | 0.5 - 2.5 | K·m/W | Depends on moisture content and soil type |
| Thermal capacity | 1.5e6 - 3.0e6 | J/(m³·K) | Higher for moist soils |
| Ambient temperature | 5 - 20 | °C | Varies with depth and season |

### Cable Loading

| Parameter | Typical Range | Unit | Notes |
|-----------|---------------|------|-------|
| Current load | 0 - 1000 | A | Depends on cable rating |
| Load factor | 0 - 1 | - | Ratio of actual to rated load |

### Pipe Dimensions

| Parameter | Typical Range | Unit | Notes |
|-----------|---------------|------|-------|
| Outer radius | 0.05 - 0.20 | m | Common SDR values: 11, 17, 21 |
| SDR | 7.3 - 41 | - | Lower SDR = thicker walls |

## Additional Resources

- **[Example Calculation](../examples/example_calculation.ipynb)**: See all parameters in action
- **[Build Cable Example](../examples/build_cable_example.ipynb)**: Learn to define custom cables
- **[External Heat Sources](../examples/external_heat_sources_example.ipynb)**: Advanced modeling techniques

## Input Validation

The Cable Thermal Model uses Pydantic and Pandera for input validation. If you provide invalid inputs, you'll receive informative error messages indicating:

- Which parameter is invalid
- What the valid range or type should be
- Suggestions for correction

Always check validation errors carefully—they help ensure accurate simulations!

## Next Steps

With your understanding of model inputs, you're ready to:

1. Review the [Example Calculation](../examples/example_calculation.ipynb)
2. Set up your own cable configuration
3. Run your first thermal simulation

For detailed information on model outputs and interpretation, refer to the API documentation.
