<!--
SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project

SPDX-License-Identifier: MPL-2.0
-->

# Installation and Overview

This guide will help you install and configure the Cable Thermal Model (CTM) for your environment.

## System Requirements

- **Python Version**: Python 3.11 or higher
- **Operating System**: Windows, Linux, or macOS
- **Package Manager**: Poetry

## Installation

### For Users

#### Installation via pip

Install the package from PyPI:

```bash
pip install cable-thermal-model
```

#### Installation via Poetry (Recommended)

For better dependency management, use Poetry:

**Adding to an Existing Poetry Project:**

```bash
poetry add cable-thermal-model
```

**Creating a New Poetry Project:**

If you're starting from scratch, first create a new Poetry project:

```bash
# Create a new Poetry project
poetry new my-cable-thermal-model
cd my-cable-thermal-model

# Install the project dependencies
poetry install

# Add cable-thermal-model
poetry add cable-thermal-model
```

### For Developers

If you plan to contribute to the Cable Thermal Model, follow these steps to set up your development environment:

#### 1. Clone the Repository

```bash
git clone https://github.com/alliander-opensource/cable-thermal-model
cd cable-thermal-model
```

#### 2. Install Dependencies with Poetry

We recommend using Poetry for development:

```bash
# Install all dependencies including development tools
poetry install --with dev

```

#### 3. Enable Pre-commit Hooks

Pre-commit hooks ensure code quality and proper licensing headers:

```bash
# Install pre-commit hooks
pre-commit install --install-hooks
```

The **reuse** pre-commit hook ensures that all files have proper copyright headers (MPL-2.0 license).

## Verifying Your Installation

After installation, verify that the package is correctly installed by importing the main classes:

```python
from cable_thermal_model import (
    CircuitType,
    BondingType,
    StaticEnvSoil,
    StaticEnvAir,
    ModelFactory,
    CableKey
)

# Check the installed version
from cable_thermal_model  import __version__
print(f"Cable Thermal Model version: {__version__}")
```

If the import succeeds without errors, your installation is complete!

## Package Overview

The Cable Thermal Model package provides a clean, intuitive interface for thermal simulations:

### Main Components

```python
cable_thermal_model.
    [enum]      CircuitType           # Circuit configuration (single, linear, trefoil, etc.)
    [enum]      BondingType           # Bonding configuration (no bonding, single/two-sided)
    [enum]      CircuitYReference     # Y-position reference for circuits
    [enum]      CableLayer            # Cable layer identifiers (conductor, insulation, etc.)
    [enum]      PipeFillType          # Pipe filling material types
    [enum]      CablePosition         # Position of cables in a circuit
    [class]     PipeInputSchema       # Schema for pipe configuration
    [class]     StaticEnvSoil         # Static environment for cables in soil
    [class]     StaticEnvAir          # Static environment for cables in air
    [class]     ModelFactory          # Factory for creating thermal models
    [class]     CableKey              # Identifier for cables in circuits
    [class]     StateSoil             # State object for soil model results
    [class]     StateAir              # State object for air model results
    [str]       __version__           # Package version string
```

### Public vs. Private Interfaces

The package exports both **public** and **private** interfaces:

- **Public interfaces**: Intended for use by package users (listed above)
- **Private interfaces**: Internal use only, denoted by a leading underscore (e.g., `_private_function`)

!!! warning "Private Interface Usage"
    Calling or modifying private interfaces may lead to unexpected behavior and is not recommended for regular users. Contributors are welcome to modify private interfaces when contributing to the package's internal development.

## Project Structure

Understanding the project layout helps when exploring examples and source code:

```
cable-thermal-model/
├── cable_thermal_model/          # Main package source code
│   ├── cable/             # Cable and circuit definitions
│   ├── environment/       # Environment modeling (soil/air)
│   ├── model/             # Core thermal model implementation
│   ├── utils/             # Utility functions
│   └── validation/        # Input validation schemas
├── data/                   # Cable specifications and material properties
│   ├── example_cables.csv # Example cable database
│   ├── material_properties.csv
│   └── circuits/          # Example circuit configurations
├── docs/                   # Documentation source files
│   └── examples/          # Jupyter notebook examples
├── tests/                  # Test suite
└── pyproject.toml         # Project configuration and dependencies
```

## Basic Workflow

A typical Cable Thermal Model workflow consists of three main steps:

### 1. Define the Static Environment

Create a static environment specifying cable types, circuit geometry, and configuration:

```python
from cable_thermal_model import StaticEnvSoil

static_env = StaticEnvSoil()
# Add circuits to the environment (see examples)
```

### 2. Create a Scenario

Define time-dependent parameters like loads and ambient conditions:

```python
import pandas as pd

scenario = pd.DataFrame({
    'load_circuit_1': [200, 250, 300],  # Amperes
    'ambient_temperature': [15, 16, 17],  # Celsius
    'soil_thermal_resistivity': [0.75, 0.75, 0.75],  # mK/W
    # ... other required parameters
})
```

### 3. Run the Model

Create and execute the thermal model:

```python
from cable_thermal_model import ModelFactory

model = ModelFactory.create_model(static_env=static_env, scenario=scenario)
solution = model.run()
temperature_result = solution.result
```

## Data Files

The package includes several data files with cable specifications and material properties:

- **`example_cables.csv`**: Common cable types with constructional details
- **`IEC_conductor_resistance_table.csv`**: IEC standard conductor resistances
- **`material_properties.csv`**: Thermal and electrical properties of materials
- **`pipe_filling_material_properties.csv`**: Properties for pipe filling materials

These files are located in the `data/` directory and can be referenced in your calculations.

## Next Steps

Now that you have installed the Cable Thermal Model, continue to:

- **[Model Input](model_input.md)**: Learn about required parameters and data structures
- **[Example Calculation](../examples/example_calculation.ipynb)**: Follow a complete example
- **[Build Cable Example](../examples/build_cable_example.ipynb)**: Learn to define custom cables

## Troubleshooting

### Import Errors

If you encounter import errors:

1. Ensure Python 3.11 or higher is installed: `python --version`
2. Verify the package is installed: `poetry show`
3. Check that you're using the correct virtual environment

### Pre-commit Hook Failures

If pre-commit hooks fail:

- Ensure all files have proper SPDX license headers
- Run `pre-commit run --all-files` to check all files
- Consult `CONTRIBUTING.md` for contribution guidelines

## Getting Help

If you encounter issues or have questions:

- **Internal Users**: Contact the Verbindingsteam via Teams
- **External Users**: Check the GitHub Issues page

Happy modeling!
