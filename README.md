<!--
SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project

SPDX-License-Identifier: MPL-2.0
-->

# Cable thermal model

Cable Thermal Model is a physical model which can be used to calculate cable
temperature profiles in situations with dynamic profiles for loads, ambient temperature,
soil properties. The model uses an implicit Euler finite difference approach to discretize the heat equation.
Moreover, it relies on the [NEN-IEC](https://www.nen.nl/) norms for cable ratings.
Cable Thermal Model supports different bonding types,
circuit types and environments in soil as well as in air.

# Getting started

## System Requirements

- **Python Version**: Python 3.11 or higher
- **Operating System**: Windows, Linux, or macOS
- **Package Manager**: Poetry

## Installation

### For Users

#### Installation via Poetry (Recommended)

For better dependency management, use Poetry:

```bash
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

After installation, verify that the package is correctly installed by importing some of the main classes:

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
## Privates and Publics
The package exports both private and public interfaces. Public interfaces are intended for use by users of the package, while private interfaces are meant for internal use within the package. Calling/modifying private interfaces may lead to unexpected behavior and is not recommended. Private interfaces are denoted by a leading underscore in their name (e.g., `_private_function`). Contributors are welcome to modify private interfaces if they are contributing to the internal development of the package.
