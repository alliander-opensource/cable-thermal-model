<!--
SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project

SPDX-License-Identifier: MPL-2.0
-->

# Welcome to Cable Thermal Model

**Cable Thermal Model** is a Python library for modeling cable conductor and sheath temperatures based on cable specifications, circuit configuration, load profiles, and ambient conditions. The model uses an implicit Euler finite difference approach to discretize the heat equation, enabling dynamic temperature calculations in various operational scenarios.

## Features

The Cable Thermal Model is designed to model heat generation in cables. In short, it has the following features:

- Create power and distribution cable circuits based on user specifications;
- Model heat generation based on static or dynamic load profiles and ambient temperature profiles;
- Support for cables in soil and air environments with different thermal properties;
- Handle multiple cable circuits with various configurations (single, linear, trefoil);
- Model cables in pipes with different filling materials (air, water, sand, bentonite);
- Account for different bonding types (no bonding, single-sided, two-sided bonding);
- Calculate temperatures for different cable layers (conductor, insulation, sheath, armor);
- Support for soil stratification with multiple layers having different thermal properties.

## Who is using the Cable Thermal Model?

The Cable Thermal Model is designed with three major use cases in mind:

**Modelling conductor temperature**: To maximize utilization of cable infrastructure, we need to understand the thermal limits under various loading conditions. The conductor temperature is the primary indicator of cable thermal stress and determines the safe current-carrying capacity (ampacity). This model provides accurate estimates of conductor temperatures based on load profiles and environmental conditions, enabling operators to confidently utilize cables closer to their thermal limits without risking asset damage.

**Network planning and design validation**: When designing new cable installations or modifying existing networks, engineers need to verify that cable configurations meet thermal requirements. The model allows testing different cable types, burial depths, circuit arrangements, and environmental conditions to optimize designs before installation. This reduces the risk of thermal overloads and ensures compliance with safety standards.

**Asset management and operational decision-making**: Cable thermal behavior is crucial for making informed operational decisions about load distribution, maintenance scheduling, and asset lifetime management. By modeling cable temperatures under actual field conditions, operators can identify thermal hotspots, assess the impact of load changes, and develop strategies to extend asset life. The model supports integration with monitoring systems and SCADA platforms, enabling real-time thermal assessment and proactive asset management.

## Getting Started

For installation instructions and usage examples, please visit our documentation:
- [Installation instructions](get_started/installation_and_overview.md) - Everything you need to get up and running
- [Model input](get_started/model_input.md) - How to prepare the input parameters for the model
- [Example calculations](examples/example_calculation.ipynb) - Complete working examples showing basic usage
- [Building custom cables](examples/build_cable_example.ipynb) - How to define cable specifications
- [External heat sources](examples/external_heat_sources_example.ipynb) - Advanced modeling techniques

## License

This project is licensed under the Mozilla Public License, version 2.0 - see [LICENSE](https://github.com/alliander-opensource/cable-thermal-model/blob/main/LICENSE) for details.

## Licenses third-party libraries

This project includes third-party libraries, which are licensed under their own respective Open-Source licenses. SPDX-License-Identifier headers are used to show which license is applicable.

The concerning license files can be found in the [LICENSES](https://github.com/alliander-opensource/cable-thermal-model/tree/main/LICENSES) directory.

## Contributing

Please read [CONTRIBUTING](https://github.com/alliander-opensource/cable-thermal-model/blob/main/CONTRIBUTING.md) and [PROJECT GOVERNANCE](https://github.com/alliander-opensource/cable-thermal-model/blob/main/PROJECT_GOVERNANCE.md) for details on the process for submitting pull requests to us.

## Contact

For questions, support, or security concerns:

- **General Support**: Contact the team at cable-thermal-model@alliander.com
- **Security Issues**: Please refer to our [SECURITY](https://github.com/alliander-opensource/cable-thermal-model/blob/main/SECURITY.md) policy
- **Bug Reports & Feature Requests**: Use [GitHub Issues](https://github.com/alliander-opensource/cable-thermal-model/issues)
