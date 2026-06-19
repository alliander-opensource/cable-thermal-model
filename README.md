<!--
SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project

SPDX-License-Identifier: MPL-2.0
-->

# Dynamic cable temperature model

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Alliander_cable-thermal-model&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=alliander_cable-thermal-model)
[![Code Smells](https://sonarcloud.io/api/project_badges/measure?project=Alliander_cable-thermal-model&metric=code_smells)](https://sonarcloud.io/summary/new_code?id=alliander_cable-thermal-model)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=Alliander_cable-thermal-model&metric=coverage)](https://sonarcloud.io/summary/new_code?id=alliander_cable-thermal-model)
[![Duplicated Lines (%)](https://sonarcloud.io/api/project_badges/measure?project=Alliander_cable-thermal-model&metric=duplicated_lines_density)](https://sonarcloud.io/summary/new_code?id=alliander_cable-thermal-model)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=Alliander_cable-thermal-model&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=alliander_cable-thermal-model)

The DKM (dynamisch kabeltemperatuur model) is a physical model used for computing dynamic cable temperatures.
The model computes cable temperatures by approximating the heat equation iteratively throughout time. This makes it
possible to find cable temperature profiles in situations with dynamic profiles for loads, ambient temperature, soil
properties in differing environments. The DKM supports all common cable types, environments with multiple cable
circuits, different soil layers, pipes and lots more.

## Overview of package interfaces

```python
cable_thermal_model.
    [enum]      CircuitType
    [enum]      BondingType
    [enum]      CircuitYReference
    [enum]      CableLayer
    [enum]      PipeFillType
    [enum]      CablePosition
    [class]     PipeInputSchema
    [class]     StaticEnvSoil
    [class]     StaticEnvAir
    [class]     ModelFactory
    [class]     CableKey
    [class]     StateSoil
    [class]     StateAir
    [str]       __version__
```

## Users outside the Verbindingsteam

This project is still under active development and model outcomes can't be trusted blindly as they are highly dependent
on the correct configuration of cables and input parameters.
To get started, we advise you to visit the [DKM Usage guide](https://alliander.atlassian.net/wiki/spaces/INNO/pages/4028236124).
Background on the model physics can be found on [Confluence](https://alliander.atlassian.net/wiki/spaces/INNO/pages/2885655299).

Please reach out to a member of the Verbindingsteam using our ["DKM developers vragen" Teams channel](https://teams.microsoft.com/l/channel/19%3A990a2597abe6468081d1ad748fa6c888%40thread.tacv2/DCM%20developers%20vragen?groupId=9f096660-1466-4292-8708-e9e7bf4e233f&tenantId=697f104b-d7cb-48c8-ac9f-bd87105bafdc) if you encounter any problems, if you need to use functionality
not yet added to the official releases or to help get you started!

### Disclaimer

Note that model outcomes cannot be used in cable ampacity studies, policy documents and/or operational decisions unless:

- These have been provided by the Verbindingsteam
- Or after discussion about the context and input with members of the Verbindingsteam

In any case it is important to:

- Highlight the uncertainties and limitations of the model.
- The model outcomes or script have been reviewed.

Any feedback is highly appreciated!

## Getting started for USERS

To get started you will need to start by gaining Alliander Artifactory access from your project if this hasn't been
done yet. You can do this by following the instructions on Confluence:

- [Dutch instructions](https://alliander.atlassian.net/wiki/spaces/INNO/pages/3885827852)
- [English instructions](https://alliander.atlassian.net/wiki/spaces/SA/pages/3838312623)

After you have gained access to the Alliander Artifactory you can install the package by running one of the following
commands (depending on whether you are using pip or Poetry):

**Installation via Pip:**

```pip install cable-thermal-model```

**Installation via Poetry:**

```poetry add cable-thermal-model```

**You can check your installation by running the following command from the CLI:**

```from cable_thermal_model import CircuitType, BondingType, StaticEnvSoil, ModelFactory, CableKey```

**Basic usage and example**
Please examine this files in: [here](docs/examples).

## Getting started for DEVELOPERS

To set up your local environment we recommend using poetry.

```shell
# Install dependencies
poetry install --with dev

# Enable pre-commit hooks
pre-commit install
```

This project will be published under the open source license MPL-2.0. The **reuse** pre-commit hook will make sure that the correct headers are present in the files. If this hook fails, please add the appropriate headers/files to the project.

## Inputs and outputs

For information on the inputs and outputs of the model, please refer to the Usage Guide at:

- [DKM Usage guide](https://alliander.atlassian.net/wiki/spaces/INNO/pages/4028236124).

## Privates and Publics
The package exports both private and public interfaces. Public interfaces are intended for use by users of the package, while private interfaces are meant for internal use within the package. Calling/modifying private interfaces may lead to unexpected behavior and is not recommended. Private interfaces are denoted by a leading underscore in their name (e.g., `_private_function`). Contributors are welcome to modify private interfaces if they are contributing to the internal development of the package.
