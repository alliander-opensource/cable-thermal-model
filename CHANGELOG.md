<!--
SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project

SPDX-License-Identifier: MPL-2.0
-->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.0](https://github.com/alliander-opensource/cable-thermal-model/compare/v2.1.1...v2.2.0) (2026-09-07)


### Features

* add initial CodeQL workflow ([#78](https://github.com/alliander-opensource/cable-thermal-model/issues/78)) ([90bd9af](https://github.com/alliander-opensource/cable-thermal-model/commit/90bd9afa513515eb3874cead2dec4379771f911e))
* make states serializable ([#88](https://github.com/alliander-opensource/cable-thermal-model/issues/88)) ([a80032e](https://github.com/alliander-opensource/cable-thermal-model/commit/a80032e51027cc12457983fb73004a3f1d4081d2))


### Bug Fixes

* file format of uv lock ([#101](https://github.com/alliander-opensource/cable-thermal-model/issues/101)) ([fd8600f](https://github.com/alliander-opensource/cable-thermal-model/commit/fd8600f2b7cf8cb65b4398073fc3f9b4f2b13490))
* fix the release please config ([#98](https://github.com/alliander-opensource/cable-thermal-model/issues/98)) ([37eefef](https://github.com/alliander-opensource/cable-thermal-model/commit/37eefefd46e1a011433a5c3be7291e9076959ef5))
* release please ([#102](https://github.com/alliander-opensource/cable-thermal-model/issues/102)) ([4accc5c](https://github.com/alliander-opensource/cable-thermal-model/commit/4accc5cea97ca987c01f7e1e89e85733be8d3bad))
* treat cables with PE insulation material as XLPE cables for screen loss function ([#86](https://github.com/alliander-opensource/cable-thermal-model/issues/86)) ([bf52607](https://github.com/alliander-opensource/cable-thermal-model/commit/bf526079f0d7e6e6d1b74582326e5df8c1b6f53a))


### Dependencies

* bump ruff from 0.15.22 to 0.16.5 ([#90](https://github.com/alliander-opensource/cable-thermal-model/issues/90)) ([c006533](https://github.com/alliander-opensource/cable-thermal-model/commit/c0065339c09ab9031ec91bb14529b7579937420c))
* bump structlog from 25.5.0 to 26.1.0 ([#89](https://github.com/alliander-opensource/cable-thermal-model/issues/89)) ([370a972](https://github.com/alliander-opensource/cable-thermal-model/commit/370a9721cad14df0fba51dd8c07acd21b7256f77))
* update uv-build requirement ([#93](https://github.com/alliander-opensource/cable-thermal-model/issues/93)) ([f389d07](https://github.com/alliander-opensource/cable-thermal-model/commit/f389d07f0fae313289907c403c8ff765d36f6350))


### Miscellaneous Chores

* add uv lock to release please PR ([#100](https://github.com/alliander-opensource/cable-thermal-model/issues/100)) ([e091729](https://github.com/alliander-opensource/cable-thermal-model/commit/e091729ced25500725e981997ba0695ced58f9d4))
* bump version ([#77](https://github.com/alliander-opensource/cable-thermal-model/issues/77)) ([5228e4e](https://github.com/alliander-opensource/cable-thermal-model/commit/5228e4e36fb57b4892173f9d6bc65e15138626f0))
* release please and trigger publishing workflow on release ([#87](https://github.com/alliander-opensource/cable-thermal-model/issues/87)) ([60c8d44](https://github.com/alliander-opensource/cable-thermal-model/commit/60c8d44fe3ca478181b2044c85ee2d0727d95146))
* update dependencies ([#84](https://github.com/alliander-opensource/cable-thermal-model/issues/84)) ([0ab52e1](https://github.com/alliander-opensource/cable-thermal-model/commit/0ab52e13ae3cf910dbfe615bccdf46d795c03d66))
* update to the new managed runners ([#80](https://github.com/alliander-opensource/cable-thermal-model/issues/80)) ([a579c16](https://github.com/alliander-opensource/cable-thermal-model/commit/a579c16801e5a528e046b0a4508ff5e4a35d3d63))

## [1.15.2] - 2026-06-23

### Added

- First release of the cable-thermal-model package.
- Dynamic cable temperature model (DKM) for computing cable temperatures with iterative heat equation approximation.
- Support for common cable types, multiple cable circuits, different soil layers, and pipes.
- Pydantic-based schemas for cable specifications and environmental inputs.
- Comprehensive documentation and examples.
- SPDX license headers and MPL-2.0 licensing.

### Features

- `CircuitType`, `BondingType`, `CircuitYReference`, `CableLayer`, `PipeFillType`, `CablePosition` enums
- `PipeInputSchema`, `StaticEnvSoil`, `StaticEnvAir` configuration classes
- `ModelFactory` for creating cable temperature models
- `CableKey` for cable identification
- `StateSoil`, `StateAir` state tracking classes
- Full test coverage with pytest
- Pre-commit hooks and linting with ruff and mypy
- Poetry-based dependency management

[1.15.2]: https://github.com/alliander-opensource/cable-thermal-model/releases/tag/v1.15.2
