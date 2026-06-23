<!--
SPDX-FileCopyrightText: 2014 Coraline Ada Ehmke

SPDX-License-Identifier: CC-BY-4.0
-->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
