<!--
SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project

SPDX-License-Identifier: MPL-2.0
-->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0](https://github.com/alliander-opensource/cable-thermal-model/compare/v2.2.0...v3.0.0) (2026-09-08)


### ⚠ BREAKING CHANGES

* 

### Features

* add initial CodeQL workflow ([#78](https://github.com/alliander-opensource/cable-thermal-model/issues/78)) ([90bd9af](https://github.com/alliander-opensource/cable-thermal-model/commit/90bd9afa513515eb3874cead2dec4379771f911e))
* add measurement points to soil model ([#49](https://github.com/alliander-opensource/cable-thermal-model/issues/49)) ([7145189](https://github.com/alliander-opensource/cable-thermal-model/commit/71451898a584028e3b23642c0efab7d8e5f1a8b2))
* add method add_circuit_from_cable_specs to StaticEnv ([#72](https://github.com/alliander-opensource/cable-thermal-model/issues/72)) ([21b67c5](https://github.com/alliander-opensource/cable-thermal-model/commit/21b67c5c490cd9a80c5b3d5e5da6d22a870b4801))
* added info to pyproject and created initial CHANGELOG ([#25](https://github.com/alliander-opensource/cable-thermal-model/issues/25)) ([93883a5](https://github.com/alliander-opensource/cable-thermal-model/commit/93883a5c47bf6815780893084b983a8e7de5b693))
* added publishing workflow ([#22](https://github.com/alliander-opensource/cable-thermal-model/issues/22)) ([c060369](https://github.com/alliander-opensource/cable-thermal-model/commit/c0603692b60e670294cf39a4b13271dafdbe95e2))
* **docs:** Patch to fix gh-pages upload ([#69](https://github.com/alliander-opensource/cable-thermal-model/issues/69)) ([95e46a2](https://github.com/alliander-opensource/cable-thermal-model/commit/95e46a2334a72bdfdf4991c13a0b169f695afc8c))
* **docstrings:** Patched some small docstring issues ([#28](https://github.com/alliander-opensource/cable-thermal-model/issues/28)) ([d48a4fb](https://github.com/alliander-opensource/cable-thermal-model/commit/d48a4fb72aabbbd5063e5004eec541650d83aa63))
* initial commit ([763ecd4](https://github.com/alliander-opensource/cable-thermal-model/commit/763ecd42e49ef2162fcc82476bb0bcbb0d380d34))
* make states serializable ([#88](https://github.com/alliander-opensource/cable-thermal-model/issues/88)) ([a80032e](https://github.com/alliander-opensource/cable-thermal-model/commit/a80032e51027cc12457983fb73004a3f1d4081d2))
* Migrated codebase to uv ([#43](https://github.com/alliander-opensource/cable-thermal-model/issues/43)) ([21021bf](https://github.com/alliander-opensource/cable-thermal-model/commit/21021bfb8fc49da3e464864f2f311bb25a94f71a))
* **mkdocs:** QoL improvements ([#42](https://github.com/alliander-opensource/cable-thermal-model/issues/42)) ([7b3c791](https://github.com/alliander-opensource/cable-thermal-model/commit/7b3c79159802a7c3caa1fe0ba8fb359ea8d92a9c))
* re-added gh actions ([#36](https://github.com/alliander-opensource/cable-thermal-model/issues/36)) ([436a781](https://github.com/alliander-opensource/cable-thermal-model/commit/436a781d3d0aa5eae96f3ee880e01f43ee8c2145))
* refactor of fdcable into cable and subclass ([45dba79](https://github.com/alliander-opensource/cable-thermal-model/commit/45dba79980bb2b68c1498cdd39758a35c9b1a994))
* split cable.py into separate modules ([#64](https://github.com/alliander-opensource/cable-thermal-model/issues/64)) ([37af0a2](https://github.com/alliander-opensource/cable-thermal-model/commit/37af0a2bca84325e62fb94f46bebb95038c7f4f3))
* update AbstractModel and related classes to use TypeVar for static_env. ([#39](https://github.com/alliander-opensource/cable-thermal-model/issues/39)) ([4ac26ba](https://github.com/alliander-opensource/cable-thermal-model/commit/4ac26bafa753d5bc6e84f678d3f084c66e6dfb48))


### Bug Fixes

* bumped version to 2.0.0 ([#67](https://github.com/alliander-opensource/cable-thermal-model/issues/67)) ([5d651cd](https://github.com/alliander-opensource/cable-thermal-model/commit/5d651cdc7e36854c1d110dc875ddf668f0750dab))
* change hash to be consistent over different sessions ([#75](https://github.com/alliander-opensource/cable-thermal-model/issues/75)) ([c00fd32](https://github.com/alliander-opensource/cable-thermal-model/commit/c00fd323eb531a01fb2cf668b27e0b6ec4216e9c))
* file format of uv lock ([#101](https://github.com/alliander-opensource/cable-thermal-model/issues/101)) ([fd8600f](https://github.com/alliander-opensource/cable-thermal-model/commit/fd8600f2b7cf8cb65b4398073fc3f9b4f2b13490))
* fix the release please config ([#98](https://github.com/alliander-opensource/cable-thermal-model/issues/98)) ([37eefef](https://github.com/alliander-opensource/cable-thermal-model/commit/37eefefd46e1a011433a5c3be7291e9076959ef5))
* improve finite difference method logic for cases where r_i is no… ([#71](https://github.com/alliander-opensource/cable-thermal-model/issues/71)) ([71767fb](https://github.com/alliander-opensource/cable-thermal-model/commit/71767fbd9421553a2717c23472a006f230fb2281))
* release please ([#102](https://github.com/alliander-opensource/cable-thermal-model/issues/102)) ([4accc5c](https://github.com/alliander-opensource/cable-thermal-model/commit/4accc5cea97ca987c01f7e1e89e85733be8d3bad))
* separate scenario from Model object and improve update of soil properties ([#68](https://github.com/alliander-opensource/cable-thermal-model/issues/68)) ([e12b8be](https://github.com/alliander-opensource/cable-thermal-model/commit/e12b8beb54316cfc296a31514067a2001c6c54b6))
* sonar token ([#45](https://github.com/alliander-opensource/cable-thermal-model/issues/45)) ([b1780eb](https://github.com/alliander-opensource/cable-thermal-model/commit/b1780ebd4728aa9fa71eec199ce0774c725adf1c))
* the trigger for publish workflow ([#103](https://github.com/alliander-opensource/cable-thermal-model/issues/103)) ([9768930](https://github.com/alliander-opensource/cable-thermal-model/commit/9768930ae3671fc7783ea259480999dc9af8ee7d))
* treat cables with PE insulation material as XLPE cables for screen loss function ([#86](https://github.com/alliander-opensource/cable-thermal-model/issues/86)) ([bf52607](https://github.com/alliander-opensource/cable-thermal-model/commit/bf526079f0d7e6e6d1b74582326e5df8c1b6f53a))
* type hinting for scenario validation in ModelFactory ([#48](https://github.com/alliander-opensource/cable-thermal-model/issues/48)) ([92e1213](https://github.com/alliander-opensource/cable-thermal-model/commit/92e1213908f9fb03fa2c473e2ba318406649aa92))
* update default cable source file path and refactor scenario setting method. ([#59](https://github.com/alliander-opensource/cable-thermal-model/issues/59)) ([4a041c5](https://github.com/alliander-opensource/cable-thermal-model/commit/4a041c5c1a922ac22ff0d919652e5b719dadf900))
* various fixes to documentation ([#29](https://github.com/alliander-opensource/cable-thermal-model/issues/29)) ([73cbdc6](https://github.com/alliander-opensource/cable-thermal-model/commit/73cbdc60922ce98ddd8f84d6ce51432f53155951))


### Dependencies

* bump ruff from 0.15.22 to 0.16.5 ([#90](https://github.com/alliander-opensource/cable-thermal-model/issues/90)) ([c006533](https://github.com/alliander-opensource/cable-thermal-model/commit/c0065339c09ab9031ec91bb14529b7579937420c))
* bump structlog from 25.5.0 to 26.1.0 ([#89](https://github.com/alliander-opensource/cable-thermal-model/issues/89)) ([370a972](https://github.com/alliander-opensource/cable-thermal-model/commit/370a9721cad14df0fba51dd8c07acd21b7256f77))
* update uv-build requirement ([#93](https://github.com/alliander-opensource/cable-thermal-model/issues/93)) ([f389d07](https://github.com/alliander-opensource/cable-thermal-model/commit/f389d07f0fae313289907c403c8ff765d36f6350))


### Documentation

* add documentation page about FD method ([#37](https://github.com/alliander-opensource/cable-thermal-model/issues/37)) ([6234847](https://github.com/alliander-opensource/cable-thermal-model/commit/6234847e87feb38d4835889418dbd40896cab2da))
* add heat equation docs ([#35](https://github.com/alliander-opensource/cable-thermal-model/issues/35)) ([1c31224](https://github.com/alliander-opensource/cable-thermal-model/commit/1c31224767c91fda3115895c5cf7e9dc024919cb))
* add stateful documentation ([#23](https://github.com/alliander-opensource/cable-thermal-model/issues/23)) ([59755dc](https://github.com/alliander-opensource/cable-thermal-model/commit/59755dc0d7b200d7fe8bc7b63a76334b2a2cd82b))


### Miscellaneous Chores

* add pre-commit package ecosystem to dependabot configuration ([#63](https://github.com/alliander-opensource/cable-thermal-model/issues/63)) ([5b92394](https://github.com/alliander-opensource/cable-thermal-model/commit/5b92394ac0698418e7337f2542b399a58de88a42))
* add sonarcloud scan to cable thermal model ([#34](https://github.com/alliander-opensource/cable-thermal-model/issues/34)) ([46c2f8f](https://github.com/alliander-opensource/cable-thermal-model/commit/46c2f8f71f7e7236ab6bb87deb3c83cd2dab7cd2))
* add uv lock to release please PR ([#100](https://github.com/alliander-opensource/cable-thermal-model/issues/100)) ([e091729](https://github.com/alliander-opensource/cable-thermal-model/commit/e091729ced25500725e981997ba0695ced58f9d4))
* added instructions for pip and poetry ([#33](https://github.com/alliander-opensource/cable-thermal-model/issues/33)) ([9188658](https://github.com/alliander-opensource/cable-thermal-model/commit/91886583206f1fbce548b15850af0d289880741f))
* bump version ([#77](https://github.com/alliander-opensource/cable-thermal-model/issues/77)) ([5228e4e](https://github.com/alliander-opensource/cable-thermal-model/commit/5228e4e36fb57b4892173f9d6bc65e15138626f0))
* Bumped version to 2.1.0 ([#74](https://github.com/alliander-opensource/cable-thermal-model/issues/74)) ([b148433](https://github.com/alliander-opensource/cable-thermal-model/commit/b148433c16ce0e6a9de55e1a7e6e66f54dd4dbe4))
* privatize model methods ([#21](https://github.com/alliander-opensource/cable-thermal-model/issues/21)) ([f1fd730](https://github.com/alliander-opensource/cable-thermal-model/commit/f1fd730a0c98bef5a024b529a588dbe5b71b90eb))
* release main ([#99](https://github.com/alliander-opensource/cable-thermal-model/issues/99)) ([b932211](https://github.com/alliander-opensource/cable-thermal-model/commit/b932211416f17af7de285a743a891f88bb25dc6d))
* release please and trigger publishing workflow on release ([#87](https://github.com/alliander-opensource/cable-thermal-model/issues/87)) ([60c8d44](https://github.com/alliander-opensource/cable-thermal-model/commit/60c8d44fe3ca478181b2044c85ee2d0727d95146))
* remove sonar and reconfigure mkdocs generation ([#47](https://github.com/alliander-opensource/cable-thermal-model/issues/47)) ([64ce3c7](https://github.com/alliander-opensource/cable-thermal-model/commit/64ce3c7c66182369115c14a9736cc5afd5a59126))
* remove unused cff file ([#26](https://github.com/alliander-opensource/cable-thermal-model/issues/26)) ([c24cf90](https://github.com/alliander-opensource/cable-thermal-model/commit/c24cf90497f0a65190784c04f071dfba2b7f6e8b))
* update dependencies ([#84](https://github.com/alliander-opensource/cable-thermal-model/issues/84)) ([0ab52e1](https://github.com/alliander-opensource/cable-thermal-model/commit/0ab52e13ae3cf910dbfe615bccdf46d795c03d66))
* update package ecosystem from pip to uv in dependabot configuration ([#61](https://github.com/alliander-opensource/cable-thermal-model/issues/61)) ([f93b52f](https://github.com/alliander-opensource/cable-thermal-model/commit/f93b52f554e7a718fcd6d76dd1fd1593cb6c4fc0))
* update packages and libraries ([#60](https://github.com/alliander-opensource/cable-thermal-model/issues/60)) ([8cd0019](https://github.com/alliander-opensource/cable-thermal-model/commit/8cd0019d7a328aeed0fc669493a023efbf965963))
* update the release please workflow ([#104](https://github.com/alliander-opensource/cable-thermal-model/issues/104)) ([c3321ab](https://github.com/alliander-opensource/cable-thermal-model/commit/c3321ab7c8042580e68fd9dc610ed4464cb9b556))
* update to the new managed runners ([#80](https://github.com/alliander-opensource/cable-thermal-model/issues/80)) ([a579c16](https://github.com/alliander-opensource/cable-thermal-model/commit/a579c16801e5a528e046b0a4508ff5e4a35d3d63))

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
