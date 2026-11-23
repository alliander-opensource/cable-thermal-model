<!--
SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project

SPDX-License-Identifier: MPL-2.0
-->
# Usage of Pandera Schemas in addition to Pydantic Models in CTM

This project uses Pandera for runtime validation of DataFrames and for type annotations in function and constructor signatures.

## Pandera Schema Inventory

The project currently defines Pandera DataFrame schemas in two places.

1. Input scenario schemas in `cable_thermal_model/model/schemas/model_input_schemas.py`:
1. `AbstractScenarioModel`: shared scenario checks for all models.
1. `ScenarioModelAir`: adds required, nonnegative `solar_radiation_intensity`.
1. `ScenarioModelSoil`: adds required soil thermal resistivity and capacity columns.

2. Output result schema in `cable_thermal_model/model/schemas/model_output_schemas.py`:
-. `TemperatureResultSchema`: validates the model result DataFrame shape and content.

## When Validation Happens

Validation occurs in two phases: input scenario validation and output result validation.
Input validation happens at the start of each `model.run(scenario)` call, before thermal computations.
Output validation happens at the end of each model run.

The model extends the base input schema with required `load_<circuit_name>` columns
and optional `screen_current_<circuit_name>_<position>` columns for known cable keys.
Unknown columns are rejected. Screen-current inputs must be finite, nonnegative RMS
amperes over the interval ending at their timestamp. The first row is initialization-only.
Absent columns retain calculated circulating losses. Zero is an explicit custom current, not absence.
Eddy losses remain standards-derived. Custom-current columns are rejected for DC runs, screenless
cables, and equivalent shared-pipe trefoil models. See
[custom screen currents](get_started/model_input.md#optional-custom-screen-currents)
for the loss equation and input assumptions.

## Explicit Validation Outside Models

Some utilities and examples also validate the base environmental schema explicitly:

- `ScenarioModelAir.validate(df)` for air use cases,
- `ScenarioModelSoil.validate(df)` for soil use cases.

These classes are defined in `cable_thermal_model.model.schemas.model_input_schemas`.
Standalone validation does not check environment-specific cable keys, circuit loads,
or screen-current compatibility. `model.run()` performs those checks automatically.

## Note on Other "Schemas" in the Codebase

Not every `*schema*` module uses Pandera. Many input and state schemas are Pydantic models (for example cable construction input schemas, circuit input schemas, pipe schemas, run options, and state models). In short:

1. Pandera excels in, and is used for DataFrame-shaped validation.
2. Pydantic is used for object/model configuration validation.
3. Pandera schema's can be used in conjunction with Pydantic models.
