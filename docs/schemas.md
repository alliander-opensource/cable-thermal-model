<!--
SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project

SPDX-License-Identifier: MPL-2.0
-->
# Usage of Pandera Schemas in addition to Pydantic Models in CTM

This project uses Pandera for runtime validation of DataFrames and for type annotations in function and constructor signatures.

## Pandera Schema Inventory

The project currently defines Pandera DataFrame schemas in two places.

1. Input scenario schemas in `cable_thermal_model/model/schemas/model_input_schemas.py`:
1. `AbstractScenarioSchema`: shared scenario checks for all models.
1. `ScenarioSchemaAir`: currently the same checks as `AbstractScenarioSchema` (reserved for future air-specific checks).
1. `ScenarioSchemaSoil`: extends the abstract schema and adds required soil columns and numeric checks for those columns.

2. Output result schema in `cable_thermal_model/model/schemas/model_output_schemas.py`:
-. `TemperatureResultSchema`: validates the model result DataFrame shape and content.

## When Validation Happens

Validation occurs in two phases: input scenario validation and output result validation. Input validation happens at model construction time and whenever the scenario is replaced. This means invalid input is rejected early, before running thermal computations.
Output validation happens at the end of each model run.


## Explicit Validation Outside Models

Some utilities and examples also call schema validation explicitly before model creation. For example, a scenario can be validated with:

- `ScenarioSchemaAir.validate(df)` for air use cases,
- `ScenarioSchemaSoil.validate(df)` for soil use cases.

This is optional for callers, because model initialization will validate again.

## Note on Other "Schemas" in the Codebase

Not every `*schema*` module uses Pandera. Many input and state schemas are Pydantic models (for example cable construction input schemas, circuit input schemas, pipe schemas, run options, and state models). In short:

1. Pandera excels in, and is used for DataFrame-shaped validation.
2. Pydantic is used for object/model configuration validation.
3. Pandera schema's can be used in conjunction with Pydantic models.
