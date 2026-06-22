<!--
SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project

SPDX-License-Identifier: MPL-2.0
-->
# Stateful computations

In the Cable Thermal Model we support both stateless and stateful computations.
In a stateless computation (which is the default) the initial temperature of all cables equals the initial ambient temperature at the start of the scenario.
In a stateful computation you can use the results from a previous computation (the state) as a starting point for your calculations.

Refer to [this example calculation](./../../examples/example_calculation#stateful-computations) for example usage and
to the [API Docementation](./../../api_reference/cable_thermal_model/model/schemas/state_schemas) for information about the `State` object.
