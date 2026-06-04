<!--
SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project

SPDX-License-Identifier: MPL-2.0
-->

This folder/directory contains the different components of the thermal cable model.
In the cable directory the cable dataclass is constructed and circuits can be made from this.
In the environment directory the environment around the cable is constructed.

The model directory contains the classes for the cables and models that are used to solve the heat equation using FD.

Directories that are included but are not actively being used at the moment are:

- data_sourcing which contains methods to gather weather and soil data
- utils which contains utility functions that are only used by code that is not actively being used
These are not deleted as we might reuse this logic in the future.

TODO:
Explain what the differences are between the folders and how they interact.
