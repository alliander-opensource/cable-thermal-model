<!--
SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project

SPDX-License-Identifier: MPL-2.0
-->

This folder/directory contains the different components of the thermal cable model.
In the cable directory the cable dataclass is constructed and circuits can be made from this.
In the environment directory the environment around the cable is constructed.

The model directory contains the classes for the cables and models that are used to solve the heat equation using FD.

The utils directory contains functionality that is exclusively used by code not actively being used.
We include the directory nonetheless to enable future reuse of the logic.
