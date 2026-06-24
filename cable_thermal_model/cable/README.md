<!--
SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project

SPDX-License-Identifier: MPL-2.0
-->

This folder/directory contains the logic to build the cable configuration based on properties such as the core material,
which isolation is used, how the cables are configured (e.g. trefoil vs single).

The main logic is in the CableBuilder and CircuitBuilder classes in respectively `cable.py` and `cable_circuit.py`.
The CableBuilder sets the correct material properties. The CircuitBuilder constructs multiple cables or cores using
the knowledge about the configuration and uses the CableBuilder to do this.
