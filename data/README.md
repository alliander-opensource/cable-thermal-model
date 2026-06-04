<!--
SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project

SPDX-License-Identifier: MPL-2.0
-->

# Data file documentation

This directory contains the different data objects that are used to construct cables and environment properties.

## example_cables.csv

Contains constructional information that entails details about which layers a cable consists of, what the dimensions of those
layers are and which materials the different layers are made of. Often times dimensions are not specified directly but
need to be derived. Parsing the information from this csv to the information required by the CableBuilder happens in
the CableSpecParser class.

The column structure of the file equals the column structure of cable exports from [Vision Cable Analysis](https://www.phasetophase.nl/producten-diensten/producten/vision-cable-analysis/), developed by [Phase to Phase](https://www.phasetophase.nl/).

## material_properties.csv

A table describing material properties for all materials encountered in the list of example cables.

## IEC_conductor_resistance_table.csv

IEC 60228:2023 tables 3 and 4 describe the maximum resistance of conductors for different conductor surfaces and
materials. This resistance value is to be used in calculations of heating losses. In practice the exact properties
of materials can differ between manufacturers which means that conductor surfaces need not be the precise surface
specified by the cable type. The IEC table gives a reference to which resistance is expected for a given cable.

## /circuits/*.csv

A set of environment descriptions used in tests.

## pipe_filling_material_properties.csv

A table describing U, V, Y parameters for different filling materials. These properties are specfied in IEC60287 and
are required to compute the temperature-dependent thermal resistance of the material.
