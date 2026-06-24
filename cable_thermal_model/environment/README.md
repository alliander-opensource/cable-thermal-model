<!--
SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project

SPDX-License-Identifier: MPL-2.0
-->

This directory contains the logic to build the environment properties around the cable, such as the resistivity
of the soil. An abstract base class StaticEnv is used which is implemented which is extended by a subclasses StaticEnvSoil and StaticEnvAir that represent circuits in soil and air respectively.
