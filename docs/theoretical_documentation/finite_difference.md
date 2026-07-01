<!--
SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project

SPDX-License-Identifier: MPL-2.0
-->
Any numerical method for solving the heat equation approximates the heat equation by dubdividing the time and spaces domains in a finite amount of cells within which the solution is approximated.
This process of subdividing is called discretizing.
In every cell the PDE is often linearized and solved with a simple function (typically a low order polynomial).
These solutions per cell together then form the approximation to the solution of the heat equation.
A larger number of cells will typically lead to a better approximation.

On this page we explain the _finite difference_ approach to approximating solutions to the heat equation.

## Discretizing the domain

In Cable Thermal Model we calculate temperatures for cross sections of cable circuits.
Thus we are considering a 2-dimensional (space) domain.
We simplify this to a 1-dimensional domain by using polar coordinates.
To explain this, consider the figure below.

![Using polar coordinates to model cable temperatures](../assets/cable-in-soil-polar-coordinates.drawio)
<p align="center"><em>Four images of a cross section of an undergroung power cable shows how the 2-dimensional domain is simplified into a 1-dimensional domain. <strong>A</strong>: Cross section of a cable in soil (brown) under a blue sky. <strong>B</strong>: Consider polar coordinates $r$ and $\phi$. <strong>C &amp; D</strong>: A symmetrical 1-dimensional representation of the domain is deduced from figure <strong>B</strong>.</em></p>
