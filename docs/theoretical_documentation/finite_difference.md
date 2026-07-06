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

## Using radial symmetry

In Cable Thermal Model we calculate temperatures for cross sections of cable circuits.
Thus we are considering a 2-dimensional (space) domain.
We simplify this to a 1-dimensional domain by using polar coordinates.
To explain this, consider the figure below, which shows how the 2-dimensional domain of a cable lying in soil can be simplified into a one-dimensional domain.

![Using polar coordinates to model cable temperatures](../assets/cable-in-soil-polar-coordinates.drawio)
<p align="center"><em>Four images of a cross section of an undergroung power cable shows how the 2-dimensional domain is simplified into a 1-dimensional domain. <strong>A</strong>: Cross section of a cable in soil (brown) under a blue sky. <strong>B</strong>: Consider polar coordinates $r$ and $\phi$. <strong>C &amp; D</strong>: A symmetrical 1-dimensional representation of the domain is deduced from figure <strong>B</strong>.</em></p>

Do note that we make the assumption that our domain, the heat generation and therefore also our solution are radially symmetrical. They thus depend on the radius $r$ exclusively. Whenever a more complex asymmetrical situation is considered, the finite difference approach as explained on this page may need some extra steps to yield satisfying results.

## Discretizing the domain

Keeping in mind that we are working in polar coordinates, the discretized domain consists of a typically non-uniformous vector
$$
\boldsymbol{r} = (r_0, \dots, r_n)
$$
of length $n+1$. The $r_i$ will be represent layers in the cable (for small i) and possibly soil (for larger i, if we are considering a circuit in soil).

We also record the thermal properties at the gridpoint locations in vectors of the same length. For example $\boldsymbol{\rho} = (\rho_0, \dots, \rho_n)$ represents the thermal resistivity at the gridpoints of $\boldsymbol{r}$.

We write $\boldsymbol{\Delta r}$ for the _spacing vector_, which contains the differences between consecutive gridpoints:
$$
\boldsymbol{\Delta r} = (\Delta r_0, \dots, \Delta r_{n-1}) = (r_1-r_0, \dots, r_n-r_{n-1})\,.
$$

We also define _intermediate gridpoints_:

$$
\begin{align*}
r_i^+ &= \frac{r_i+r_{i+1}}{2}=r_i+\Delta r_i/2,   &\text{ for } i=0, \dots, n-1\,, \\
r_i^- &= \frac{r_{i-1}+r_i}{2}=r_i-\Delta r_{i-1}/2,  &\text{ for } i=1, \dots, n\,.
\end{align*}
$$

Note that with this definition we have that $r_{i+1}^-=r_i^+$ for $1\leq i\leq n-1$.
We call $r_i^+$ and $r_i^-$ the _outer radius_ and _inner radius_ respectively of the grid point $r_i$.
We say the _thickness_ of the gridpoint $r_i$ is $r_i^+-r_i^-$. Then the volume corresponding to the gridpoint $r_i$ is given by $\pi (r_i^++r_i^-)(r_i^+-r_i^-)$.

## Grid points per layer

Each cable layer is represented by a number of gridpoints.
If the cable lies in soil, the soil is also represented by a number of grid points.
The gridpoints near the boundary of a layer are placed at most 0.1 mm from that boundary.
In between the outermost gridpoints of a single layer, the gridpoints are linearly distributed,
except for soil layers if they are present.
For soil layers we use a logarithmic distribution of the grid points.
In the python code of Cable Thermal Model we refer to $\boldsymbol{r}$ as `radii_grid` and to $\boldsymbol{\rho}$ as `rho_grid`.

### Example

Consider a cable with only three layers: a conductor, insulation and a sheath. Suppose that the three cable layers have radii 0.4, 0.8 and 1 m (for the purpose of this example we are dealing with a huge power cable).

![Image of example cable](../assets/example-cable.drawio)
