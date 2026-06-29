<!--
SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project

SPDX-License-Identifier: MPL-2.0
-->
# The heat equation

In order to compute real-time cable temperatures the temperature model has been designed to be able to handle dynamic loads and changing environmental conditions. The foundation of these dynamic computations is the heat equation. The heat equation is a partial differential equation which completely controls heat flow in a closed system.

The heat equation is given by

$$
c\frac{\partial \theta}{\partial t} = \nabla \cdot \left(\frac{1}{\rho}\nabla \theta\right) +W_{int}\,
$$
where

- $\theta$ is the temperature of the medium in K. In CTM, $\theta$ is actually defined as the temperature difference with respect to the ambient temperature.
- $\rho$ is the thermal resistivity of the medium in Km/W.
- $c$ is the volumetric heat capacity of the medium in J/m<sup>3</sup>K.
- $W_{int}$ is the internal heat generated in the medium in W/m<sup>3</sup>.

Note that both $c$ and $\rho$ may depend on the position. In particular we cannot we cannot move the factor $1/\rho$ on the right hand side of the equation outside the divergence operator $\nabla$.

We use polar coordinates to exploit the radial symmetry of power cables. The heat equation in polar coordinates becomes

$$
c\frac{\partial \theta}{\partial t}(r, \phi, t) = \frac{1}{r}\frac{\partial}{\partial r}\left(\frac{r}{\rho}\frac{\partial\theta}{\partial r}\right) + \frac{1}{r^2} \frac{\partial}{\partial \phi}\left(\frac{r}{\rho}\frac{\partial\theta}{\partial \phi}\right) +W_{int}\,,
$$
where $\phi$ represents the angle and $r$ the distance to the origin. We furthermore assume that the solution is radially symmetric, so that the term $\frac{1}{r^2} \frac{\partial}{\partial \phi}\left(\frac{r}{\rho}\frac{\partial\theta}{\partial \phi}\right)$ vanishes. We are left with the equation
$$
c\frac{\partial \theta}{\partial t} = \frac{1}{r}\frac{\partial}{\partial r}\left(\frac{r}{\rho}\frac{\partial\theta}{\partial r}\right) + W_{int}\,.
$$
The choice of $\theta$ for the temperature is standard in the literature for power cables, where $T$ is reserved for thermal resistance.

## Influence of parameters

The intuition behind the parameters $\rho$ and $c$ are as follows. The thermal resistivity controls how quickly heat dissipates through a medium. The volumetric heat capacity on the other hand measures how much energy needs to be added to increase the temperature of a unit volume of the material by 1 Kelvin.

We can illustrate the influence of these parameters by varying the values in a realistic setting. We plot the temperature profile of a cable carrying a cyclic load. As can be seen in the figures below __increasing thermal resistivity__ shifts the whole dynamic temperature profile __upwards__, whereas __increasing volumetric heat capacity__ stretches the dynamic temperature profile __downwards__.

### Varying thermal resistivity
![Influence of varying thermal resistivity on temperature profile](../assets/varying-resistivity.png)

### Varying thermal capacity
![Influence of varying thermal capacity on temperature profile](../assets/varying-capacity.png)
