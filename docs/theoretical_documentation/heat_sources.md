<!--
SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project

SPDX-License-Identifier: MPL-2.0
-->

# Internal heat generation

As a directed electrical current of magnitude $I$ passes through a power cable heat is generated in its conductor according to Ohm's law
$$
P = I^2R\,,
$$
where $R$ is the electrical resistance per meter ($\Omega/m$) and $P$ denotes the heat loss per meter ($W/m$).
$R$ is calculated as
$$
R = \rho_{elec}/A_{nom}\,,
$$
where $\rho_{elec}$ (see [Electrical resistance](#electrical-resistance)) is the _electrical resistance_ of the conductor material and $A_{nom}$ is the nominal cross sectional area of the cable conductor as specified in the IEC-60228 norm _Conductors for insulated cables_.

### Alternating currents

In practice, we are dealing with AC currents instead of DC currents most of the time. Therefore we apply correction factors that take into account the alternating current skin and proximity effects. These corrections are based on the 60287-1-1 IEC norm Section 5.1.3 and Section 5.1.5 for the skin and proximity effect respectively.

### Heat generation in cable screen

When dealing with alternating currents, the alternating magnetic field induces currents in the cable screen (when present). We model the heat loss in the cable screen as a fraction of the heat loss in the cable conductor. This fraction is specified in Section 5.3 by the 60287-1-1 IEC norm and depends on the geometry of the cable, the circuit configuration and the bonding type.


### Adding heat to cable layers
In Cable Thermal Model we distribute the heat generation evenly across a cable layer. For a specific cable layer (typically the conductor or screen) this means that

$$
W_{int, i} = P/A_{layer}\,\text{ for each } i \text { inside the cable layer.}
$$
Here $A_{layer}$ denotes the surface area of the cable layer and $W_{int}$ denotes the internal heating vector, see [Finite difference method](./theoretical_documentation/finite_difference.md).

> **_NOTE:_** Here we use the surface area of the layer as modelled in CTM, instead of the equivalent surface area used to calculate the heat generation for conductors mentioned above in [Heat generation inside the conductor](#heat-generation-inside-the-conductor). If $r_{inner}$ and $r_{outer}$ denote the inner and outer radius of the cable layer, the area is thus given by $ A_{layer} = \pi(r_{outer}^2-r_{inner}^2)$.

### Electrical resistance

The electrical resistance $\rho_{elec}$ of the conductor material is dependent on the temperature and computed as
$$
\rho_{elec} = \rho_{20}(1+\alpha)(T-20)\,,
$$
where $\alpha$ denotes the _temperature coefficient_ of the conductor material and $\rho_{20}$ is the electrical resistance of the conductor material at 20 °C. Both constants $\alpha$ and $\rho_{20}$ can be found in the `material_properties.csv` file inside the `data` directory.


# External heat sources

As heat produced by a single cable dissipates through the ground, it also heats nearby cables. When modelling a 2-dimensional environment as a 2-dimensional grid, one can place all the heat sources in this grid and the heat equation automatically accounts for mutual heating of cables. However, in the implementation of Cable Thermal Model we use a 1-dimenional radial grid, which is only suitable for temperature changed which are radially symmetric. Of course, mutual heating of cables is not radially symmetric. Therefore, we approximate mutual heating using an algorithm that we describe below.

## The algorithm

We use an example to illustrate the algorithm that we are using a 1-dimensional approach to compensate for the fact that we are modelling a situation that is not radially symmetric. In the example we consider two cables A and B as depicted below.

<div class="center-drawio" markdown="1">![](../assets/mutual-heating.drawio)</div>

### Step 1. Internal heating

We first consider the two cables separately and calculate the temperature of the cables and the surrounding soil according to method explained on [Finite difference method](./theoretical_documentation/finite_difference.md). Let $s'_A(r)$ and $s'_B(r)$ denote the solutions at radius $r$ of cables A and B respectively.

### Step 2. Mutual heating and mirror cables

In the second step we truncate each cable so that only the gridpoints internal to the cable and a single soil gridpoint remain. Moreover, for each cable we consider a _mirror image_ of that cable as if it were above the ground. Starting with cable A, we calculate all the distances to the other cables (now four in total) as depicted in the image below. See also Figure 1 in IEC norm 60287-2-1.

<div class="center-drawio" markdown="1">![](../assets/mutual-heating-mirrored.drawio)</div>

We then solve a system of linear equations for a second time, but for the truncated cable instead, and with a different outer boundary condition. Instead of setting it to zero, which would correspond to the ambient temperature, we use the solution from Step 1 of cable B at distance $d$. Note that in reality, the temperature of cable B at the sheath of cable A will be higher at points on the sheath that are closer to cable B and lower at points that are further away from cable B. However, the temperature at distance $d$ provides a good average and is also used by the IEC norms implementing mutual heating.

To further compensate for the fact that we assume the the temperature of the ground surface equals the ambient temperature, we _substract_ the solutions from Step 1 of cable A and cable B at distances $d_{A,A'}$ and $d_{A,B'}$  respectively. The boundary condition is thus given by

$$
s'_{B}(r)-s'_{A}(d_{A,A'})-s'_{B}(d_{B,B'})\,.
$$

We repeat this calculation for cable B, resulting in _mutual heating solutions_ $s''_{A}(r)$ and $s''_B(r)$, for $r$ less than the respective cable radii.

### Step 3. Add the results

Finally, we add the solutions of Step 1 and Step 2 to the ambient temperature $T_{amb}$ to obtain the final solution. For instance, for cable A we get the solution

$$
s_A(r) = s'_A(r) + s''_A(r) + T_{amb}\,,
$$

for $r$ less than the radius of cable A.

### Step 4. Repeat.

The above steps are repeated for each timestep
