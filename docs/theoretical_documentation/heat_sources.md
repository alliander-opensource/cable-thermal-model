<!--
SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project

SPDX-License-Identifier: MPL-2.0
-->

# Internal heat generation

## Heat generation inside the conductor
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

In practice, we are dealing with AC currents instead of DC currents most of the time. Therefore we apply correction factors that take into account the alternating current skin and proximity effects. These corrections are based on the IEC-60287-1-1 norm:

- Section 5.1.5 for the proximity effect.
- Section 5.1.3 for the skin effect.

## Heat generation in cable screen

When dealing with alternating currents, the alternating magnetic field induces currents in the cable screen (when present). We model the heat loss in the cable screen as a fraction of the heat loss in the cable conductor. This fraction is specified in Section 5.3 by the 60287-1-1 IEC norm and depends on the geometry of the cable, the circuit configuration and the bonding type.


## Adding heat to cable layers
In Cable Thermal Model we distribute the heat generation evenly across a cable layer. For a specific cable layer (typically the conductor or screen) this means that

$$
W_{int, i} = P/A_{cable\_layer}\,\text{ for each } i \text { inside the cable layer.}
$$
Here $A_{cable\_layer}$ denotes the surface area of the cable layer and $W_{int}$ denotes the internal heating vector, see [Finite difference method](./theoretical_documentation/finite_difference.md).

> **_NOTE:_** Here we use the surface area of the layer as modelled in CTM, instead of the equivalent surface area used to calculate the heat generation for conductors mentioned above in [Heat generation inside the conductor](#heat-generation-inside-the-conductor). If $r_{inner}$ and $r_{outer}$ denote the inner and outer radius of the cable layer, the area is thus given by $ A_{cable\_layer} = \pi(r_{outer}^2-r_{inner}^2)$.


# External heat sources


## Electrical resistance

The electrical resistance $\rho_{elec}$ of the conductor material is dependent on the temperature and computed as
$$
\rho_{elec} = \rho_{20}(1+\alpha)(T-20)\,,
$$
where $\alpha$ denotes the _temperature coefficient_ of the conductor material and $\rho_{20}$ is the electrical resistance of the conductor material at 20 °C. Both constants $\alpha$ and $\rho_{20}$ can be found in the `material_properties.csv` file inside the `data` directory.
