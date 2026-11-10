<!--
SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project

SPDX-License-Identifier: MPL-2.0
-->

## Cables in Air
Cables are mostly installed underground. However, they can be installed in air as well. Typical applications are cable connections at substations in cable cellars or cable connections to transformers.

### Heat dissipation by the environment

For cables in air, heat is dissipated to the environment through natural convection of the surrounding air. The heat dissipation depends on the temperature difference between the cable sheath and the ambient temperature. The thermal behaviour inside the cable is similar to cables that are surrounded by the soil.

In CTM, the behaviour of the air around the cables is not modelled directly, but applied as a boundary condition. The heat exchange between the cable sheath and the environment (in W/m) is described as

$$
W_{dissipation} = \frac{\theta_n}{T_4}
$$

where $\theta_n$ denotes the difference between the cable sheath temperature and the ambient temperature. $T_4$ can be calculated as described in NEN-IEC 60287-2-1 (2023) section 4.2.1:

$$
T_4 = \frac{1}{\pi D_e h \theta_n^{1/4}}
$$

In this equation, $h$ denotes the heat convection coefficient and is calculated with

$$
h = \frac{Z}{D_e^{C_g}} + E
$$

Values for $Z$, $C_g$ and $E$ are given in Table 3 of NEN-IEC 60287-2-1 (2023). $D_e$ is the outer diameter of the cable (or pipe) in meters.

### Cable heating by solar radiation
Next to the heat dissipation by the environment, cables are heating up when exposed to solar radiation.

$$
W_{solar \ radiation} = \sigma D_e E_e
$$

$\sigma$ denotes the absorption coefficient of solar radiation and is provided for different cable surfaces in Table 4 of NEN-IEC 60287-2-1 (2023). $E_e$ is the intensity of solar radiation in W/m².

### Boundary condition in the FD model

At the boundary of the cable, there can not be any discontinuities. I.e. heat coming into the boundary (from the cable) has to leave the boundary to the environment. Heat transfer in the cable is merely assumed to be conduction:

$$
\frac{-2\pi r_n}{\rho_n}\frac{\partial \theta}{\partial r}\bigg|_{r=r_n} = \frac{\theta_n}{T_4} - \sigma D_e E_e
$$

The right hand side of the heat balance at the **last element** of the finite difference system then looks like

$$
\frac{1}{r} \frac{\partial}{\partial r} \frac{r}{\rho} \frac{\partial \theta}{\partial r} + W_{int} = \frac{1}{S_n} \bigg[\sigma D_e E_e - \frac{\theta_n}{T_4} + \frac{2\pi r_n^-}{\rho_n^-}\frac{\theta_{n-1} - \theta_n}{r_n - r_{n-1}} \bigg] + W_{int, n}
$$

where the area $S_n$ is defined as the area of the last element. The last element does not exceed the outer boundary $r_n$.

$$
S_n = \pi(r_n + r_n^-)(r_n - r_n^-)
$$

### Extending the right hand side of the matrix equation in [Overview](finite_difference.md#extending-the-matrix-equation)

We can now add an equation to the matrix to solve for $\theta_n$ with

$$
\begin{align*}
\beta_n &= -\frac{2\pi}{S_n} \bigg[r_n h \theta_n^{1/4} + \frac{r_n^-}{\rho_n^-\Delta r_{n-1}}\bigg],\\
\gamma_n &= \frac{2\pi}{S_n} \frac{r_n^-}{\rho_n^-\Delta r_{n-1}}.
\end{align*}
$$

Resulting in

$$
\begin{pmatrix}
\beta_0 & \alpha_0 & 0 & \dots & 0\\
\gamma_1 & \beta_1 & \alpha_1& \dots & 0\\
0 & \gamma_2 & \beta_2 & \ddots&\vdots\\
\vdots&&\ddots&\ddots&\alpha_{n-1}\\
0&\dots&&\gamma_{n}&\beta_{n}
\end{pmatrix}
\begin{pmatrix}
\theta_0\\
\\
\vdots\\
\\
\theta_{n}
\end{pmatrix}
+
\begin{pmatrix}
0\\
0\\
\vdots\\
0\\
2\sigma r_nE_e/S_n
\end{pmatrix}
+\begin{pmatrix}
W_{int,0}\\
\\
\vdots\\
\\
W_{int,n}
\end{pmatrix}
$$

The $\theta_n^{1/4}$ term in $\beta_n$ is calculated iteratively within each timestep.
