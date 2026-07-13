<!--
SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project

SPDX-License-Identifier: MPL-2.0
-->
Any numerical method for solving the heat equation approximates the heat equation by subdividing the time and spaces domains in a finite amount of cells within which the solution is approximated.
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

<div class="center-drawio" markdown="1">![](../assets/cable-in-soil-polar-coordinates.drawio)</div>
<figcaption align="center"><em>Four images of a cross section of an underground power cable shows how the 2-dimensional domain is simplified into a 1-dimensional domain. <strong>A</strong>: Cross section of a cable in soil (brown) under a blue sky. <strong>B</strong>: Consider polar coordinates $r$ and $\phi$. <strong>C &amp; D</strong>: A symmetrical 1-dimensional representation of the domain is deduced from figure <strong>B</strong>.</em></figcaption>

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

## Intermediate thermal resistivity

We define the thermal resistivity at these intermediate grid points using the logarithmic relation of thermal resistivity of multiple circular layers (See _Rating of Electric Power Cables Ampacity Computations for Transmission (1997), Section 3.2.1_, George J. Anders):

$$
\begin{align*}
\rho_i^+ &= \frac{\rho_i\log(r_i^+/r_{i})+\rho_{i+1}\log(r_{i+1}/r_{i}^+)}{\log(r_{i+1}/r_{i})}, &\text{ for } i = 0, \dots, n-1\,,\\
\rho_i^- &= \frac{\rho_{i-1}\log(r_{i}^-/r_{i-1})+\rho_{i}\log(r_i/r_{i}^-)}{\log(r_{i}/r_{i-1})}, &\text{ for } i = 1, \dots, n\,.\\
\end{align*}
$$

Note that $\rho_i^+=\rho_{i+1}^-$ for all $1\leq i\leq n-1$ and that $\rho_i^+=\rho_i$ whenever $\rho_i=\rho_{i+1}$, which happens when $r_i$ and $r_{i+1}$ are part of the same cable layer. In other words, only at the cable boundary the intermediate thermal resistivities might be different from the thermal resistivity at neighbouring grid points.

## Grid points per layer

Each cable layer is represented by a number of gridpoints.
If the cable lies in soil, the soil is also represented by a number of grid points.
The gridpoints near the boundary of a layer are placed at most 0.1 mm from that boundary.
We always set $r_0=0$.
In between the outermost gridpoints of a single layer, the gridpoints are linearly distributed,
except for soil layers if they are present.
For soil layers we use a logarithmic distribution of the grid points.
In the python code of Cable Thermal Model we refer to $\boldsymbol{r}$ as `radii_grid` and to $\boldsymbol{\rho}$ as `rho_grid`.

### Example

Consider a cable with only three layers: a conductor, insulation and a sheath. Suppose that the three cable layers have radii 0.4, 0.8 and 1m (for the purpose of this example we are dealing with a huge power cable). We also add a soil layer around the cable, with a boundary at 2m.

<div class="center-drawio" markdown="1">![](../assets/example-cable.drawio)</div>
<figcaption align="center"><em>A picture of the simple cable with three layer and the first soil layer surrounding it.</em></figcaption>

Suppose that we use 5 grid points for each layer, then the distribution of the grid points might look something like in the figure below. When using the a maximal distance to the layer boundaries of 0.1mm, the insulation layer for instance is represented by the vector `radii_grid[5:10] = (0.4001, 0.50005, 0.6, 0.69995, 0.7999)`.

<div class="center-drawio" markdown="1"> ![](../assets/arrow.drawio) </div>
<figcaption align="center"><em>A schematic representation of the grid points in the different layers.</em></figcaption>

Now that we have determined the grid points in the vector $\boldsymbol{r}$, we need to specify the thermal properties at each grid point. These properties depend on the materials that the layers are made of. As an example, consider the thermal resistivity $\boldsymbol{\rho} = (\rho_0, \dots, \rho_n)$. Then $\rho_i$ is the thermal resistivity of the piece of cable that is represented by $r_i$. In our example, suppose that the conductor is made of aluminium, the insulation is made of unfilled XLPE and the sheath is consists of PVC. The thermal resistivity of these materials is recorded in the `material_properties.csv` file in the `data` directory. Let's assume that the theraml resistivity of the soil equals 0.5 J/m<sup>3</sup>K. Then

$$
\boldsymbol{\rho} = (0.004219409, 0.004219409, 0.004219409, 0.004219409, 0.004219409, 3.5, 3.5, 3.5, 3.5, 3.5, 5, 5, 5, 5, 5, 0.5, 0.5, 0.5, 0.5, 0.5)\,.
$$

> **_NOTE:_** When running the model, the user will need to specify the thermal properties of the soil in the `scenario` dataframe.

## Approximating the heat equation


The [heat equation](./heat_equation.md) that governs the heat dynamics inside and around the cable. Assuming a radially symmetric solution, the heat equation is given in polar coordinates by

$$
c\frac{\partial \theta}{\partial t} = \frac{1}{r}\frac{\partial}{\partial r}\left(\frac{r}{\rho}\frac{\partial\theta}{\partial r}\right) + W_{int}\,.
$$

In order to approximate the solution $\theta$, we replace the differential operator $\frac{\partial}{\partial r}$ with a _finite difference_ operator $D_r$.
For convenience we introduce some shorthand notation.
Whenever $\psi(r)$ is a function, we write $\psi_i$ and $\psi_i^{\pm}$ for the evaluation of $\psi$ at the gridpoint $r_i$ resp. $r_i^\pm$.
We also write $\psi^+$ and $\psi^-$ for the functions that are defined by $\psi^+(r_i) = \psi^+_i$ and $\psi^-(r_i) = \psi^-_i$. The finite difference operator is then defined by

$$
D_r\theta = \frac{\theta^+-\theta^-}{r^+-r^-}, \text{ so that } (D_r \theta)(r_i) = \frac{\theta_i^+-\theta_i^-}{r_i^+-r_i^-}\,.
$$

Using this approximation of the derivative with respect to $r$, we can turn the right hand side of the heat equation, without internal heat generation $W_{int}$, into a system of linear equations. Namely, for $1\leq i\leq n$ we get:

$$
\begin{align*}
\frac{1}{r}D_r\left(\frac{r}{\rho}D_r\theta\right)(r_i)&=\frac{1}{r_i(r_i^+-r_i^-)}\left(\frac{r_i^+}{\rho_i^+}(D_r\theta)(r_i^+)-\frac{r_i^-}{\rho^-_i}(D_r\theta)(r_i^-)\right)\\
&=\frac{1}{r_i(r_i^+-r_i^-)}\left(\frac{r_i^+}{\rho_i^+}\frac{\theta_{i+1}-\theta_i}{r_{i+1}-r_i}-\frac{r_i^-}{\rho^-_i}\frac{\theta_i-\theta_{i-1}}{r_i-r_{i-1}}\right)\\
&= \alpha_i\theta_{i+1} +\beta_i\theta_i + \gamma_i\theta_{i-1}\,,
\end{align*}
$$

where

$$
\begin{align*}
\alpha_i &= \frac{r^+_i}{\rho^+_ir_i(r_i^+-r_i^-)\Delta r_i}\,,\\
\beta_i &= \frac{-1}{r_i(r_i^+-r_i^-)}\left(\frac{r_i^+}{\rho_i^+\Delta r_i}+\frac{r_i^-}{\rho_i^- \Delta r_{i-1}}\right)\,,\\
\gamma_i&=\frac{r_i^-}{\rho_i^-r_i(r_i^+-r_i^-)\Delta r_{i-1}}\,.
\end{align*}
$$

Note that the coefficient $\rho$ is evaluated at intermediate grid points, whereas the temperature function $\theta$ is evaluated at the grid points themselves. We can write this system of linear equations in matrix form $c\frac{\partial \theta}{\partial t} = A \theta +b$, where $A$ is the tridiagonal matrix with upper diagonal $\alpha$, diagonal $\beta$ and lower diagonal $\gamma$. The right hand side of the heat equation then looks as follows

$$
\begin{pmatrix}
\beta_1 & \alpha_1 & 0 & \dots & 0\\
\gamma_2 & \beta_2 & \alpha_2& \dots & 0\\
0 & \gamma_3 & \beta_3 & \ddots&\vdots\\
\vdots&&\ddots&\ddots&\alpha_{n-2}\\
0&\dots&&\gamma_{n-1}&\beta_{n-1}
\end{pmatrix}
\begin{pmatrix}
\theta_1\\
\\
\vdots\\
\\
\theta_{n-1}
\end{pmatrix}
+
\begin{pmatrix}
\gamma_1\theta_0\\
0\\
\vdots\\
0\\
\alpha_{n-1}\theta_n
\end{pmatrix}
+\begin{pmatrix}
W_{int,1}\\
\\
\vdots\\
\\
W_{int,n-1}
\end{pmatrix}\,.
$$

### Boundary conditions

The matrix equation above still contains two unknowns, namely $\theta_0$ and $\theta_N$. These correspond to the _boundary conditions_ of the problem we are trying to solve. We deal with this as follows.

#### Outside boundary
At the outer boundary $r_n$ we use a _Dirichlet boundary condition_, i.e., we set $\theta_n$ to be equal to a specified ambient temperature.

#### Inside boundary
For our usecase for cables we typically have that $r_0=0$. Due to the axial symmetry of the problem, we have that $\frac{\partial \theta}{\partial r}\big|_{r=0}=0$. This is a _Neumann boundary condition_. Assuming that $\rho$ is constant to $\rho_0$ near the origin, using the product rule we get that

$$
\begin{align*}
\frac{1}{r}\frac{\partial }{\partial r}\left(\frac{r}{\rho_0}\frac{\partial \theta}{\partial r}\right)&=\frac{1}{\rho_0 r}\frac{\partial }{\partial r}\left(r\frac{\partial \theta}{\partial r}\right)\\
&=\frac{1}{\rho_0 r}\left(\frac{\partial \theta}{\partial r} + r\frac{\partial^2\theta}{\partial r^2}\right)\\
&=\frac{1}{\rho_0}\left(\frac{1}{r}\frac{\partial \theta}{\partial r} +\frac{\partial^2\theta}{\partial r^2}\right)\,.
\end{align*}
$$

Then evaluating in $r>0$ and taking the limit $r\rightarrow 0$ we use l'Hôpital's rule to find

$$
\begin{align*}
\lim_{r\rightarrow 0} \frac{1}{\rho_0}\left(\frac{\frac{\partial \theta}{\partial r}}{r} +\frac{\partial^2\theta}{\partial r^2}\right)&= \frac{2}{\rho_0}\frac{\partial^2\theta}{\partial r^2}\bigg|_{r=0}\,.
\end{align*}
$$

Replacing again $\frac{\partial}{\partial r}$ with the finite difference approximation we obtain

$$
\begin{align*}
\frac{2}{\rho_0}D_r(D_r\theta)(r_0)&= \frac{2}{\rho_0}\frac{(D_r\theta)(r_0^+)-(D_r\theta)(r_0^-)}{r_0^+-r_0^-}\\
&=\frac{2}{\rho_0\Delta r_0}\left(\frac{\theta_1-\theta_0}{r_1-r_0}-\frac{\theta_0-\theta_{-1}}{r_0-r_{-1}}\right)\\
&=\frac{4}{\rho_0(\Delta r_0)^2}(\theta_1-\theta_0)\,,
\end{align*}
$$

where we have used the symmetries $r_0^+-r_0^-=r_1-r_0=r_0-r_{-1}=\Delta r_0$ and $\theta_{-1}=\theta_1$.

#### Extending the matrix equation
Writing

$$\alpha_0 = \beta_0 = \frac{4}{\rho_0 (\Delta r_0)^2}$$

we can extend the right hand side of the matrix equation to

$$
\begin{pmatrix}
\beta_0 & \alpha_0 & 0 & \dots & 0\\
\gamma_1 & \beta_1 & \alpha_1& \dots & 0\\
0 & \gamma_2 & \beta_2 & \ddots&\vdots\\
\vdots&&\ddots&\ddots&\alpha_{n-2}\\
0&\dots&&\gamma_{n-1}&\beta_{n-1}
\end{pmatrix}
\begin{pmatrix}
\theta_0\\
\\
\vdots\\
\\
\theta_{n-1}
\end{pmatrix}
+
\begin{pmatrix}
0\\
0\\
\vdots\\
0\\
\alpha_{n-1}\theta_n
\end{pmatrix}
+\begin{pmatrix}
W_{int,0}\\
\\
\vdots\\
\\
W_{int,n-1}
\end{pmatrix}\,,
$$

which now only has $\theta_n$ as a boundary value.

## Apprimating the time derivative

So far we have only approximated the right-hand side of the heat equation. In order to solve for non-steady state solutions we also need to approximate the left-hand side that involves the time derivative. We will approximate the solution on a finite, possibly non-uniform vector

$$ t = (t_0, \dots, t_m)\,,$$

with time differences given by the vector

$$ \Delta t = (\Delta t_0, \dots, \Delta t_{m-1}) = (t_1-t_0, \dots , t_m-t_{m-1})\,.$$

Our approximation of the left hand side of the heat equation is then given by

$$ c (D_t \theta)(t_j) = c\frac{\theta(t_{j+1})-\theta(t_j)}{\Delta t_j}, \text{ for } j=0, \dots, m-1$$

and where $\theta(t)$ now denotes the vector $(\theta_0(t), \dots, \theta_{n-1}(t))$ and c denotes the diagonal matrix $ c = diag(c_0, \dots, c_{n-1})$ with entries the thermal capacities at the spatial gridpoints.

## Backwards Euler method

In Cable Thermal Model we use the [backward Euler method](https://en.wikipedia.org/wiki/Backward_Euler_method) to approximate the solution to the heat equation. This amounts to solving the system of linear equations

$$c \frac{\theta(t_{j+1})-\theta(t_j)}{\Delta t_j} = A \theta(t_{j+1})+b $$

at each timestep $t_j$. Here $A$ and $b$ are defined as [above](#extending-the-matrix-equation). $c$ is the diagonal matrix $ c = diag(c_0, \dots, c_{n-1})$ with entries the thermal capacities at the spatial gridpoints. We can rewrite this equation to

$$ (c-\Delta t_j A)\theta(t_{j+1}) = \Delta t_j b + c\theta(t_j)\,, $$

so that the solution at time $t_{j+1}$ can be determined from the solution at time $t_j$ by

$$ \theta(t_{j+1}) = (c-\Delta t_j A)^{-1}(\Delta t_j b + c\theta(t_j))\,. $$

Computing the matrix inverse of $c-\Delta t_j A$ is computationally expensive however. Instead we use the efficient [`scipy.linalg.solve_banded`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_banded.html), which makes use of the tridiagonal structure of the matrix $c-\Delta t_j A$.
