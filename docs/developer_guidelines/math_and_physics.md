<!--
SPDX-FileCopyrightText: Contributors to the Cable Thermal Model project

SPDX-License-Identifier: MPL-2.0
-->

# Handling math and physics in CTM code

A lot of the modelling we do involves equations surrounding physics directly. As such there can be a thin line between keeping formulas recognizable and code “Pythonesque”.

To keep the CTM project accessible for as many contributors as possible and help individuals determine whether they could support a module or method we have set up guidelines on how we handle code that is related to physics or math.

## Developer guidelines

To help developers and keep the code as transparent as possible we have set up extra developer guidelines for the following situations:

- CTM Code needs to cover math and/or physics based equations in a way that directly refers to existing research or technical reference documents. A common example of this is most of the code where we follow IEC norms.

In these cases a clear link between the source documents and the code takes precedence over the usual guidelines and the following changes apply to that part of the code only:

- Math/Physics based code should always exist outside of code that follows regular guidelines unless
  there is a very good reason to not do so. Generate separate methods for each reference or any group
  of references that matches the intended usage.
- Each of these methods should have extensive documentation referencing the research or technical
  documentation they cover. Use the 'References' section for links to reference sites and Notes to help link formulas to code parameters where possible and not already clear.
- If part of the source formulas can be shortened or sped up via Python, only do so with a clear
  explanation.
- If these methods need to be called directly by users, the method's interface parameters should
  still use Pythonic names. Within the method these parameters can then either be cast to more
  reference appropriate names or the documentation can be updated to link these parameters to the referenced formulas and documentation.

## Examples

A good example of code where these guidelines would apply can be found in:
/cable_thermal_model/model/cables/abstract_cable.py

For this example we'll look at an older version of the method "_cable_screen_loss_method_cross_bonding_or_one_sided_bonding_linear_leading" and how it should be altered:

```python
def _cable_screen_loss_method_cross_bonding_or_one_sided_bonding_linear_leading(
        self, Tc: float, Ts: float
    ) -> float:
        """The screen loss retrieval method for cables with cross bonding or one-sided bonding.
        Linear version, conductor carrying the leading phase.

        Notes:
            Calculates eddy currents in the earthing sheath based on the NEN-IEC 60287-1-1 (2023) - [section 5.3.7.1].

        """
        Rs = self._get_resistance_screen(Ts)

        m = self.omega / Rs * 1e-7
        Delta1, Delta2 = 0, 0
        if m > self._M_THRESHOLD:
            Delta1 = 4.7 * m**0.7 * (self.d / (2 * self.s)) ** (0.16 * m + 2)
            Delta2 = 21 * m**3.3 * (self.d / (2 * self.s)) ** (1.47 * m + 5.06)

        lambda0 = 1.5 * (m**2 / (1 + m**2)) * (self.d / (2 * self.s)) ** 2
        lambda1_eddy = self._get_lambda1_eddy(Ts, Tc, lambda0, (Delta1 + Delta2))

        return lambda1_eddy
```



Already a lot was properly handled. The function variables and parameters are already based on the parameters and forces mentioned in the NEN IEC documentation rather than using Python naming conventions.

Two sub-methods are used for specific calculations mentioned in other sections of the NEN IEC reference document. These are loaded into the NEN referenced variables immediately and then used, rather than used directly within the code.

This is practical because this code:

```python
Rs = self._get_resistance_screen(Ts)

m = self.omega / Rs * 1e-7
```

Keeps the declaration of "m" far more transparent than this code:

```python
m = self.omega / self._get_resistance_screen(Ts) * 1e-7
```



Some improvements still need to be made though:

- Method arguments are still missing.
- The NEN IEC needs to be referenced in the References section.



Resulting in the following adjusted method:

```python
def _cable_screen_loss_method_cross_bonding_or_one_sided_bonding_linear_leading(
        self, Tc: float, Ts: float
    ) -> float:
        """The screen loss retrieval method for cables with cross bonding or one-sided bonding.

        Linear version, conductor carrying the leading phase.

        Args:
            Tc (float): The conductor temperature in degrees Celsius.
            Ts (float): The screen temperature in degrees Celsius.

        Notes:
            Calculates eddy currents in the earthing sheath based on the NEN-IEC 60287-1-1 (2023) - [section 5.3.7.1].

        References:
            - NEN-IEC 60287-1-1 (2023) - [section 5.3.7.1]

        """
        Rs = self._get_resistance_screen(Ts)

        m = self.omega / Rs * 1e-7
        Delta1, Delta2 = 0, 0
        if m > self._M_THRESHOLD:
            Delta1 = 4.7 * m**0.7 * (self.d / (2 * self.s)) ** (0.16 * m + 2)
            Delta2 = 21 * m**3.3 * (self.d / (2 * self.s)) ** (1.47 * m + 5.06)

        lambda0 = 1.5 * (m**2 / (1 + m**2)) * (self.d / (2 * self.s)) ** 2
        lambda1_eddy = self._get_lambda1_eddy(Ts, Tc, lambda0, (Delta1 + Delta2))

        return lambda1_eddy
```



## Summary

The main objective here is to keep this code separated from regular code and clearly related to the
referenced math or physics based papers. This prevents Pythonic naming conventions from hiding the original source and purpose of the code and allows for easy verification that the code matches the reference documents it was based on.
