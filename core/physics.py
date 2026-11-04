"""core/physics.py – Physical constants, fluid properties, and calculation settings.

Classes
-------
PhysicsSettings
    Dataclass holding gravitational acceleration, kinematic viscosity,
    and friction calculation parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class PhysicsSettings:
    """Physical constants and calculation settings.

    Parameters
    ----------
    g : float
        Gravitational acceleration in m/s^2. EPANET defines standard gravity
        as 32.2 ft/s^2, which in SI metric units is 32.2 * 0.3048 = 9.81456 m/s^2.
        Using this exact value ensures bitwise agreement with EPANET and the
        MATLAB reference.
    nu : float
        Kinematic viscosity of water at 20°C in m^2/s (1.0219e-6).
    friction_mode : str
        Friction formulation: 'constant' (default, f = friction_factor),
        'swamee_jain', or 'darcy_weisbach'.
    friction_factor : float
        Darcy friction factor (dimensionless) used when friction_mode is
        'constant' (default 0.02).
    """

    g: float = 32.2 * 0.3048
    nu: float = 1.0219e-6
    friction_mode: str = "constant"
    friction_factor: float = 0.02

    def __post_init__(self) -> None:
        if self.friction_mode not in {"constant", "swamee_jain", "darcy_weisbach"}:
            raise ValueError(
                f"Unknown friction_mode: {self.friction_mode!r}. "
                "Must be 'constant', 'swamee_jain', or 'darcy_weisbach'."
            )
        if self.g <= 0:
            raise ValueError(f"g must be positive, got {self.g}")
        if self.nu <= 0:
            raise ValueError(f"nu must be positive, got {self.nu}")
        if self.friction_factor <= 0:
            raise ValueError(f"friction_factor must be positive, got {self.friction_factor}")


# EPANET 2.2 hydcoeffs.c constants for Darcy-Weisbach friction factor
_DW_A1 = 1000.0 * math.pi
_DW_A2 = 500.0 * math.pi
_DW_A8 = 5.74 * (math.pi / 4.0) ** 0.9
_DW_A9 = -2.0 / math.log(10.0)
_DW_AA = -2.0 * 0.9 * 2.0 / math.log(10.0)
_DW_AB = 5.74 / (4000.0 ** 0.9)
_DW_AC = _DW_AA * _DW_AB


def friction_factor(
    diameter: float,
    roughness: float,
    Q: float,
    settings: Optional[PhysicsSettings] = None,
) -> float:
    """Compute Darcy-Weisbach friction factor f for pipe flow Q.

    Parameters
    ----------
    diameter : float
        Pipe internal diameter in metres.
    roughness : float
        Absolute pipe roughness (epsilon) in metres.
    Q : float
        Volumetric flow rate in m^3/s.
    settings : PhysicsSettings, optional
        Physical parameters and friction mode. Defaults to PhysicsSettings().

    Returns
    -------
    float
        Dimensionless friction factor f.
    """
    if settings is None:
        settings = PhysicsSettings()

    mode = settings.friction_mode
    if mode == "constant":
        return float(settings.friction_factor)

    q = abs(float(Q))
    d = float(diameter)
    eps = float(roughness)
    nu = float(settings.nu)

    # Reynolds number Re = 4*|Q| / (pi * D * nu)
    re = 4.0 * q / (math.pi * d * nu)

    if mode == "swamee_jain":
        # Turbulent formula at every Q:
        # f = 0.25 / log10( eps/(3.7*D) + 5.74 / Re^0.9 )^2
        re_safe = max(re, 1e-12)
        arg = eps / (3.7 * d) + 5.74 / (re_safe ** 0.9)
        return 0.25 / (math.log10(arg) ** 2)

    if mode == "darcy_weisbach":
        # EPANET 2.2 rule (function frictionFactor in hydcoeffs.c):
        # w = q / s = Re * pi / 4
        s = nu * d
        w = q / s if s > 0 else 0.0
        e_rel = eps / d

        # Re >= 4000: Swamee & Jain approximation
        if w >= _DW_A1:
            y1 = _DW_A8 / (w ** 0.9)
            y2 = e_rel / 3.7 + y1
            y3 = _DW_A9 * math.log(y2)
            return 1.0 / (y3 * y3)

        # 2000 < Re < 4000: Dunlop cubic interpolation
        if w > _DW_A2:
            y2 = e_rel / 3.7 + _DW_AB
            y3 = _DW_A9 * math.log(y2)
            fa = 1.0 / (y3 * y3)
            fb = (2.0 + _DW_AC / (y2 * y3)) * fa
            r = w / _DW_A2
            x1 = 7.0 * fa - fb
            x2 = 0.128 - 17.0 * fa + 2.5 * fb
            x3 = -0.128 + 13.0 * fa - 2.0 * fb
            x4 = 0.032 - 3.0 * fa + 0.5 * fb
            return x1 + r * (x2 + r * (x3 + r * x4))

        # Re <= 2000: Laminar flow (64 / Re)
        if re <= 0.0:
            return 64.0 / 1e-12
        return 64.0 / re

    raise ValueError(f"Unknown friction_mode: {mode!r}")


def pipe_friction_term(
    diameter: float,
    roughness: float,
    Q: float,
    settings: Optional[PhysicsSettings] = None,
) -> float:
    """Compute friction term gbar_i(Q) for a single pipe.

    gbar_i(Q) = -(f_i(Q) / (2 * D_i * Area_i)) * Q * |Q|

    For laminar flow (Re <= 2000) under Darcy-Weisbach mode, the expression
    (64/Re) * Q*|Q| reduces to (32 * nu / D^2) * Q, which is strictly linear
    in Q and avoids any 0/0 indeterminacy at Q = 0.
    """
    if settings is None:
        settings = PhysicsSettings()

    q = float(Q)
    d = float(diameter)
    eps = float(roughness)
    nu = float(settings.nu)
    area = math.pi * (d ** 2) / 4.0

    mode = settings.friction_mode
    if mode == "constant":
        ff = float(settings.friction_factor)
        coeff = -(ff / (2.0 * d * area))
        return coeff * q * abs(q)

    if mode == "swamee_jain":
        if abs(q) < 1e-16:
            return 0.0
        ff = friction_factor(d, eps, q, settings)
        coeff = -(ff / (2.0 * d * area))
        return coeff * q * abs(q)

    if mode == "darcy_weisbach":
        s = nu * d
        w = abs(q) / s if s > 0 else 0.0
        if w <= _DW_A2:
            # Laminar regime: (64 / Re) / (2 * D * Area) * Q * |Q| = 32 * nu / D^2 * Q
            return -(32.0 * nu / (d ** 2)) * q

        ff = friction_factor(d, eps, q, settings)
        coeff = -(ff / (2.0 * d * area))
        return coeff * q * abs(q)

    raise ValueError(f"Unknown friction_mode: {mode!r}")
