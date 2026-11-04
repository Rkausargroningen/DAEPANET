"""tests/test_physics.py – Tests for pipe friction laws and fluid mechanics formulas."""

import math
import numpy as np
import pytest

from core.physics import PhysicsSettings, friction_factor, pipe_friction_term
from core.models import Network
from core.dae_model import build_switched_dae, steady_state


def _build_specimen_network() -> Network:
    net = Network()
    net.add_node("reservoir", id="R1", elevation=20.0, head=50.0)
    net.add_node("junction", id="J1", elevation=5.0)
    net.add_node("reservoir", id="R2", elevation=0.0, head=46.0)

    net.add_pipe(
        "R1", "J1", id="P1",
        length=400.0, diameter=0.4, roughness=0.000359
    )
    net.add_pipe(
        "J1", "R2", id="P2",
        length=250.0, diameter=0.4, roughness=0.000359,
        valve_start=True, valve_end=True
    )
    return net


def test_physics_settings_modes():
    s1 = PhysicsSettings(friction_mode="constant", friction_factor=0.02)
    assert s1.friction_mode == "constant"
    assert s1.friction_factor == 0.02

    s2 = PhysicsSettings(friction_mode="swamee_jain")
    assert s2.friction_mode == "swamee_jain"

    s3 = PhysicsSettings(friction_mode="darcy_weisbach")
    assert s3.friction_mode == "darcy_weisbach"

    with pytest.raises(ValueError):
        PhysicsSettings(friction_mode="invalid")
    with pytest.raises(ValueError):
        PhysicsSettings(g=-1.0)
    with pytest.raises(ValueError):
        PhysicsSettings(nu=0.0)
    with pytest.raises(ValueError):
        PhysicsSettings(friction_factor=0.0)


def test_swamee_jain_formula_and_operating_point():
    """Swamee-Jain steady state of the specimen: Q = 0.196438, f = 0.019773, Re ~ 6.1e5."""
    settings = PhysicsSettings(friction_mode="swamee_jain")
    D = 0.40
    eps = 0.359e-3
    Q = 0.196438

    Re = 4.0 * Q / (math.pi * D * settings.nu)
    assert 6.1e5 <= Re <= 6.15e5

    f = friction_factor(D, eps, Q, settings)
    assert pytest.approx(0.019773, abs=1e-5) == f

    # Check zero flow does not produce NaN
    f_zero = friction_factor(D, eps, 0.0, settings)
    assert not math.isnan(f_zero)
    term_zero = pipe_friction_term(D, eps, 0.0, settings)
    assert term_zero == 0.0


def test_darcy_weisbach_epanet_rule():
    """EPANET Darcy-Weisbach rule: laminar 64/Re, transition Dunlop, turbulent Swamee-Jain."""
    settings = PhysicsSettings(friction_mode="darcy_weisbach")
    D = 0.40
    eps = 0.359e-3
    nu = settings.nu

    # 1. Laminar spot check: f(Re=1000) = 0.064
    Q_1000 = 1000.0 * (math.pi * D * nu) / 4.0
    f_1000 = friction_factor(D, eps, Q_1000, settings)
    assert pytest.approx(0.064, abs=1e-6) == f_1000

    # 2. Turbulent spot check: f(Re=1e6) equals Swamee-Jain value
    Q_1e6 = 1.0e6 * (math.pi * D * nu) / 4.0
    f_dw_1e6 = friction_factor(D, eps, Q_1e6, settings)
    settings_sj = PhysicsSettings(friction_mode="swamee_jain")
    f_sj_1e6 = friction_factor(D, eps, Q_1e6, settings_sj)
    assert pytest.approx(f_sj_1e6, abs=1e-12) == f_dw_1e6

    # 3. Finite and positive for Re in [1, 1e8]
    re_values = np.logspace(0, 8, 200)
    for re_val in re_values:
        q_val = re_val * (math.pi * D * nu) / 4.0
        f_val = friction_factor(D, eps, q_val, settings)
        assert not math.isnan(f_val)
        assert f_val > 0.0

    # 4. Zero flow behavior: no NaN, and laminar pipe friction term is strictly linear
    f_0 = friction_factor(D, eps, 0.0, settings)
    assert not math.isnan(f_0)
    term_0 = pipe_friction_term(D, eps, 0.0, settings)
    assert term_0 == 0.0

    # In laminar regime, gbar = -(32*nu / D^2) * Q
    q_small = 1e-5
    term_small = pipe_friction_term(D, eps, q_small, settings)
    expected_linear = -(32.0 * nu / (D ** 2)) * q_small
    assert pytest.approx(expected_linear, rel=1e-9) == term_small


def test_steady_state_swamee_jain_specimen():
    """Steady state with Swamee-Jain friction converges to Q = 0.196438 within 1e-5."""
    net = _build_specimen_network()
    settings = PhysicsSettings(friction_mode="swamee_jain")
    dae = build_switched_dae(net, settings)

    x_ss = steady_state(dae, (1, 1))
    Q1 = x_ss[0]
    Q2 = x_ss[1]
    assert pytest.approx(0.196438, abs=1e-5) == Q1
    assert pytest.approx(0.196438, abs=1e-5) == Q2


def test_steady_state_darcy_weisbach_specimen():
    """Steady state with Darcy-Weisbach friction operates in turbulent regime (Q ~ 0.196438)."""
    net = _build_specimen_network()
    settings = PhysicsSettings(friction_mode="darcy_weisbach")
    dae = build_switched_dae(net, settings)

    x_ss = steady_state(dae, (1, 1))
    Q1 = x_ss[0]
    assert pytest.approx(0.196438, abs=1e-5) == Q1
