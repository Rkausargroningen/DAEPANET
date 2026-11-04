"""tests/test_dae_model.py – Tests for numeric switched-DAE model, regularity, and steady-state solver."""

import math
import numpy as np
import pytest

from core.models import Network
from core.physics import PhysicsSettings
from core.scenario import ValveId, Scenario, SwitchEvent
from core.dae_model import (
    check_supported,
    build_switched_dae,
    steady_state,
    UnsupportedNetworkError,
)


def _build_specimen_network() -> Network:
    """Build the MATLAB reference specimen network (not part of examples.py)."""
    net = Network()
    net.add_node("reservoir", id="R1", elevation=20.0, head=50.0)
    net.add_node("junction", id="J1", elevation=5.0)
    net.add_node("reservoir", id="R2", elevation=0.0, head=46.0)

    # Pipe 1: R1 -> J1 (L=400, D=0.4, roughness=0.000359)
    net.add_pipe(
        "R1", "J1", id="P1",
        length=400.0, diameter=0.4, roughness=0.000359
    )
    # Pipe 2: J1 -> R2 (L=250, D=0.4, roughness=0.000359) with valves at both ends
    net.add_pipe(
        "J1", "R2", id="P2",
        length=250.0, diameter=0.4, roughness=0.000359,
        valve_start=True, valve_end=True
    )
    return net


def buildEA2_matlab(s1: int, s2: int, b1: float, b2: float, dz1: float, dz2: float, R1: float, R2: float):
    """Helper copied directly from reference/switchedDAE3_vs_EPANET.m (line 311)."""
    E = np.zeros((9, 9), dtype=float)
    E[0, 0] = 1.0
    E[1, 1] = 1.0
    A = np.array([
        [0.0,     0.0,  b1,  0.0, -b1,  0.0,  0.0,  0.0,  0.0],
        [0.0,     0.0, 0.0,   b2, 0.0,  -b2,  0.0,  0.0,  0.0],
        [1.0-s1,  0.0, 0.0,   s1, 0.0,  0.0,  -s1,  0.0,  0.0],
        [0.0,  1.0-s2, 0.0,  0.0, 0.0,   s2,  0.0,  0.0,  -s2],
        [0.0,     0.0, 1.0,  0.0, 0.0,  0.0,  0.0, -1.0,  0.0],
        [0.0,     0.0, 0.0,  0.0, 1.0,  0.0, -1.0,  0.0,  0.0],
        [1.0,    -1.0, 0.0,  0.0, 0.0,  0.0,  0.0,  0.0,  0.0],
        [0.0,     0.0, 0.0,  0.0, 0.0,  0.0,  0.0,  1.0,  0.0],
        [0.0,     0.0, 0.0,  0.0, 0.0,  0.0,  0.0,  0.0,  1.0],
    ], dtype=float)
    f = np.zeros(9, dtype=float)
    f[0] = dz1
    f[1] = dz2
    f[7] = R1
    f[8] = R2
    return E, A, f


def test_specimen_labels_and_reference_numbers():
    net = _build_specimen_network()
    settings = PhysicsSettings(friction_mode="constant", friction_factor=0.02)
    model = build_switched_dae(net, settings)

    # 1. Labels equal
    expected_labels = [
        'Pipe_1_Flow', 'Pipe_2_Flow',
        'Pipe_1_L', 'Pipe_2_L',
        'Pipe_1_R', 'Pipe_2_R',
        'J1', 'R1', 'R2'
    ]
    assert model.labels == expected_labels

    # 2. Dimensions and valves
    assert model.K == 9
    assert model.valves == [ValveId("P2", "start"), ValveId("P2", "end")]

    # 3. Reference numbers comparison (tol 1e-6)
    E, A, f = model.matrices()  # all-open mode
    Area = math.pi * 0.4**2 / 4.0
    b1_expected = 32.2 * 0.3048 * Area / 400.0
    b2_expected = 32.2 * 0.3048 * Area / 250.0

    assert math.isclose(b1_expected, 3.083335e-03, rel_tol=1e-6)
    assert math.isclose(b2_expected, 4.933336e-03, rel_tol=1e-6)
    assert math.isclose(f[0], 4.625002e-02, rel_tol=1e-6)
    assert math.isclose(f[1], 2.466668e-02, rel_tol=1e-6)
    assert math.isclose(f[7], -30.0, rel_tol=1e-6)
    assert math.isclose(f[8], -46.0, rel_tol=1e-6)

    # E is diagonal with 1 on first 2 elements
    assert np.all(np.diag(E) == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.count_nonzero(E - np.diag(np.diag(E))) == 0


def test_matrix_equality_with_matlab_buildEA2():
    """Verify that after row permutation and junction continuity substitution,
    the app's (E, A, f) matrices match MATLAB's buildEA2 bitwise for modes (1,1), (0,1), (1,0).

    Permutation: perm = [0, 1, 5, 3, 4, 2, 6, 7, 8]
    (MATLAB row k = app row perm[k])
    Row 2 adjustment: add (1 - s1) * row 6 (junction continuity), because MATLAB uses
    Q1 in valve 1 row whereas this app uses Q2.
    """
    net = _build_specimen_network()
    settings = PhysicsSettings(friction_mode="constant", friction_factor=0.02)
    model = build_switched_dae(net, settings)

    v1 = ValveId("P2", "start")
    v2 = ValveId("P2", "end")

    Area = math.pi * 0.4**2 / 4.0
    g = 32.2 * 0.3048
    b1 = g * Area / 400.0
    b2 = g * Area / 250.0
    dz1 = b1 * 15.0
    dz2 = b2 * 5.0
    R1 = -30.0
    R2 = -46.0

    perm = [0, 1, 5, 3, 4, 2, 6, 7, 8]

    for s1, s2 in [(1, 1), (0, 1), (1, 0)]:
        E_app, A_app, f_app = model.matrices({v1: s1, v2: s2})
        E_mat, A_mat, f_mat = buildEA2_matlab(s1, s2, b1, b2, dz1, dz2, R1, R2)

        # Permute app rows
        E_perm = E_app[perm, :]
        A_perm = A_app[perm, :].copy()
        f_perm = f_app[perm]

        # Add (1 - s1) * continuity row (row 6) to valve 1 row (row 2)
        A_perm[2, :] += (1.0 - s1) * A_perm[6, :]

        diff_E = np.max(np.abs(E_perm - E_mat))
        diff_A = np.max(np.abs(A_perm - A_mat))
        diff_f = np.max(np.abs(f_perm - f_mat))

        assert diff_E == 0.0, f"Mode ({s1}, {s2}) diff_E = {diff_E}"
        assert diff_A == 0.0, f"Mode ({s1}, {s2}) diff_A = {diff_A}"
        assert diff_f == 0.0, f"Mode ({s1}, {s2}) diff_f = {diff_f}"


def test_regularity_and_check_scenario():
    net = _build_specimen_network()
    settings = PhysicsSettings(friction_mode="constant", friction_factor=0.02)
    model = build_switched_dae(net, settings)

    v1 = ValveId("P2", "start")
    v2 = ValveId("P2", "end")

    assert model.is_regular({v1: 1, v2: 1}) is True
    assert model.is_regular({v1: 0, v2: 1}) is True
    assert model.is_regular({v1: 1, v2: 0}) is True
    assert model.is_regular({v1: 0, v2: 0}) is False

    # Scenarios 1 to 5 from matlab_specimen must all be accepted
    for k in range(1, 6):
        sc = Scenario.matlab_specimen(k, v1, v2)
        problems = model.check_scenario(sc)
        assert problems == [], f"Scenario {k} had problems: {problems}"

    # Scenario reaching (0, 0) must be flagged with plain-language problem
    bad_sc = Scenario(
        title="Reaches (0,0)",
        initial={v1: 1, v2: 0},
        events=[SwitchEvent(time=50.0, valve=v1, state=0)],
        t_end=100.0,
    )
    problems = model.check_scenario(bad_sc)
    assert len(problems) > 0
    assert any("all valves closed" in p or "singular" in p for p in problems)

    # Scenario with unknown valve must be flagged
    unknown_sc = Scenario(
        title="Unknown Valve",
        initial={v1: 1, ValveId("P99", "start"): 1},
        events=[],
        t_end=100.0,
    )
    problems_unknown = model.check_scenario(unknown_sc)
    assert any("does not exist in the network" in p for p in problems_unknown)


def test_steady_state_specimen_reference_values():
    net = _build_specimen_network()
    settings = PhysicsSettings(friction_mode="constant", friction_factor=0.02)
    model = build_switched_dae(net, settings)

    v1 = ValveId("P2", "start")
    v2 = ValveId("P2", "end")

    # Mode (1, 1):
    # Q1 = Q2 = 0.195321 m3/s, x3 = x8 = 30, x5 = x7 = x4 = 42.5385, x6 = x9 = 46.
    sol_11 = steady_state(model, {v1: 1, v2: 1})
    assert math.isclose(sol_11[0], 0.195321, rel_tol=1e-4)
    assert math.isclose(sol_11[1], 0.195321, rel_tol=1e-4)
    assert math.isclose(sol_11[2], 30.0, rel_tol=1e-4)     # Pl1 (x3)
    assert math.isclose(sol_11[7], 30.0, rel_tol=1e-4)     # R1  (x8)
    assert math.isclose(sol_11[4], 42.5385, rel_tol=1e-4)   # Pr1 (x5)
    assert math.isclose(sol_11[6], 42.5385, rel_tol=1e-4)   # J1  (x7)
    assert math.isclose(sol_11[3], 42.5385, rel_tol=1e-4)   # Pl2 (x4)
    assert math.isclose(sol_11[5], 46.0, rel_tol=1e-4)     # Pr2 (x6)
    assert math.isclose(sol_11[8], 46.0, rel_tol=1e-4)     # R2  (x9)

    # Mode (0, 1):
    # Q = 0, x3 = x8 = 30, x5 = x7 = 45, x4 = 41, x6 = x9 = 46.
    sol_01 = steady_state(model, {v1: 0, v2: 1})
    assert abs(sol_01[0]) < 1e-9
    assert abs(sol_01[1]) < 1e-9
    assert math.isclose(sol_01[2], 30.0, rel_tol=1e-4)     # Pl1 (x3)
    assert math.isclose(sol_01[7], 30.0, rel_tol=1e-4)     # R1  (x8)
    assert math.isclose(sol_01[4], 45.0, rel_tol=1e-4)     # Pr1 (x5)
    assert math.isclose(sol_01[6], 45.0, rel_tol=1e-4)     # J1  (x7)
    assert math.isclose(sol_01[3], 41.0, rel_tol=1e-4)     # Pl2 (x4)
    assert math.isclose(sol_01[5], 46.0, rel_tol=1e-4)     # Pr2 (x6)
    assert math.isclose(sol_01[8], 46.0, rel_tol=1e-4)     # R2  (x9)

    # Mode (1, 0):
    # Q = 0, x3 = x8 = 30, x5 = x7 = x4 = 45, x6 = 50, x9 = 46.
    sol_10 = steady_state(model, {v1: 1, v2: 0})
    assert abs(sol_10[0]) < 1e-9
    assert abs(sol_10[1]) < 1e-9
    assert math.isclose(sol_10[2], 30.0, rel_tol=1e-4)     # Pl1 (x3)
    assert math.isclose(sol_10[7], 30.0, rel_tol=1e-4)     # R1  (x8)
    assert math.isclose(sol_10[4], 45.0, rel_tol=1e-4)     # Pr1 (x5)
    assert math.isclose(sol_10[6], 45.0, rel_tol=1e-4)     # J1  (x7)
    assert math.isclose(sol_10[3], 45.0, rel_tol=1e-4)     # Pl2 (x4)
    assert math.isclose(sol_10[5], 50.0, rel_tol=1e-4)     # Pr2 (x6)
    assert math.isclose(sol_10[8], 46.0, rel_tol=1e-4)     # R2  (x9)

    # Mode (0, 0) raises ValueError
    with pytest.raises(ValueError, match="mode is singular"):
        steady_state(model, {v1: 0, v2: 0})


def test_generic_topology():
    """Generic topology: R1 -> J1, J1 -> J2, J1 -> R2, J2 -> R2 (a loop and a branch)."""
    net = Network()
    net.add_node("reservoir", id="R1", elevation=30.0, head=60.0)
    net.add_node("junction", id="J1", elevation=10.0)
    net.add_node("junction", id="J2", elevation=5.0)
    net.add_node("reservoir", id="R2", elevation=0.0, head=40.0)

    # 4 pipes
    net.add_pipe("R1", "J1", id="P1", length=500.0, diameter=0.35, roughness=0.0002)
    net.add_pipe("J1", "J2", id="P2", length=300.0, diameter=0.25, roughness=0.0002)
    net.add_pipe("J1", "R2", id="P3", length=400.0, diameter=0.30, roughness=0.0002)
    net.add_pipe("J2", "R2", id="P4", length=350.0, diameter=0.25, roughness=0.0002)

    settings = PhysicsSettings(friction_mode="constant", friction_factor=0.02)
    model = build_switched_dae(net, settings)

    assert model.K == 3 * 4 + 2 + 2  # 16 states
    E, A, f = model.matrices()
    assert E.shape == (16, 16)
    assert A.shape == (16, 16)
    assert len(f) == 16

    assert model.is_regular() is True

    sol = steady_state(model)
    assert len(sol) == 16

    # Verify continuity at J1: P1 arrives (+), P2 leaves (-), P3 leaves (-)
    # Q1 - Q2 - Q3 == 0
    q1, q2, q3, q4 = sol[0], sol[1], sol[2], sol[3]
    assert math.isclose(q1 - q2 - q3, 0.0, abs_tol=1e-9)

    # Verify continuity at J2: P2 arrives (+), P4 leaves (-)
    # Q2 - Q4 == 0
    assert math.isclose(q2 - q4, 0.0, abs_tol=1e-9)


def test_check_supported_reasons():
    # 1. Specimen is fully supported
    specimen = _build_specimen_network()
    assert check_supported(specimen) == []

    # 2. Network with pump gives readable reason
    net_pump = _build_specimen_network()
    net_pump.pipes[0].has_pump = True
    reasons_pump = check_supported(net_pump)
    assert len(reasons_pump) > 0
    assert any("Pumps are not supported" in r for r in reasons_pump)

    # 3. Network with tank gives readable reason
    net_tank = _build_specimen_network()
    net_tank.add_node("tank", id="T1", elevation=10.0, capacity=1000.0)
    reasons_tank = check_supported(net_tank)
    assert len(reasons_tank) > 0
    assert any("Tanks are not supported" in r for r in reasons_tank)

    # 4. Network with missing parameters gives readable reason
    net_missing = _build_specimen_network()
    net_missing.pipes[0].length = None
    reasons_missing = check_supported(net_missing)
    assert len(reasons_missing) > 0
    assert any("length is not set" in r for r in reasons_missing)

    # 5. Network without reservoir gives readable reason
    net_no_res = Network()
    net_no_res.add_node("junction", id="J1", elevation=10.0)
    net_no_res.add_node("junction", id="J2", elevation=5.0)
    net_no_res.add_pipe("J1", "J2", id="P1", length=100.0, diameter=0.3, roughness=0.0001)
    reasons_no_res = check_supported(net_no_res)
    assert len(reasons_no_res) > 0
    assert any("at least one reservoir" in r for r in reasons_no_res)

    # build_switched_dae raises UnsupportedNetworkError
    with pytest.raises(UnsupportedNetworkError):
        build_switched_dae(net_missing)


def test_friction_modes_supported():
    """Verify PhysicsSettings supports 'constant', 'swamee_jain', and 'darcy_weisbach'."""
    s_c = PhysicsSettings(friction_mode="constant")
    assert s_c.friction_mode == "constant"
    s_sj = PhysicsSettings(friction_mode="swamee_jain")
    assert s_sj.friction_mode == "swamee_jain"
    s_dw = PhysicsSettings(friction_mode="darcy_weisbach")
    assert s_dw.friction_mode == "darcy_weisbach"

    with pytest.raises(ValueError, match="Unknown friction_mode"):
        PhysicsSettings(friction_mode="unknown_mode")
