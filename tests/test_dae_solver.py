"""tests/test_dae_solver.py – Comprehensive unit tests for Theorem 8 DAE solver."""

import time
import math
import numpy as np
import pytest

from core.models import Network
from core.physics import PhysicsSettings
from core.scenario import Scenario, SwitchEvent, ValveId
from core.dae_model import build_switched_dae, steady_state, UnsupportedNetworkError
from core.dae_solver import (
    simulate,
    analyze_mode,
    rest_state,
    orthT,
    nullT,
    rnk,
    preimg,
    extendBasis,
    SolverError,
    ScenarioError,
)


def _build_specimen_network(friction_mode: str = "constant") -> Network:
    """Build the MATLAB reference specimen network."""
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


def test_linear_algebra_helpers():
    """Verify faithful behavior of orthT, nullT, rnk, preimg, extendBasis."""
    # OrthT and NullT on known matrix
    A = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=float)
    assert rnk(A) == 1

    O = orthT(A)
    assert O.shape == (2, 1)
    assert pytest.approx(1.0) == float(np.linalg.norm(O))

    N = nullT(A)
    assert N.shape == (2, 1)
    assert pytest.approx(0.0, abs=1e-10) == float(np.linalg.norm(A @ N))

    # Preimage: { y : A y in span(O) }
    P = preimg(A, O)
    assert P.shape == (2, 2)  # full space R^2 maps into col space of A

    # Extend basis
    B = np.array([[1.0], [0.0], [0.0]])
    Tg = np.array([[2.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    ext = extendBasis(B, Tg)
    assert ext.shape == (3, 1)
    combined = np.hstack([B, ext])
    assert rnk(combined) == 2


def test_scenario_2():
    """Step 3(a): Scenario 2 (no switch, from rest, end 200 s)."""
    net = _build_specimen_network()
    dae = build_switched_dae(net)
    v1 = ValveId("P2", "start")
    v2 = ValveId("P2", "end")

    sc2 = Scenario("Scenario 2", initial={v1: 1, v2: 1}, t_end=200.0)
    res = simulate(dae, sc2)

    # Reference numbers
    expected = {
        10.0: 0.072295,
        20.0: 0.127168,
        30.0: 0.160730,
        60.0: 0.191668,
        100.0: 0.195156,
        200.0: 0.195321,
    }
    for t_val, q_exp in expected.items():
        state = res.at(t_val)
        Q1, Q2 = state[0], state[1]
        assert pytest.approx(q_exp, abs=5e-6) == Q1
        assert pytest.approx(q_exp, abs=5e-6) == Q2

    # Heads x5 (Pipe_1_R) and x4 (Pipe_2_L) = 42.5385 at every sample
    x5 = res.series("Pipe_1_R")
    x4 = res.series("Pipe_2_L")
    assert np.allclose(x5, 42.5385, atol=1e-3)
    assert np.allclose(x4, 42.5385, atol=1e-3)

    # Impulses list is empty or all-zero
    assert len(res.impulses) == 0


def test_scenario_1():
    """Step 3(b): Scenario 1 (V1 closes at 60 s, end 120)."""
    net = _build_specimen_network()
    dae = build_switched_dae(net)
    v1 = ValveId("P2", "start")
    v2 = ValveId("P2", "end")

    sc1 = Scenario("Scenario 1", initial={v1: 1, v2: 1}, events=[SwitchEvent(60.0, v1, 0)], t_end=120.0)
    res = simulate(dae, sc1)

    # At t=60: Q(60-) = 0.191668
    assert len(res.impulses) == 1
    imp = res.impulses[0]
    assert imp.time == 60.0
    assert pytest.approx(0.191668, abs=5e-6) == imp.x_minus[0]
    assert imp.cross_check_error < 1e-9

    # Impulse on x5 (Pipe_1_R) ~ +62.16 and on x4 (Pipe_2_L) ~ -38.85
    # Note: J1 is tied upstream of V1 so it carries the same impulse as Pipe_1_R (+62.16)
    idx_p1r = dae.labels.index("Pipe_1_R")
    idx_p2l = dae.labels.index("Pipe_2_L")
    idx_j1 = dae.labels.index("J1")

    assert pytest.approx(62.16, abs=0.1) == imp.coefficients[idx_p1r]
    assert pytest.approx(-38.85, abs=0.1) == imp.coefficients[idx_p2l]
    assert pytest.approx(62.16, abs=0.1) == imp.coefficients[idx_j1]

    # Flow states and reservoir states carry no impulse
    for idx in [0, 1, dae.labels.index("Pipe_1_L"), dae.labels.index("Pipe_2_R"), dae.labels.index("R1"), dae.labels.index("R2")]:
        assert abs(imp.coefficients[idx]) < 1e-9

    # After switch (t > 60): Q1 = Q2 = 0, x5 = 45.0, x4 = 41.0 constant
    post_mask = res.t > 60.0
    assert np.allclose(res.series("Pipe_1_Flow")[post_mask], 0.0, atol=1e-8)
    assert np.allclose(res.series("Pipe_2_Flow")[post_mask], 0.0, atol=1e-8)
    assert np.allclose(res.series("Pipe_1_R")[post_mask], 45.0, atol=1e-3)
    assert np.allclose(res.series("Pipe_2_L")[post_mask], 41.0, atol=1e-3)


def test_scenario_3():
    """Step 3(c): Scenario 3 (equilibrium first, V1 closes at 300, end 400)."""
    net = _build_specimen_network()
    dae = build_switched_dae(net)
    v1 = ValveId("P2", "start")
    v2 = ValveId("P2", "end")

    sc3 = Scenario("Scenario 3", initial={v1: 1, v2: 1}, events=[SwitchEvent(300.0, v1, 0)], t_end=400.0)
    res = simulate(dae, sc3)

    assert len(res.impulses) == 1
    imp = res.impulses[0]
    assert imp.time == 300.0
    assert pytest.approx(0.195321, abs=1e-6) == imp.x_minus[0]

    # Impulses: x5 = +63.347 and x4 = -39.592
    idx_p1r = dae.labels.index("Pipe_1_R")
    idx_p2l = dae.labels.index("Pipe_2_L")
    assert pytest.approx(63.347, abs=1e-3) == imp.coefficients[idx_p1r]
    assert pytest.approx(-39.592, abs=1e-3) == imp.coefficients[idx_p2l]
    assert imp.cross_check_error < 1e-9

    # After switch: Q = 0, x5 = 45, x4 = 41
    post_mask = res.t > 300.0
    assert np.allclose(res.series("Pipe_1_Flow")[post_mask], 0.0, atol=1e-8)
    assert np.allclose(res.series("Pipe_1_R")[post_mask], 45.0, atol=1e-3)
    assert np.allclose(res.series("Pipe_2_L")[post_mask], 41.0, atol=1e-3)


def test_scenario_4():
    """Step 3(d): Scenario 4 (V2 closes at 300, end 400)."""
    net = _build_specimen_network()
    dae = build_switched_dae(net)
    v1 = ValveId("P2", "start")
    v2 = ValveId("P2", "end")

    sc4 = Scenario("Scenario 4", initial={v1: 1, v2: 1}, events=[SwitchEvent(300.0, v2, 0)], t_end=400.0)
    res = simulate(dae, sc4)

    assert len(res.impulses) == 1
    imp = res.impulses[0]
    assert imp.time == 300.0
    assert imp.cross_check_error < 1e-9

    idx_x5 = dae.labels.index("Pipe_1_R")
    idx_x4 = dae.labels.index("Pipe_2_L")
    idx_x7 = dae.labels.index("J1")
    idx_x6 = dae.labels.index("Pipe_2_R")

    # Independent hand derivation:
    # x5 = x4 = x7 = +63.347 (= Q / b1)
    # x6 = +102.94 (= Q * (1/b1 + 1/b2))
    assert pytest.approx(63.347, abs=1e-3) == imp.coefficients[idx_x5]
    assert pytest.approx(63.347, abs=1e-3) == imp.coefficients[idx_x4]
    assert pytest.approx(63.347, abs=1e-3) == imp.coefficients[idx_x7]
    assert pytest.approx(102.94, abs=0.01) == imp.coefficients[idx_x6]

    # After switch: Q = 0, x5 = x4 = 45, x6 = 50
    post_mask = res.t > 300.0
    assert np.allclose(res.series("Pipe_1_Flow")[post_mask], 0.0, atol=1e-8)
    assert np.allclose(res.series("Pipe_1_R")[post_mask], 45.0, atol=1e-3)
    assert np.allclose(res.series("Pipe_2_L")[post_mask], 45.0, atol=1e-3)
    assert np.allclose(res.series("Pipe_2_R")[post_mask], 50.0, atol=1e-3)


def test_scenario_5():
    """Step 3(e): Scenario 5 (start with V1 closed, V1 opens at 60 s, end 300)."""
    net = _build_specimen_network()
    dae = build_switched_dae(net)
    v1 = ValveId("P2", "start")
    v2 = ValveId("P2", "end")

    sc5 = Scenario("Scenario 5", initial={v1: 0, v2: 1}, events=[SwitchEvent(60.0, v1, 1)], t_end=300.0)
    res = simulate(dae, sc5)

    # Before 60: Q = 0, x5 = 45, x4 = 41 (rest state of mode (0,1))
    pre_mask = res.t < 60.0
    assert np.allclose(res.series("Pipe_1_Flow")[pre_mask], 0.0, atol=1e-8)
    assert np.allclose(res.series("Pipe_1_R")[pre_mask], 45.0, atol=1e-3)
    assert np.allclose(res.series("Pipe_2_L")[pre_mask], 41.0, atol=1e-3)

    # Switch at 60: no Dirac impulse (algebraic jump only)
    assert len(res.impulses) == 1
    imp = res.impulses[0]
    assert np.all(np.abs(imp.coefficients) < 1e-9)

    # At t=60, x5 and x4 jump to 42.5385 and Q(60) = 0
    idx_switch = np.where(res.t == 60.0)[0]
    assert len(idx_switch) == 2
    state_plus = res.X[idx_switch[1]]
    assert pytest.approx(0.0, abs=1e-8) == state_plus[0]
    assert pytest.approx(42.5385, abs=1e-3) == state_plus[dae.labels.index("Pipe_1_R")]
    assert pytest.approx(42.5385, abs=1e-3) == state_plus[dae.labels.index("Pipe_2_L")]

    # Q(60 + tau)
    tau_checks = {
        10.0: 0.072295,
        50.0: 0.187462,
        100.0: 0.195156,
        240.0: 0.195321,
    }
    for tau, q_exp in tau_checks.items():
        state = res.at(60.0 + tau)
        assert pytest.approx(q_exp, abs=5e-6) == state[0]


def test_consistency_checks_scenarios_1_3_5():
    """Step 3(f): Consistency checks on scenarios 1, 3, 5."""
    net = _build_specimen_network()
    dae = build_switched_dae(net)
    v1 = ValveId("P2", "start")
    v2 = ValveId("P2", "end")

    scenarios = [
        Scenario("Scenario 1", initial={v1: 1, v2: 1}, events=[SwitchEvent(60.0, v1, 0)], t_end=120.0),
        Scenario("Scenario 3", initial={v1: 1, v2: 1}, events=[SwitchEvent(300.0, v1, 0)], t_end=400.0),
        Scenario("Scenario 5", initial={v1: 0, v2: 1}, events=[SwitchEvent(60.0, v1, 1)], t_end=300.0),
    ]

    for sc in scenarios:
        res = simulate(dae, sc)
        t_switch = sc.events[0].time
        idx_switch = np.where(res.t == t_switch)[0]
        i_split = idx_switch[1]

        seg_slices = [(res.t[:i_split], res.X[:i_split], res.segments[0][2]),
                      (res.t[i_split:], res.X[i_split:], res.segments[1][2])]

        for t_seg, X_seg, mode_state in seg_slices:
            E, A, f = dae.matrices(mode_state)
            alg_rows = np.where(np.all(E == 0, axis=1))[0]
            diff_rows = np.where(np.any(E != 0, axis=1))[0]

            # (i) Algebraic row residual < 1e-8 at every sample
            for x in X_seg:
                xi = dae.M @ x
                res_val = A @ x + dae.N @ dae.friction(xi) + f
                assert float(np.max(np.abs(res_val[alg_rows]))) < 1e-8

            # (ii) Centered finite difference on differential rows (away from boundaries)
            dt = t_seg[1] - t_seg[0]
            if len(t_seg) > 80 and len(diff_rows) > 0:
                for i in range(40, len(t_seg) - 40):
                    x_dot = (X_seg[i+1] - X_seg[i-1]) / (2.0 * dt)
                    lhs = E @ x_dot
                    x = X_seg[i]
                    rhs = A @ x + dae.N @ dae.friction(dae.M @ x) + f
                    for r in diff_rows:
                        denom = max(abs(rhs[r]), 1e-4)
                        rel_err = abs(lhs[r] - rhs[r]) / denom
                        assert rel_err < 1e-4

    # (iii) Last state of Scenarios 2 and 3 equals steady_state within 1e-5
    sc2 = Scenario("Scenario 2", initial={v1: 1, v2: 1}, t_end=200.0)
    res2 = simulate(dae, sc2)
    x_ss_2 = steady_state(dae, (1, 1))
    assert np.allclose(res2.X[-1], x_ss_2, atol=1e-5)

    sc3 = Scenario("Scenario 3", initial={v1: 1, v2: 1}, events=[SwitchEvent(300.0, v1, 0)], t_end=400.0)
    res3 = simulate(dae, sc3)
    x_ss_3 = steady_state(dae, (0, 1))
    assert np.allclose(res3.X[-1], x_ss_3, atol=1e-5)


def test_cross_check_and_timing_specimens():
    """Step 3(g): Formula (6) cross-check error < 1e-9; all 5 scenarios run under 10s."""
    net = _build_specimen_network()
    dae = build_switched_dae(net)
    v1 = ValveId("P2", "start")
    v2 = ValveId("P2", "end")

    scenarios = {
        1: Scenario("Sc1", initial={v1: 1, v2: 1}, events=[SwitchEvent(60.0, v1, 0)], t_end=120.0),
        2: Scenario("Sc2", initial={v1: 1, v2: 1}, t_end=200.0),
        3: Scenario("Sc3", initial={v1: 1, v2: 1}, events=[SwitchEvent(300.0, v1, 0)], t_end=400.0),
        4: Scenario("Sc4", initial={v1: 1, v2: 1}, events=[SwitchEvent(300.0, v2, 0)], t_end=400.0),
        5: Scenario("Sc5", initial={v1: 0, v2: 1}, events=[SwitchEvent(60.0, v1, 1)], t_end=300.0),
    }

    for sc_id, sc in scenarios.items():
        t0 = time.perf_counter()
        res = simulate(dae, sc)
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0, f"Scenario {sc_id} took {elapsed:.2f}s > 10s"

        for imp in res.impulses:
            assert imp.cross_check_error < 1e-9


def test_friction_laws_simulation():
    """Step 3(h): Scenario 1 simulation with Swamee-Jain friction."""
    net = _build_specimen_network()
    v1 = ValveId("P2", "start")
    v2 = ValveId("P2", "end")
    sc1 = Scenario("Scenario 1", initial={v1: 1, v2: 1}, events=[SwitchEvent(60.0, v1, 0)], t_end=120.0)

    # Constant friction
    dae_const = build_switched_dae(net, PhysicsSettings(friction_mode="constant"))
    res_const = simulate(dae_const, sc1)
    q60_const = res_const.impulses[0].x_minus[0]

    # Swamee-Jain
    dae_sj = build_switched_dae(net, PhysicsSettings(friction_mode="swamee_jain"))
    res_sj = simulate(dae_sj, sc1)
    q60_sj = res_sj.impulses[0].x_minus[0]

    assert pytest.approx(0.191668, abs=5e-6) == q60_const
    assert pytest.approx(0.192477, abs=5e-5) == q60_sj
    assert abs(q60_sj - q60_const) < 0.002


def test_generic_topology():
    """Step 3(i): Generic topology (R1 -> J1, J1 -> J2, J1 -> R2, J2 -> R2)."""
    net = Network()
    net.add_node("reservoir", id="R1", elevation=30.0, head=60.0)
    net.add_node("junction", id="J1", elevation=15.0)
    net.add_node("junction", id="J2", elevation=10.0)
    net.add_node("reservoir", id="R2", elevation=0.0, head=20.0)

    net.add_pipe("R1", "J1", id="P1", length=300.0, diameter=0.35, roughness=0.0002, valve_start=True)
    net.add_pipe("J1", "J2", id="P2", length=200.0, diameter=0.30, roughness=0.0002)
    net.add_pipe("J1", "R2", id="P3", length=400.0, diameter=0.30, roughness=0.0002)
    net.add_pipe("J2", "R2", id="P4", length=250.0, diameter=0.25, roughness=0.0002)

    dae = build_switched_dae(net)
    v1 = ValveId("P1", "start")

    # All open: simulate for 100s
    sc_open = Scenario("All open", initial={v1: 1}, t_end=100.0)
    res_open = simulate(dae, sc_open)

    x_ss = steady_state(dae, {v1: 1})
    assert np.allclose(res_open.X[-1], x_ss, atol=1e-4)

    # Junction continuity: J1 (Q1 - Q2 - Q3 = 0), J2 (Q2 - Q4 = 0)
    q_end = res_open.X[-1]
    j1_res = q_end[0] - q_end[1] - q_end[2]
    j2_res = q_end[1] - q_end[3]
    assert abs(j1_res) < 1e-9
    assert abs(j2_res) < 1e-9

    # Close valve at start of P1 at t = 50s: must run or raise SolverError cleanly
    sc_close = Scenario("Close at 50", initial={v1: 1}, events=[SwitchEvent(50.0, v1, 0)], t_end=100.0)
    try:
        res_close = simulate(dae, sc_close)
        assert len(res_close.t) > 0
    except SolverError as e:
        assert isinstance(str(e), str) and len(str(e)) > 0


def test_readable_error_paths():
    """Step 3(j): Errors with plain-language readable messages (no matrix jargon in 1st sentence)."""
    net = _build_specimen_network()
    dae = build_switched_dae(net)
    v1 = ValveId("P2", "start")
    v2 = ValveId("P2", "end")

    # 1. Unknown valve in scenario
    v_unknown = ValveId("P99", "start")
    sc_bad = Scenario("Unknown valve", initial={v1: 1, v2: 1, v_unknown: 1}, t_end=10.0)
    with pytest.raises(ScenarioError) as exc_info:
        simulate(dae, sc_bad)
    msg = str(exc_info.value)
    assert "does not exist in the network" in msg

    # 2. Scenario reaching singular mode (0,0)
    sc_00 = Scenario("Mode 0,0", initial={v1: 0, v2: 0}, t_end=10.0)
    with pytest.raises(ScenarioError) as exc_info:
        simulate(dae, sc_00)
    msg = str(exc_info.value)
    # Plain text asserted: isolates part of network / no unique solution
    assert "no unique solution" in msg

    # 3. Pump network raises UnsupportedNetworkError with readable message
    net_pump = _build_specimen_network()
    net_pump.pipes[0].has_pump = True
    with pytest.raises(UnsupportedNetworkError) as exc_info:
        build_switched_dae(net_pump)
    msg = str(exc_info.value)
    assert "Pumps are not supported" in msg


def test_simulation_result_helpers():
    """Verify helpers on SimulationResult: series, pressure_head, total_head, at."""
    net = _build_specimen_network()
    dae = build_switched_dae(net)
    v1 = ValveId("P2", "start")
    sc2 = Scenario("Scenario 2", initial={v1: 1, ValveId("P2", "end"): 1}, t_end=50.0)
    res = simulate(dae, sc2)

    # Series
    q1 = res.series("Pipe_1_Flow")
    assert len(q1) == len(res.t)

    # Pressure head on junction J1
    p_j1 = res.pressure_head("J1")
    assert np.all(p_j1 > 0)

    # Total head on J1 (pressure + elevation 5.0)
    h_j1 = res.total_head("J1")
    assert np.allclose(h_j1, p_j1 + 5.0)

    # Flow states have no pressure or total head
    with pytest.raises(ValueError, match="has no pressure head"):
        res.pressure_head("Pipe_1_Flow")
    with pytest.raises(ValueError, match="has no hydraulic head"):
        res.total_head("Pipe_1_Flow")

    # At helper
    sample = res.at(25.0)
    assert len(sample) == dae.K
