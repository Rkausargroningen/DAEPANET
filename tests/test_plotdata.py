"""Tests for core/plotdata.py – plot series, jump pairs, differences, and impulse arrows."""

import pytest
import numpy as np

from core.examples import specimen_network, specimen_scenario, specimen_physics
from core.comparison import run_comparison
from core.plotdata import (
    default_states,
    series,
    difference_rows,
    max_abs_difference_after_last_switch,
    impulse_arrows,
    engine_label,
    decimate_series,
)


@pytest.fixture(scope="module")
def specimen_res_sc1():
    net, pos = specimen_network()
    sc = specimen_scenario(1)
    phys = specimen_physics()
    return run_comparison(net, sc, phys, positions=pos)


def test_default_states_order(specimen_res_sc1):
    res = specimen_res_sc1
    states = default_states(res)
    assert states == ["Pipe_1_Flow", "Pipe_2_Flow", "Pipe_1_R", "Pipe_2_L", "Pipe_2_R"]


def test_jump_pairs_scenario1(specimen_res_sc1):
    res = specimen_res_sc1

    # 1. Pipe_1_R at t=60: before 42.5385, after 45.0000
    dae_segs, jumps_p1r, (te, ye) = series(res, "Pipe_1_R", kind="pressure")
    assert len(jumps_p1r) == 1
    t_s, y_before, y_after = jumps_p1r[0]
    assert t_s == 60.0
    assert y_before == pytest.approx(42.5385, abs=1e-4)
    assert y_after == pytest.approx(45.0000, abs=1e-4)

    # 2. Pipe_2_L at t=60: before 42.5385, after 41.0000
    _, jumps_p2l, _ = series(res, "Pipe_2_L", kind="pressure")
    assert len(jumps_p2l) == 1
    t_s, y_before, y_after = jumps_p2l[0]
    assert t_s == 60.0
    assert y_before == pytest.approx(42.5385, abs=1e-4)
    assert y_after == pytest.approx(41.0000, abs=1e-4)

    # 3. Pipe_1_Flow (Q1) at t=60: before 0.191668, after 0.0
    _, jumps_q1, _ = series(res, "Pipe_1_Flow", kind="pressure")
    assert len(jumps_q1) == 1
    t_s, y_before, y_after = jumps_q1[0]
    assert t_s == 60.0
    assert y_before == pytest.approx(0.191668, abs=1e-5)
    assert y_after == pytest.approx(0.0, abs=1e-6)


def test_difference_row_t0_and_last_switch(specimen_res_sc1):
    res = specimen_res_sc1

    # Difference rows
    rows = difference_rows(res, labels=["Pipe_1_Flow", "Pipe_1_R"])
    assert len(rows) > 0

    # At t=0 for Q1 (Pipe_1_Flow): DAE 0.0, EPANET ~0.19533
    row_t0_q1 = next(r for r in rows if abs(r[0] - 0.0) < 1e-6 and r[1] == "Pipe_1_Flow")
    t, lbl, dae_val, ep_val, diff = row_t0_q1
    assert dae_val == pytest.approx(0.0, abs=1e-6)
    assert ep_val == pytest.approx(0.19533, abs=1e-3)
    assert diff == pytest.approx(dae_val - ep_val, abs=1e-9)

    # After last switch: maximum difference is below 1e-4 (scenario 1: about 5e-9)
    max_diff = max_abs_difference_after_last_switch(res)
    assert max_diff < 1e-4
    assert max_diff == pytest.approx(4.7e-9, abs=2e-9)


def test_total_head_offset(specimen_res_sc1):
    res = specimen_res_sc1

    # Pipe_1_R elevation is 5.0 (J1 elevation)
    # Total head should equal pressure + 5.0
    segs_pres, jumps_pres, ep_pres = series(res, "Pipe_1_R", kind="pressure")
    segs_tot, jumps_tot, ep_tot = series(res, "Pipe_1_R", kind="total")

    for (t_p, y_p), (t_t, y_t) in zip(segs_pres, segs_tot):
        np.testing.assert_allclose(t_p, t_t)
        np.testing.assert_allclose(y_t, y_p + 5.0, atol=1e-9)

    for (ts_p, yb_p, ya_p), (ts_t, yb_t, ya_t) in zip(jumps_pres, jumps_tot):
        assert ts_p == ts_t
        assert yb_t == pytest.approx(yb_p + 5.0, abs=1e-9)
        assert ya_t == pytest.approx(ya_p + 5.0, abs=1e-9)

    np.testing.assert_allclose(ep_tot[1], ep_pres[1] + 5.0, atol=1e-9)

    # Flows unaffected
    segs_q_p, _, ep_q_p = series(res, "Pipe_1_Flow", kind="pressure")
    segs_q_t, _, ep_q_t = series(res, "Pipe_1_Flow", kind="total")
    np.testing.assert_allclose(segs_q_p[0][1], segs_q_t[0][1], atol=1e-9)
    np.testing.assert_allclose(ep_q_p[1], ep_q_t[1], atol=1e-9)


def test_impulse_arrows_scenarios():
    net, pos = specimen_network()
    phys = specimen_physics()

    # Scenario 1: Pipe_1_R +62.16, Pipe_2_L -38.85
    res1 = run_comparison(net, specimen_scenario(1), phys, positions=pos)
    arrs1 = impulse_arrows(res1)
    assert len(arrs1["Pipe_1_R"]) == 1
    assert arrs1["Pipe_1_R"][0][1] == pytest.approx(62.1626, abs=1e-2)
    assert len(arrs1["Pipe_2_L"]) == 1
    assert arrs1["Pipe_2_L"][0][1] == pytest.approx(-38.8516, abs=1e-2)
    assert len(arrs1["Pipe_2_R"]) == 0

    # Scenario 2: no impulses
    res2 = run_comparison(net, specimen_scenario(2), phys, positions=pos)
    arrs2 = impulse_arrows(res2)
    assert all(len(v) == 0 for v in arrs2.values())

    # Scenario 3: +63.35, -39.59
    res3 = run_comparison(net, specimen_scenario(3), phys, positions=pos)
    arrs3 = impulse_arrows(res3)
    assert arrs3["Pipe_1_R"][0][1] == pytest.approx(63.35, abs=1e-2)
    assert arrs3["Pipe_2_L"][0][1] == pytest.approx(-39.59, abs=1e-2)

    # Scenario 4: x5 +63.35, x4 +63.35, x6 +102.94
    res4 = run_comparison(net, specimen_scenario(4), phys, positions=pos)
    arrs4 = impulse_arrows(res4)
    assert arrs4["Pipe_1_R"][0][1] == pytest.approx(63.35, abs=1e-2)
    assert arrs4["Pipe_2_L"][0][1] == pytest.approx(63.35, abs=1e-2)
    assert arrs4["Pipe_2_R"][0][1] == pytest.approx(102.94, abs=1e-2)


def test_engine_label(specimen_res_sc1):
    lbl = engine_label(specimen_res_sc1)
    assert "real EPANET 2.3.5" in lbl


def test_decimation_preserves_extrema_and_boundaries():
    # 1. Test with synthetic signal containing prominent peaks and sharp valleys
    n = 20000
    t = np.linspace(0.0, 100.0, n)
    y = np.sin(t) * np.exp(-t / 50.0)
    # Inject extreme spike at an arbitrary index
    y[5432] = 999.0
    y[12345] = -888.0

    t_dec, y_dec = decimate_series(t, y, max_points=2000)
    assert len(t_dec) <= 2000
    # First and last points preserved exactly
    assert t_dec[0] == t[0]
    assert y_dec[0] == y[0]
    assert t_dec[-1] == t[-1]
    assert y_dec[-1] == y[-1]
    # Extreme values preserved exactly
    assert np.max(y_dec) == 999.0
    assert np.min(y_dec) == -888.0

    # 2. Test with real Scenario 3 specimen data across all mode segments
    net, pos = specimen_network()
    phys = specimen_physics()
    res3 = run_comparison(net, specimen_scenario(3), phys, positions=pos)

    # For each state in default states, check every segment
    for st in default_states(res3):
        dae_segs, jumps, _ = series(res3, st, kind="pressure")
        for t_seg, y_seg in dae_segs:
            if len(t_seg) <= 4000:
                continue
            t_dec, y_dec = decimate_series(t_seg, y_seg, max_points=4000)
            assert len(t_dec) <= 4000
            # Boundary values preserved
            assert t_dec[0] == t_seg[0]
            assert y_dec[0] == y_seg[0]
            assert t_dec[-1] == t_seg[-1]
            assert y_dec[-1] == y_seg[-1]
            # Extrema preserved
            assert np.min(y_dec) == pytest.approx(np.min(y_seg), rel=1e-12)
            assert np.max(y_dec) == pytest.approx(np.max(y_seg), rel=1e-12)

