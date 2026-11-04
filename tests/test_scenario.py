"""tests/test_scenario.py – Unit tests for ValveId, SwitchEvent, Mode, and Scenario."""

import pytest
from core.scenario import ValveId, SwitchEvent, Mode, Scenario


def test_valve_id_properties_and_parsing():
    v1 = ValveId("P2", "start")
    assert v1.pipe_id == "P2"
    assert v1.end == "start"
    assert str(v1) == "P2:start"
    assert repr(v1) == "ValveId(pipe_id='P2', end='start')"

    v2 = ValveId.from_str("P5:end")
    assert v2.pipe_id == "P5"
    assert v2.end == "end"
    assert v2 == ValveId("P5", "end")

    # Invalid end
    with pytest.raises(ValueError, match="Valve end must be 'start' or 'end'"):
        ValveId("P1", "middle")

    # Invalid from_str
    with pytest.raises(ValueError, match="Invalid valve string representation"):
        ValveId.from_str("InvalidNoColon")


def test_switch_event_validation():
    v = ValveId("P1", "start")
    ev = SwitchEvent(time=10.0, valve=v, state=0)
    assert ev.time == 10.0
    assert ev.valve == v
    assert ev.state == 0

    with pytest.raises(ValueError, match="Switch event state must be 0 or 1"):
        SwitchEvent(time=10.0, valve=v, state=2)
    with pytest.raises(ValueError, match="Switch event state must be 0 or 1"):
        SwitchEvent(time=10.0, valve=v, state=True)
    with pytest.raises(ValueError, match="Switch event time must be a non-negative finite number"):
        SwitchEvent(time=-5.0, valve=v, state=1)


def test_scenario_validation_and_modes():
    v1 = ValveId("P2", "start")
    v2 = ValveId("P2", "end")

    # Valid scenario
    sc = Scenario(
        title="Test Scenario",
        initial={v1: 1, v2: 1},
        events=[
            SwitchEvent(time=60.0, valve=v1, state=0),
            SwitchEvent(time=30.0, valve=v2, state=0),
        ],
        t_end=120.0,
    )
    # Events should be sorted automatically
    assert sc.events[0].time == 30.0
    assert sc.events[1].time == 60.0

    modes = sc.modes()
    assert len(modes) == 3
    # Mode 1: [0, 30] state: v1=1, v2=1
    assert modes[0].t_start == 0.0
    assert modes[0].t_end == 30.0
    assert modes[0].state == {v1: 1, v2: 1}

    # Mode 2: [30, 60] state: v1=1, v2=0
    assert modes[1].t_start == 30.0
    assert modes[1].t_end == 60.0
    assert modes[1].state == {v1: 1, v2: 0}

    # Mode 3: [60, 120] state: v1=0, v2=0
    assert modes[2].t_start == 60.0
    assert modes[2].t_end == 120.0
    assert modes[2].state == {v1: 0, v2: 0}


def test_scenario_modes_grouping_and_dropping():
    v1 = ValveId("P1", "start")
    v2 = ValveId("P1", "end")

    # Events at same timestamp should group
    # Event that doesn't change state should drop
    sc = Scenario(
        title="Grouping and Dropping",
        initial={v1: 1, v2: 1},
        events=[
            SwitchEvent(time=20.0, valve=v1, state=1),  # already 1, no-op
            SwitchEvent(time=40.0, valve=v1, state=0),
            SwitchEvent(time=40.0, valve=v2, state=0),  # grouped at t=40
        ],
        t_end=100.0,
    )
    modes = sc.modes()
    assert len(modes) == 2
    assert modes[0].t_start == 0.0
    assert modes[0].t_end == 40.0
    assert modes[0].state == {v1: 1, v2: 1}

    assert modes[1].t_start == 40.0
    assert modes[1].t_end == 100.0
    assert modes[1].state == {v1: 0, v2: 0}


def test_scenario_validation_errors():
    v1 = ValveId("P1", "start")
    v2 = ValveId("P2", "end")

    # t_end <= 0
    with pytest.raises(ValueError, match="Scenario end time t_end must be > 0"):
        Scenario("Invalid", initial={v1: 1}, t_end=0)

    # Empty title
    with pytest.raises(ValueError, match="Scenario title must be a non-empty string"):
        Scenario("", initial={v1: 1}, t_end=100)

    # Event time out of bounds (<= 0 or >= t_end)
    with pytest.raises(ValueError, match="strictly inside"):
        Scenario("Invalid", initial={v1: 1}, events=[SwitchEvent(0.0, v1, 0)], t_end=100)
    with pytest.raises(ValueError, match="strictly inside"):
        Scenario("Invalid", initial={v1: 1}, events=[SwitchEvent(100.0, v1, 0)], t_end=100)

    # Valve in event without initial state
    with pytest.raises(ValueError, match="has no initial state"):
        Scenario("Invalid", initial={v1: 1}, events=[SwitchEvent(50.0, v2, 0)], t_end=100)


def test_scenario_to_from_dict_round_trip():
    v1 = ValveId("P2", "start")
    v2 = ValveId("P2", "end")
    sc = Scenario(
        title="Round Trip Scenario",
        initial={v1: 1, v2: 0},
        events=[SwitchEvent(time=45.0, valve=v2, state=1)],
        t_end=150.0,
    )
    d = sc.to_dict()
    assert d["title"] == "Round Trip Scenario"
    assert d["t_end"] == 150.0
    assert d["initial"] == {"P2:start": 1, "P2:end": 0}
    assert d["events"] == [{"time": 45.0, "valve": "P2:end", "state": 1}]

    reconstructed = Scenario.from_dict(d)
    assert reconstructed.title == sc.title
    assert reconstructed.t_end == sc.t_end
    assert reconstructed.initial == sc.initial
    assert reconstructed.events == sc.events


def test_matlab_specimen_scenarios():
    v1 = ValveId("P2", "start")
    v2 = ValveId("P2", "end")

    for k in range(1, 6):
        sc = Scenario.matlab_specimen(k, v1, v2)
        assert sc.title
        assert sc.t_end > 0
        modes = sc.modes()
        assert len(modes) >= 1

    with pytest.raises(ValueError, match="k must be in 1..5"):
        Scenario.matlab_specimen(0, v1, v2)
    with pytest.raises(ValueError, match="k must be in 1..5"):
        Scenario.matlab_specimen(6, v1, v2)
