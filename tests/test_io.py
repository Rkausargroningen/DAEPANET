"""Tests for project file serialization and deserialization (core/io.py)."""

import json
import pytest
from core.models import Network
from core.matrices import build_matrices
from core.io import save_project, load_project


def test_save_load_round_trip(tmp_path):
    net = Network()
    net.add_node("reservoir", id="R1")
    net.add_node("junction", id="J1")
    net.add_node("junction", id="J2")
    net.add_pipe("R1", "J1", id="P1")
    net.add_pipe("J1", "J2", id="P2", has_pump=True, pump_power=12.0)

    positions = {"R1": (0.0, 0.0), "J1": (1.0, 2.0), "J2": (3.0, 4.0)}
    file_path = tmp_path / "test_net.wnet"

    save_project(file_path, net, positions)
    assert file_path.is_file()

    loaded_net, loaded_pos = load_project(file_path)

    # Check structure match
    assert len(loaded_net.nodes) == 3
    assert len(loaded_net.pipes) == 2
    assert loaded_pos == positions

    # Check matrix identity before mutating loaded_net
    orig_res = build_matrices(net.to_dict())
    loaded_res = build_matrices(loaded_net.to_dict())
    assert orig_res["labels"] == loaded_res["labels"]
    assert orig_res["E_matrix"] == loaded_res["E_matrix"]
    assert orig_res["A_matrix"] == loaded_res["A_matrix"]

    # Check ID counters restored
    next_j = loaded_net.add_node("junction")
    assert next_j.id == "J3"


def test_corrupt_file(tmp_path):
    file_path = tmp_path / "corrupt.wnet"
    file_path.write_text("{ invalid json ...", encoding="utf-8")

    with pytest.raises(ValueError, match="Corrupt JSON"):
        load_project(file_path)


def test_wrong_format(tmp_path):
    file_path = tmp_path / "wrong_format.wnet"
    data = {"format": "invalid-format", "version": 1, "nodes": [], "pipes": []}
    file_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid format identifier"):
        load_project(file_path)


def test_unknown_node_in_pipe(tmp_path):
    file_path = tmp_path / "bad_node.wnet"
    data = {
        "format": "water-network",
        "version": 1,
        "nodes": [{"id": "J1", "type": "junction"}],
        "pipes": [{"id": "P1", "start": "J1", "end": "NON_EXISTENT"}],
    }
    file_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="references unknown end node"):
        load_project(file_path)


def test_literal_v1_json_loads_correctly_with_none_fields(tmp_path):
    # Hand-written literal v1 JSON with extra unknown keys to verify backward compatibility and tolerance
    v1_json = """{
      "format": "water-network",
      "version": 1,
      "nodes": [
        {"id": "R1", "type": "reservoir", "capacity": 1000.0, "elevation": 50.0, "extra_node_meta": 999},
        {"id": "J1", "type": "junction", "capacity": 1000.0, "elevation": 45.0}
      ],
      "pipes": [
        {"id": "P1", "start": "R1", "end": "J1", "has_pump": false, "pump_power": 0.0, "extra_pipe_meta": "test"}
      ],
      "counters": {"J": 1, "R": 1, "P": 1, "T": 0},
      "positions": {
        "R1": [10.5, 20.5],
        "J1": [30.0, 40.0]
      },
      "extra_top_level": {"ignored": true}
    }"""
    file_path = tmp_path / "legacy_v1.wnet"
    file_path.write_text(v1_json, encoding="utf-8")

    loaded_net, loaded_pos = load_project(file_path)

    assert len(loaded_net.nodes) == 2
    assert len(loaded_net.pipes) == 1

    r1 = next(n for n in loaded_net.nodes if n.id == "R1")
    j1 = next(n for n in loaded_net.nodes if n.id == "J1")
    p1 = loaded_net.pipes[0]

    # Verify new fields are None on v1 load
    assert r1.head is None
    assert j1.head is None
    assert p1.length is None
    assert p1.diameter is None
    assert p1.roughness is None

    # Verify positions and existing fields loaded properly
    assert loaded_pos == {"R1": (10.5, 20.5), "J1": (30.0, 40.0)}
    assert r1.elevation == 50.0
    assert j1.elevation == 45.0


def test_v2_save_load_round_trip_preserves_new_fields(tmp_path):
    net = Network()
    net.add_node("reservoir", id="R1", elevation=20.0, head=50.0)
    net.add_node("junction", id="J1", elevation=15.0)
    net.add_pipe(
        "R1", "J1", id="P1",
        has_pump=True, pump_power=25.0,
        length=250.5, diameter=0.45, roughness=0.00015
    )

    positions = {"R1": (100.0, 200.0), "J1": (300.0, 400.0)}
    file_path = tmp_path / "v2_test.wnet"

    save_project(file_path, net, positions)

    # Inspect the saved JSON to verify format version 2
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["format"] == "water-network"
    assert data["version"] == 2
    assert data["nodes"][0]["head"] == 50.0
    assert data["pipes"][0]["length"] == 250.5
    assert data["pipes"][0]["diameter"] == 0.45
    assert data["pipes"][0]["roughness"] == 0.00015

    # Load back and verify round trip
    loaded_net, loaded_pos = load_project(file_path)
    assert loaded_pos == positions

    r_loaded = next(n for n in loaded_net.nodes if n.id == "R1")
    j_loaded = next(n for n in loaded_net.nodes if n.id == "J1")
    p_loaded = loaded_net.pipes[0]

    assert r_loaded.head == 50.0
    assert r_loaded.elevation == 20.0
    assert j_loaded.head is None
    assert j_loaded.elevation == 15.0

    assert p_loaded.id == "P1"
    assert p_loaded.has_pump is True
    assert p_loaded.pump_power == 25.0
    assert p_loaded.length == 250.5
    assert p_loaded.diameter == 0.45
    assert p_loaded.roughness == 0.00015


def test_unsupported_version_rejected(tmp_path):
    file_path = tmp_path / "unsupported_v3.wnet"
    data = {
        "format": "water-network",
        "version": 3,
        "nodes": [],
        "pipes": [],
    }
    file_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported project file version"):
        load_project(file_path)


def test_io_edge_bends_round_trip(tmp_path):
    from core.io import load_project_full

    net = Network()
    net.add_node("reservoir", id="R1")
    net.add_node("junction", id="J1")
    net.add_pipe("R1", "J1", id="P1")

    pos = {"R1": (0.0, 0.0), "J1": (1.0, 1.0)}
    bends = {"P1": 0.45}

    # 1. Save with bends
    file_with = tmp_path / "net_with_bends.wnet"
    save_project(file_with, net, pos, edge_bends=bends)

    with open(file_with, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert "edge_bends" in payload
    assert payload["edge_bends"] == {"P1": 0.45}

    data = load_project_full(file_with)
    assert data.edge_bends == {"P1": 0.45}
    assert data.positions == pos

    # load_project still returns 2-tuple (Network, dict)
    loaded_net, loaded_pos = load_project(file_with)
    assert len(loaded_net.nodes) == 2
    assert loaded_pos == pos

    # 2. Save without bends
    file_without = tmp_path / "net_without_bends.wnet"
    save_project(file_without, net, pos, edge_bends=None)

    with open(file_without, "r", encoding="utf-8") as f:
        payload_no = json.load(f)
    assert "edge_bends" not in payload_no

    data_no = load_project_full(file_without)
    assert data_no.edge_bends == {}


def test_io_unknown_pipe_id_ignored_in_bends(tmp_path):
    from core.io import load_project_full

    data = {
        "format": "water-network",
        "version": 2,
        "nodes": [{"id": "J1", "type": "junction"}, {"id": "J2", "type": "junction"}],
        "pipes": [{"id": "P1", "start": "J1", "end": "J2"}],
        "edge_bends": {"P1": -0.3, "NON_EXISTENT_PIPE": 0.8},
    }
    file_path = tmp_path / "unknown_pipe_bends.wnet"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    proj = load_project_full(file_path)
    assert proj.edge_bends == {"P1": -0.3}
    assert "NON_EXISTENT_PIPE" not in proj.edge_bends


def test_io_corrupt_bend_values_raise_value_error(tmp_path):
    from core.io import load_project_full

    base_data = {
        "format": "water-network",
        "version": 2,
        "nodes": [{"id": "J1", "type": "junction"}, {"id": "J2", "type": "junction"}],
        "pipes": [{"id": "P1", "start": "J1", "end": "J2"}],
    }

    # Non-dict edge_bends
    f1 = tmp_path / "corrupt_bends_list.wnet"
    d1 = dict(base_data, edge_bends=[0.2])
    f1.write_text(json.dumps(d1), encoding="utf-8")
    with pytest.raises(ValueError, match="'edge_bends' must be an object"):
        load_project_full(f1)

    # String bend value
    f2 = tmp_path / "corrupt_bends_str.wnet"
    d2 = dict(base_data, edge_bends={"P1": "invalid_number"})
    f2.write_text(json.dumps(d2), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a finite number"):
        load_project_full(f2)

    # Boolean bend value
    f3 = tmp_path / "corrupt_bends_bool.wnet"
    d3 = dict(base_data, edge_bends={"P1": True})
    f3.write_text(json.dumps(d3), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a finite number"):
        load_project_full(f3)

    # NaN / Inf bend value
    f4 = tmp_path / "corrupt_bends_nan.json"
    # Using raw string because json.dumps produces non-standard NaN/Infinity
    f4.write_text('{"format": "water-network", "version": 2, "nodes": [{"id": "J1", "type": "junction"}, {"id": "J2", "type": "junction"}], "pipes": [{"id": "P1", "start": "J1", "end": "J2"}], "edge_bends": {"P1": NaN}}', encoding="utf-8")
    with pytest.raises(ValueError, match="must be a finite number"):
        load_project_full(f4)




def test_valve_flags_and_scenario_io_round_trip(tmp_path):
    from core.scenario import ValveId, Scenario, SwitchEvent
    from core.io import save_project, load_project_full, load_project

    net = Network()
    net.add_node("reservoir", id="R1", elevation=20.0, head=50.0)
    net.add_node("junction", id="J1", elevation=5.0)
    # P1 has valve at start, P2 has valve at end
    p1 = net.add_pipe("R1", "J1", id="P1", length=100.0, diameter=0.3, roughness=0.0001, valve_start=True, valve_end=False)
    p2 = net.add_pipe("J1", "R1", id="P2", length=200.0, diameter=0.3, roughness=0.0001, valve_start=False, valve_end=True)

    v1 = ValveId("P1", "start")
    v2 = ValveId("P2", "end")
    sc = Scenario(
        title="Test IO Scenario",
        initial={v1: 1, v2: 0},
        events=[SwitchEvent(time=40.0, valve=v2, state=1)],
        t_end=100.0,
    )

    file_path = tmp_path / "valves_and_scenario.wnet"
    save_project(file_path, net, scenario=sc)

    # Inspect raw JSON: valve_start on P1 is True, valve_end is not present (or omitted when False)
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    pipe_p1_raw = next(p for p in raw["pipes"] if p["id"] == "P1")
    pipe_p2_raw = next(p for p in raw["pipes"] if p["id"] == "P2")
    assert pipe_p1_raw.get("valve_start") is True
    assert "valve_end" not in pipe_p1_raw
    assert "valve_start" not in pipe_p2_raw
    assert pipe_p2_raw.get("valve_end") is True
    assert "scenario" in raw
    assert raw["scenario"]["title"] == "Test IO Scenario"

    # load_project_full
    proj = load_project_full(file_path)
    assert len(proj.network.pipes) == 2
    loaded_p1 = next(p for p in proj.network.pipes if p.id == "P1")
    loaded_p2 = next(p for p in proj.network.pipes if p.id == "P2")
    assert loaded_p1.valve_start is True
    assert loaded_p1.valve_end is False
    assert loaded_p2.valve_start is False
    assert loaded_p2.valve_end is True

    assert proj.scenario is not None
    assert proj.scenario.title == sc.title
    assert proj.scenario.initial == sc.initial
    assert proj.scenario.events == sc.events

    # load_project still returns 2-tuple (network, positions)
    loaded_net, loaded_pos = load_project(file_path)
    assert isinstance(loaded_net, Network)
    assert isinstance(loaded_pos, dict)


def test_legacy_file_without_valves_or_scenario_loads_defaults(tmp_path):
    from core.io import load_project_full

    # Literal JSON without valve flags or scenario
    data = {
        "format": "water-network",
        "version": 2,
        "nodes": [{"id": "J1", "type": "junction"}, {"id": "J2", "type": "junction"}],
        "pipes": [{"id": "P1", "start": "J1", "end": "J2"}],
    }
    file_path = tmp_path / "no_valves.wnet"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    proj = load_project_full(file_path)
    assert proj.network.pipes[0].valve_start is False
    assert proj.network.pipes[0].valve_end is False
    assert proj.scenario is None


def test_corrupt_scenario_rejected(tmp_path):
    from core.io import load_project_full

    data = {
        "format": "water-network",
        "version": 2,
        "nodes": [{"id": "J1", "type": "junction"}, {"id": "J2", "type": "junction"}],
        "pipes": [{"id": "P1", "start": "J1", "end": "J2"}],
        "scenario": "not a dictionary",
    }
    file_path = tmp_path / "corrupt_scenario.wnet"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="'scenario' must be an object"):
        load_project_full(file_path)
