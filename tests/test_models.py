"""Tests for Network model additions: remove_pipe, remove_node, clear, ID reset on clear, non-contiguous matrix building."""

import math
import pytest
from core.models import Network, Node, Pipe
from core.matrices import build_matrices
from core.checks import numeric_pencil, is_regular
from core.examples import example_paper_network, example_broken_network


def test_remove_pipe():
    net = Network()
    net.add_node("reservoir", "R1")
    net.add_node("junction", "J1")
    p1 = net.add_pipe("R1", "J1")
    assert len(net.pipes) == 1

    net.remove_pipe(p1.id)
    assert len(net.pipes) == 0

    with pytest.raises(ValueError, match="Unknown pipe ID"):
        net.remove_pipe("P99")


def test_remove_node_removes_attached_pipes():
    net = Network()
    net.add_node("reservoir", "R1")
    net.add_node("junction", "J1")
    net.add_node("junction", "J2")
    net.add_pipe("R1", "J1", id="P1")
    net.add_pipe("J1", "J2", id="P2")
    assert len(net.pipes) == 2

    # Removing J1 should remove P1 and P2
    net.remove_node("J1")
    assert len(net.nodes) == 2
    assert len(net.pipes) == 0

    with pytest.raises(ValueError, match="Unknown node ID"):
        net.remove_node("J99")


def test_clear_resets_counters():
    net = Network()
    net.add_node("reservoir")  # R1
    net.add_node("junction")   # J1
    net.add_pipe("R1", "J1")   # P1

    net.clear()
    assert len(net.nodes) == 0
    assert len(net.pipes) == 0

    # Next items created after clear must start again at J1, R1, P1
    j1 = net.add_node("junction")
    r1 = net.add_node("reservoir")
    p1 = net.add_pipe("R1", "J1")
    assert j1.id == "J1"
    assert r1.id == "R1"
    assert p1.id == "P1"


def test_individual_deletion_does_not_reuse_ids():
    net = Network()
    j1 = net.add_node("junction")  # J1
    j2 = net.add_node("junction")  # J2
    assert j1.id == "J1"
    assert j2.id == "J2"

    net.remove_node("J2")
    j3 = net.add_node("junction")  # Should be J3, NOT J2
    assert j3.id == "J3"

    p1 = net.add_pipe("J1", "J3")  # P1
    assert p1.id == "P1"
    net.remove_pipe("P1")
    p2 = net.add_pipe("J1", "J3")  # Should be P2, NOT P1
    assert p2.id == "P2"


def test_non_contiguous_ids_build_matrices_and_is_regular():
    net = Network()
    net.add_node("reservoir", "R1")
    net.add_node("junction", "J1")
    net.add_node("junction", "J2")
    net.add_node("junction", "J3")

    net.add_pipe("R1", "J1")  # P1
    net.add_pipe("J1", "J2")  # P2
    net.add_pipe("J2", "J3")  # P3

    # Remove middle junction J2 (removes P2 and P3)
    net.remove_node("J2")
    assert {n.id for n in net.nodes} == {"R1", "J1", "J3"}
    assert {p.id for p in net.pipes} == {"P1"}

    # Add a new pipe connecting J1 to J3
    net.add_pipe("J1", "J3")  # P4

    # Build matrices and verify regularity
    res = build_matrices(net.to_dict())
    assert res["K_dimension"] > 0
    E_num, A_num = numeric_pencil(res)
    assert is_regular(E_num, A_num)


def test_update_node_happy_path():
    net = Network()
    j1 = net.add_node("junction", "J1")
    r1 = net.add_node("reservoir", "R1")

    # Update junction elevation and capacity
    net.update_node("J1", elevation=25.0, capacity=500.0)
    assert j1.elevation == 25.0
    assert j1.capacity == 500.0

    # Update reservoir head, elevation, capacity
    net.update_node("R1", head=75.0, elevation=20.0, capacity=2000.0)
    assert r1.head == 75.0
    assert r1.elevation == 20.0
    assert r1.capacity == 2000.0

    # Setting fields to None (unset) is allowed
    net.update_node("J1", elevation=None, capacity=None)
    assert j1.elevation is None
    assert j1.capacity is None

    net.update_node("R1", head=None)
    assert r1.head is None


def test_update_node_value_error_paths():
    net = Network()
    net.add_node("junction", "J1")
    net.add_node("reservoir", "R1")
    net.add_node("tank", "T1")

    # Unknown node ID
    with pytest.raises(ValueError, match="Unknown node ID"):
        net.update_node("J99", elevation=10.0)

    # Unknown field
    with pytest.raises(ValueError, match="Unknown field"):
        net.update_node("J1", invalid_prop=123)

    # Head on junction or tank must raise ValueError
    with pytest.raises(ValueError, match="head is only allowed on reservoirs"):
        net.update_node("J1", head=50.0)
    with pytest.raises(ValueError, match="head is only allowed on reservoirs"):
        net.update_node("T1", head=50.0)

    # Negative capacity
    with pytest.raises(ValueError, match="capacity must be >= 0"):
        net.update_node("J1", capacity=-1.0)

    # Non-finite values
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite number"):
            net.update_node("J1", elevation=bad)
        with pytest.raises(ValueError, match="finite number"):
            net.update_node("J1", capacity=bad)
        with pytest.raises(ValueError, match="finite number"):
            net.update_node("R1", head=bad)

    # Non-numeric types (boolean or string)
    with pytest.raises(ValueError, match="must be a number"):
        net.update_node("J1", elevation="abc")
    with pytest.raises(ValueError, match="must be a number"):
        net.update_node("J1", elevation=True)


def test_update_pipe_happy_path():
    net = Network()
    net.add_node("junction", "J1")
    net.add_node("junction", "J2")
    p1 = net.add_pipe("J1", "J2", id="P1")

    net.update_pipe(
        "P1",
        length=350.0,
        diameter=0.3,
        roughness=0.0002,
        has_pump=True,
        pump_power=15.5
    )
    assert p1.length == 350.0
    assert p1.diameter == 0.3
    assert p1.roughness == 0.0002
    assert p1.has_pump is True
    assert p1.pump_power == 15.5

    # Roughness = 0 is allowed
    net.update_pipe("P1", roughness=0.0)
    assert p1.roughness == 0.0

    # Setting to None is allowed
    net.update_pipe("P1", length=None, diameter=None, roughness=None)
    assert p1.length is None
    assert p1.diameter is None
    assert p1.roughness is None


def test_update_pipe_value_error_paths():
    net = Network()
    net.add_node("junction", "J1")
    net.add_node("junction", "J2")
    net.add_pipe("J1", "J2", id="P1")

    # Unknown pipe ID
    with pytest.raises(ValueError, match="Unknown pipe ID"):
        net.update_pipe("P99", length=100.0)

    # Unknown field
    with pytest.raises(ValueError, match="Unknown field"):
        net.update_pipe("P1", bogus=50)

    # Length <= 0
    with pytest.raises(ValueError, match="length must be > 0"):
        net.update_pipe("P1", length=0.0)
    with pytest.raises(ValueError, match="length must be > 0"):
        net.update_pipe("P1", length=-5.0)

    # Diameter <= 0
    with pytest.raises(ValueError, match="diameter must be > 0"):
        net.update_pipe("P1", diameter=0.0)
    with pytest.raises(ValueError, match="diameter must be > 0"):
        net.update_pipe("P1", diameter=-0.1)

    # Roughness < 0
    with pytest.raises(ValueError, match="roughness must be >= 0"):
        net.update_pipe("P1", roughness=-0.001)

    # Pump power < 0
    with pytest.raises(ValueError, match="pump_power must be >= 0"):
        net.update_pipe("P1", pump_power=-1.0)

    # Non-finite values
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite number"):
            net.update_pipe("P1", length=bad)
        with pytest.raises(ValueError, match="finite number"):
            net.update_pipe("P1", diameter=bad)
        with pytest.raises(ValueError, match="finite number"):
            net.update_pipe("P1", roughness=bad)
        with pytest.raises(ValueError, match="finite number"):
            net.update_pipe("P1", pump_power=bad)

    # Non-numeric types (boolean or string)
    with pytest.raises(ValueError, match="must be a number"):
        net.update_pipe("P1", length="100m")
    with pytest.raises(ValueError, match="must be a number"):
        net.update_pipe("P1", length=True)
    with pytest.raises(ValueError, match="has_pump must be a boolean"):
        net.update_pipe("P1", has_pump="yes")


def test_add_node_and_add_pipe_validation():
    net = Network()
    # Invalid node arguments at addition time
    with pytest.raises(ValueError, match="head is only allowed on reservoirs"):
        net.add_node("junction", head=10.0)
    with pytest.raises(ValueError, match="capacity must be >= 0"):
        net.add_node("junction", capacity=-10.0)
    with pytest.raises(ValueError, match="finite number"):
        net.add_node("reservoir", elevation=float("nan"))

    net.add_node("junction", "J1")
    net.add_node("junction", "J2")

    # Invalid pipe arguments at addition time
    with pytest.raises(ValueError, match="length must be > 0"):
        net.add_pipe("J1", "J2", length=0.0)
    with pytest.raises(ValueError, match="diameter must be > 0"):
        net.add_pipe("J1", "J2", diameter=-1.0)
    with pytest.raises(ValueError, match="roughness must be >= 0"):
        net.add_pipe("J1", "J2", roughness=-0.005)

    # Direct dataclass instantiation validation
    with pytest.raises(ValueError, match="head is only allowed on reservoirs"):
        Node(id="J3", type="junction", head=10.0)
    with pytest.raises(ValueError, match="length must be > 0"):
        Pipe(id="P1", start="J1", end="J2", length=-10.0)


def test_regression_example_networks_matrices_identical():
    # 1. Paper network
    net_paper = example_paper_network()
    res_paper_baseline = build_matrices(net_paper.to_dict())

    # Verify example network itself has None for new parameters
    for p in net_paper.pipes:
        assert p.length is None
        assert p.diameter is None
        assert p.roughness is None
    for n in net_paper.nodes:
        assert n.head is None

    # Populate physical parameters
    for i, p in enumerate(net_paper.pipes):
        net_paper.update_pipe(p.id, length=100.0 * (i + 1), diameter=0.3, roughness=0.0001)
    net_paper.update_node("R1", head=65.0)

    res_paper_parametrized = build_matrices(net_paper.to_dict())

    # Assert matrix outputs are identical
    for key in ("A_matrix", "E_matrix", "F_matrix", "G_matrix", "K_dimension", "labels", "components"):
        assert res_paper_baseline[key] == res_paper_parametrized[key]

    # Verify fresh example_paper_network() is unmodified
    fresh_paper = example_paper_network()
    for p in fresh_paper.pipes:
        assert p.length is None
    assert fresh_paper.nodes[0].head is None

    # 2. Broken network
    net_broken = example_broken_network()
    res_broken_baseline = build_matrices(net_broken.to_dict())

    for i, p in enumerate(net_broken.pipes):
        net_broken.update_pipe(p.id, length=50.0 + i, diameter=0.2, roughness=0.0002)
    net_broken.update_node("R1", head=80.0)

    res_broken_parametrized = build_matrices(net_broken.to_dict())

    for key in ("A_matrix", "E_matrix", "F_matrix", "G_matrix", "K_dimension", "labels", "components"):
        assert res_broken_baseline[key] == res_broken_parametrized[key]

    # Verify fresh example_broken_network() is unmodified
    fresh_broken = example_broken_network()
    for p in fresh_broken.pipes:
        assert p.length is None
    assert fresh_broken.nodes[0].head is None



# ---------------------------------------------------------------------------
# STEP 1a regression tests: default elevation is None (not 50.0 from v1.0.0)
# ---------------------------------------------------------------------------

def test_add_node_without_elevation_gives_none():
    net = Network()
    j = net.add_node("junction")
    assert j.elevation is None, f"Expected None, got {j.elevation!r}"

    r = net.add_node("reservoir")
    assert r.elevation is None, f"Expected None, got {r.elevation!r}"

    t = net.add_node("tank")
    assert t.elevation is None, f"Expected None, got {t.elevation!r}"


def test_from_dict_node_with_explicit_elevation_loads_it():
    net = Network.from_dict({
        "nodes": [{"id": "J1", "type": "junction", "elevation": 50.0}],
        "pipes": [],
    })
    assert net.nodes[0].elevation == 50.0


def test_from_dict_node_without_elevation_key_gives_none():
    net = Network.from_dict({
        "nodes": [{"id": "J1", "type": "junction"}],
        "pipes": [],
    })
    assert net.nodes[0].elevation is None


def test_build_matrices_unchanged_with_and_without_elevation():
    net_no_elev = Network()
    net_no_elev.add_node("reservoir", id="R1")
    net_no_elev.add_node("junction", id="J1")
    net_no_elev.add_pipe("R1", "J1", id="P1")

    net_with_elev = Network()
    net_with_elev.add_node("reservoir", id="R1", elevation=20.0)
    net_with_elev.add_node("junction", id="J1", elevation=5.0)
    net_with_elev.add_pipe("R1", "J1", id="P1")

    m_no = build_matrices(net_no_elev)
    m_with = build_matrices(net_with_elev)

    for key in ("A_matrix", "E_matrix", "F_matrix", "G_matrix", "K_dimension", "labels"):
        assert m_no[key] == m_with[key], f"Mismatch on {key!r}"


# ---------------------------------------------------------------------------
# Pipe valve flags tests
# ---------------------------------------------------------------------------

def test_pipe_valve_flags_creation_and_update():
    net = Network()
    net.add_node("junction", "J1")
    net.add_node("junction", "J2")

    p = net.add_pipe("J1", "J2", id="P1", valve_start=True, valve_end=False)
    assert p.valve_start is True
    assert p.valve_end is False

    net.update_pipe("P1", valve_start=False, valve_end=True)
    assert p.valve_start is False
    assert p.valve_end is True

    # Invalid non-boolean values
    with pytest.raises(ValueError, match="valve_start must be a boolean"):
        net.update_pipe("P1", valve_start="yes")
    with pytest.raises(ValueError, match="valve_end must be a boolean"):
        net.update_pipe("P1", valve_end=1)

    with pytest.raises(ValueError, match="valve_start must be a boolean"):
        net.add_pipe("J1", "J2", valve_start=123)
    with pytest.raises(ValueError, match="valve_end must be a boolean"):
        net.add_pipe("J1", "J2", valve_end="false")


def test_pipe_valve_flags_to_dict_and_from_dict():
    net = Network()
    net.add_node("junction", "J1")
    net.add_node("junction", "J2")
    net.add_pipe("J1", "J2", id="P1", valve_start=True, valve_end=True)
    net.add_pipe("J2", "J1", id="P2", valve_start=False, valve_end=False)

    d = net.to_dict()
    p1_d = next(p for p in d["pipes"] if p["id"] == "P1")
    p2_d = next(p for p in d["pipes"] if p["id"] == "P2")

    assert p1_d.get("valve_start") is True
    assert p1_d.get("valve_end") is True
    assert "valve_start" not in p2_d
    assert "valve_end" not in p2_d

    reconstructed = Network.from_dict(d)
    rp1 = next(p for p in reconstructed.pipes if p.id == "P1")
    rp2 = next(p for p in reconstructed.pipes if p.id == "P2")
    assert rp1.valve_start is True
    assert rp1.valve_end is True
    assert rp2.valve_start is False
    assert rp2.valve_end is False
