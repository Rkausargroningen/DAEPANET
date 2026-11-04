"""Tests for core/params.py parameter completeness checks."""

from core.models import Network
from core.params import missing_parameters


def test_missing_parameters_empty_network():
    net = Network()
    assert missing_parameters(net) == []


def test_missing_parameters_v1_network():
    # v1 data has default elevations (50.0) but no pipe dimensions and no reservoir head
    v1_dict = {
        "format": "water-network",
        "version": 1,
        "nodes": [
            {"id": "R1", "type": "reservoir", "elevation": 50.0, "capacity": 1000.0},
            {"id": "J1", "type": "junction", "elevation": 50.0, "capacity": 1000.0},
        ],
        "pipes": [
            {"id": "P1", "start": "R1", "end": "J1", "has_pump": False, "pump_power": 0.0}
        ],
        "counters": {"R": 1, "J": 1, "P": 1, "T": 0},
    }
    net = Network.from_dict(v1_dict)
    missing = missing_parameters(net)

    assert "Pipe P1: length is not set" in missing
    assert "Pipe P1: diameter is not set" in missing
    assert "Pipe P1: roughness is not set" in missing
    assert "Reservoir R1: head is not set" in missing
    # Junction J1 has elevation 50.0, so no missing elevation for J1
    assert not any("J1" in m for m in missing)
    # Reservoir R1 has elevation 50.0, so no missing elevation for R1
    assert "Reservoir R1: elevation is not set" not in missing


def test_missing_parameters_fully_specified():
    net = Network()
    net.add_node("reservoir", id="R1", elevation=10.0, head=50.0)
    net.add_node("junction", id="J1", elevation=5.0)
    net.add_pipe("R1", "J1", id="P1", length=100.0, diameter=0.3, roughness=0.0001)

    assert missing_parameters(net) == []


def test_missing_parameters_partial_elements():
    net = Network()
    net.add_node("reservoir", id="R1", elevation=None, head=None)
    net.add_node("junction", id="J1", elevation=None)
    net.add_pipe("R1", "J1", id="P1", length=100.0, diameter=None, roughness=0.001)

    missing = missing_parameters(net)
    assert missing == [
        "Pipe P1: diameter is not set",
        "Reservoir R1: head is not set",
        "Reservoir R1: elevation is not set",
        "Junction J1: elevation is not set",
    ]


def test_missing_parameters_accepts_dict():
    net = Network()
    net.add_node("junction", id="J1", elevation=None)
    missing = missing_parameters(net.to_dict())
    assert missing == ["Junction J1: elevation is not set"]
