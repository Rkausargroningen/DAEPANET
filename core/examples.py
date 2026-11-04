"""Sample networks for testing and GUI demo presets."""

from __future__ import annotations

from typing import Dict, Tuple

from core.models import Network
from core.scenario import Scenario, ValveId
from core.physics import PhysicsSettings


def example_paper_network() -> Network:
    """Create a valid, regular paper-inspired network with 1 reservoir, 1 tank, 4 junctions, 1 pump, 8 pipes."""
    net = Network()
    net.add_node("reservoir", id="R1", elevation=50.0)
    net.add_node("tank", id="T1", capacity=1000.0)
    net.add_node("junction", id="J1")
    net.add_node("junction", id="J2")
    net.add_node("junction", id="J3")
    net.add_node("junction", id="J4")

    net.add_pipe("R1", "J1", id="P1")
    net.add_pipe("J1", "J2", id="P2")
    net.add_pipe("J2", "J3", id="P3", has_pump=True, pump_power=15.0)
    net.add_pipe("J3", "J4", id="P4")
    net.add_pipe("J4", "T1", id="P5")
    net.add_pipe("J1", "J3", id="P6")
    net.add_pipe("J2", "J4", id="P7")
    net.add_pipe("J3", "T1", id="P8")

    return net


def example_broken_network() -> Network:
    """Create a broken network containing an unsupplied island and a pump cycle."""
    net = Network()
    net.add_node("reservoir", id="R1")
    net.add_node("junction", id="J1")
    net.add_node("junction", id="J2")
    net.add_node("junction", id="J3")
    net.add_node("junction", id="J4")
    net.add_node("junction", id="J5")

    # Supply chain with pump loop
    net.add_pipe("R1", "J1", id="P1")
    net.add_pipe("J1", "J2", id="P2", has_pump=True, pump_power=10.0)
    net.add_pipe("J2", "J3", id="P3", has_pump=True, pump_power=10.0)
    net.add_pipe("J3", "J1", id="P4", has_pump=True, pump_power=10.0)

    # Disconnected island without supply
    net.add_pipe("J4", "J5", id="P5")

    net.default_positions = example_broken_network_positions()
    return net


def example_broken_network_positions() -> Dict[str, Tuple[float, float]]:
    """Readable layout for the broken network with no overlapping nodes or labels."""
    return {
        "R1": (-0.8, 0.4),
        "J1": (-0.3, 0.4),
        "J2": (0.2, 0.7),
        "J3": (0.2, 0.1),
        "J4": (0.7, 0.4),
        "J5": (0.7, -0.2),
    }


example_broken_network.positions = example_broken_network_positions()


def specimen_network() -> Tuple[Network, Dict[str, Tuple[float, float]]]:
    """Create the MATLAB reference specimen network with left-to-right positions.

    Nodes:
      - R1: reservoir (elevation 20.0 m, head 50.0 m)
      - J1: junction (elevation 5.0 m)
      - R2: reservoir (elevation 0.0 m, head 46.0 m)

    Pipes:
      - P1: R1 -> J1 (length 400.0 m, diameter 0.4 m, roughness 0.000359 m)
      - P2: J1 -> R2 (length 250.0 m, diameter 0.4 m, roughness 0.000359 m)
        with valve_start=True and valve_end=True
    """
    net = Network()
    net.add_node("reservoir", id="R1", elevation=20.0, head=50.0)
    net.add_node("junction", id="J1", elevation=5.0)
    net.add_node("reservoir", id="R2", elevation=0.0, head=46.0)

    net.add_pipe(
        "R1", "J1", id="P1",
        length=400.0, diameter=0.4, roughness=0.000359,
    )
    net.add_pipe(
        "J1", "R2", id="P2",
        length=250.0, diameter=0.4, roughness=0.000359,
        valve_start=True, valve_end=True,
    )

    positions = {
        "R1": (-0.8, 0.0),
        "J1": (0.0, 0.0),
        "R2": (0.8, 0.0),
    }

    return net, positions


def specimen_scenario(k: int = 1) -> Scenario:
    """Factory for MATLAB reference specimen scenarios (k = 1..5)."""
    v1 = ValveId("P2", "start")
    v2 = ValveId("P2", "end")
    return Scenario.matlab_specimen(k, v1, v2)


def specimen_physics() -> PhysicsSettings:
    """Physics settings for the specimen reference (constant friction f = 0.02)."""
    return PhysicsSettings(friction_mode="constant", friction_factor=0.02)
