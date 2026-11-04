"""File I/O for saving and loading Water Network project files (.wnet)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List

from core.models import Network
from core.scenario import Scenario
from core.physics import PhysicsSettings


SUPPORTED_FORMAT = "water-network"
CURRENT_VERSION = 2
SUPPORTED_VERSIONS = {1, 2}


@dataclass
class ProjectData:
    """Container for fully loaded project data including network, positions, edge bends, scenario, and physics."""

    network: Network
    positions: Dict[str, Tuple[float, float]]
    edge_bends: Dict[str, float]
    scenario: Optional[Scenario] = None
    physics: Optional[PhysicsSettings] = None


def save_project(
    path: str | Path,
    network: Network,
    positions: Optional[Dict[str, Tuple[float, float]]] = None,
    edge_bends: Optional[Dict[str, float]] = None,
    scenario: Optional[Scenario] = None,
    physics: Optional[PhysicsSettings] = None,
) -> None:
    """Save a network and optional canvas node positions, edge bends, scenario, and physics to a JSON file.

    Parameters
    ----------
    path : str or Path
        Destination file path (.wnet).
    network : Network
        Network instance to serialize.
    positions : dict, optional
        Mapping of node IDs to (x, y) canvas coordinates.
    edge_bends : dict, optional
        Mapping of pipe IDs to dimensionless bend ratios.
    scenario : Scenario, optional
        Valve switching scenario.
    physics : PhysicsSettings, optional
        Fluid properties and calculation settings.
    """
    path = Path(path)
    data = network.to_dict()

    pos_dict = {}
    if positions:
        for nid, (x, y) in positions.items():
            pos_dict[nid] = [float(x), float(y)]

    payload = {
        "format": SUPPORTED_FORMAT,
        "version": CURRENT_VERSION,
        "nodes": data["nodes"],
        "pipes": data["pipes"],
        "counters": data["counters"],
        "positions": pos_dict,
    }

    if edge_bends:
        bends_dict = {
            str(pid): float(val)
            for pid, val in edge_bends.items()
            if abs(float(val)) > 1e-6
        }
        if bends_dict:
            payload["edge_bends"] = bends_dict

    if scenario is not None:
        payload["scenario"] = scenario.to_dict() if hasattr(scenario, "to_dict") else scenario

    if physics is not None:
        payload["physics"] = {
            "friction_mode": str(physics.friction_mode),
            "friction_factor": float(physics.friction_factor),
            "nu": float(physics.nu),
            "g": float(physics.g),
        }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_project_full(path: str | Path) -> ProjectData:
    """Load network, node positions, and edge bends from a JSON file.

    Parameters
    ----------
    path : str or Path
        File path to load (.wnet).

    Returns
    -------
    ProjectData
        Data container with loaded Network instance, node positions, and edge bends.

    Raises
    ------
    ValueError
        If format, version, or structure is invalid or corrupt.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"File not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Corrupt JSON file {path.name}: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to read file {path.name}: {e}") from e

    if not isinstance(payload, dict):
        raise ValueError("Invalid project file: top-level JSON structure must be an object.")

    fmt = payload.get("format")
    if fmt != SUPPORTED_FORMAT:
        raise ValueError(f"Invalid format identifier {fmt!r}, expected {SUPPORTED_FORMAT!r}.")

    ver = payload.get("version")
    if ver not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"Unsupported project file version {ver!r}, expected one of {sorted(SUPPORTED_VERSIONS)}."
        )

    nodes_data = payload.get("nodes")
    if not isinstance(nodes_data, list):
        raise ValueError("Invalid project file: 'nodes' must be a list.")

    pipes_data = payload.get("pipes")
    if not isinstance(pipes_data, list):
        raise ValueError("Invalid project file: 'pipes' must be a list.")

    # Validate node IDs & types
    node_ids = set()
    valid_types = {"junction", "tank", "reservoir"}
    for nd in nodes_data:
        if not isinstance(nd, dict) or "id" not in nd or "type" not in nd:
            raise ValueError("Corrupt node definition in project file.")
        nid = str(nd["id"])
        ntype = str(nd["type"])
        if nid in node_ids:
            raise ValueError(f"Duplicate node ID in project file: {nid!r}")
        if ntype not in valid_types:
            raise ValueError(f"Unknown node type {ntype!r} for node {nid!r}")
        node_ids.add(nid)

    # Validate pipe IDs & start/end references
    pipe_ids = set()
    for pd in pipes_data:
        if not isinstance(pd, dict) or "id" not in pd or "start" not in pd or "end" not in pd:
            raise ValueError("Corrupt pipe definition in project file.")
        pid = str(pd["id"])
        start = str(pd["start"])
        end = str(pd["end"])

        if pid in pipe_ids:
            raise ValueError(f"Duplicate pipe ID in project file: {pid!r}")
        if start not in node_ids:
            raise ValueError(f"Pipe {pid!r} references unknown start node {start!r}")
        if end not in node_ids:
            raise ValueError(f"Pipe {pid!r} references unknown end node {end!r}")
        pipe_ids.add(pid)

    net = Network.from_dict(payload)

    # Parse positions
    positions: Dict[str, Tuple[float, float]] = {}
    raw_pos = payload.get("positions", {})
    if isinstance(raw_pos, dict):
        for nid, coords in raw_pos.items():
            if nid in node_ids and isinstance(coords, (list, tuple)) and len(coords) == 2:
                positions[nid] = (float(coords[0]), float(coords[1]))

    # Parse edge_bends
    edge_bends: Dict[str, float] = {}
    raw_bends = payload.get("edge_bends")
    if raw_bends is not None:
        if not isinstance(raw_bends, dict):
            raise ValueError("Invalid project file: 'edge_bends' must be an object.")
        for pid, val in raw_bends.items():
            if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val):
                raise ValueError(
                    f"Invalid bend value for pipe {pid!r}: must be a finite number, got {val!r}"
                )
            if str(pid) in pipe_ids:
                edge_bends[str(pid)] = float(val)

    # Parse scenario
    scenario = None
    raw_scenario = payload.get("scenario")
    if raw_scenario is not None:
        if isinstance(raw_scenario, dict):
            scenario = Scenario.from_dict(raw_scenario)
        else:
            raise ValueError("Invalid project file: 'scenario' must be an object.")

    # Parse physics
    physics = None
    raw_physics = payload.get("physics")
    if raw_physics is not None:
        if not isinstance(raw_physics, dict):
            raise ValueError("Invalid project file: 'physics' must be an object.")
        try:
            f_mode = raw_physics.get("friction_mode", "constant")
            f_fac = raw_physics.get("friction_factor", 0.02)
            nu_val = raw_physics.get("nu", 1.0219e-6)
            g_val = raw_physics.get("g", 32.2 * 0.3048)
            physics = PhysicsSettings(
                friction_mode=str(f_mode),
                friction_factor=float(f_fac),
                nu=float(nu_val),
                g=float(g_val),
            )
        except Exception as e:
            raise ValueError(f"Invalid physics configuration in project file: {e}") from e

    return ProjectData(
        network=net,
        positions=positions,
        edge_bends=edge_bends,
        scenario=scenario,
        physics=physics,
    )


def load_project(path: str | Path) -> Tuple[Network, Dict[str, Tuple[float, float]]]:
    """Load a network and node positions from a JSON file.

    Parameters
    ----------
    path : str or Path
        File path to load (.wnet).

    Returns
    -------
    tuple of (Network, dict)
        Loaded Network instance and node positions mapping.

    Raises
    ------
    ValueError
        If format, version, or structure is invalid or corrupt.
    """
    data = load_project_full(path)
    return data.network, data.positions

