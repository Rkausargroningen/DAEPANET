"""core/epanet_export.py – Generic EPANET .inp file generator and element mapping.

Generates EPANET 2.2 input (.inp) files from Network, Scenario, and PhysicsSettings,
reproducing the MATLAB writeInp2 layout generically for arbitrary topologies.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union

from core.models import Network
from core.physics import PhysicsSettings
from core.scenario import Scenario, ValveId, Mode
from core.dae_model import check_supported, build_switched_dae, UnsupportedNetworkError


class ScenarioError(ValueError):
    """Raised when a scenario has invalid valve timings or settings for EPANET."""
    pass


@dataclass
class ExportMap:
    """Mapping between DAE state labels and EPANET link/node elements.

    Parameters
    ----------
    state_to_id : dict[str, str]
        DAE state label -> EPANET element ID.
    state_to_type : dict[str, str]
        DAE state label -> 'link' (for flow) or 'node' (for pressure head).
    valve_to_link : dict[ValveId, str]
        ValveId -> EPANET valve link ID.
    """

    state_to_id: Dict[str, str] = field(default_factory=dict)
    state_to_type: Dict[str, str] = field(default_factory=dict)
    valve_to_link: Dict[ValveId, str] = field(default_factory=dict)

    def get_element(self, state_label: str) -> Tuple[str, str]:
        """Return (epanet_id, element_type) for the given DAE state label."""
        if state_label not in self.state_to_id:
            raise KeyError(f"State label {state_label!r} not in ExportMap: {list(self.state_to_id.keys())}")
        return self.state_to_id[state_label], self.state_to_type[state_label]


def _format_hms(seconds: float) -> str:
    """Format time in seconds to EPANET 'h:mm:ss' format."""
    total_sec = int(round(seconds))
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    return f"{h}:{m:02d}:{s:02d}"


def _validate_epanet_id(element_id: str) -> str:
    """Validate EPANET identifier (max 31 chars, no spaces, no semicolons)."""
    if not element_id or len(element_id) > 31:
        raise ValueError(f"EPANET ID {element_id!r} exceeds 31 characters (len={len(element_id)}).")
    if any(c in element_id for c in " ;\t\r\n"):
        raise ValueError(f"EPANET ID {element_id!r} contains invalid whitespace or semicolon.")
    return element_id


def export_epanet(
    network: Union[Network, dict],
    scenario: Scenario,
    settings: Optional[PhysicsSettings] = None,
    path: Optional[str] = None,
    positions: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[str, ExportMap]:
    """Build the EPANET .inp text and ExportMap mapping DAE states to EPANET IDs.

    Parameters
    ----------
    network : Network or dict
        Network container with physical parameters.
    scenario : Scenario
        Switching scenario with initial valve states, events, and duration.
    settings : PhysicsSettings, optional
        Fluid properties and friction mode. Defaults to PhysicsSettings().
    path : str, optional
        If provided, write the .inp file to this path on disk.
    positions : dict, optional
        Node ID -> (x, y) coordinates for the [COORDINATES] section.

    Returns
    -------
    tuple of (inp_text, export_map)
    """
    if isinstance(network, dict):
        network = Network.from_dict(network)

    if settings is None:
        settings = PhysicsSettings()

    # 1. Validation checks
    unsupported = check_supported(network)
    if unsupported:
        raise UnsupportedNetworkError(unsupported)

    dae = build_switched_dae(network, settings)
    scenario_problems = dae.check_scenario(scenario)
    if scenario_problems:
        raise ScenarioError("; ".join(scenario_problems))

    # 2. Collect elements and build collision-safe IDs
    node_map = {n.id: n for n in network.nodes}
    pipe_map = {p.id: p for p in network.pipes}

    used_node_ids: set[str] = set()
    used_link_ids: set[str] = set()

    def claim_node_id(candidate: str) -> str:
        _validate_epanet_id(candidate)
        if candidate in used_node_ids:
            raise ValueError(f"EPANET node ID collision detected: {candidate!r} already used.")
        used_node_ids.add(candidate)
        return candidate

    def claim_link_id(candidate: str) -> str:
        _validate_epanet_id(candidate)
        if candidate in used_link_ids:
            raise ValueError(f"EPANET link ID collision detected: {candidate!r} already used.")
        used_link_ids.add(candidate)
        return candidate

    export_map = ExportMap()

    # Section accumulators
    junction_lines: List[str] = []
    reservoir_lines: List[str] = []
    pipe_lines: List[str] = []
    valve_lines: List[str] = []
    status_lines: List[str] = []
    control_lines: List[str] = []
    coord_lines: List[str] = []

    # Map each reservoir node to its EPANET reservoir ID and associated pressure junction ID
    res_epanet_id: Dict[str, str] = {}
    res_node_id: Dict[str, str] = {}

    for rnode in dae.reservoirs:
        # Reservoir element
        r_id = claim_node_id(f"RES_{rnode.id}")
        head_val = float(rnode.head or 0.0)
        reservoir_lines.append(f" {r_id:<8s} {head_val:.6f}")
        res_epanet_id[rnode.id] = r_id

        # Associated pressure junction at the reservoir's elevation
        r_junction_id = claim_node_id(f"{rnode.id}_node")
        r_elev = float(rnode.elevation or 0.0)
        junction_lines.append(f" {r_junction_id:<8s} {r_elev:.6f}   0")
        res_node_id[rnode.id] = r_junction_id

        # Loss-free connector pipe (1 cm long, 5 m diameter, 0 minor loss)
        c_pipe_id = claim_link_id(f"C_{rnode.id}")
        pipe_roughness_mm = (dae.pipes[0].roughness or 0.000359) * 1000.0
        pipe_lines.append(
            f" {c_pipe_id:<8s} {r_id:<8s} {r_junction_id:<8s} 0.010000   5000.000000   {pipe_roughness_mm:.6f}   0   Open"
        )

        export_map.state_to_id[rnode.id] = r_junction_id
        export_map.state_to_type[rnode.id] = "node"

    # User junctions
    junc_epanet_id: Dict[str, str] = {}
    for jnode in dae.junctions:
        j_id = claim_node_id(jnode.id)
        j_elev = float(jnode.elevation or 0.0)
        junction_lines.append(f" {j_id:<8s} {j_elev:.6f}   0")
        junc_epanet_id[jnode.id] = j_id

        export_map.state_to_id[jnode.id] = j_id
        export_map.state_to_type[jnode.id] = "node"

    def get_boundary_junction(node_id: str) -> str:
        if node_id in res_node_id:
            return res_node_id[node_id]
        return junc_epanet_id[node_id]

    # Process pipes and their valves
    for pipe in dae.pipes:
        d_mm = float(pipe.diameter) * 1000.0
        rough_mm = float(pipe.roughness if pipe.roughness is not None else 0.000359) * 1000.0
        len_m = float(pipe.length)

        start_boundary = get_boundary_junction(pipe.start)
        end_boundary = get_boundary_junction(pipe.end)

        # 1. Start side valve / node
        if pipe.valve_start:
            pl_node = claim_node_id(f"{pipe.id}_L")
            start_elev = float(node_map[pipe.start].elevation or 0.0)
            junction_lines.append(f" {pl_node:<8s} {start_elev:.6f}   0")

            v_start_id = claim_link_id(f"V_{pipe.id}_start")
            valve_lines.append(f" {v_start_id:<8s} {start_boundary:<8s} {pl_node:<8s} {d_mm:.6f}   TCV   0   0")

            vid = ValveId(pipe.id, "start")
            export_map.valve_to_link[vid] = v_start_id
            pipe_from_node = pl_node
        else:
            pl_node = start_boundary
            pipe_from_node = start_boundary

        export_map.state_to_id[f"Pipe_{dae.pipe_col[pipe.id] + 1}_L"] = pl_node
        export_map.state_to_type[f"Pipe_{dae.pipe_col[pipe.id] + 1}_L"] = "node"

        # 2. End side valve / node
        if pipe.valve_end:
            pr_node = claim_node_id(f"{pipe.id}_R")
            end_elev = float(node_map[pipe.end].elevation or 0.0)
            junction_lines.append(f" {pr_node:<8s} {end_elev:.6f}   0")

            v_end_id = claim_link_id(f"V_{pipe.id}_end")
            valve_lines.append(f" {v_end_id:<8s} {pr_node:<8s} {end_boundary:<8s} {d_mm:.6f}   TCV   0   0")

            vid = ValveId(pipe.id, "end")
            export_map.valve_to_link[vid] = v_end_id
            pipe_to_node = pr_node
        else:
            pr_node = end_boundary
            pipe_to_node = end_boundary

        export_map.state_to_id[f"Pipe_{dae.pipe_col[pipe.id] + 1}_R"] = pr_node
        export_map.state_to_type[f"Pipe_{dae.pipe_col[pipe.id] + 1}_R"] = "node"

        # 3. Main pipe line
        p_id = claim_link_id(pipe.id)
        if settings.friction_mode == "constant":
            # Constant friction written as 1 cm stub pipe with minor loss K = f*L/D
            k_minor = settings.friction_factor * len_m / pipe.diameter
            pipe_lines.append(
                f" {p_id:<8s} {pipe_from_node:<8s} {pipe_to_node:<8s} 0.010000   {d_mm:.6f}   {rough_mm:.6f}   {k_minor:.8f}   Open"
            )
        else:
            # Darcy-Weisbach or Swamee-Jain real pipe
            pipe_lines.append(
                f" {p_id:<8s} {pipe_from_node:<8s} {pipe_to_node:<8s} {len_m:.6f}   {d_mm:.6f}   {rough_mm:.6f}   0   Open"
            )

        export_map.state_to_id[f"Pipe_{dae.pipe_col[pipe.id] + 1}_Flow"] = p_id
        export_map.state_to_type[f"Pipe_{dae.pipe_col[pipe.id] + 1}_Flow"] = "link"

    # Status section (initial valve states)
    for vid, v_link_id in export_map.valve_to_link.items():
        init_st = dae._get_valve_switch(scenario.initial, vid.pipe_id, vid.end)
        status_str = "OPEN" if init_st == 1 else "CLOSED"
        status_lines.append(f" {v_link_id:<12s} {status_str}")

    # Controls section (scheduled events)
    for event in scenario.events:
        # Check integer time
        if abs(event.time - round(event.time)) > 1e-6:
            raise ScenarioError(
                f"EPANET requires switch times in whole seconds, got {event.time}s for valve {event.valve}"
            )
        v_link_id = export_map.valve_to_link.get(event.valve)
        if not v_link_id:
            raise ScenarioError(f"Valve {event.valve} in event does not exist in network valves.")
        st_str = "OPEN" if event.state == 1 else "CLOSED"
        t_hms = _format_hms(event.time)
        control_lines.append(f" LINK {v_link_id} {st_str} AT TIME {t_hms}")

    # Coordinates section
    # If positions provided, map given coordinates; else use running layout
    coord_idx = 0
    assigned_coords: Dict[str, Tuple[float, float]] = {}

    # 1. Base user nodes
    for nid, node in node_map.items():
        if positions and nid in positions:
            assigned_coords[nid] = (float(positions[nid][0]), float(positions[nid][1]))
        else:
            elev_val = float(node.elevation or 0.0)
            assigned_coords[nid] = (float(coord_idx * 100.0), elev_val)
            coord_idx += 1

    # Write reservoirs and their junctions
    for rnode in dae.reservoirs:
        base_x, base_y = assigned_coords[rnode.id]
        r_id = res_epanet_id[rnode.id]
        r_junc = res_node_id[rnode.id]
        coord_lines.append(f" {r_id:<12s} {base_x - 30.0:.2f}   {base_y:.2f}")
        coord_lines.append(f" {r_junc:<12s} {base_x:.2f}   {base_y:.2f}")

    # Write user junctions
    for jnode in dae.junctions:
        base_x, base_y = assigned_coords[jnode.id]
        coord_lines.append(f" {jnode.id:<12s} {base_x:.2f}   {base_y:.2f}")

    # Write valve junctions
    for pipe in dae.pipes:
        p_from_pos = assigned_coords[pipe.start]
        p_to_pos = assigned_coords[pipe.end]
        dx = p_to_pos[0] - p_from_pos[0]
        dy = p_to_pos[1] - p_from_pos[1]

        if pipe.valve_start:
            pl_id = f"{pipe.id}_L"
            coord_lines.append(f" {pl_id:<12s} {p_from_pos[0] + 0.2 * dx:.2f}   {p_from_pos[1] + 0.2 * dy:.2f}")
        if pipe.valve_end:
            pr_id = f"{pipe.id}_R"
            coord_lines.append(f" {pr_id:<12s} {p_to_pos[0] - 0.2 * dx:.2f}   {p_to_pos[1] - 0.2 * dy:.2f}")

    # Assemble INP file text
    duration_hms = _format_hms(scenario.t_end)

    lines: List[str] = [
        "[TITLE]",
        f"WaterNetworkModeler – {scenario.title}",
        "",
        "[JUNCTIONS]",
        ";ID       Elev(m)    Demand",
        *junction_lines,
        "",
        "[RESERVOIRS]",
        ";ID       Head(m)",
        *reservoir_lines,
        "",
        "[PIPES]",
        ";ID       From     To       Length(m)    Diam(mm)     Rough(mm)    MinorLoss   Status",
        *pipe_lines,
        "",
        "[VALVES]",
        ";ID       From     To       Diam(mm)     Type  Setting  MinorLoss",
        *valve_lines,
        "",
        "[STATUS]",
        *status_lines,
        "",
        "[CONTROLS]",
        *control_lines,
        "",
        "[TIMES]",
        f" Duration             {duration_hms}",
        " Hydraulic Timestep   0:00:01",
        " Report Timestep      0:00:01",
        "",
        "[OPTIONS]",
        " Units              CMS",
        " Headloss           D-W",
        f" Viscosity          {settings.nu:.6e}",
        " Specific Gravity   1",
        " Trials             200",
        " Accuracy           0.0000001",
        " Unbalanced         Continue 10",
        "",
        "[COORDINATES]",
        ";Node     X        Y",
        *coord_lines,
        "",
        "[END]",
        "",
    ]

    inp_text = "\n".join(lines)

    if path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(inp_text)

    return inp_text, export_map


def write_inp(
    network: Union[Network, dict],
    scenario: Scenario,
    settings: Optional[PhysicsSettings] = None,
    path: Optional[str] = None,
    positions: Optional[Dict[str, Tuple[float, float]]] = None,
) -> str:
    """Generate EPANET .inp file text (and write to disk if path is provided).

    Parameters
    ----------
    network : Network or dict
        Network container.
    scenario : Scenario
        Switching scenario.
    settings : PhysicsSettings, optional
        Physical parameters.
    path : str, optional
        Target file path.
    positions : dict, optional
        Custom node coordinates.

    Returns
    -------
    str
        Complete EPANET input file text.
    """
    inp_text, _ = export_epanet(network, scenario, settings, path=path, positions=positions)
    return inp_text
