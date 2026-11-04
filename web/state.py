"""Session state for the Streamlit water-network modeler."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import networkx as nx
import streamlit as st

from core.models import Network
from core.physics import PhysicsSettings

Position = Tuple[float, float]


def init_state() -> None:
    """Create the working network the first time the page runs."""
    if "network" in st.session_state:
        return
    st.session_state.network = Network()
    st.session_state.positions = {}
    st.session_state.edge_bends = {}
    st.session_state.selection = None
    st.session_state.canvas_seq = None
    st.session_state.fit_token = 0
    st.session_state.project_name = "Untitled"
    st.session_state.upload_token = None
    st.session_state.comparison = None
    st.session_state.scenario_title = "Custom scenario"
    st.session_state.t_end = 120
    st.session_state.initial_states = {}
    st.session_state.events_records = []
    st.session_state.valve_sig = ()
    st.session_state.friction_label = "Constant friction factor"
    st.session_state.friction_factor = 0.02
    st.session_state.nu_text = "1.0219e-06"
    st.session_state.g = 32.2 * 0.3048
    st.session_state.preset_choice = "Custom"
    st.session_state.plot_kind = "Pressure head"
    st.session_state.show_impulses = True


def _drop_editor_widgets() -> None:
    """Forget widget values that belong to the previous network."""
    prefixes = ("vs_", "ve_", "init_", "prop_", "events_editor", "node_table", "pipe_table")
    for key in list(st.session_state.keys()):
        if key.startswith(prefixes):
            del st.session_state[key]


def layout_positions(network: Network, positions: Dict[str, Position]) -> Dict[str, Position]:
    """Give every node a coordinate, keeping positions the user has already set."""
    ids = [node.id for node in network.nodes]
    kept = {nid: positions[nid] for nid in ids if nid in positions}
    missing = [nid for nid in ids if nid not in kept]
    if not missing:
        return kept
    if len(ids) == 1:
        kept[ids[0]] = (0.0, 0.0)
        return kept

    graph = nx.Graph()
    graph.add_nodes_from(ids)
    for pipe in network.pipes:
        graph.add_edge(pipe.start, pipe.end)
    laid_out = nx.spring_layout(graph, seed=1)
    for nid in missing:
        x_coord, y_coord = laid_out[nid]
        kept[nid] = (float(x_coord), float(y_coord))
    return kept


def sync_layout() -> None:
    """Fill in coordinates for new nodes and drop bends for deleted pipes."""
    network: Network = st.session_state.network
    st.session_state.positions = layout_positions(network, st.session_state.positions)
    pipe_ids = {pipe.id for pipe in network.pipes}
    st.session_state.edge_bends = {
        pid: float(value)
        for pid, value in st.session_state.edge_bends.items()
        if pid in pipe_ids
    }
    selection = st.session_state.selection
    if selection:
        kind = selection.get("kind")
        selected_id = selection.get("id")
        if kind == "node" and selected_id not in {node.id for node in network.nodes}:
            st.session_state.selection = None
        if kind == "pipe" and selected_id not in pipe_ids:
            st.session_state.selection = None


def replace_network(
    network: Network,
    positions: Optional[Dict[str, Position]] = None,
    bends: Optional[Dict[str, float]] = None,
    name: str = "Untitled",
) -> None:
    """Swap the working network and clear results that belonged to the old one."""
    clean_positions: Dict[str, Position] = {}
    for nid, xy in (positions or {}).items():
        clean_positions[str(nid)] = (float(xy[0]), float(xy[1]))
    st.session_state.network = network
    st.session_state.positions = layout_positions(network, clean_positions)
    st.session_state.edge_bends = {
        str(pid): float(value) for pid, value in (bends or {}).items()
    }
    st.session_state.selection = None
    st.session_state.fit_token = int(st.session_state.fit_token) + 1
    st.session_state.project_name = name
    st.session_state.comparison = None
    st.session_state.events_records = []
    st.session_state.initial_states = {}
    st.session_state.valve_sig = ()
    _drop_editor_widgets()


def relayout() -> None:
    """Recompute a spring layout and clear custom pipe bends."""
    st.session_state.positions = {}
    st.session_state.edge_bends = {}
    st.session_state.fit_token = int(st.session_state.fit_token) + 1
    sync_layout()


def consume_canvas_event(event) -> bool:
    """Apply a canvas message once. Returns True when this run changed the network."""
    if not isinstance(event, dict):
        return False
    seq = event.get("seq")
    if seq is None or seq == st.session_state.get("canvas_seq"):
        return False
    st.session_state.canvas_seq = seq

    network: Network = st.session_state.network
    action = event.get("action")
    if action == "move":
        node_ids = {node.id for node in network.nodes}
        for nid, xy in (event.get("positions") or {}).items():
            if nid in node_ids and isinstance(xy, (list, tuple)) and len(xy) == 2:
                st.session_state.positions[str(nid)] = (float(xy[0]), float(xy[1]))
    elif action == "bend":
        pipe_ids = {pipe.id for pipe in network.pipes}
        for pid, value in (event.get("bends") or {}).items():
            if pid in pipe_ids:
                st.session_state.edge_bends[str(pid)] = float(value)
                st.session_state[f"bend_{pid}"] = float(value)
    elif action == "clear":
        st.session_state.selection = None
        return True

    selected_id = event.get("id")
    kind = event.get("kind")
    if selected_id and kind in {"node", "pipe"}:
        st.session_state.selection = {"kind": kind, "id": selected_id}
    return True


def friction_mode() -> str:
    labels = {
        "Constant friction factor": "constant",
        "Swamee-Jain": "swamee_jain",
        "Darcy-Weisbach as in EPANET": "darcy_weisbach",
    }
    return labels.get(st.session_state.get("friction_label", ""), "constant")


def current_physics() -> PhysicsSettings:
    """Physics settings currently shown in the comparison form."""
    return PhysicsSettings(
        g=float(st.session_state.g),
        nu=float(str(st.session_state.nu_text).strip()),
        friction_mode=friction_mode(),
        friction_factor=float(st.session_state.friction_factor),
    )


def apply_physics(settings: PhysicsSettings) -> None:
    reverse = {
        "constant": "Constant friction factor",
        "swamee_jain": "Swamee-Jain",
        "darcy_weisbach": "Darcy-Weisbach as in EPANET",
    }
    st.session_state.friction_label = reverse.get(settings.friction_mode, "Constant friction factor")
    st.session_state.friction_factor = float(settings.friction_factor)
    st.session_state.nu_text = f"{float(settings.nu):.6g}"
    st.session_state.g = float(settings.g)
