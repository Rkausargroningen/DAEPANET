"""Streamlit pages for the water-network modeler. Calculation stays in core/."""

from __future__ import annotations

import io
import re
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from core.checks import incidence_matrices, is_regular, numeric_pencil, validate_network
from core.comparison import problems, run_comparison
from core.epanet_export import export_epanet
from core.examples import specimen_scenario
from core.matrices import build_matrices
from core.models import Network
from core.plotdata import (
    default_states,
    difference_rows,
    engine_label,
    impulse_arrows,
    max_abs_difference_after_last_switch,
    series,
)
from core.scenario import Scenario, SwitchEvent, ValveId
from web.canvas import network_canvas
from web.formatting import format_optional, matrix_to_csv, matrix_to_latex, parse_optional_number
from web.state import consume_canvas_event, current_physics, sync_layout

PRESETS = [
    "Custom",
    "MATLAB specimen 1: valve 1 closed at t = 60 s (started from rest)",
    "MATLAB specimen 2: no switching, relaxation from rest",
    "MATLAB specimen 3: equilibrium first, then valve 1 closes at t = 300 s",
    "MATLAB specimen 4: equilibrium first, then valve 2 closes at t = 300 s",
    "MATLAB specimen 5: valve 1 opened at t = 60 s (start with valve 1 closed)",
]

FRICTION_LABELS = [
    "Constant friction factor",
    "Swamee-Jain",
    "Darcy-Weisbach as in EPANET",
]


def network_valves(network: Network) -> List[ValveId]:
    valves: List[ValveId] = []
    for pipe in network.pipes:
        if pipe.valve_start:
            valves.append(ValveId(pipe.id, "start"))
        if pipe.valve_end:
            valves.append(ValveId(pipe.id, "end"))
    return valves


def store_scenario(scenario: Scenario) -> None:
    """Copy a scenario into the form fields before those widgets are drawn."""
    st.session_state.scenario_title = scenario.title
    st.session_state.t_end = int(scenario.t_end)
    st.session_state.initial_states = {str(valve): int(state) for valve, state in scenario.initial.items()}
    st.session_state.events_records = [
        {"time": float(event.time), "valve": str(event.valve), "state": int(event.state)}
        for event in scenario.events
    ]
    st.session_state.pop("events_editor", None)
    for key in list(st.session_state.keys()):
        if key.startswith("init_"):
            del st.session_state[key]


def apply_preset_choice() -> None:
    """Fill the scenario form from a MATLAB specimen. V1 and V2 are the first two valves."""
    label = st.session_state.get("preset_choice", "Custom")
    if label not in PRESETS or label == "Custom":
        st.session_state.preset_warning = ""
        return
    valves = network_valves(st.session_state.network)
    if len(valves) < 2:
        st.session_state.preset_warning = (
            "A preset needs two valves. V1 is the first valve in the network and V2 is the second."
        )
        return
    scenario = Scenario.matlab_specimen(PRESETS.index(label), valves[0], valves[1])
    store_scenario(scenario)
    st.session_state.preset_warning = f"V1 = {valves[0]}, V2 = {valves[1]}"


def load_specimen_scenario() -> None:
    """Use specimen preset 1, whose valves are P2:start and P2:end."""
    st.session_state.preset_choice = PRESETS[1]
    store_scenario(specimen_scenario(1))
    st.session_state.preset_warning = "V1 = P2:start, V2 = P2:end"


def assemble_scenario(network: Network) -> Optional[Scenario]:
    """Build a scenario from the form. Returns None when the network has no valves."""
    valves = network_valves(network)
    if not valves:
        return None
    initial = {}
    for valve in valves:
        initial[valve] = int(st.session_state.initial_states.get(str(valve), 1))
    events = []
    for record in st.session_state.events_records:
        valve_name = str(record.get("valve", "")).strip()
        if not valve_name:
            continue
        events.append(
            SwitchEvent(
                time=float(record["time"]),
                valve=ValveId.from_str(valve_name),
                state=int(record["state"]),
            )
        )
    title = str(st.session_state.scenario_title).strip() or "Custom scenario"
    return Scenario(title=title, initial=initial, events=events, t_end=float(st.session_state.t_end))


def _canvas_spec(network: Network) -> dict:
    selection = st.session_state.selection
    return {
        "nodes": [
            {
                "id": node.id,
                "type": node.type,
                "elevation": node.elevation,
                "head": node.head,
                "capacity": node.capacity,
            }
            for node in network.nodes
        ],
        "pipes": [
            {
                "id": pipe.id,
                "start": pipe.start,
                "end": pipe.end,
                "has_pump": pipe.has_pump,
                "pump_power": pipe.pump_power,
                "length": pipe.length,
                "diameter": pipe.diameter,
                "roughness": pipe.roughness,
                "valve_start": pipe.valve_start,
                "valve_end": pipe.valve_end,
            }
            for pipe in network.pipes
        ],
        "positions": {nid: [xy[0], xy[1]] for nid, xy in st.session_state.positions.items()},
        "bends": {pid: float(value) for pid, value in st.session_state.edge_bends.items()},
        "selected": selection,
        "fitToken": int(st.session_state.fit_token),
    }


def _remember_table_selection(event, memory_key: str, ids: List[str], kind: str) -> None:
    selection = getattr(event, "selection", None)
    rows = list(getattr(selection, "rows", []) or [])
    previous = list(st.session_state.get(memory_key, []))
    if rows == previous:
        return
    st.session_state[memory_key] = rows
    if rows:
        st.session_state.selection = {"kind": kind, "id": ids[rows[0]]}
    elif previous:
        current = st.session_state.selection or {}
        if current.get("kind") == kind:
            st.session_state.selection = None


def render_network() -> None:
    network: Network = st.session_state.network
    sync_layout()
    event = network_canvas(_canvas_spec(network))
    if consume_canvas_event(event):
        st.rerun()

    st.caption(
        "Drag a node to move it. Select a pipe and drag the round handle to bend it; "
        "double-click the handle to straighten it. Drag empty background to pan. Hold Ctrl (⌘ on Mac) and scroll, or pinch, to zoom."
    )
    button_row = st.columns(2)
    if button_row[0].button("Re-layout", key="relayout"):
        from web.state import relayout
        relayout()
        st.rerun()
    if button_row[1].button("Fit", key="fit_view"):
        st.session_state.fit_token = int(st.session_state.fit_token) + 1
        st.rerun()

    editor, builder = st.columns([1.15, 1])
    with builder:
        _add_forms(network)
    with editor:
        _selection_editor(network)

    st.subheader("Nodes")
    node_ids = [node.id for node in network.nodes]
    node_frame = pd.DataFrame(
        [
            {
                "ID": node.id,
                "Type": node.type,
                "Elevation (m)": node.elevation,
                "Head (m)": node.head,
                "Capacity": node.capacity,
            }
            for node in network.nodes
        ]
    )
    if node_frame.empty:
        st.info("No nodes yet.")
    else:
        node_event = st.dataframe(
            node_frame,
            hide_index=True,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            key="node_table",
        )
        _remember_table_selection(node_event, "node_table_rows", node_ids, "node")

    st.subheader("Pipes")
    pipe_ids = [pipe.id for pipe in network.pipes]
    pipe_frame = pd.DataFrame(
        [
            {
                "ID": pipe.id,
                "From": pipe.start,
                "To": pipe.end,
                "Pump": pipe.has_pump,
                "Power (kW)": pipe.pump_power if pipe.has_pump else None,
                "Length (m)": pipe.length,
                "Diameter (m)": pipe.diameter,
                "Roughness (m)": pipe.roughness,
                "Valve start": pipe.valve_start,
                "Valve end": pipe.valve_end,
            }
            for pipe in network.pipes
        ]
    )
    if pipe_frame.empty:
        st.info("No pipes yet.")
    else:
        pipe_event = st.dataframe(
            pipe_frame,
            hide_index=True,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            key="pipe_table",
        )
        _remember_table_selection(pipe_event, "pipe_table_rows", pipe_ids, "pipe")


def _add_forms(network: Network) -> None:
    with st.expander("Add node", expanded=not network.nodes):
        with st.form("add_node", clear_on_submit=True):
            node_type = st.selectbox("Type", ["junction", "tank", "reservoir"])
            node_id = st.text_input("ID", placeholder="Leave blank to number it automatically")
            elevation = st.text_input("Elevation (m)", placeholder="blank = not set")
            head = st.text_input("Reservoir head (m)", placeholder="reservoirs only; blank = not set")
            capacity = st.text_input("Tank capacity", placeholder="blank = 1000")
            if st.form_submit_button("Add node"):
                _commit_new_node(network, node_type, node_id, elevation, head, capacity)

    with st.expander("Add pipe", expanded=False):
        node_ids = [node.id for node in network.nodes]
        if len(node_ids) < 2:
            st.info("Add at least two nodes before connecting them.")
        else:
            with st.form("add_pipe", clear_on_submit=True):
                start, end = st.columns(2)
                start_id = start.selectbox("From", node_ids)
                end_id = end.selectbox("To", node_ids, index=1 if len(node_ids) > 1 else 0)
                has_pump = st.checkbox("Pump on this pipe")
                power = st.text_input("Pump power (kW)", value="0")
                length = st.text_input("Length (m)", placeholder="blank = not set")
                diameter = st.text_input("Diameter (m)", placeholder="blank = not set")
                roughness = st.text_input("Roughness ε (m)", placeholder="blank = not set, e.g. 3.59e-4")
                valve_start = st.checkbox("Valve at start")
                valve_end = st.checkbox("Valve at end")
                if st.form_submit_button("Add pipe"):
                    _commit_new_pipe(
                        network, start_id, end_id, has_pump, power,
                        length, diameter, roughness, valve_start, valve_end,
                    )


def _commit_new_node(network, node_type, node_id, elevation, head, capacity) -> None:
    try:
        kwargs = {}
        if node_id.strip():
            kwargs["id"] = node_id.strip()
        elev_value = parse_optional_number(elevation, "Elevation")
        head_value = parse_optional_number(head, "Head") if node_type == "reservoir" else None
        cap_value = 1000.0
        if node_type == "tank":
            parsed = parse_optional_number(capacity, "Capacity", minimum=0)
            cap_value = 1000.0 if parsed is None else parsed
        network.add_node(node_type, elevation=elev_value, head=head_value, capacity=cap_value, **kwargs)
        sync_layout()
        st.session_state.fit_token = int(st.session_state.fit_token) + 1
        st.rerun()
    except ValueError as exc:
        st.error(str(exc))


def _commit_new_pipe(network, start_id, end_id, has_pump, power, length, diameter, roughness, valve_start, valve_end) -> None:
    try:
        power_value = parse_optional_number(power, "Pump power", minimum=0)
        network.add_pipe(
            start_id,
            end_id,
            has_pump=has_pump,
            pump_power=0.0 if power_value is None else power_value,
            length=parse_optional_number(length, "Length", exclusive_min=0),
            diameter=parse_optional_number(diameter, "Diameter", exclusive_min=0),
            roughness=parse_optional_number(roughness, "Roughness", minimum=0),
            valve_start=bool(valve_start) and not has_pump,
            valve_end=bool(valve_end) and not has_pump,
        )
        st.rerun()
    except ValueError as exc:
        st.error(str(exc))


def _selection_editor(network: Network) -> None:
    selection = st.session_state.selection
    st.subheader("Properties")
    if not selection:
        st.info("Select a node or a pipe on the canvas or in a table.")
        return
    if selection["kind"] == "node":
        node = next((item for item in network.nodes if item.id == selection["id"]), None)
        if node is None:
            st.info("That node is no longer in the network.")
            return
        st.markdown(f"**{node.id}** · {node.type}")
        with st.form(f"prop_node_{node.id}"):
            elevation = st.text_input("Elevation (m)", value=format_optional(node.elevation))
            head = None
            capacity = None
            if node.type == "reservoir":
                head = st.text_input("Head (m)", value=format_optional(node.head))
            if node.type == "tank":
                capacity = st.text_input("Capacity", value=format_optional(node.capacity))
            apply = st.form_submit_button("Apply")
        if apply:
            try:
                fields = {"elevation": parse_optional_number(elevation, "Elevation")}
                if node.type == "reservoir":
                    fields["head"] = parse_optional_number(head or "", "Head")
                if node.type == "tank":
                    parsed = parse_optional_number(capacity or "", "Capacity", minimum=0)
                    fields["capacity"] = 1000.0 if parsed is None else parsed
                network.update_node(node.id, **fields)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        if st.button("Delete node", key="delete_node"):
            network.remove_node(node.id)
            sync_layout()
            st.rerun()
        return

    pipe = next((item for item in network.pipes if item.id == selection["id"]), None)
    if pipe is None:
        st.info("That pipe is no longer in the network.")
        return
    st.markdown(f"**{pipe.id}** · {pipe.start} → {pipe.end}")
    bend_key = f"bend_{pipe.id}"
    if bend_key not in st.session_state:
        st.session_state[bend_key] = float(st.session_state.edge_bends.get(pipe.id, 0.0))
    bend = st.slider("Bend", min_value=-0.8, max_value=0.8, step=0.01, key=bend_key)
    st.session_state.edge_bends[pipe.id] = float(bend)
    with st.form(f"prop_pipe_{pipe.id}"):
        length = st.text_input("Length (m)", value=format_optional(pipe.length))
        diameter = st.text_input("Diameter (m)", value=format_optional(pipe.diameter))
        roughness = st.text_input("Roughness ε (m)", value=format_optional(pipe.roughness))
        has_pump = st.checkbox("Pump", value=pipe.has_pump)
        power = st.text_input("Pump power (kW)", value=format_optional(pipe.pump_power))
        valve_start = st.checkbox("Valve at start", value=pipe.valve_start, disabled=has_pump)
        valve_end = st.checkbox("Valve at end", value=pipe.valve_end, disabled=has_pump)
        apply = st.form_submit_button("Apply")
    if apply:
        try:
            network.update_pipe(
                pipe.id,
                length=parse_optional_number(length, "Length", exclusive_min=0),
                diameter=parse_optional_number(diameter, "Diameter", exclusive_min=0),
                roughness=parse_optional_number(roughness, "Roughness", minimum=0),
                has_pump=has_pump,
                pump_power=parse_optional_number(power, "Pump power", minimum=0) or 0.0,
                valve_start=bool(valve_start) and not has_pump,
                valve_end=bool(valve_end) and not has_pump,
            )
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    if st.button("Delete pipe", key="delete_pipe"):
        network.remove_pipe(pipe.id)
        sync_layout()
        st.rerun()


def _matrix_block(name: str, data, labels: list[str], col_labels: list[str] | None = None) -> None:
    if not data:
        st.info(f"{name} is empty.")
        return
    is_2d = isinstance(data[0], (list, tuple))
    if is_2d:
        width = len(data[0])
        columns = list(col_labels)[:width] if col_labels else (
            labels[:width] if len(labels) >= width else [str(i) for i in range(width)]
        )
        if len(columns) < width:
            columns = columns + [str(i) for i in range(len(columns), width)]
        frame = pd.DataFrame(data, index=labels[: len(data)], columns=columns)
        col_labels = columns
    else:
        frame = pd.DataFrame({name: list(data)}, index=labels[: len(data)])
        col_labels = [name]
    st.dataframe(frame, width="stretch")
    latex = matrix_to_latex(data)
    with st.expander(f"{name} as LaTeX"):
        st.code(latex, language="latex")
    csv_col, tex_col = st.columns(2)
    csv_col.download_button(
        f"Download {name}.csv",
        matrix_to_csv(labels, col_labels, data),
        file_name=f"{name.lower()}_matrix.csv",
        mime="text/csv",
        key=f"csv_{name}",
    )
    tex_col.download_button(
        f"Download {name}.tex",
        latex,
        file_name=f"{name.lower()}_matrix.tex",
        mime="text/plain",
        key=f"tex_{name}",
    )


def render_matrices() -> None:
    network: Network = st.session_state.network
    if not network.pipes:
        st.info("Add at least one pipe to see the matrices.")
        return
    result = build_matrices(network.to_dict())
    parts = result["components"]
    st.markdown(
        f"**K = {result['K_dimension']}** · "
        f"pipes {len(network.pipes)}, pumps {parts['pumped']}, "
        f"junctions {parts['junctions']}, reservoirs {parts['reservoirs']}, tanks {parts['tanks']}"
    )
    labels = result["labels"]
    tabs = st.tabs(["E", "A", "F", "G", "Patterns"])
    with tabs[0]:
        _matrix_block("E", result["E_matrix"], labels)
    with tabs[1]:
        _matrix_block("A", result["A_matrix"], labels)
    with tabs[2]:
        _matrix_block("F", result["F_matrix"], labels)
    with tabs[3]:
        _matrix_block("G", result["G_matrix"], labels)
    with tabs[4]:
        e_binary = np.array([[0 if str(value) == "0" else 1 for value in row] for row in result["E_matrix"]])
        a_binary = np.array([[0 if str(value) == "0" else 1 for value in row] for row in result["A_matrix"]])
        figure, axes = plt.subplots(1, 2, figsize=(9, 4.2))
        axes[0].imshow(e_binary, cmap="binary", interpolation="nearest")
        axes[0].set_title("E non-zero pattern")
        axes[1].imshow(a_binary, cmap="binary", interpolation="nearest")
        axes[1].set_title("A non-zero pattern")
        figure.tight_layout()
        st.pyplot(figure, alt="Non-zero pattern of the E and A matrices")
        plt.close(figure)


def render_checks() -> None:
    network: Network = st.session_state.network
    if not network.nodes or not network.pipes:
        st.info("Add pipes to run the checks.")
        return
    import networkx as nx

    matrices = build_matrices(network.to_dict())
    e_num, a_num = numeric_pencil(matrices)
    regular = is_regular(e_num, a_num)
    if regular:
        st.success("Regular — the system is solvable (under the assumptions below).")
    else:
        st.error("NOT regular")

    graph = nx.Graph()
    graph.add_nodes_from(node.id for node in network.nodes)
    graph.add_edges_from((pipe.start, pipe.end) for pipe in network.pipes)
    components = nx.number_connected_components(graph)
    modified = incidence_matrices(network)["modified"]
    rank = int(np.linalg.matrix_rank(modified)) if modified.size else 0
    st.caption(
        f"Connected components: {components} · "
        f"Modified incidence matrix rank: {rank} / {modified.shape[0]} rows"
    )

    warnings = validate_network(network)
    st.subheader("Validation warnings")
    if not warnings:
        st.info("No topology warnings.")
    else:
        st.dataframe(pd.DataFrame({"Warning": warnings}), hide_index=True)
        choice = st.selectbox("Warning to highlight", warnings, key="warn_pick")
        if st.button("Highlight on the network", key="warn_go"):
            found = re.findall(r"\b[JTRP]\d+\b", choice)
            node_ids = {node.id for node in network.nodes}
            pipe_ids = {pipe.id for pipe in network.pipes}
            for element_id in found:
                if element_id in node_ids:
                    st.session_state.selection = {"kind": "node", "id": element_id}
                    break
                if element_id in pipe_ids:
                    st.session_state.selection = {"kind": "pipe", "id": element_id}
                    break
            st.rerun()

    with st.expander("What do these checks mean?", expanded=True):
        st.markdown(
            """
- **Supply.** Every connected group of nodes needs a reservoir or a tank.
- **Pumps.** Pumps must not form a loop, including pumps in series between two supplies.
- **Numerical test.** The pencil `(E, A)` is tested with every valve open, every pump on, and each friction symbol set to 1.
            """
        )


def render_incidence() -> None:
    network: Network = st.session_state.network
    if not network.nodes or not network.pipes:
        st.info("Add nodes and pipes to view incidence matrices.")
        return
    result = incidence_matrices(network)
    node_labels = list(result["node_labels"])
    edge_labels = list(result["edge_labels"])
    reference = result["reference_node"]
    st.caption("Sign convention: −1 is the start node, +1 is the end node.")
    tabs = st.tabs(["Full I(G)", "Reduced Iʳ(G)", "Modified Iᵐ(G)", "Pressure differences"])
    with tabs[0]:
        _matrix_block("I", result["incidence"].tolist(), node_labels, edge_labels)
    with tabs[1]:
        reduced_rows = [label for label in node_labels if label != reference]
        st.caption(f"Reference node removed: {reference}")
        _matrix_block("Ir", result["reduced"].tolist(), reduced_rows, edge_labels)
    with tabs[2]:
        junction_rows = [node.id for node in network.nodes if node.type == "junction"]
        removed = [node.id for node in network.nodes if node.type in {"reservoir", "tank"}]
        st.caption("Reservoirs and tanks removed: " + (", ".join(removed) if removed else "none"))
        _matrix_block("Im", result["modified"].tolist(), junction_rows, edge_labels)
    with tabs[3]:
        expressions = [[f"dP_{pipe.id} = P_{pipe.end} - P_{pipe.start}"] for pipe in network.pipes]
        st.caption("ΔP = Iᵀ P, end pressure minus start pressure.")
        _matrix_block("dP", expressions, [pipe.id for pipe in network.pipes], ["Expression"])


def _sync_valves(network: Network) -> None:
    changed = False
    for pipe in network.pipes:
        if pipe.has_pump and (pipe.valve_start or pipe.valve_end):
            network.update_pipe(pipe.id, valve_start=False, valve_end=False)
            changed = True
        for end, field, prefix in (
            ("start", "valve_start", "vs_"),
            ("end", "valve_end", "ve_"),
        ):
            key = f"{prefix}{pipe.id}"
            kwargs = {"key": key, "disabled": pipe.has_pump}
            if key not in st.session_state:
                kwargs["value"] = False if pipe.has_pump else bool(getattr(pipe, field))
            checked = st.checkbox(f"{pipe.id} valve at {end} ({pipe.start} → {pipe.end})", **kwargs)
            if pipe.has_pump:
                continue
            if checked != bool(getattr(pipe, field)):
                network.update_pipe(pipe.id, **{field: checked})
                changed = True
    if changed:
        st.rerun()
    valves = network_valves(network)
    valve_names = tuple(str(valve) for valve in valves)
    if valve_names != st.session_state.get("valve_sig"):
        st.session_state.valve_sig = valve_names
        st.session_state.events_records = [
            record for record in st.session_state.events_records
            if str(record.get("valve")) in set(valve_names)
        ]
        st.session_state.initial_states = {
            name: int(st.session_state.initial_states.get(name, 1)) for name in valve_names
        }
        st.session_state.pop("events_editor", None)


def _scenario_form(network: Network) -> None:
    valves = network_valves(network)
    if valves:
        mapping = f"V1 = {valves[0]}"
        if len(valves) > 1:
            mapping += f", V2 = {valves[1]}"
        st.caption(mapping)
    else:
        st.caption("V1 and V2 are the first two valves. This network has none yet.")
    if st.session_state.get("preset_warning"):
        st.info(st.session_state.preset_warning)

    st.selectbox("Preset", PRESETS, key="preset_choice", on_change=apply_preset_choice)
    st.text_input("Title", key="scenario_title")
    if valves:
        st.markdown("**Initial state at t = 0**")
        for valve in valves:
            label = str(valve)
            kwargs = {"key": f"init_{label}"}
            if kwargs["key"] not in st.session_state:
                kwargs["index"] = 0 if int(st.session_state.initial_states.get(label, 1)) == 1 else 1
            choice = st.selectbox(label, ["open", "closed"], **kwargs)
            st.session_state.initial_states[label] = 1 if choice == "open" else 0
        valve_options = [str(valve) for valve in valves]
        records = st.session_state.events_records or []
        frame = pd.DataFrame(records, columns=["time", "valve", "state"]) if records else pd.DataFrame(
            {"time": pd.Series(dtype=float), "valve": pd.Series(dtype=str), "state": pd.Series(dtype=int)}
        )
        edited = st.data_editor(
            frame,
            num_rows="dynamic",
            width="stretch",
            key="events_editor",
            column_config={
                "time": st.column_config.NumberColumn("Time (s)", min_value=0.0, step=1.0, format="%.0f"),
                "valve": st.column_config.SelectboxColumn("Valve", options=valve_options, required=True),
                "state": st.column_config.SelectboxColumn("New state (1 open, 0 closed)", options=[1, 0], required=True),
            },
        )
        st.session_state.events_records = edited.dropna(how="all").to_dict("records")
    st.number_input("End time (s)", min_value=1, max_value=86400, step=1, key="t_end")


def _physics_form() -> None:
    st.selectbox("Friction law", FRICTION_LABELS, key="friction_label")
    if st.session_state.friction_label == "Constant friction factor":
        st.caption("One friction factor for every pipe and every flow.")
        st.number_input("Friction factor f", min_value=0.0001, max_value=1.0, step=0.001, format="%.4f", key="friction_factor")
    elif st.session_state.friction_label == "Swamee-Jain":
        st.caption("Explicit Swamee–Jain approximation of the Darcy friction factor.")
    else:
        st.caption("EPANET's Darcy–Weisbach formula, including the laminar and transition regions.")
    with st.expander("Advanced fluid properties"):
        st.number_input("Gravity g (m/s²)", min_value=1.0, max_value=30.0, step=0.000001, format="%.6f", key="g")
        st.text_input("Kinematic viscosity ν (m²/s)", key="nu_text")


def render_comparison() -> None:
    network: Network = st.session_state.network
    with st.expander("How to compare with EPANET", expanded=False):
        st.markdown(
            """
1. Fill elevation, reservoir head, length, diameter, and roughness. Leave a field blank only when it is unknown.
2. Tick a valve at the start or end of a pipe. Pumps and tanks cannot be compared.
3. Choose a MATLAB specimen or write switch times in whole seconds.
4. Run the comparison.
5. Read the plots, the difference table, the equilibrium of each valve mode, and any Dirac impulses.
6. Download the figure, the tables, or an EPANET `.inp` file.
            """
        )

    left, right = st.columns([1, 1.3], gap="large")
    with left:
        st.subheader("Valves")
        if not network.pipes:
            st.info("Add pipes before placing valves.")
        else:
            _sync_valves(network)
        st.subheader("Scenario")
        _scenario_form(network)
        st.subheader("Physics")
        _physics_form()

        scenario = None
        scenario_error = None
        try:
            scenario = assemble_scenario(network)
        except ValueError as exc:
            scenario_error = str(exc)
        physics = None
        physics_error = None
        try:
            physics = current_physics()
        except ValueError as exc:
            physics_error = str(exc)

        found = []
        if physics_error:
            found.append(physics_error)
        if scenario_error:
            found.append(scenario_error)
        if physics is not None:
            found.extend(problems(network, scenario, physics))
        if found:
            st.warning("Not ready to compare:\n\n" + "\n\n".join(f"- {item}" for item in found))
        else:
            st.success("Ready to compare.")

        if st.button("Run comparison", type="primary", disabled=bool(found), key="run_comparison"):
            with st.spinner("Running the switched DAE and EPANET…"):
                st.session_state.comparison = run_comparison(
                    network, scenario, physics, positions=dict(st.session_state.positions)
                )
            st.session_state.plot_states = default_states(st.session_state.comparison)

    result = st.session_state.comparison
    with right:
        if result is None:
            st.info("Run a comparison to see plots and tables.")
            return
        try:
            fresh_physics = current_physics()
            fresh_scenario = assemble_scenario(network)
        except ValueError:
            fresh_physics = None
            fresh_scenario = None
        if fresh_physics is not None and result.is_outdated(network, fresh_scenario, fresh_physics):
            st.warning("The network or the settings changed after this result. Run again to refresh it.")
        _render_results(result)


def _render_results(result) -> None:
    st.caption(
        f"{engine_label(result)} · DAE {result.timings['dae_solve_sec']:.2f} s · "
        f"EPANET {result.timings['epanet_run_sec']:.2f} s"
    )
    if result.warnings:
        for warning in result.warnings:
            st.warning(warning)
    try:
        gap = max_abs_difference_after_last_switch(result)
        st.metric("Largest |DAE − EPANET| after the last switch", f"{gap:.3e}")
    except Exception:
        pass

    labels = list(result.dae.labels)
    if "plot_states" not in st.session_state:
        st.session_state.plot_states = default_states(result)
    st.session_state.plot_states = [label for label in st.session_state.plot_states if label in labels] or default_states(result)
    chosen = st.multiselect("States", labels, key="plot_states")
    kind_label = st.segmented_control("Head", ["Pressure head", "Total head"], key="plot_kind")
    show_impulses = st.checkbox("Show Dirac impulses", key="show_impulses")
    kind = "total" if kind_label == "Total head" else "pressure"
    if chosen:
        figure = _comparison_figure(result, chosen, kind, show_impulses)
        st.pyplot(figure, alt="DAE and EPANET trajectories for the selected states")
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        st.download_button(
            "Download plot PNG",
            buffer.getvalue(),
            file_name="comparison.png",
            mime="image/png",
            key="download_plot",
        )
    else:
        st.info("Choose at least one state to plot.")

    diff = pd.DataFrame(difference_rows(result, chosen or None), columns=["Time (s)", "State", "DAE", "EPANET", "DAE − EPANET"])
    equilibrium = pd.DataFrame(result.equilibrium)
    impulses = pd.DataFrame(result.impulses)
    parameters = pd.DataFrame(result.params, columns=["Element", "Parameter", "Value", "Unit"])
    tabs = st.tabs(["Differences", "Equilibrium", "Dirac impulses", "Parameters"])
    with tabs[0]:
        st.dataframe(diff, hide_index=True)
        st.download_button("Download differences CSV", diff.to_csv(index=False).encode("utf-8-sig"), "differences.csv", key="dl_diff")
    with tabs[1]:
        st.dataframe(equilibrium, hide_index=True)
        st.download_button("Download equilibrium CSV", equilibrium.to_csv(index=False).encode("utf-8-sig"), "equilibrium.csv", key="dl_eq")
    with tabs[2]:
        if impulses.empty:
            st.info("No Dirac impulse above the reporting threshold.")
        else:
            st.dataframe(impulses, hide_index=True)
        st.download_button("Download impulses CSV", impulses.to_csv(index=False).encode("utf-8-sig"), "impulses.csv", key="dl_imp")
    with tabs[3]:
        st.dataframe(parameters, hide_index=True)

    dae_frame = pd.DataFrame(result.dae.X, columns=result.dae.labels)
    dae_frame.insert(0, "t", result.dae.t)
    ep_frame = pd.DataFrame(result.epanet.X, columns=result.epanet.labels)
    ep_frame.insert(0, "t", result.epanet.t)
    dae_col, ep_col, inp_col = st.columns(3)
    dae_col.download_button("DAE series CSV", dae_frame.to_csv(index=False).encode("utf-8-sig"), "dae_series.csv", key="dl_dae")
    ep_col.download_button("EPANET series CSV", ep_frame.to_csv(index=False).encode("utf-8-sig"), "epanet_series.csv", key="dl_ep")
    try:
        inp_text, _export_map = export_epanet(
            result.snapshot, result.scenario, result.settings, positions=dict(st.session_state.positions)
        )
        inp_col.download_button("EPANET .inp", inp_text, file_name="network.inp", mime="text/plain", key="dl_inp")
    except Exception as exc:
        inp_col.caption(f"INP export is unavailable: {exc}")


def _comparison_figure(result, labels: List[str], kind: str, show_impulses: bool):
    count = len(labels)
    figure, axes = plt.subplots(count, 1, figsize=(8.2, max(2.4 * count, 3.2)), sharex=True, squeeze=False)
    arrows = impulse_arrows(result, labels) if show_impulses else {}
    for axis, label in zip(axes[:, 0], labels):
        segments, jumps, (time_ep, value_ep) = series(result, label, kind)
        plotted_dae = False
        for time_dae, value_dae in segments:
            axis.plot(time_dae, value_dae, color="#0B6E99", lw=1.6, label="DAE" if not plotted_dae else None)
            plotted_dae = True
        axis.plot(time_ep, value_ep, color="#E07A3D", lw=1.3, ls="--", label="EPANET")
        for instant, before, after in jumps:
            axis.plot([instant, instant], [before, after], color="#0B6E99", lw=1.0)
        # Arrow length is half the axis height, as in the desktop plot. Sizing
        # it by the coefficient pushed it far outside the axes, and the tight
        # bounding box then stretched the image into a tall blank strip.
        low, high = axis.get_ylim()
        height = 0.5 * ((high - low) if abs(high - low) > 1e-9 else 1.0)
        for instant, coefficient, base in arrows.get(label, []):
            tip = base + np.sign(coefficient or 1.0) * height
            axis.annotate(
                "",
                xy=(instant, tip),
                xytext=(instant, base),
                arrowprops={"arrowstyle": "-|>", "color": "#6B3FA0", "lw": 1.6},
            )
            low, high = min(low, tip), max(high, tip)
        axis.set_ylim(low, high)
        axis.set_ylabel(label, fontsize=8)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best", fontsize=8)
    axes[-1, 0].set_xlabel("Time (s)")
    figure.tight_layout()
    return figure
