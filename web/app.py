"""Water Network Modeler in the browser.

Run from the repository root:

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="Water Network Modeler",
    page_icon=":material/water_drop:",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app_info import APP_NAME, __version__
from core.epanet_runner import get_epanet_version
from core.examples import (
    example_broken_network,
    example_broken_network_positions,
    example_paper_network,
    specimen_network,
    specimen_physics,
)
from core.io import load_project_full, save_project
from core.params import missing_parameters
from web.state import apply_physics, init_state, replace_network
from web.views import (
    assemble_scenario,
    current_physics,
    load_specimen_scenario,
    render_checks,
    render_comparison,
    render_incidence,
    render_matrices,
    render_network,
    store_scenario,
)

init_state()


def _project_bytes() -> bytes:
    scenario = None
    physics = None
    try:
        scenario = assemble_scenario(st.session_state.network)
    except Exception:
        scenario = None
    try:
        physics = current_physics()
    except Exception:
        physics = None
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "network.wnet"
        save_project(
            path,
            st.session_state.network,
            dict(st.session_state.positions),
            dict(st.session_state.edge_bends),
            scenario,
            physics,
        )
        return path.read_bytes()


def _open_upload(upload) -> None:
    token = hashlib.sha256(upload.getvalue()).hexdigest()
    if token == st.session_state.upload_token:
        return
    st.session_state.upload_token = token
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "upload.wnet"
        path.write_bytes(upload.getvalue())
        project = load_project_full(path)
    replace_network(project.network, project.positions, project.edge_bends, upload.name)
    st.session_state.preset_choice = "Custom"
    if project.physics is not None:
        apply_physics(project.physics)
    if project.scenario is not None:
        store_scenario(project.scenario)


def _epanet_label() -> str:
    if "epanet_label" not in st.session_state:
        try:
            st.session_state.epanet_label = get_epanet_version()
        except Exception as exc:
            st.session_state.epanet_label = f"not available ({exc.__class__.__name__})"
    return st.session_state.epanet_label


st.title(APP_NAME)
st.caption(f"Version {__version__} · graph model, switched DAE, and EPANET comparison")

with st.sidebar:
    st.header("Project")
    if st.button("New empty network", width="stretch", key="new_network"):
        from core.models import Network
        replace_network(Network(), name="Untitled")
        st.session_state.preset_choice = "Custom"
        st.session_state.scenario_title = "Custom scenario"
        st.session_state.t_end = 120

    st.markdown("**Examples**")
    if st.button("Paper network", width="stretch", key="load_paper"):
        replace_network(example_paper_network(), name="Paper network")
        st.session_state.preset_choice = "Custom"
    if st.button("Broken network", width="stretch", key="load_broken"):
        replace_network(
            example_broken_network(),
            example_broken_network_positions(),
            name="Broken network",
        )
        st.session_state.preset_choice = "Custom"
    if st.button("Specimen for EPANET", width="stretch", key="load_specimen"):
        specimen, positions = specimen_network()
        replace_network(specimen, positions, name="Specimen")
        apply_physics(specimen_physics())
        load_specimen_scenario()

    upload = st.file_uploader("Open a .wnet file", type=["wnet", "json"])
    if upload is not None:
        try:
            _open_upload(upload)
        except Exception as exc:
            st.error(f"Could not open that file: {exc}")

    st.download_button(
        "Download .wnet",
        data=_project_bytes(),
        file_name=f"{st.session_state.project_name}.wnet",
        mime="application/json",
        width="stretch",
        key="download_wnet",
    )

    missing = missing_parameters(st.session_state.network)
    st.subheader("Parameters")
    if not st.session_state.network.nodes:
        st.caption("The network is empty.")
    elif not missing:
        st.caption("Every value needed for a comparison is set.")
    else:
        st.caption(f"{len(missing)} value(s) still blank.")
        for item in missing[:8]:
            st.caption(f"• {item}")
        if len(missing) > 8:
            st.caption(f"• … and {len(missing) - 8} more")

    with st.expander("About"):
        st.markdown(
            f"**{APP_NAME}** {__version__}\n\n"
            "The same calculation code as the desktop app: symbolic matrices, "
            "regularity checks, the switched DAE solver, and the EPANET engine.\n\n"
            f"EPANET engine: {_epanet_label()}."
        )
        notices = ROOT / "THIRD_PARTY_NOTICES.txt"
        if notices.is_file():
            st.markdown("**Third-party notices**")
            st.text(notices.read_text(encoding="utf-8"))

network_tab, matrix_tab, checks_tab, incidence_tab, comparison_tab = st.tabs(
    ["Network", "Matrices", "Checks", "Incidence", "Comparison with EPANET"],
    key="main_tabs",
    on_change="rerun",
)
if network_tab.open:
    with network_tab:
        render_network()
if matrix_tab.open:
    with matrix_tab:
        render_matrices()
if checks_tab.open:
    with checks_tab:
        render_checks()
if incidence_tab.open:
    with incidence_tab:
        render_incidence()
if comparison_tab.open:
    with comparison_tab:
        render_comparison()
