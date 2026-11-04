"""Bidirectional network canvas embedded in Streamlit."""

from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

_COMPONENT = components.declare_component(
    "water_network_canvas",
    path=str(Path(__file__).resolve().parent / "network_canvas"),
)


def network_canvas(spec: dict, key: str = "network_canvas"):
    """Draw the network and return the latest drag, bend, or selection event."""
    return _COMPONENT(spec=spec, key=key, default=None)
