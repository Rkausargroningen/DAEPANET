"""core/params.py – Physical parameter completeness checking.

Units:
- Pipe length: metres (m)
- Pipe diameter: metres (m)
- Pipe roughness: metres (m) [absolute surface roughness]
- Node elevation: metres (m)
- Reservoir total head: metres (m)
"""

from __future__ import annotations

from typing import List, Union
from core.models import Network


def missing_parameters(network: Union[Network, dict]) -> List[str]:
    """Check a network for missing physical parameters required for simulation.

    Rules:
    - Every pipe needs length, diameter, and roughness.
    - Every reservoir needs head and elevation.
    - Every junction needs elevation.

    Returns
    -------
    list of str
        Human-readable messages detailing each missing parameter, or an empty
        list if all required parameters are specified.
    """
    if isinstance(network, dict):
        network = Network.from_dict(network)

    missing: List[str] = []

    for pipe in network.pipes:
        if pipe.length is None:
            missing.append(f"Pipe {pipe.id}: length is not set")
        if pipe.diameter is None:
            missing.append(f"Pipe {pipe.id}: diameter is not set")
        if pipe.roughness is None:
            missing.append(f"Pipe {pipe.id}: roughness is not set")

    for node in network.nodes:
        if node.type == "reservoir":
            if node.head is None:
                missing.append(f"Reservoir {node.id}: head is not set")
            if node.elevation is None:
                missing.append(f"Reservoir {node.id}: elevation is not set")
        elif node.type == "junction":
            if node.elevation is None:
                missing.append(f"Junction {node.id}: elevation is not set")

    return missing
