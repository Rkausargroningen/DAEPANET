"""core – pure calculation kernel for the Water Network application.

Exposes:
    Node, Pipe, Network  – data model
    build_matrices        – symbolic DAE matrices (E, A, F, G)
    numeric_pencil        – evaluate symbolic matrices at s=1, c=1
    is_regular            – test matrix pencil regularity
    incidence_matrices    – standard incidence / reduced / modified
    validate_network      – human-readable network warnings
"""

from .models import Node, Pipe, Network
from .matrices import build_matrices
from .checks import numeric_pencil, is_regular, incidence_matrices, validate_network

__all__ = [
    "Node", "Pipe", "Network",
    "build_matrices",
    "numeric_pencil", "is_regular",
    "incidence_matrices", "validate_network",
]
