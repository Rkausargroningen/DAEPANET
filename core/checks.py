"""core/checks.py – Numerical and structural analysis for water networks.

Public API
----------
numeric_pencil(matrices)
    Substitute s=1, c=1 into the symbolic E and A matrices and return float
    numpy arrays.

is_regular(E, A)
    Test whether the matrix pencil (lambda*E - A) is regular (non-identically-
    zero determinant) using ``np.linalg.slogdet`` with a fixed random seed.

incidence_matrices(network)
    Build the full, reduced, and modified incidence matrices.

validate_network(network)
    Return a list of human-readable warning strings covering isolated nodes,
    disconnected supply-free components, pump cycles (cyclomatic check), and
    rank deficiency.
"""

from __future__ import annotations

import re
from typing import Union

import networkx as nx
import numpy as np

from .models import Network, Node, Pipe

# ---------------------------------------------------------------------------
# Numeric pencil evaluation
# ---------------------------------------------------------------------------

# Regex substitution: symbols like c1, c2, s1, s2 → 1
# "1-s1" → "1-1" = "0"; "s1" → "1"; "c1" → "1"; "-c1" → "-1"
_SYM_RE = re.compile(r"c\d+|(?<!\d)s\d+")


def _eval_sym(token: str) -> float:
    """Substitute c_i=1, s_i=1 into a symbolic string and evaluate."""
    if token == "0":
        return 0.0
    # Replace every symbolic token with 1
    replaced = _SYM_RE.sub("1", token)
    try:
        return float(eval(replaced))  # noqa: S307
    except Exception:
        return 0.0


def numeric_pencil(matrices: dict) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate symbolic E and A matrices with all valves open, pumps on.

    Substitutes ``s_i = 1`` (switch closed / pump on) and ``c_i = 1``
    (conductance coefficient) into every entry of the symbolic A matrix,
    and returns float numpy arrays.

    Parameters
    ----------
    matrices : dict
        As returned by :func:`~core.matrices.build_matrices`.

    Returns
    -------
    E_num, A_num : np.ndarray
        Float64 arrays of shape ``(K, K)``.
    """
    E_sym = matrices["E_matrix"]
    A_sym = matrices["A_matrix"]

    K = len(E_sym)
    E_num = np.array(E_sym, dtype=float)   # already 0/1 integers
    A_num = np.zeros((K, K), dtype=float)
    for r in range(K):
        for c in range(K):
            A_num[r, c] = _eval_sym(A_sym[r][c])
    return E_num, A_num


# ---------------------------------------------------------------------------
# Regularity check
# ---------------------------------------------------------------------------

# Fixed random seed guarantees reproducible results across runs.
_REGULARITY_SEED = 42
_REGULARITY_N_SAMPLES = 8
_REGULARITY_TOL = 1e-7  # |det| below this is treated as zero (see ASSUMPTIONS.md)


def is_regular(E: np.ndarray, A: np.ndarray, *, tol: float = _REGULARITY_TOL) -> bool:
    """Return True if the pencil ``(lambda*E - A)`` is non-singular for at
    least one of several sampled complex lambda values.

    Strategy
    --------
    We sample ``_REGULARITY_N_SAMPLES`` complex values from a fixed-seed RNG,
    evaluate ``slogdet(lambda*E - A)`` for each, and return True as soon as
    we find one where ``exp(log|det|) > tol``.  Using ``slogdet`` avoids
    floating-point overflow on large matrices.

    Parameters
    ----------
    E, A : np.ndarray
        Float64 square matrices (same shape).
    tol : float
        Threshold for treating the determinant as non-zero. Default 1e-7.

    Returns
    -------
    bool
    """
    rng = np.random.default_rng(_REGULARITY_SEED)
    K = E.shape[0]
    # Sample real and imaginary parts from a moderate range to avoid
    # accidentally landing on true eigenvalues
    lams_real = rng.uniform(-5.0, 5.0, _REGULARITY_N_SAMPLES)
    lams_imag = rng.uniform(0.5, 5.0, _REGULARITY_N_SAMPLES)   # keep Im != 0

    for lr, li in zip(lams_real, lams_imag):
        lam = complex(lr, li)
        M = lam * E - A            # complex128 matrix
        sign, logabsdet = np.linalg.slogdet(M)
        if sign != 0 and logabsdet > np.log(tol):
            return True
    return False


# ---------------------------------------------------------------------------
# Incidence matrices
# ---------------------------------------------------------------------------

def incidence_matrices(network: Union[Network, dict]) -> dict:
    """Build standard network incidence matrices.

    Sign convention: ``-1`` at the start (initial) node of each edge,
    ``+1`` at the end (terminal) node.

    Parameters
    ----------
    network : Network or dict

    Returns
    -------
    dict with keys:
        ``incidence``     – full (n × m) incidence matrix (numpy int8)
        ``reduced``       – (n-1) × m  – reference row deleted
        ``modified``      – (n_j) × m  – all reservoir and tank rows deleted
        ``node_labels``   – list of node IDs in row order
        ``edge_labels``   – list of pipe IDs in column order
        ``reference_node``– ID of the deleted reference row in *reduced*
    """
    if isinstance(network, dict):
        network = Network.from_dict(network)

    nodes: list[Node] = network.nodes
    pipes: list[Pipe] = network.pipes

    n = len(nodes)
    m = len(pipes)

    node_labels = [nd.id for nd in nodes]
    edge_labels  = [p.id  for p in pipes]
    node_idx     = {nd.id: i for i, nd in enumerate(nodes)}

    B = np.zeros((n, m), dtype=np.int8)
    for j, pipe in enumerate(pipes):
        B[node_idx[pipe.start], j] = -1
        B[node_idx[pipe.end],   j] = +1

    # Reduced: delete reference row
    # Preference: first reservoir → first tank → first node
    reservoirs = [nd for nd in nodes if nd.type == "reservoir"]
    tanks      = [nd for nd in nodes if nd.type == "tank"]
    if reservoirs:
        ref_node = reservoirs[0].id
    elif tanks:
        ref_node = tanks[0].id
    else:
        ref_node = nodes[0].id

    ref_idx = node_idx[ref_node]
    B_reduced = np.delete(B, ref_idx, axis=0)

    # Modified: delete ALL reservoir and tank rows
    supply_indices = sorted(
        node_idx[nd.id]
        for nd in nodes
        if nd.type in ("reservoir", "tank")
    )
    B_modified = np.delete(B, supply_indices, axis=0)

    return {
        "incidence":      B,
        "reduced":        B_reduced,
        "modified":       B_modified,
        "node_labels":    node_labels,
        "edge_labels":    edge_labels,
        "reference_node": ref_node,
    }


# ---------------------------------------------------------------------------
# Network validation
# ---------------------------------------------------------------------------

def validate_network(network: Union[Network, dict]) -> list[str]:
    """Return a list of human-readable warning strings.

    Checks performed
    ----------------
    1. **Isolated nodes** – nodes with no pipe attached.
    2. **Supply-free components** – undirected connected components that
       contain neither a reservoir nor a tank.
    3. **Pump cycles** – cyclomatic-number check (paper method):
       merge ALL reservoir and tank nodes into one virtual supply super-node,
       build an undirected multigraph of pump edges only, and flag every
       connected component where  E - V + C > 0  (E=edges, V=vertices,
       C=components of *that* sub-graph).  A cross-check with the numpy rank
       of the pump columns of the modified incidence matrix is also performed;
       a discrepancy is reported as an additional warning.
    4. **Rank deficiency** – rank of the modified incidence matrix is less
       than the number of junctions.

    Parameters
    ----------
    network : Network or dict

    Returns
    -------
    list[str]
        Empty list ↔ no warnings.
    """
    if isinstance(network, dict):
        network = Network.from_dict(network)

    nodes: list[Node] = network.nodes
    pipes: list[Pipe] = network.pipes

    warnings: list[str] = []

    # Build lookup sets
    node_map = {n.id: n for n in nodes}
    node_ids = set(node_map)

    # Collect node adjacency (undirected)
    adjacent: dict[str, set[str]] = {n.id: set() for n in nodes}
    for pipe in pipes:
        adjacent[pipe.start].add(pipe.end)
        adjacent[pipe.end].add(pipe.start)

    # ---- 1. Isolated nodes ------------------------------------------------
    isolated = [n.id for n in nodes if not adjacent[n.id]]
    if isolated:
        warnings.append(
            f"Isolated node(s) with no pipe attached: {', '.join(isolated)}"
        )

    # ---- 2. Supply-free connected components ------------------------------
    # Build undirected graph (all edges)
    G_full = nx.Graph()
    G_full.add_nodes_from(n.id for n in nodes)
    for pipe in pipes:
        G_full.add_edge(pipe.start, pipe.end)

    supply_ids = {
        n.id for n in nodes if n.type in ("reservoir", "tank")
    }
    for component in nx.connected_components(G_full):
        if len(component) == 1:
            # Single isolated node – already reported above
            continue
        if not component & supply_ids:
            member_nodes = sorted(component)
            warnings.append(
                f"Nodes {', '.join(member_nodes)} are not connected to any "
                f"supply (no reservoir or tank in this component)."
            )

    # ---- 3. Pump cycles (cyclomatic-number method from paper) --------------
    #
    # Algorithm:
    #   a) Collapse all supply nodes (reservoir + tank) into ONE virtual node
    #      called "__supply__".
    #   b) Keep only pump edges; build an undirected multigraph to handle
    #      parallel pump edges.
    #   c) For each connected component of that pump multigraph:
    #        cyclomatic_number = E - V + 1
    #      (For a connected graph with one component C=1, so E-V+C = E-V+1.)
    #      If cyclomatic_number > 0 a pump cycle exists.
    #
    # Cross-check: rank of pump columns in the modified incidence matrix.
    #   If rank(pump_cols) < number_of_pump_edges,
    #   there is linear dependence among pump columns → another indicator
    #   of a pump cycle (reported separately if it disagrees with above).

    _SUPPLY = "__supply__"
    pump_pipes = [p for p in pipes if p.has_pump]

    # Build undirected pump multigraph with supply collapsed
    PG = nx.MultiGraph()
    for pp in pump_pipes:
        u = _SUPPLY if node_map[pp.start].type in ("reservoir", "tank") else pp.start
        v = _SUPPLY if node_map[pp.end].type   in ("reservoir", "tank") else pp.end
        PG.add_edge(u, v, pipe_id=pp.id)

    # Add isolated pump-graph nodes that might have no pump edges
    # (not needed for cycle detection but kept for completeness)
    flagged_pump_ids: list[str] = []

    for component_nodes in nx.connected_components(PG):
        sub = PG.subgraph(component_nodes)
        V = sub.number_of_nodes()
        E_edges = sub.number_of_edges()
        cyclomatic = E_edges - V + 1   # for a connected component C=1
        if cyclomatic > 0:
            involved = [
                d["pipe_id"]
                for _, _, d in sub.edges(data=True)
            ]
            flagged_pump_ids.extend(involved)
            warnings.append(
                f"Pump cycle detected (cyclomatic number {cyclomatic}) "
                f"involving pumps: {', '.join(sorted(set(involved)))}."
            )

    # Cross-check with rank of pump columns in modified incidence
    inc = incidence_matrices(network)
    B_mod = inc["modified"]          # junctions-only rows
    edge_labels = inc["edge_labels"]

    if pump_pipes and B_mod.shape[0] > 0:
        pump_col_indices = [edge_labels.index(pp.id) for pp in pump_pipes]
        pump_block = B_mod[:, pump_col_indices]
        rank_pump = int(np.linalg.matrix_rank(pump_block))
        n_pump_cols = len(pump_pipes)
        if rank_pump < n_pump_cols:
            # Column dependence → pump cycle exists; cross-check warning only
            # if cyclomatic method did NOT flag it (they should agree)
            if not flagged_pump_ids:
                warnings.append(
                    f"Rank check: pump columns of modified incidence are rank "
                    f"{rank_pump} < {n_pump_cols} pump pipes – possible pump "
                    f"cycle not caught by cyclomatic method."
                )

    # ---- 4. Rank of modified incidence ------------------------------------
    junctions = [n for n in nodes if n.type == "junction"]
    if junctions and B_mod.shape[0] > 0:
        rank_mod = int(np.linalg.matrix_rank(B_mod))
        n_junctions = len(junctions)
        if rank_mod < n_junctions:
            warnings.append(
                f"Modified incidence matrix rank ({rank_mod}) < number of "
                f"junction rows ({n_junctions})."
            )

    return warnings
