"""core/matrices.py – Symbolic DAE-matrix builder for water distribution networks.

Row / column order (reproduces legacy ``create_network_matrices`` exactly for
networks without tanks, and fixes the two documented bugs for general networks):

    Columns / rows
    ~~~~~~~~~~~~~~
    0 .. N_np-1                  : non-pumped pipe flow  Q_i
    N_np .. N_p-1                : pumped pipe flow      Q_k  (pump flow)
    N_p .. N_p + N_np - 1        : left pressure  P_l_i  (one per non-pumped pipe)
    N_p + N_np .. N_p + 2*N_np-1 : right pressure P_r_i  (one per non-pumped pipe)
    N_p + 2*N_np .. .. + N_j - 1 : junction pressure P_J
    .. + N_r - 1                 : reservoir pressure P_R
    .. + N_t - 1                 : tank pressure P_T   (also tank state rows)

where  N_p  = total number of pipes,
       N_np = number of non-pumped pipes.

Bug fixes versus legacy
-----------------------
ORDER BUG  – legacy assumed the first ``non_pumped`` entries in the pipes list
             were the non-pumped ones and parsed numbers from IDs.  Here we
             split into separate lists and build ``pipe_flow_col`` / ``node_col``
             dict lookups so the matrix is independent of creation order.
TANK BUG   – legacy only wrote ``+1`` when a tank is the *start* of a pipe and
             left the row zero when a tank *only receives* flow.  Here the tank
             row follows the same sign convention as junction rows:
             ``+TANK_INFLOW_SIGN`` for a pipe *ending* at the tank and
             ``-TANK_INFLOW_SIGN`` for a pipe *starting* at the tank.

Module-level constant (see ASSUMPTIONS.md for rationale)
---------------------------------------------------------
TANK_INFLOW_SIGN = +1   (can be flipped here to switch convention)
"""

from __future__ import annotations

from typing import Union

from .models import Network, Node, Pipe

# ---------------------------------------------------------------------------
# Convention constant
# ---------------------------------------------------------------------------
TANK_INFLOW_SIGN: int = +1
"""Sign applied to the flow column entry in a tank row when a pipe *ends* at
that tank (+1 = inflow positive).  Set to -1 to reverse convention."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_matrices(network: Union[Network, dict]) -> dict:
    """Build symbolic E, A, F, G matrices for *network*.

    Parameters
    ----------
    network : Network or dict
        A :class:`~core.models.Network` instance or a plain dict as returned by
        ``Network.to_dict()``.

    Returns
    -------
    dict with keys:
        ``A_matrix``   – K×K list-of-lists of strings
        ``E_matrix``   – K×K list-of-lists (int 0/1)
        ``F_matrix``   – K-element list of strings
        ``G_matrix``   – K-element list of strings
        ``components`` – dict with counts (non_pumped, pumped, junctions, …)
        ``K_dimension``– int K
        ``labels``     – list[str] of column labels matching the column order
                         described in the module docstring
    """
    # ------------------------------------------------------------------
    # Normalise input
    # ------------------------------------------------------------------
    if isinstance(network, dict):
        network = Network.from_dict(network)

    nodes: list[Node] = network.nodes
    pipes: list[Pipe] = network.pipes

    # ------------------------------------------------------------------
    # Partition pipes (preserve creation order within each group)
    # ------------------------------------------------------------------
    non_pumped: list[Pipe] = [p for p in pipes if not p.has_pump]
    pumped:     list[Pipe] = [p for p in pipes if p.has_pump]

    N_np = len(non_pumped)   # number of non-pumped pipes
    N_pu = len(pumped)       # number of pumped pipes
    N_p  = N_np + N_pu       # total pipes

    # ------------------------------------------------------------------
    # Node counts and ordered sub-lists
    # ------------------------------------------------------------------
    junctions  = [n for n in nodes if n.type == "junction"]
    reservoirs = [n for n in nodes if n.type == "reservoir"]
    tanks      = [n for n in nodes if n.type == "tank"]

    N_j = len(junctions)
    N_r = len(reservoirs)
    N_t = len(tanks)

    # ------------------------------------------------------------------
    # Matrix dimension (matches legacy formula exactly)
    #   K = len(pipes) + 2*non_pumped + junctions + reservoirs + tanks
    # ------------------------------------------------------------------
    K = N_p + 2 * N_np + N_j + N_r + N_t

    # ------------------------------------------------------------------
    # Column index lookup tables  (BUG FIX: dict-based, never parse IDs)
    # ------------------------------------------------------------------
    # Pipe flow columns:  non-pumped first, then pumped
    pipe_flow_col: dict[str, int] = {}
    for i, p in enumerate(non_pumped):
        pipe_flow_col[p.id] = i
    for i, p in enumerate(pumped):
        pipe_flow_col[p.id] = N_np + i

    # Pressure columns:  Pl_i, Pr_i, junctions, reservoirs, tanks
    _pl_offset  = N_p              # first Pl column
    _pr_offset  = N_p + N_np      # first Pr column
    _j_offset   = N_p + 2 * N_np
    _r_offset   = _j_offset + N_j
    _t_offset   = _r_offset + N_r

    pl_col: dict[str, int] = {p.id: _pl_offset + i for i, p in enumerate(non_pumped)}
    pr_col: dict[str, int] = {p.id: _pr_offset + i for i, p in enumerate(non_pumped)}

    node_col: dict[str, int] = {}
    for i, n in enumerate(junctions):
        node_col[n.id] = _j_offset + i
    for i, n in enumerate(reservoirs):
        node_col[n.id] = _r_offset + i
    for i, n in enumerate(tanks):
        node_col[n.id] = _t_offset + i

    # ------------------------------------------------------------------
    # Initialise matrices
    # ------------------------------------------------------------------
    E: list[list] = [[0] * K for _ in range(K)]
    A: list[list] = [["0"] * K for _ in range(K)]
    F: list[str]  = ["0"] * K
    G: list[str]  = ["0"] * K

    # ------------------------------------------------------------------
    # E diagonal
    # ------------------------------------------------------------------
    # Non-pumped pipe rows get E[i][i] = 1
    for i in range(N_np):
        E[i][i] = 1

    # Tank rows get E[t][t] = 1
    for i in range(N_t):
        row = _t_offset + i
        E[row][row] = 1

    # ------------------------------------------------------------------
    # Non-pumped pipe flow rows  (rows 0 .. N_np-1)
    # ------------------------------------------------------------------
    for i, pipe in enumerate(non_pumped):
        row = i                     # flow row for pipe i
        # G: exact legacy string
        G[row] = f"-C_aQ{i + 1}|Q{i + 1}|"
        # Pl and Pr coefficients  (c{i+1} … keep 1-based from creation order)
        A[row][pl_col[pipe.id]] = f"c{i + 1}"
        A[row][pr_col[pipe.id]] = f"-c{i + 1}"

    # ------------------------------------------------------------------
    # Pumped pipe flow rows  (rows N_np .. N_p-1)
    # ------------------------------------------------------------------
    for i, pipe in enumerate(pumped):
        row      = N_np + i
        sym_idx  = N_np + i + 1    # symbol number (continues from non-pumped)
        G[row]   = "-S_puG_pu(Q_pu)"
        # Diagonal  1-s_k  (A only; E[row][row] stays 0)
        A[row][row] = f"1-s{sym_idx}"

        # Start-node pressure
        if pipe.start in node_col:
            A[row][node_col[pipe.start]] = f"-s{sym_idx}"
        # End-node pressure
        if pipe.end in node_col:
            A[row][node_col[pipe.end]] = f"s{sym_idx}"

    # ------------------------------------------------------------------
    # Pipe end-side rows  (rows N_p .. N_p+N_np-1)
    # ------------------------------------------------------------------
    for i, pipe in enumerate(non_pumped):
        row     = N_p + i
        sym_idx = i + 1
        A[row][pipe_flow_col[pipe.id]] = f"1-s{sym_idx}"
        A[row][pr_col[pipe.id]]        = f"s{sym_idx}"
        # Connection to end node
        if pipe.end in node_col:
            A[row][node_col[pipe.end]] = f"-s{sym_idx}"

    # ------------------------------------------------------------------
    # Pipe start-side rows  (rows N_p+N_np .. N_p+2*N_np-1)
    # ------------------------------------------------------------------
    for i, pipe in enumerate(non_pumped):
        row = N_p + N_np + i
        A[row][pl_col[pipe.id]] = "1"
        # Connection to start node
        if pipe.start in node_col:
            A[row][node_col[pipe.start]] = "-1"

    # ------------------------------------------------------------------
    # Junction rows
    # ------------------------------------------------------------------
    for j_idx, jnode in enumerate(junctions):
        row = _j_offset + j_idx
        for pipe in pipes:
            if pipe.end == jnode.id:
                A[row][pipe_flow_col[pipe.id]] = "1"
            elif pipe.start == jnode.id:
                A[row][pipe_flow_col[pipe.id]] = "-1"

    # ------------------------------------------------------------------
    # Reservoir rows
    # ------------------------------------------------------------------
    for r_idx, rnode in enumerate(reservoirs):
        row      = _r_offset + r_idx
        col      = node_col[rnode.id]
        A[row][col] = "1"
        F[row]      = f"-P_{chr(97 + r_idx)}"   # -P_a, -P_b, …

    # ------------------------------------------------------------------
    # Tank rows  (BUG FIX: use TANK_INFLOW_SIGN for both endpoints)
    # ------------------------------------------------------------------
    for t_idx, tnode in enumerate(tanks):
        row = _t_offset + t_idx
        for pipe in pipes:
            if pipe.end == tnode.id:
                # Flow arrives at tank
                A[row][pipe_flow_col[pipe.id]] = str(TANK_INFLOW_SIGN)
            elif pipe.start == tnode.id:
                # Flow leaves tank
                A[row][pipe_flow_col[pipe.id]] = str(-TANK_INFLOW_SIGN)

    # ------------------------------------------------------------------
    # Labels vector (column names)
    # The legacy display_E_matrix interleaves Pl/Pr wrongly; the display_A_matrix
    # was correct.  We reproduce the *correct* A-matrix column order:
    #   all flows (non-pumped, then pumped), all Pl, all Pr,
    #   junctions, reservoirs, tanks.
    # ------------------------------------------------------------------
    labels: list[str] = []
    for i, p in enumerate(non_pumped):
        labels.append(f"Pipe_{i + 1}_Flow")
    for i, p in enumerate(pumped):
        labels.append(f"Pumped_{i + 1}_Flow")
    for i, p in enumerate(non_pumped):
        labels.append(f"Pipe_{i + 1}_L")
    for i, p in enumerate(non_pumped):
        labels.append(f"Pipe_{i + 1}_R")
    for j in junctions:
        labels.append(j.id)
    for r in reservoirs:
        labels.append(r.id)
    for t in tanks:
        labels.append(t.id)

    assert len(labels) == K, f"labels {len(labels)} != K {K}"

    # ------------------------------------------------------------------
    # Return
    # ------------------------------------------------------------------
    return {
        "A_matrix": A,
        "E_matrix": E,
        "F_matrix": F,
        "G_matrix": G,
        "components": {
            "non_pumped": N_np,
            "pumped":     N_pu,
            "junctions":  N_j,
            "reservoirs": N_r,
            "tanks":      N_t,
        },
        "K_dimension": K,
        "labels": labels,
    }
