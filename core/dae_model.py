"""core/dae_model.py – Switched DAE numeric model builder and steady-state solver.

Classes
-------
UnsupportedNetworkError
    Raised when a network cannot be converted to a numeric DAE model.
SwitchedDAE
    Numeric DAE model representing E x' = A_sigma x + N gbar(M x) + f.

Functions
---------
check_supported(network)
    Return a list of plain-language reasons why a network cannot be solved.
build_switched_dae(network, settings)
    Construct and return a SwitchedDAE instance or raise UnsupportedNetworkError.
steady_state(model, mode)
    Solve for the stationary hydraulic state using pure-NumPy Newton iteration.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np

from core.models import Network, Node, Pipe
from core.physics import PhysicsSettings, pipe_friction_term
from core.params import missing_parameters
from core.checks import validate_network
from core.scenario import ValveId, Scenario, Mode


class UnsupportedNetworkError(ValueError):
    """Raised when a network topology or parameter configuration is unsupported."""

    def __init__(self, reasons: List[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = list(reasons)


def check_supported(network: Union[Network, dict]) -> List[str]:
    """Check whether a network is supported by the numeric switched-DAE model.

    Rules
    -----
    - Pumps are not supported in the numeric DAE model.
    - Tanks are not supported in the numeric DAE model.
    - Missing physical parameters (length, diameter, roughness, elevation, head) are not allowed.
    - At least one reservoir must be present.
    - Network validation warnings that make the network unsolvable (unsupplied components,
      isolated nodes) are rejected.

    Returns
    -------
    list of str
        Plain-language descriptions of every reason the network cannot be converted,
        or an empty list if the network is fully supported.
    """
    if isinstance(network, dict):
        network = Network.from_dict(network)

    reasons: List[str] = []

    # 1. Pumps
    pumps = [p.id for p in network.pipes if p.has_pump]
    if pumps:
        reasons.append(
            f"Pumps are not supported in the numeric DAE model (found: {', '.join(pumps)})."
        )

    # 2. Tanks
    tanks = [n.id for n in network.nodes if n.type == "tank"]
    if tanks:
        reasons.append(
            f"Tanks are not supported in the numeric DAE model (found: {', '.join(tanks)})."
        )

    # 3. Reservoir presence
    reservoirs = [n.id for n in network.nodes if n.type == "reservoir"]
    if not reservoirs:
        reasons.append("Network must contain at least one reservoir.")

    # 4. Missing physical parameters
    missing = missing_parameters(network)
    if missing:
        reasons.extend(missing)

    # 5. Topological unsolvability from validate_network
    topo_warnings = validate_network(network)
    for w in topo_warnings:
        if "not connected to any supply" in w or "Isolated node" in w:
            reasons.append(w)

    return reasons


class SwitchedDAE:
    """Numeric switched DAE model: E x' = A_sigma x + N gbar(M x) + f.

    Maintains identical column and state ordering as symbolic build_matrices:
        0 .. N_p - 1: pipe flows Q_i
        N_p .. 2*N_p - 1: left pressures P_l_i
        2*N_p .. 3*N_p - 1: right pressures P_r_i
        3*N_p .. 3*N_p + N_j - 1: junction pressures P_J
        .. + N_r - 1: reservoir pressures P_R
    """

    def __init__(self, network: Network, settings: PhysicsSettings) -> None:
        self.network = network
        self.settings = settings

        self.pipes: List[Pipe] = list(network.pipes)
        self.junctions: List[Node] = [n for n in network.nodes if n.type == "junction"]
        self.reservoirs: List[Node] = [n for n in network.nodes if n.type == "reservoir"]

        self.N_p = len(self.pipes)
        self.N_j = len(self.junctions)
        self.N_r = len(self.reservoirs)
        self.K = 3 * self.N_p + self.N_j + self.N_r

        # Index offsets
        self.pl_offset = self.N_p
        self.pr_offset = 2 * self.N_p
        self.j_offset = 3 * self.N_p
        self.r_offset = self.j_offset + self.N_j

        self.pipe_col: Dict[str, int] = {p.id: i for i, p in enumerate(self.pipes)}
        self.pl_col: Dict[str, int] = {p.id: self.pl_offset + i for i, p in enumerate(self.pipes)}
        self.pr_col: Dict[str, int] = {p.id: self.pr_offset + i for i, p in enumerate(self.pipes)}
        self.node_col: Dict[str, int] = {}
        for i, j in enumerate(self.junctions):
            self.node_col[j.id] = self.j_offset + i
        for i, r in enumerate(self.reservoirs):
            self.node_col[r.id] = self.r_offset + i

        self.node_map: Dict[str, Node] = {n.id: n for n in network.nodes}

        # Labels
        self.labels: List[str] = []
        for i in range(self.N_p):
            self.labels.append(f"Pipe_{i + 1}_Flow")
        for i in range(self.N_p):
            self.labels.append(f"Pipe_{i + 1}_L")
        for i in range(self.N_p):
            self.labels.append(f"Pipe_{i + 1}_R")
        for j in self.junctions:
            self.labels.append(j.id)
        for r in self.reservoirs:
            self.labels.append(r.id)

        # Elevation per state (np.nan for flows)
        elev = np.full(self.K, np.nan, dtype=float)
        for i, p in enumerate(self.pipes):
            elev[self.pl_offset + i] = float(self.node_map[p.start].elevation or 0.0)
            elev[self.pr_offset + i] = float(self.node_map[p.end].elevation or 0.0)
        for i, j in enumerate(self.junctions):
            elev[self.j_offset + i] = float(j.elevation or 0.0)
        for i, r in enumerate(self.reservoirs):
            elev[self.r_offset + i] = float(r.elevation or 0.0)
        self._elevation = elev

        # Valves present in network
        self.valves: List[ValveId] = []
        for p in self.pipes:
            if p.valve_start:
                self.valves.append(ValveId(p.id, "start"))
            if p.valve_end:
                self.valves.append(ValveId(p.id, "end"))

        # Projection matrices M and N
        self.M = np.zeros((self.N_p, self.K), dtype=float)
        self.M[:, :self.N_p] = np.eye(self.N_p)

        self.N = np.zeros((self.K, self.N_p), dtype=float)
        self.N[:self.N_p, :] = np.eye(self.N_p)

    @property
    def elevation(self) -> np.ndarray:
        """Elevation for each state in metres (NaN for pipe flow states)."""
        return self._elevation.copy()

    def _get_valve_switch(self, mode: Optional[Union[Dict[Any, int], Mode, Tuple[int, ...], List[int]]], pipe_id: str, end: str) -> int:
        if mode is None:
            return 1
        if isinstance(mode, (tuple, list)):
            if len(mode) == len(self.valves):
                vid = ValveId(pipe_id, end)
                if vid in self.valves:
                    return int(mode[self.valves.index(vid)])
            return 1
        if isinstance(mode, Mode):
            state_dict = mode.state
        elif hasattr(mode, "state"):
            state_dict = mode.state
        else:
            state_dict = mode

        vid = ValveId(pipe_id, end)
        if isinstance(state_dict, dict):
            if vid in state_dict:
                return int(state_dict[vid])
            s_key = f"{pipe_id}:{end}"
            if s_key in state_dict:
                return int(state_dict[s_key])
            t_key = (pipe_id, end)
            if t_key in state_dict:
                return int(state_dict[t_key])
        return 1

    def matrices(self, mode: Optional[Union[Dict[Any, int], Mode]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct numeric E, A, and f matrices for a given valve switching mode.

        Parameters
        ----------
        mode : dict or Mode, optional
            Mapping of ValveId to binary state (1=open, 0=closed).
            Valves not specified default to 1 (open).

        Returns
        -------
        tuple of (E, A, f)
            E : (K, K) float array
            A : (K, K) float array
            f : (K,) float array
        """
        E = np.zeros((self.K, self.K), dtype=float)
        A = np.zeros((self.K, self.K), dtype=float)
        f = np.zeros(self.K, dtype=float)

        g = self.settings.g

        # 1. Pipe momentum rows (0 .. N_p - 1)
        for i, pipe in enumerate(self.pipes):
            E[i, i] = 1.0
            Area = math.pi * (pipe.diameter ** 2) / 4.0
            b = g * Area / pipe.length
            z_start = float(self.node_map[pipe.start].elevation or 0.0)
            z_end = float(self.node_map[pipe.end].elevation or 0.0)
            Dz = z_start - z_end

            A[i, self.pl_col[pipe.id]] = b
            A[i, self.pr_col[pipe.id]] = -b
            f[i] = b * Dz

        # 2. Pipe end-side rows (N_p .. 2*N_p - 1)
        for i, pipe in enumerate(self.pipes):
            row = self.N_p + i
            s = self._get_valve_switch(mode, pipe.id, "end") if pipe.valve_end else 1
            A[row, self.pipe_col[pipe.id]] = 1.0 - s
            A[row, self.pr_col[pipe.id]] = float(s)
            A[row, self.node_col[pipe.end]] = -float(s)

        # 3. Pipe start-side rows (2*N_p .. 3*N_p - 1)
        for i, pipe in enumerate(self.pipes):
            row = 2 * self.N_p + i
            s = self._get_valve_switch(mode, pipe.id, "start") if pipe.valve_start else 1
            A[row, self.pipe_col[pipe.id]] = 1.0 - s
            A[row, self.pl_col[pipe.id]] = float(s)
            A[row, self.node_col[pipe.start]] = -float(s)

        # 4. Junction rows (3*N_p .. 3*N_p + N_j - 1)
        for j_idx, jnode in enumerate(self.junctions):
            row = self.j_offset + j_idx
            for pipe in self.pipes:
                if pipe.end == jnode.id:
                    A[row, self.pipe_col[pipe.id]] = 1.0
                elif pipe.start == jnode.id:
                    A[row, self.pipe_col[pipe.id]] = -1.0

        # 5. Reservoir rows (3*N_p + N_j .. 3*N_p + N_j + N_r - 1)
        for r_idx, rnode in enumerate(self.reservoirs):
            row = self.r_offset + r_idx
            col = self.node_col[rnode.id]
            A[row, col] = 1.0
            head = float(rnode.head or 0.0)
            elev = float(rnode.elevation or 0.0)
            f[row] = -(head - elev)

        return E, A, f

    def friction(self, Q: np.ndarray) -> np.ndarray:
        """Compute Darcy friction vector gbar(Q) for pipe flows Q."""
        gbar = np.zeros(self.N_p, dtype=float)
        for i, pipe in enumerate(self.pipes):
            d = float(pipe.diameter)
            eps = float(pipe.roughness if pipe.roughness is not None else 0.0)
            q = float(Q[i])
            gbar[i] = pipe_friction_term(d, eps, q, self.settings)
        return gbar

    def is_regular(self, mode: Optional[Union[Dict[Any, int], Mode]] = None) -> bool:
        """Scale-aware pencil regularity test for (lambda*E - A).

        Evaluates row-normalized (lambda*E - A) at multiple fixed complex test points.
        A mode is deemed regular if sigma_min / sigma_max > 1e-10 for at least one test point.
        """
        E, A, _ = self.matrices(mode)
        test_lambdas = [1.0, 2.0, 0.7 + 1.2j, -1.5 + 0.8j]
        for lam in test_lambdas:
            M_pencil = lam * E - A
            row_norms = np.linalg.norm(M_pencil, axis=1, keepdims=True)
            row_norms = np.where(row_norms == 0, 1.0, row_norms)
            M_norm = M_pencil / row_norms
            s = np.linalg.svd(M_norm, compute_uv=False)
            if s[0] > 0 and (s[-1] / s[0]) > 1e-10:
                return True
        return False

    def explain_singular(self, mode: Optional[Union[Dict[Any, int], Mode]] = None) -> str:
        """Return a plain-language explanation of why a mode cannot be computed."""
        if self.is_regular(mode):
            return "Valid: flow paths and pressures are well-posed."
        return (
            "Closing valves isolates a network section or disconnects "
            "reservoir supplies, so pressures cannot be uniquely determined."
        )

    def check_scenario(self, scenario: Scenario) -> List[str]:
        """Check scenario validity against network valves and mode solvability."""
        problems: List[str] = []
        network_valves = set(self.valves)

        # Check valves referenced
        scenario_valves = set(scenario.initial.keys()) | {e.valve for e in scenario.events}
        for v in sorted(scenario_valves, key=str):
            if v not in network_valves:
                problems.append(f"Valve {v} in scenario does not exist in the network")

        # Check solvability of each mode
        for m in scenario.modes():
            if not self.is_regular(m.state):
                # Check if both valves of any pipe are closed
                shut_pipes = []
                for p in self.network.pipes:
                    if p.valve_start and p.valve_end:
                        st_start = None
                        st_end = None
                        for v_k, v_val in m.state.items():
                            v_str = str(v_k)
                            if v_str == f"{p.id}:start":
                                st_start = v_val
                            elif v_str == f"{p.id}:end":
                                st_end = v_val
                        if st_start == 0 and st_end == 0:
                            shut_pipes.append(p.id)
                if shut_pipes:
                    p_name = shut_pipes[0]
                    problems.append(
                        f"From {m.t_start:.0f} s to {m.t_end:.0f} s both valves of pipe {p_name} are closed (all valves closed: no unique solution). "
                        "The water between them is then shut in on both sides, so the model cannot compute that interval. "
                        "Open at least one valve."
                    )
                else:
                    all_closed = all(s == 0 for s in m.state.values())
                    if all_closed and len(m.state) > 0:
                        problems.append(
                            f"From {m.t_start:.0f} s to {m.t_end:.0f} s all valves are closed. "
                            "No water can flow through any pipe, so pressures cannot be uniquely determined for that interval. "
                            "Open at least one valve."
                        )
                    else:
                        closed_valves = [str(v) for v, s in m.state.items() if s == 0]
                        cv_str = ", ".join(closed_valves) if closed_valves else "specified"
                        problems.append(
                            f"From {m.t_start:.0f} s to {m.t_end:.0f} s closing valve(s) {cv_str} "
                            "cuts off flow or isolates network sections, "
                            "so the model cannot compute that interval. Open at least one valve."
                        )

        return problems


def build_switched_dae(
    network: Union[Network, dict],
    settings: Optional[PhysicsSettings] = None,
) -> SwitchedDAE:
    """Build a numeric switched-DAE model for *network* or raise UnsupportedNetworkError."""
    if isinstance(network, dict):
        network = Network.from_dict(network)

    if settings is None:
        settings = PhysicsSettings()

    reasons = check_supported(network)
    if reasons:
        raise UnsupportedNetworkError(reasons)

    return SwitchedDAE(network, settings)


def steady_state(model: SwitchedDAE, mode: Optional[Union[Dict[Any, int], Mode]] = None) -> np.ndarray:
    """Compute steady-state solution vector x via Newton iteration using pure NumPy.

    Parameters
    ----------
    model : SwitchedDAE
        The switched DAE system.
    mode : dict or Mode, optional
        Valve configuration.

    Returns
    -------
    numpy.ndarray
        Solution vector x of length K.

    Raises
    ------
    ValueError
        If the specified mode is singular.
    RuntimeError
        If the Newton iteration fails to converge.
    """
    if not model.is_regular(mode):
        raise ValueError(
            f"Cannot solve steady state: mode is singular ({model.explain_singular(mode)})"
        )

    E, A, f = model.matrices(mode)
    M = model.M
    N = model.N
    N_p = model.N_p

    # Initial guess
    x = np.zeros(model.K, dtype=float)
    x[:N_p] = 0.1
    res_heads = []
    for r in model.reservoirs:
        if r.head is not None and r.elevation is not None:
            res_heads.append(r.head - r.elevation)
    default_p = float(np.mean(res_heads)) if res_heads else 30.0
    x[N_p:] = default_p

    max_iter = 100
    tol = 1e-11

    for it in range(max_iter):
        xi = M @ x
        g_val = model.friction(xi)
        res = A @ x + N @ g_val + f

        if float(np.max(np.abs(res))) < tol:
            return x

        # Build Jacobian J = A + N * d(gbar)/dQ * M via numerical finite difference
        h = 1e-7
        J = A.copy()
        for i in range(N_p):
            xi_plus = xi.copy()
            xi_minus = xi.copy()
            xi_plus[i] += h
            xi_minus[i] -= h
            g_plus = model.friction(xi_plus)[i]
            g_minus = model.friction(xi_minus)[i]
            dg_dQ = (g_plus - g_minus) / (2.0 * h)
            J[i, i] += dg_dQ

        try:
            dx = np.linalg.solve(J, -res)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(J, -res, rcond=None)[0]

        x = x + dx
        if float(np.max(np.abs(dx))) < tol:
            xi = M @ x
            g_val = model.friction(xi)
            res = A @ x + N @ g_val + f
            if float(np.max(np.abs(res))) < 1e-8:
                return x

    # Final residual check
    xi = M @ x
    g_val = model.friction(xi)
    res = A @ x + N @ g_val + f
    if float(np.max(np.abs(res))) < 1e-8:
        return x

    raise RuntimeError(
        f"Newton iteration did not converge for steady state after {max_iter} iterations "
        f"(max residual: {np.max(np.abs(res)):.3e})"
    )
