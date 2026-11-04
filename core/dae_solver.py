"""core/dae_solver.py – Theorem 8 switched DAE solver with Dirac impulses.

Pure NumPy implementation of the quasi-Weierstrass form (QWF), Wong sequences,
Dirac impulse calculation, and adaptive Dormand-Prince 5(4) integrator for
hydraulic networks with switching valves.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np

from core.scenario import Scenario, Mode, ValveId
from core.dae_model import SwitchedDAE


class SolverError(RuntimeError):
    """Raised when a numerical or DAE structural condition fails during analysis or solution."""
    pass


class ScenarioError(ValueError):
    """Raised when a scenario contains invalid valves or modes that cannot be simulated."""
    pass


# =============================================================================
#  2.1 LINEAR-ALGEBRA HELPERS (SVD-based, tolerance 1e-10 * max(1, s_max))
# =============================================================================
# Faithful port of orthT, nullT, rnk, preimg, extendBasis from
# reference/switchedDAE3_vs_EPANET.m (lines 1006-1035).
# The cutoff tolerance is tol = 1e-10 * max(1.0, s_max), where s_max is the
# maximum singular value of the matrix (or 1.0 if empty).


def orthT(X: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the column space of X.

    Uses economic SVD with tolerance 1e-10 * max(1.0, s_max).
    """
    if X.size == 0 or X.shape[1] == 0:
        return np.zeros((X.shape[0], 0), dtype=float)
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    tol = 1e-10 * max(1.0, float(s[0]))
    k = int(np.sum(s > tol))
    return U[:, :k]


def nullT(X: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the null space of X.

    Uses full SVD with tolerance 1e-10 * max(1.0, s_max).
    """
    m, n = X.shape
    if m == 0 or n == 0:
        return np.eye(n, dtype=float)
    U, s, Vt = np.linalg.svd(X, full_matrices=True)
    V = Vt.T
    s_full = np.zeros(n, dtype=float)
    s_full[:len(s)] = s
    s_max = float(np.max(s)) if len(s) > 0 else 0.0
    tol = 1e-10 * max(1.0, s_max)
    return V[:, s_full <= tol]


def rnk(X: np.ndarray) -> int:
    """Numerical rank of X using SVD with tolerance 1e-10 * max(1.0, s_max)."""
    if X.size == 0 or X.shape[1] == 0:
        return 0
    s = np.linalg.svd(X, compute_uv=False)
    if len(s) == 0:
        return 0
    tol = 1e-10 * max(1.0, float(s[0]))
    return int(np.sum(s > tol))


def preimg(Amat: np.ndarray, Img: np.ndarray) -> np.ndarray:
    """Basis of the preimage { y : Amat @ y in span(Img) }."""
    if Img.shape[1] == 0:
        return orthT(nullT(Amat))
    stacked = np.hstack([Amat, -Img])
    Z = nullT(stacked)
    return orthT(Z[:Amat.shape[1], :])


def extendBasis(B: np.ndarray, Tg: np.ndarray) -> np.ndarray:
    """Columns of Tg that extend basis B to a basis of span([B, Tg])."""
    cur = B.copy()
    cols: List[np.ndarray] = []
    for j in range(Tg.shape[1]):
        col = Tg[:, j:j+1]
        test = np.hstack([cur, col]) if cur.shape[1] > 0 else col
        if rnk(test) > rnk(cur):
            cur = test
            cols.append(col)
    if cols:
        return np.hstack(cols)
    return np.zeros((B.shape[0], 0), dtype=float)


# =============================================================================
#  2.2 MODE DATA & ANALYSIS (Step 1 of Theorem 8: modified QWF)
# =============================================================================


@dataclass
class ModeData:
    """Decomposition matrices and diagnostic values for one valve switching mode."""

    mode_state: Any
    n1: int
    n12: int
    n22: int
    S: np.ndarray
    T: np.ndarray
    Tinv: np.ndarray
    J: np.ndarray
    N1: np.ndarray
    N2: np.ndarray
    Tv: np.ndarray
    T1w: np.ndarray
    T2w: np.ndarray
    Sv: np.ndarray
    S1w: np.ndarray
    S2w: np.ndarray
    Pi: np.ndarray
    Piimp: np.ndarray
    Eimp: np.ndarray
    E: np.ndarray
    A: np.ndarray
    f: np.ndarray
    fv: np.ndarray
    f1w: np.ndarray
    f2w: np.ndarray
    Nv: np.ndarray
    N1w: np.ndarray
    N2w: np.ndarray

    # Diagnostic quantities matching reference/switchedDAE3_vs_EPANET.m
    det_E_minus_A: float = 0.0
    norm_M_Eimp: float = 0.0
    norm_N1w: float = 0.0
    norm_M_T2w: float = 0.0
    norm_N2: float = 0.0
    eig_J: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=complex))
    nonzero_Eimp_rows: List[int] = field(default_factory=list)
    nonzero_Eimp_labels: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Format diagnostic numbers exactly as in the reference MATLAB script."""
        lines = [
            f"mode: n1 = {self.n1}, n12 = {self.n12}, n22 = {self.n22}",
            f"   (R) regular      : |det(E-A)| = {self.det_E_minus_A:.4e}",
            f"   (M) M*Eimp = 0   : ||M*Eimp|| = {self.norm_M_Eimp:.2e}",
            f"   (N) S1w*N  = 0   : ||N1w||    = {self.norm_N1w:.2e}",
            f"       M*T2w  = 0   : ||M*T2w||  = {self.norm_M_T2w:.2e}",
            f"       N2 = 0       : ||N2||     = {self.norm_N2:.2e}",
        ]
        if self.n1 > 0:
            eigs_str = "  ".join(f"{v.real:+.6f}{v.imag:+.6f}i" for v in self.eig_J)
            lines.append(f"       eig(J)       : {eigs_str}")
        else:
            lines.append("       eig(J)       : (none) -- n1 = 0, this mode is PURELY ALGEBRAIC")
        lines.append(f"       nonzero Eimp rows: {self.nonzero_Eimp_rows}")
        return "\n".join(lines)


def analyze_mode(model: SwitchedDAE, mode_state: Any) -> ModeData:
    """Port of qwf: Wong sequences and modified quasi-Weierstrass form of (E, A).

    Decomposes (E, A) into differential (v, size n1) and algebraic (w1, w2)
    coordinates using the Wong sequences of (E', A').

    Results are cached on the model.
    """
    if not hasattr(model, "_mode_cache"):
        model._mode_cache = {}

    # Cache key: normalized tuple of valve states
    cache_key: Tuple[Any, ...]
    if isinstance(mode_state, (tuple, list)):
        cache_key = tuple(mode_state)
    elif isinstance(mode_state, Mode):
        cache_key = tuple((k, mode_state.state[k]) for k in sorted(mode_state.state.keys(), key=str))
    elif isinstance(mode_state, dict):
        cache_key = tuple((k, mode_state[k]) for k in sorted(mode_state.keys(), key=str))
    else:
        cache_key = (mode_state,)

    if cache_key in model._mode_cache:
        return model._mode_cache[cache_key]

    # Regularity check
    if not model.is_regular(mode_state):
        raise SolverError(
            f"Mode {mode_state} is not regular: pencil (lambda*E - A) is singular; "
            "the algebraic equations have no unique solution."
        )

    E, A, f = model.matrices(mode_state)
    n = E.shape[0]
    Et = E.T
    At = A.T

    # 1. First Wong sequence (limit V)
    V = np.eye(n, dtype=float)
    for _ in range(n + 2):
        Vn = preimg(At, orthT(Et @ V))
        if rnk(Vn) == rnk(V) and rnk(np.hstack([V, Vn])) == rnk(V):
            V = Vn
            break
        V = Vn

    # 2. Second Wong sequence (limit W)
    W = np.zeros((n, 0), dtype=float)
    for _ in range(n + 2):
        Wn = preimg(Et, orthT(At @ W))
        if rnk(Wn) == rnk(W):
            W = Wn
            break
        W = Wn

    # Transformation matrices S = [Vh W1 W2]', T = [Et*Vh At*W1 At*W2]^(-T)
    Vh = orthT(V)
    Wh = orthT(W)
    W1 = orthT(nullT(Et))
    W2 = extendBasis(W1, Wh)

    S = np.hstack([Vh, W1, W2]).T
    col_mat = np.hstack([Et @ Vh, At @ W1, At @ W2]).T
    try:
        T = np.linalg.inv(col_mat)
    except np.linalg.LinAlgError:
        raise SolverError("Failed to invert transformation matrix T during quasi-Weierstrass decomposition.")

    n1 = Vh.shape[1]
    n12 = W1.shape[1]
    n22 = W2.shape[1]

    SET = S @ E @ T
    SAT = S @ A @ T
    Tinv = np.linalg.inv(T)

    J = SAT[:n1, :n1]
    N1 = SET[n1 + n12:, n1:n1 + n12]
    N2 = SET[n1 + n12:, n1 + n12:]

    Tv = T[:, :n1]
    T1w = T[:, n1:n1 + n12]
    T2w = T[:, n1 + n12:]

    Sv = S[:n1, :]
    S1w = S[n1:n1 + n12, :]
    S2w = S[n1 + n12:, :]

    # Projectors of Definition 3
    sel = np.diag([1.0] * n1 + [0.0] * (n - n1))
    Pi = T @ sel @ Tinv
    Piimp = T @ (np.eye(n) - sel) @ S
    Eimp = Piimp @ E

    # Coordinates of f and N
    fv = Sv @ f
    f1w = S1w @ f
    f2w = S2w @ f

    Nv = Sv @ model.N
    N1w = S1w @ model.N
    N2w = S2w @ model.N

    # Diagnostics
    det_val = float(abs(np.linalg.det(E - A)))
    norm_M_Eimp = float(np.linalg.norm(model.M @ Eimp))
    norm_N1w = float(np.linalg.norm(N1w))
    norm_M_T2w = float(np.linalg.norm(model.M @ T2w))
    norm_N2 = float(np.linalg.norm(N2))
    eig_J = np.linalg.eigvals(J) if n1 > 0 else np.zeros(0, dtype=complex)

    row_norms = np.linalg.norm(Eimp, axis=1)
    nonzero_rows = [int(i) for i in np.where(row_norms > 1e-9)[0]]
    nonzero_labels = [model.labels[i] for i in nonzero_rows]

    # Required Theorem 8 checks
    if norm_N2 > 1e-9:
        raise SolverError(
            "the valve change creates an infinitely strong impulse (index above 2): "
            f"N2 norm is {norm_N2:.2e}"
        )
    if norm_N1w > 1e-9:
        raise SolverError(
            "Assumption (N) violated: pipe friction enters algebraic equations directly "
            f"(||N1w|| = {norm_N1w:.2e})."
        )
    if norm_M_Eimp > 1e-9:
        raise SolverError(
            "Assumption (M) violated: Dirac impulses affect flow momentum equations directly "
            f"(||M*Eimp|| = {norm_M_Eimp:.2e})."
        )

    md = ModeData(
        mode_state=mode_state,
        n1=n1,
        n12=n12,
        n22=n22,
        S=S,
        T=T,
        Tinv=Tinv,
        J=J,
        N1=N1,
        N2=N2,
        Tv=Tv,
        T1w=T1w,
        T2w=T2w,
        Sv=Sv,
        S1w=S1w,
        S2w=S2w,
        Pi=Pi,
        Piimp=Piimp,
        Eimp=Eimp,
        E=E,
        A=A,
        f=f,
        fv=fv,
        f1w=f1w,
        f2w=f2w,
        Nv=Nv,
        N1w=N1w,
        N2w=N2w,
        det_E_minus_A=det_val,
        norm_M_Eimp=norm_M_Eimp,
        norm_N1w=norm_N1w,
        norm_M_T2w=norm_M_T2w,
        norm_N2=norm_N2,
        eig_J=eig_J,
        nonzero_Eimp_rows=nonzero_rows,
        nonzero_Eimp_labels=nonzero_labels,
    )
    model._mode_cache[cache_key] = md
    return md


# =============================================================================
#  2.3 REST STATE (Port of restState)
# =============================================================================


def rest_state(model: SwitchedDAE, mode_state: Any) -> np.ndarray:
    """Consistent initial state with zero flows.

    Takes v = 0 and evaluates the algebraic components from (10b) and (10c).
    """
    D = analyze_mode(model, mode_state)
    v = np.zeros(D.n1, dtype=float)
    w1 = -D.f1w
    xi = model.M @ (D.Tv @ v + D.T1w @ w1) if D.n1 > 0 else model.M @ (D.T1w @ w1)
    w2 = -(D.f2w + D.N2w @ model.friction(xi))
    return D.Tv @ v + D.T1w @ w1 + D.T2w @ w2


# =============================================================================
#  2.4 ADAPTIVE DORMAND-PRINCE 5(4) INTEGRATOR (pure NumPy ode45 equivalent)
# =============================================================================

# Dormand-Prince tableau constants
_DP_C = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1], dtype=float)
_DP_A = [
    np.array([], dtype=float),
    np.array([1/5], dtype=float),
    np.array([3/40, 9/40], dtype=float),
    np.array([44/45, -56/15, 32/9], dtype=float),
    np.array([19372/6561, -25360/2187, 64448/6561, -212/729], dtype=float),
    np.array([9017/3168, -355/33, 46732/5247, 49/176, -5103/18656], dtype=float),
    np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84], dtype=float),
]
_DP_B = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0], dtype=float)
_DP_BHAT = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40], dtype=float)
_DP_E = _DP_B - _DP_BHAT


def _hermite_eval(t0: float, t1: float, y0: np.ndarray, y1: np.ndarray, f0: np.ndarray, f1: np.ndarray, t_target: float) -> np.ndarray:
    """Cubic Hermite interpolation at t_target between [t0, t1]."""
    h = t1 - t0
    if h <= 0:
        return y0.copy()
    theta = (t_target - t0) / h
    theta2 = theta * theta
    theta3 = theta2 * theta
    h00 = 2 * theta3 - 3 * theta2 + 1.0
    h10 = theta3 - 2 * theta2 + theta
    h01 = -2 * theta3 + 3 * theta2
    h11 = theta3 - theta2
    return h00 * y0 + (h * h10) * f0 + h01 * y1 + (h * h11) * f1


def _ode45_dp54(
    odefun: Any,
    t_span: Tuple[float, float],
    y0: np.ndarray,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_step: Optional[float] = None,
) -> Tuple[List[float], List[np.ndarray], List[np.ndarray]]:
    """Adaptive Dormand-Prince 5(4) integrator in pure NumPy.

    Parameters
    ----------
    odefun : callable
        Function f(t, y) returning derivative array of same shape as y.
    t_span : (t0, t1)
        Integration interval.
    y0 : np.ndarray
        Initial state vector.
    rtol : float
        Relative error tolerance (default 1e-10).
    atol : float
        Absolute error tolerance (default 1e-12).
    max_step : float, optional
        Maximum step size limit (default (t1 - t0) / 200).

    Returns
    -------
    t_steps, y_steps, f_steps : lists of recorded steps
    """
    t0, t1 = float(t_span[0]), float(t_span[1])
    n = len(y0)
    if t1 <= t0:
        f0 = odefun(t0, y0)
        return [t0], [y0.copy()], [f0]

    if max_step is None:
        max_step = (t1 - t0) / 200.0

    t_cur = t0
    y_cur = y0.copy()
    f_cur = odefun(t_cur, y_cur)

    t_steps = [t_cur]
    y_steps = [y_cur]
    f_steps = [f_cur]

    # Initial step estimate
    h = min(max_step, max(1e-6, (t1 - t0) / 1000.0))

    while t_cur < t1:
        if t_cur + h > t1:
            h = t1 - t_cur

        k = [np.zeros(n, dtype=float) for _ in range(7)]
        k[0] = f_steps[-1]

        for i in range(1, 6):
            ti = t_cur + _DP_C[i] * h
            yi = y_cur + h * sum(_DP_A[i][j] * k[j] for j in range(i))
            k[i] = odefun(ti, yi)

        y_next = y_cur + h * sum(_DP_B[j] * k[j] for j in range(6))
        k[6] = odefun(t_cur + h, y_next)

        err = h * sum(_DP_E[j] * k[j] for j in range(7))
        scale = atol + rtol * np.maximum(np.abs(y_cur), np.abs(y_next))
        err_ratio = float(np.max(np.abs(err / scale)))

        if err_ratio <= 1.0:
            # Step accepted
            t_cur += h
            y_cur = y_next
            t_steps.append(t_cur)
            y_steps.append(y_cur)
            f_steps.append(k[6])

            if len(t_steps) % 200 == 0:
                time.sleep(0)

            if t_cur >= t1:
                break

            factor = 0.9 * (err_ratio ** -0.2) if err_ratio > 0 else 5.0
            h = min(max_step, h * min(5.0, max(0.2, factor)))
        else:
            # Step rejected
            factor = 0.9 * (err_ratio ** -0.2)
            h = h * max(0.1, factor)

        # Step size underflow check
        if h < 1e-15 * max(1.0, abs(t_cur)):
            raise SolverError(
                f"Step size underflow in ODE integration at t = {t_cur:.6e}: "
                f"step size {h:.3e} is too small to maintain tolerances."
            )

    return t_steps, y_steps, f_steps


# =============================================================================
#  2.5 SIMULATION RESULT & SOLVER (Theorem 8 Step 3)
# =============================================================================


@dataclass
class ImpulseRecord:
    """Record of a Dirac impulse at a valve switching instant."""

    time: float
    coefficients: np.ndarray
    nonzero_rows: List[str]
    x_minus: np.ndarray
    cross_check_error: float


@dataclass
class SimulationResult:
    """Complete time-domain solution of the switched DAE model."""

    t: np.ndarray
    X: np.ndarray
    labels: List[str]
    elevations: np.ndarray
    segments: List[Tuple[float, float, Any]]
    impulses: List[ImpulseRecord]

    def _col(self, label: str) -> int:
        if label in self.labels:
            return self.labels.index(label)
        # Search case-insensitive or partial match
        matches = [i for i, lb in enumerate(self.labels) if lb.lower() == label.lower()]
        if matches:
            return matches[0]
        raise KeyError(f"State label {label!r} not found in model state labels: {self.labels}")

    def series(self, label: str) -> np.ndarray:
        """Return 1-D array of the given state variable across all time samples."""
        return self.X[:, self._col(label)]

    def pressure_head(self, label: str) -> np.ndarray:
        """Return pressure head in metres. Raises ValueError for pipe flow states."""
        col = self._col(label)
        if math.isnan(self.elevations[col]):
            raise ValueError(f"State {label!r} is a volumetric flow state and has no pressure head.")
        return self.X[:, col]

    def total_head(self, label: str) -> np.ndarray:
        """Return total hydraulic head (pressure head + elevation) in metres."""
        col = self._col(label)
        if math.isnan(self.elevations[col]):
            raise ValueError(f"State {label!r} is a volumetric flow state and has no hydraulic head.")
        return self.X[:, col] + self.elevations[col]

    def at(self, t_target: float) -> np.ndarray:
        """Return the state vector at the sample closest to t_target."""
        idx = int(np.argmin(np.abs(self.t - float(t_target))))
        return self.X[idx, :].copy()


def simulate(
    model: SwitchedDAE,
    scenario: Scenario,
    pts_per_second: int = 40,
    max_pts_per_interval: int = 200000,
) -> SimulationResult:
    """Solve the switched DAE over the specified scenario.

    Implements Theorem 8 Step 3 interval by interval:
    - Verifies scenario validity via model.check_scenario.
    - Initial state = rest_state of the first mode (zero flow, consistent algebraics).
    - At every switch time, computes the Dirac impulse coefficients via (10c) and
      cross-checks with formula (6).
    - At each switch time, stores TWO samples (t- and t+) so algebraic jumps are visible.
    """
    # 2.6 Validate scenario first
    problems = model.check_scenario(scenario)
    if problems:
        raise ScenarioError("; ".join(problems))

    modes = scenario.modes()
    if not modes:
        raise ScenarioError("Scenario contains no operating intervals.")

    # Initial state on (-inf, 0): consistent rest state with zero flow
    first_mode = modes[0].state
    x_prev = rest_state(model, first_mode)

    all_t: List[np.ndarray] = []
    all_X: List[np.ndarray] = []
    impulses: List[ImpulseRecord] = []
    segments: List[Tuple[float, float, Any]] = []

    for k, m in enumerate(modes):
        t0, t1 = float(m.t_start), float(m.t_end)
        segments.append((t0, t1, m.state))
        D = analyze_mode(model, m.state)

        # 1. Dirac impulse at the switch (k > 0)
        if k > 0:
            w1 = -D.f1w
            w1minus = D.S1w @ D.A @ x_prev
            dw1 = w1 - w1minus
            coeff = D.T2w @ (D.N1 @ dw1)
            alt = -D.Eimp @ x_prev
            err = float(np.max(np.abs(coeff - alt)))

            nz_idx = [int(i) for i in np.where(np.abs(coeff) > 1e-9)[0]]
            nz_labels = [model.labels[i] for i in nz_idx]

            impulses.append(
                ImpulseRecord(
                    time=t0,
                    coefficients=coeff,
                    nonzero_rows=nz_labels,
                    x_minus=x_prev.copy(),
                    cross_check_error=err,
                )
            )

        # 2. Incoming differential state v(t0+) = (T^-1 x_prev)[:n1]
        v0 = D.Tinv[:D.n1, :] @ x_prev if D.n1 > 0 else np.zeros(0, dtype=float)

        # 3. Time grid for interval
        npts = min(int(round((t1 - t0) * pts_per_second)) + 1, max_pts_per_interval)
        npts = max(npts, 2)
        tt = np.linspace(t0, t1, npts)

        # 4. Integrate or evaluate
        w1_const = -D.f1w
        if D.n1 == 0:
            # Purely algebraic mode: flow is zero / constant, equations (10b),(10c) determine state
            xi = model.M @ (D.T1w @ w1_const)
            w2 = -(D.f2w + D.N2w @ model.friction(xi))
            x_val = D.T1w @ w1_const + D.T2w @ w2
            X_piece = np.tile(x_val, (npts, 1))
        else:
            # Differential mode: integrate (10a) v' = J v + Nv gbar(M x) + fv
            def odefun(t: float, v: np.ndarray) -> np.ndarray:
                xi_val = model.M @ (D.Tv @ v + D.T1w @ w1_const)
                return D.J @ v + D.Nv @ model.friction(xi_val) + D.fv

            t_steps, v_steps, f_steps = _ode45_dp54(
                odefun,
                (t0, t1),
                v0,
                rtol=1e-10,
                atol=1e-12,
                max_step=(t1 - t0) / 200.0,
            )

            # Sample on fixed grid tt
            t_steps_arr = np.array(t_steps)
            X_piece = np.zeros((npts, model.K), dtype=float)

            for i, st in enumerate(tt):
                if i % 1000 == 0:
                    time.sleep(0)
                idx = int(np.searchsorted(t_steps_arr, st))
                if idx == 0:
                    v_val = v_steps[0]
                elif idx >= len(t_steps_arr):
                    v_val = v_steps[-1]
                elif abs(t_steps_arr[idx] - st) < 1e-13:
                    v_val = v_steps[idx]
                else:
                    i0, i1 = idx - 1, idx
                    v_val = _hermite_eval(
                        t_steps_arr[i0],
                        t_steps_arr[i1],
                        v_steps[i0],
                        v_steps[i1],
                        f_steps[i0],
                        f_steps[i1],
                        st,
                    )

                xi_st = model.M @ (D.Tv @ v_val + D.T1w @ w1_const)
                w2_st = -(D.f2w + D.N2w @ model.friction(xi_st))
                X_piece[i, :] = D.Tv @ v_val + D.T1w @ w1_const + D.T2w @ w2_st

        all_t.append(tt)
        all_X.append(X_piece)
        x_prev = X_piece[-1, :].copy()

    # Concatenate results across all intervals
    t_full = np.concatenate(all_t)
    X_full = np.concatenate(all_X, axis=0)

    return SimulationResult(
        t=t_full,
        X=X_full,
        labels=list(model.labels),
        elevations=model.elevation,
        segments=segments,
        impulses=impulses,
    )
