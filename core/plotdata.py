"""core/plotdata.py – Pure NumPy data processing and series extraction for comparison plots.

No GUI or matplotlib imports are allowed in core.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.comparison import ComparisonResult


def default_states(result: ComparisonResult) -> List[str]:
    """Return default list of state labels to display.

    For the 2-pipe specimen network: returns the MATLAB reference 5 states in order:
    ['Pipe_1_Flow', 'Pipe_2_Flow', 'Pipe_1_R', 'Pipe_2_L', 'Pipe_2_R'].

    For other networks: all flow states first, then pressure states of pipe ends
    that carry an impulse or sit next to a valve, up to 6 panels, with the rest
    accessible via the selector.
    """
    labels = list(result.dae.labels)
    matlab_five = ["Pipe_1_Flow", "Pipe_2_Flow", "Pipe_1_R", "Pipe_2_L", "Pipe_2_R"]

    # Check if network is the specimen network (2 pipes, has all 5 matlab states)
    if (
        len(result.snapshot.pipes) == 2
        and all(s in labels for s in matlab_five)
    ):
        return list(matlab_five)

    # Generic networks
    flows = [lbl for lbl in labels if "Flow" in lbl]
    valved_or_impulse_pressures: List[str] = []

    # Identify valved pipe ends
    valved_ends = set()
    for p in result.snapshot.pipes:
        if p.valve_start:
            valved_ends.add(f"{p.id}_L")
        if p.valve_end:
            valved_ends.add(f"{p.id}_R")

    # Identify impulse rows
    for imp in result.dae.impulses:
        for r_lbl in imp.nonzero_rows:
            valved_ends.add(r_lbl)

    for lbl in labels:
        if lbl in flows:
            continue
        if lbl in valved_ends:
            valved_or_impulse_pressures.append(lbl)

    selected = flows + [p for p in valved_or_impulse_pressures if p not in flows]
    if len(selected) > 6:
        selected = selected[:6]
    elif len(selected) < 6:
        remaining = [lbl for lbl in labels if lbl not in selected]
        selected.extend(remaining[: 6 - len(selected)])

    return selected


def _dae_mode_splits(result: ComparisonResult) -> List[int]:
    """Return 0-indexed split boundaries where DAE time restarts/switches."""
    cached = getattr(result.dae, "_cached_splits", None)
    if cached is not None:
        return cached
    t = result.dae.t
    if len(t) <= 1:
        splits = [0, len(t)]
    else:
        idx = np.where(t[1:] <= t[:-1])[0] + 1
        splits = [0] + idx.tolist() + [len(t)]
    try:
        result.dae._cached_splits = splits
    except Exception:
        pass
    return splits


def series(
    result: ComparisonResult,
    label: str,
    kind: str = "pressure",
) -> Tuple[
    List[Tuple[np.ndarray, np.ndarray]],
    List[Tuple[float, float, float]],
    Tuple[np.ndarray, np.ndarray],
]:
    """Extract DAE segments, switch jump pairs, and EPANET time series for a state.

    Parameters
    ----------
    result : ComparisonResult
        Completed simulation comparison result.
    label : str
        State variable label (e.g. 'Pipe_1_Flow', 'Pipe_1_R').
    kind : str, optional
        'pressure' (default) or 'total' (total head = pressure head + elevation).
        Flow states are unaffected by kind.

    Returns
    -------
    dae_segments : list of (t_array, y_array)
        One segment per mode interval so no solid line crosses a switch.
    jump_pairs : list of (t_s, y_before, y_after)
        Values right before and after each valve switch.
    epanet_series : (t_array, y_array)
        EPANET reporting times and values.
    """
    cache = getattr(result, "_cached_series", None)
    if cache is None:
        cache = {}
        try:
            result._cached_series = cache
        except Exception:
            pass
    key = (label, kind)
    if cache is not None and key in cache:
        return cache[key]

    col = result.dae._col(label)
    elev = result.dae.elevations[col]
    offset = float(elev) if (kind == "total" and not math.isnan(elev)) else 0.0

    splits = _dae_mode_splits(result)
    dae_segments: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in range(len(splits) - 1):
        i0, i1 = splits[k], splits[k + 1]
        t_seg = result.dae.t[i0:i1].copy()
        y_seg = (result.dae.X[i0:i1, col] + offset).copy()
        dae_segments.append((t_seg, y_seg))

    jump_pairs: List[Tuple[float, float, float]] = []
    # If there are switches (len(splits) > 2), each split boundary has a jump
    for k in range(1, len(splits) - 1):
        idx_boundary = splits[k]
        t_s = float(result.dae.t[idx_boundary])
        y_before = float(result.dae.X[idx_boundary - 1, col] + offset)
        y_after = float(result.dae.X[idx_boundary, col] + offset)
        jump_pairs.append((t_s, y_before, y_after))

    # EPANET series
    t_ep = result.epanet.t.copy()
    y_ep = (result.epanet.series(label) + offset).copy()

    res_tuple = (dae_segments, jump_pairs, (t_ep, y_ep))
    if cache is not None:
        cache[key] = res_tuple
    return res_tuple


def decimate_series(
    t: np.ndarray,
    y: np.ndarray,
    max_points: int = 4000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Min-max envelope decimation for 2D line plotting.

    Reduces high-density series to at most `max_points` while strictly
    preserving:
    - The first sample (t[0], y[0])
    - The last sample (t[-1], y[-1])
    - The local minimum and maximum in each bucket, guaranteeing that global
      extrema and sharp spikes are preserved without clipping or attenuation.

    Parameters
    ----------
    t : np.ndarray
        Monotonically increasing time values.
    y : np.ndarray
        State trajectory values.
    max_points : int, optional
        Maximum number of points in decimated output (default 4000).

    Returns
    -------
    t_dec, y_dec : tuple of np.ndarray
        Decimated time and value arrays.
    """
    n = len(t)
    if n <= max_points or max_points < 4:
        return t, y

    # Reserve 2 points for boundaries (0 and n-1) and 2 for remainder bucket
    n_buckets = max(1, (max_points - 4) // 2)
    interior_len = n - 2
    bucket_len = interior_len // n_buckets
    if bucket_len < 2:
        return t, y

    usable_len = n_buckets * bucket_len
    y_int = y[1:1 + usable_len]
    y_2d = y_int.reshape(n_buckets, bucket_len)

    min_idx = np.argmin(y_2d, axis=1)
    max_idx = np.argmax(y_2d, axis=1)

    base = np.arange(n_buckets, dtype=np.int64) * bucket_len + 1
    pos_min = base + min_idx
    pos_max = base + max_idx

    indices = [0, pos_min, pos_max]

    rem = interior_len - usable_len
    if rem > 0:
        rem_slice = y[1 + usable_len:n - 1]
        indices.append([
            1 + usable_len + int(np.argmin(rem_slice)),
            1 + usable_len + int(np.argmax(rem_slice)),
        ])

    indices.append([n - 1])
    all_indices = np.concatenate([np.atleast_1d(x) for x in indices])
    chosen = np.unique(all_indices)
    return t[chosen], y[chosen]



def _strictly_increasing_dae_time_series(
    result: ComparisonResult,
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate DAE pieces ensuring strictly increasing time for interp1 (as in MATLAB)."""
    splits = _dae_mode_splits(result)
    tall_list = []
    n_segs = len(splits) - 1
    for k in range(n_segs):
        i0, i1 = splits[k], splits[k + 1]
        tk = result.dae.t[i0:i1].copy()
        if k < n_segs - 1:
            tk[-1] -= 1e-9
        tall_list.append(tk)

    tall = np.concatenate(tall_list) if tall_list else result.dae.t.copy()
    return tall, result.dae.X


def difference_rows(
    result: ComparisonResult,
    labels: Optional[List[str]] = None,
    times: Optional[List[float]] = None,
) -> List[Tuple[float, str, float, float, float]]:
    """Sample DAE and EPANET at reporting instants and compute differences.

    Parameters
    ----------
    result : ComparisonResult
        Completed comparison run.
    labels : list of str, optional
        State labels to include. Defaults to default_states(result).
    times : list of float, optional
        Target sampling times. If None, uses MATLAB rule: 7 equally spaced points
        from 0 to the first switch (or to end if none), plus end time.

    Returns
    -------
    list of (time, state_label, dae_val, epanet_val, diff)
        Where diff = dae_val - epanet_val.
    """
    if labels is None:
        labels = default_states(result)

    tall, Xall = _strictly_increasing_dae_time_series(result)
    te = result.epanet.t
    t_end = result.scenario.t_end

    if times is None:
        if result.scenario.events:
            t_first = float(result.scenario.events[0].time)
            target_times = list(np.linspace(0.0, t_first, 7))
            if t_end > t_first + 1e-4:
                target_times.append(float(t_end))
        else:
            target_times = list(np.linspace(0.0, t_end, 7))
    else:
        target_times = list(times)

    # Map target times to closest or succeeding EPANET reporting time
    used_indices = []
    for tq in target_times:
        matches = np.where(te >= tq - 1e-6)[0]
        if len(matches) > 0:
            k = int(matches[0])
            if k not in used_indices:
                used_indices.append(k)

    rows: List[Tuple[float, str, float, float, float]] = []
    for k in used_indices:
        t_rep = float(te[k])
        for lbl in labels:
            col = result.dae._col(lbl)
            dae_interp = float(np.interp(t_rep, tall, Xall[:, col]))
            ep_val = float(result.epanet.series(lbl)[k])
            diff = dae_interp - ep_val
            rows.append((t_rep, lbl, dae_interp, ep_val, diff))

    return rows


def max_abs_difference_after_last_switch(result: ComparisonResult) -> float:
    """Return maximum absolute difference across all states after the last switch."""
    tall, Xall = _strictly_increasing_dae_time_series(result)
    te = result.epanet.t

    if result.scenario.events:
        t_last = float(result.scenario.events[-1].time)
        m = (te >= t_last - 1e-6)
    else:
        m = np.ones(len(te), dtype=bool)

    if not np.any(m):
        return 0.0

    t_eval = te[m]
    max_diff = 0.0
    for col in range(len(result.dae.labels)):
        dae_vals = np.interp(t_eval, tall, Xall[:, col])
        ep_vals = result.epanet.X[m, col]
        diff_max = float(np.max(np.abs(dae_vals - ep_vals)))
        if diff_max > max_diff:
            max_diff = diff_max

    return max_diff


def impulse_arrows(
    result: ComparisonResult,
    labels: Optional[List[str]] = None,
) -> Dict[str, List[Tuple[float, float, float]]]:
    """Return Dirac impulse arrows (time, coefficient, base_value) per state label."""
    if labels is None:
        labels = default_states(result)

    arrows: Dict[str, List[Tuple[float, float, float]]] = {lbl: [] for lbl in labels}

    for imp in result.dae.impulses:
        t_s = float(imp.time)
        for lbl in labels:
            col = result.dae._col(lbl)
            coeff = float(imp.coefficients[col])
            if abs(coeff) > 1e-9:
                base_val = float(imp.x_minus[col])
                arrows[lbl].append((t_s, coeff, base_val))

    return arrows


def engine_label(result: ComparisonResult) -> str:
    """Format EPANET engine version string (e.g. 20305 -> 'real EPANET 2.3.5')."""
    ver = getattr(result.epanet, "engine_version", None)
    if ver is None:
        return "real EPANET"
    try:
        ver_int = int(ver)
        major = ver_int // 10000
        minor = (ver_int % 10000) // 100
        patch = ver_int % 100
        return f"real EPANET {major}.{minor}.{patch}"
    except Exception:
        return f"real EPANET {ver}"
