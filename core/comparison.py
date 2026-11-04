"""core/comparison.py – High-level coordinator comparing DAE solver and real EPANET engine."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np

from core.models import Network
from core.scenario import Scenario, ValveId, Mode
from core.physics import PhysicsSettings
from core.params import missing_parameters
from core.dae_model import build_switched_dae, check_supported, UnsupportedNetworkError, steady_state
from core.checks import validate_network
from core.dae_solver import simulate, SimulationResult, SolverError
from core.epanet_runner import run_epanet, EpanetResult, EpanetError
from core.epanet_export import export_epanet, ScenarioError
from core.plotdata import (
    default_states,
    series,
    decimate_series,
    difference_rows,
    max_abs_difference_after_last_switch,
    impulse_arrows,
    engine_label,
)


class ComparisonError(RuntimeError):
    """Raised when network, scenario, or settings have blocking problems preventing comparison."""

    def __init__(self, problems: List[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass
class EquilibriumRow:
    """A single row in the equilibrium comparison table."""

    interval: str
    mode: str
    label: str
    dae_end: float
    dae_eq: float
    epanet: float
    diff_end: float
    diff_eq: float

    @property
    def dae(self) -> float:
        """Backward compatibility alias for dae_end."""
        return self.dae_end

    @property
    def diff(self) -> float:
        """Backward compatibility alias for diff_eq."""
        return self.diff_eq

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interval": self.interval,
            "mode": self.mode,
            "label": self.label,
            "dae_end": self.dae_end,
            "dae_eq": self.dae_eq,
            "epanet": self.epanet,
            "diff_end": self.diff_end,
            "diff_eq": self.diff_eq,
            # Backward compatibility aliases
            "dae": self.dae_end,
            "diff": self.diff_eq,
        }


@dataclass
class ComparisonResult:
    """Complete results from running DAE solver and EPANET engine side-by-side."""

    scenario: Scenario
    settings: PhysicsSettings
    snapshot: Network
    dae: SimulationResult
    epanet: EpanetResult
    params: List[Tuple[str, str, str, str]]
    equilibrium: List[Dict[str, Any]]
    impulses: List[Dict[str, Any]]
    timings: Dict[str, float]
    warnings: List[str]
    signature: str = ""

    def is_outdated(
        self,
        network: Network,
        scenario: Optional[Scenario],
        settings: Optional[PhysicsSettings],
    ) -> bool:
        """Check if the given network, scenario, or settings differ from what was run."""
        if scenario is None or settings is None:
            return True
        current_sig = compute_signature(network, scenario, settings)
        return current_sig != self.signature


def compute_signature(
    network: Network,
    scenario: Scenario,
    settings: PhysicsSettings,
) -> str:
    """Compute a deterministic SHA-256 signature for network, scenario, and settings."""
    payload = {
        "network": network.to_dict(),
        "scenario": scenario.to_dict() if hasattr(scenario, "to_dict") else str(scenario),
        "settings": {
            "friction_mode": settings.friction_mode,
            "friction_factor": round(float(settings.friction_factor), 6),
            "nu": round(float(settings.nu), 10),
            "g": round(float(settings.g), 5),
        },
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def problems(
    network: Network,
    scenario: Optional[Scenario],
    settings: Optional[PhysicsSettings] = None,
) -> List[str]:
    """Check for plain-language blocking problems preventing EPANET vs DAE comparison.

    Returns an empty list if the setup is ready to compare.
    """
    prob_list: List[str] = []

    # 1. Empty network
    if not network.nodes or not network.pipes:
        prob_list.append("Network is empty. Add reservoirs, junctions, and pipes to model a network.")
        return prob_list

    # 2. Unsupported element types: pumps and tanks
    pump_pipes = [p.id for p in network.pipes if p.has_pump]
    if pump_pipes:
        prob_list.append(
            f"Pumps are not supported for EPANET comparison (found pump pipes: {', '.join(pump_pipes)})."
        )

    tanks = [n.id for n in network.nodes if n.type == "tank"]
    if tanks:
        prob_list.append(
            f"Tanks are not supported for EPANET comparison (found tanks: {', '.join(tanks)})."
        )

    # 3. Missing physical parameters
    missing = missing_parameters(network)
    if missing:
        count = len(missing)
        sample = missing[:3]
        suffix = "..." if count > 3 else ""
        prob_list.append(
            f"Missing physical parameters ({count}): {'; '.join(sample)}{suffix}."
        )

    # 4. Reservoir check (need at least one supply/head boundary)
    reservoirs = [n.id for n in network.nodes if n.type == "reservoir"]
    if not reservoirs:
        prob_list.append("Network has no reservoir. At least one reservoir is required for boundary head.")

    # 5. Graph connectivity / isolated nodes
    structural_warnings = validate_network(network)
    for warn in structural_warnings:
        if "isolated" in warn.lower() or "disconnected" in warn.lower():
            prob_list.append(warn)

    # 6. Valves and scenario
    network_valves: List[str] = []
    for p in network.pipes:
        if p.valve_start:
            network_valves.append(f"{p.id}:start")
        if p.valve_end:
            network_valves.append(f"{p.id}:end")

    if not network_valves:
        prob_list.append("Network has no valves. Add at least one valve to compare switching scenarios.")
    elif scenario is None:
        prob_list.append("No scenario defined. Please configure initial valve states and switching events.")
    else:
        # Check whole-second switch times
        for ev in scenario.events:
            if abs(ev.time - round(ev.time)) > 1e-6:
                prob_list.append(
                    f"EPANET requires switch times in whole seconds, got {ev.time}s for valve {ev.valve}."
                )
            if ev.time < 0:
                prob_list.append(f"Switch event time cannot be negative, got {ev.time}s.")
            if ev.time >= scenario.t_end:
                prob_list.append(
                    f"Switch event time {ev.time}s must be strictly before end time {scenario.t_end}s."
                )

        # Check unknown valves referenced in scenario
        valid_valve_strs = set(network_valves)
        for v in scenario.initial:
            v_str = str(v)
            if v_str not in valid_valve_strs:
                prob_list.append(f"Scenario initial state references valve '{v_str}', which is not in network.")
        for ev in scenario.events:
            ev_str = str(ev.valve)
            if ev_str not in valid_valve_strs:
                prob_list.append(f"Scenario event references valve '{ev_str}', which is not in network.")

        # Check scenario solvability via DAE builder if network has no missing params
        if not missing and not pump_pipes and not tanks and reservoirs and not structural_warnings:
            try:
                ps = settings if settings is not None else PhysicsSettings()
                dae = build_switched_dae(network, ps)
                sc_probs = dae.check_scenario(scenario)
                for sp in sc_probs:
                    prob_list.append(sp)
            except Exception as e:
                prob_list.append(f"Scenario validation error: {e}")

    return prob_list


def parameters_table(
    network: Network,
    settings: PhysicsSettings,
) -> List[Tuple[str, str, str, str]]:
    """Return list of rows (element, parameter, value, unit) for all physical parameters."""
    rows: List[Tuple[str, str, str, str]] = []

    # 1. Nodes
    for n in sorted(network.nodes, key=lambda x: x.id):
        elev_str = f"{n.elevation:.2f}" if n.elevation is not None else "Not set"
        rows.append((n.id, f"{n.type.capitalize()} Elevation", elev_str, "m"))
        if n.type == "reservoir":
            head_str = f"{n.head:.2f}" if n.head is not None else "Not set"
            rows.append((n.id, "Reservoir Total Head", head_str, "m"))

    # 2. Pipes
    for p in sorted(network.pipes, key=lambda x: x.id):
        l_str = f"{p.length:.2f}" if p.length is not None else "Not set"
        d_str = f"{p.diameter:.4f}" if p.diameter is not None else "Not set"
        r_str = f"{p.roughness:.6f}" if p.roughness is not None else "Not set"
        rows.append((p.id, "Length", l_str, "m"))
        rows.append((p.id, "Diameter", d_str, "m"))
        rows.append((p.id, "Roughness", r_str, "m"))
        v_start_str = "Present" if p.valve_start else "None"
        v_end_str = "Present" if p.valve_end else "None"
        rows.append((p.id, "Start Valve", v_start_str, "-"))
        rows.append((p.id, "End Valve", v_end_str, "-"))

    # 3. Physics Settings
    rows.append(("Physics", "Gravitational acceleration (g)", f"{settings.g:.5f}", "m/s^2"))
    rows.append(("Physics", "Kinematic viscosity (nu)", f"{settings.nu:.4e}", "m^2/s"))
    rows.append(("Physics", "Friction Law", settings.friction_mode, "-"))
    if settings.friction_mode == "constant":
        rows.append(("Physics", "Friction Factor (f)", f"{settings.friction_factor:.4f}", "-"))

    return rows


def run_comparison(
    network: Network,
    scenario: Scenario,
    settings: Optional[PhysicsSettings] = None,
    positions: Optional[Dict[str, Tuple[float, float]]] = None,
) -> ComparisonResult:
    """Run DAE solver and real EPANET engine side-by-side on a snapshot of the network.

    Parameters
    ----------
    network : Network
        Network instance.
    scenario : Scenario
        Switching scenario.
    settings : PhysicsSettings, optional
        Physical parameters (defaults to PhysicsSettings()).
    positions : dict, optional
        Node positions for EPANET coordinates.

    Returns
    -------
    ComparisonResult
        Full result structure with equilibrium comparisons, impulses, timings, and signature.

    Raises
    ------
    ComparisonError
        If network or scenario has blocking problems.
    """
    if settings is None:
        settings = PhysicsSettings()

    # 1. Check problems first
    probs = problems(network, scenario, settings)
    if probs:
        raise ComparisonError(probs)

    # 2. Create isolated deep copy snapshot of network
    snapshot = Network.from_dict(network.to_dict())

    # 3. Parameters table
    params = parameters_table(snapshot, settings)

    # 4. Build DAE
    dae = build_switched_dae(snapshot, settings)

    # 5. Run DAE simulation
    t0_dae = time.perf_counter()
    res_dae = simulate(dae, scenario)
    t1_dae = time.perf_counter()

    # 6. Run EPANET engine
    t0_ep = time.perf_counter()
    res_ep = run_epanet(snapshot, scenario, settings, positions=positions)
    t1_ep = time.perf_counter()

    # 7. Compute equilibrium rows
    # For every stationary interval of the scenario:
    eq_rows: List[Dict[str, Any]] = []
    modes = scenario.modes()

    for k, m in enumerate(modes):
        t_a = m.t_start
        t_b = m.t_end
        valves_order = list(scenario.initial.keys())
        mode_tuple = tuple(m.state.get(v, 0) for v in valves_order)
        mode_str = f"({', '.join(str(s) for s in mode_tuple)})"

        # Steady-state equilibrium of this mode
        try:
            x_eq = steady_state(dae, m.state)
        except Exception:
            x_eq = np.zeros(dae.K, dtype=float)

        # Sampling instant:
        # For DAE: the last sample of the interval before the next switch.
        # If t_b < scenario.t_end, there is a switch at t_b.
        # simulate() places the pre-switch sample at t_b (first sample with t == t_b).
        if t_b < scenario.t_end:
            matches = np.where(np.abs(res_dae.t - t_b) < 1e-6)[0]
            idx_dae = int(matches[0]) if len(matches) > 0 else int(np.argmin(np.abs(res_dae.t - t_b)))
            # EPANET report before switch is at t_b - 1.0 s
            idx_ep = int(np.argmin(np.abs(res_ep.t - (t_b - 1.0))))
        else:
            idx_dae = int(np.argmin(np.abs(res_dae.t - t_b)))
            idx_ep = int(np.argmin(np.abs(res_ep.t - t_b)))

        interval_label = f"[{t_a:.0f}s - {t_b:.0f}s] Mode {mode_str}"

        for idx_lbl, lbl in enumerate(dae.labels):
            dae_end_val = float(res_dae.series(lbl)[idx_dae])
            dae_eq_val = float(x_eq[idx_lbl])
            ep_val = float(res_ep.series(lbl)[idx_ep])
            diff_end_val = abs(dae_end_val - ep_val)
            diff_eq_val = abs(dae_eq_val - ep_val)

            row = EquilibriumRow(
                interval=interval_label,
                mode=mode_str,
                label=lbl,
                dae_end=dae_end_val,
                dae_eq=dae_eq_val,
                epanet=ep_val,
                diff_end=diff_end_val,
                diff_eq=diff_eq_val,
            )
            eq_rows.append(row.to_dict())

    # 8. Compute impulses table
    impulses_list: List[Dict[str, Any]] = []
    for imp in res_dae.impulses:
        for idx_lbl, lbl in enumerate(dae.labels):
            coeff = float(imp.coefficients[idx_lbl])
            if abs(coeff) > 1e-4:
                impulses_list.append({
                    "time": imp.time,
                    "state": lbl,
                    "coefficient": coeff,
                    "epanet": "none",
                })

    timings = {
        "dae_solve_sec": t1_dae - t0_dae,
        "epanet_run_sec": t1_ep - t0_ep,
        "total_sec": (t1_dae - t0_dae) + (t1_ep - t0_ep),
    }

    warnings = list(res_ep.warnings)

    signature = compute_signature(snapshot, scenario, settings)

    return ComparisonResult(
        scenario=copy.deepcopy(scenario),
        settings=copy.deepcopy(settings),
        snapshot=snapshot,
        dae=res_dae,
        epanet=res_ep,
        params=params,
        equilibrium=eq_rows,
        impulses=impulses_list,
        timings=timings,
        warnings=warnings,
        signature=signature,
    )
