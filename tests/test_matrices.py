"""tests/test_matrices.py – Pytest suite for core/ calculation module.

All tests are self-contained; the only external dependency is numpy.
The legacy streamlit_app is imported via namespace package (no __init__.py
in legacy/) with streamlit mocked out.
"""

from __future__ import annotations

import sys
import itertools
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup – project root on sys.path so "legacy" is a namespace package
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Import core (must come BEFORE any legacy import to avoid polluting sys.modules)
# ---------------------------------------------------------------------------
from core import (
    Network,
    build_matrices,
    numeric_pencil,
    is_regular,
    incidence_matrices,
    validate_network,
)

# ---------------------------------------------------------------------------
# Import legacy with streamlit mocked
# ---------------------------------------------------------------------------
sys.modules.setdefault("streamlit", MagicMock())

import legacy.streamlit_app as _legacy  # noqa: E402


# ===========================================================================
# Helper: build a simple R1->J1->J2 network (no tanks)
# ===========================================================================

def _make_r1_j1_j2() -> Network:
    net = Network()
    net.add_node("reservoir", id="R1")
    net.add_node("junction",  id="J1")
    net.add_node("junction",  id="J2")
    net.add_pipe("R1", "J1", id="P1")   # non-pumped
    net.add_pipe("J1", "J2", id="P2")   # non-pumped
    return net


def _legacy_nodes_pipes(net: Network):
    """Return (nodes, pipes) in legacy dict format."""
    d = net.to_dict()
    return d["nodes"], d["pipes"]


# ===========================================================================
# 1. REGRESSION – networks without tanks, normal pipes before pumps
# ===========================================================================

class TestRegression:
    """For networks that satisfy legacy assumptions (no tanks, non-pumped first),
    build_matrices must produce A/E/F/G/K identical to the legacy function."""

    def _compare(self, net: Network):
        nodes, pipes = _legacy_nodes_pipes(net)
        leg = _legacy.create_network_matrices(nodes, pipes)
        new = build_matrices(net)

        assert new["K_dimension"] == leg["K_dimension"], (
            f"K mismatch: new={new['K_dimension']}, legacy={leg['K_dimension']}"
        )
        assert new["E_matrix"] == leg["E_matrix"], "E matrix mismatch"
        assert new["A_matrix"] == leg["A_matrix"], "A matrix mismatch"
        assert new["F_matrix"] == leg["F_matrix"], "F matrix mismatch"
        assert new["G_matrix"] == leg["G_matrix"], "G matrix mismatch"

    def test_r1_j1_j2_no_pump(self):
        """Simple chain: R1->J1->J2, two non-pumped pipes."""
        self._compare(_make_r1_j1_j2())

    def test_two_reservoirs_two_junctions(self):
        """R1->J1, R2->J2, J1->J2 – no pumps, no tanks."""
        net = Network()
        net.add_node("reservoir", id="R1")
        net.add_node("reservoir", id="R2")
        net.add_node("junction",  id="J1")
        net.add_node("junction",  id="J2")
        net.add_pipe("R1", "J1", id="P1")
        net.add_pipe("R2", "J2", id="P2")
        net.add_pipe("J1", "J2", id="P3")
        self._compare(net)

    def test_with_one_pump_normal_first(self):
        """R1->J1 (normal), J1->J2 (pump) – pumps come AFTER normal pipes."""
        net = Network()
        net.add_node("reservoir", id="R1")
        net.add_node("junction",  id="J1")
        net.add_node("junction",  id="J2")
        net.add_pipe("R1", "J1", id="P1", has_pump=False)
        net.add_pipe("J1", "J2", id="P2", has_pump=True, pump_power=5.0)
        self._compare(net)


# ===========================================================================
# 2. ORDER-INVARIANCE
# ===========================================================================

class TestOrderInvariance:
    """Same network built with pipes in different creation orders must give
    identical regularity verdict and equivalent matrices (after consistent
    row/column permutation)."""

    def _network_from_pipe_order(self, pipe_defs: list[dict]) -> Network:
        """pipe_defs: list of dict(start, end, has_pump)."""
        node_ids: list[str] = []
        for p in pipe_defs:
            for nid in (p["start"], p["end"]):
                if nid not in node_ids:
                    node_ids.append(nid)

        net = Network()
        for nid in node_ids:
            typ = "reservoir" if nid.startswith("R") else \
                  "tank"      if nid.startswith("T") else "junction"
            net.add_node(typ, id=nid)
        for i, p in enumerate(pipe_defs):
            net.add_pipe(p["start"], p["end"], id=f"P{i+1}",
                         has_pump=p.get("has_pump", False))
        return net

    def test_regularity_invariant_to_creation_order(self):
        """R1->J1->J2: regular in both pipe creation orders."""
        pipe_defs_ab = [
            {"start": "R1", "end": "J1"},
            {"start": "J1", "end": "J2"},
        ]
        pipe_defs_ba = [
            {"start": "J1", "end": "J2"},
            {"start": "R1", "end": "J1"},
        ]
        for defs in (pipe_defs_ab, pipe_defs_ba):
            net = self._network_from_pipe_order(defs)
            mats = build_matrices(net)
            E, A = numeric_pencil(mats)
            assert is_regular(E, A), (
                f"Expected regular for R1->J1->J2 with order {defs}"
            )

    def test_matrix_permutation_consistency(self):
        """R1->J1->J2 in two orders produces same matrix up to row/col swap."""
        # Order A: P1=R1->J1, P2=J1->J2
        netA = Network()
        for nid, typ in [("R1","reservoir"),("J1","junction"),("J2","junction")]:
            netA.add_node(typ, id=nid)
        netA.add_pipe("R1","J1",id="P1"); netA.add_pipe("J1","J2",id="P2")

        # Order B: pipes swapped, but IDs also swapped for a clean permutation
        netB = Network()
        for nid, typ in [("R1","reservoir"),("J1","junction"),("J2","junction")]:
            netB.add_node(typ, id=nid)
        netB.add_pipe("J1","J2",id="P1"); netB.add_pipe("R1","J1",id="P2")

        mA = build_matrices(netA); mB = build_matrices(netB)
        EA, AA = numeric_pencil(mA);  EB, AB = numeric_pencil(mB)

        # Both must be (ir)regular
        assert is_regular(EA, AA) == is_regular(EB, AB)


# ===========================================================================
# 3. REGULARITY SCENARIOS
# ===========================================================================

class TestRegularityScenarios:

    # ------------------------------------------------------------------ 3a
    def test_r1_j1_j2_regular(self):
        """R1->J1->J2 must be regular."""
        net = _make_r1_j1_j2()
        E, A = numeric_pencil(build_matrices(net))
        assert is_regular(E, A) is True

    # ------------------------------------------------------------------ 3b
    def test_island_not_regular(self):
        """R1->J1 plus isolated J2->J3 must NOT be regular."""
        net = Network()
        net.add_node("reservoir", id="R1")
        net.add_node("junction",  id="J1")
        net.add_node("junction",  id="J2")
        net.add_node("junction",  id="J3")
        net.add_pipe("R1", "J1", id="P1")
        net.add_pipe("J2", "J3", id="P2")
        E, A = numeric_pencil(build_matrices(net))
        assert is_regular(E, A) is False

    # ------------------------------------------------------------------ 3c
    def test_pump_cycle_not_regular_all_orders(self):
        """Pump cycle J1->J2 (pump), J2->J1 (pump) + supply R1->J1 must NOT
        be regular regardless of pipe creation order."""
        pipe_triples = [
            # (start, end, has_pump)
            ("R1","J1",False), ("J1","J2",True), ("J2","J1",True),
        ]
        # Try all 6 permutations of the three pipes
        for perm in itertools.permutations(pipe_triples):
            net = Network()
            seen_nodes: list[str] = []
            for s, e, _ in perm:
                for nid in (s, e):
                    if nid not in seen_nodes:
                        seen_nodes.append(nid)
            for nid in seen_nodes:
                typ = "reservoir" if nid.startswith("R") else "junction"
                net.add_node(typ, id=nid)
            for i, (s, e, hp) in enumerate(perm):
                net.add_pipe(s, e, id=f"P{i+1}", has_pump=hp)
            E, A = numeric_pencil(build_matrices(net))
            assert is_regular(E, A) is False, (
                f"Pump cycle must be irregular in creation order {perm}"
            )

    # ------------------------------------------------------------------ 3d  (tank regular)
    def test_r1_j1_t1_regular(self):
        """R1->J1->T1 must be regular."""
        net = Network()
        net.add_node("reservoir", id="R1")
        net.add_node("junction",  id="J1")
        net.add_node("tank",      id="T1")
        net.add_pipe("R1", "J1", id="P1")
        net.add_pipe("J1", "T1", id="P2")
        E, A = numeric_pencil(build_matrices(net))
        assert is_regular(E, A) is True

    def test_t1_feeds_j1_regular(self):
        """R1->J1, T1->J1 must be regular."""
        net = Network()
        net.add_node("reservoir", id="R1")
        net.add_node("junction",  id="J1")
        net.add_node("tank",      id="T1")
        net.add_pipe("R1", "J1", id="P1")
        net.add_pipe("T1", "J1", id="P2")
        E, A = numeric_pencil(build_matrices(net))
        assert is_regular(E, A) is True

    # ------------------------------------------------------------------ 3e  pump-series (new test from corrections)
    def test_pump_series_between_supplies_irregular(self):
        """R1 -pump-> J1 -pump-> R2 must be flagged AND irregular.
        (Two supplies connected only by pumps in series creates a degenerate DAE.)
        """
        net = Network()
        net.add_node("reservoir", id="R1")
        net.add_node("junction",  id="J1")
        net.add_node("reservoir", id="R2")
        net.add_pipe("R1", "J1", id="P1", has_pump=True)
        net.add_pipe("J1", "R2", id="P2", has_pump=True)
        E, A = numeric_pencil(build_matrices(net))
        assert is_regular(E, A) is False, (
            "Pump series between two reservoirs should be irregular"
        )
        warnings = validate_network(net)
        pump_cycle_warnings = [w for w in warnings if "pump cycle" in w.lower() or "rank check" in w.lower()]
        assert pump_cycle_warnings, (
            f"Expected a pump cycle or rank warning; got: {warnings}"
        )

    # ------------------------------------------------------------------ 3f  single pump should be fine
    def test_single_pump_not_flagged_and_regular(self):
        """R1 -pump-> J1, J1->J2 (normal pipe): must NOT be flagged and must be regular."""
        net = Network()
        net.add_node("reservoir", id="R1")
        net.add_node("junction",  id="J1")
        net.add_node("junction",  id="J2")
        net.add_pipe("R1", "J1", id="P1", has_pump=True,  pump_power=10.0)
        net.add_pipe("J1", "J2", id="P2", has_pump=False)
        warnings = validate_network(net)
        pump_cycle_warnings = [w for w in warnings if "pump" in w.lower() and "cycle" in w.lower()]
        assert not pump_cycle_warnings, (
            f"Single pump must NOT be flagged as a cycle; got: {warnings}"
        )
        E, A = numeric_pencil(build_matrices(net))
        assert is_regular(E, A) is True, "Single-pump network must be regular"


# ===========================================================================
# 4. TANK ROW NON-ZERO
# ===========================================================================

class TestTankRow:
    def test_tank_row_nonzero_when_pipe_ends_at_tank(self):
        """R1->J1->T1: the T1 row of A must not be all zeros."""
        net = Network()
        net.add_node("reservoir", id="R1")
        net.add_node("junction",  id="J1")
        net.add_node("tank",      id="T1")
        net.add_pipe("R1", "J1", id="P1")
        net.add_pipe("J1", "T1", id="P2")
        mats = build_matrices(net)
        A = mats["A_matrix"]
        K = mats["K_dimension"]
        # Tank row is the last row
        tank_row = A[K - 1]
        assert any(v != "0" for v in tank_row), (
            f"Tank row must not be all zero; got: {tank_row}"
        )

    def test_tank_row_nonzero_when_pipe_starts_at_tank(self):
        """R1->J1, T1->J1: the T1 row of A must contain -1 (outflow)."""
        net = Network()
        net.add_node("reservoir", id="R1")
        net.add_node("junction",  id="J1")
        net.add_node("tank",      id="T1")
        net.add_pipe("R1", "J1", id="P1")
        net.add_pipe("T1", "J1", id="P2")
        mats = build_matrices(net)
        A = mats["A_matrix"]
        K = mats["K_dimension"]
        tank_row = A[K - 1]
        assert any(v != "0" for v in tank_row), (
            f"Tank row must not be all zero; got: {tank_row}"
        )
        assert "-1" in tank_row, (
            f"Tank row for outflow pipe should contain '-1'; got: {tank_row}"
        )


# ===========================================================================
# 5. VALIDATE_NETWORK WARNINGS
# ===========================================================================

class TestValidateNetwork:

    def test_no_warnings_for_clean_chain(self):
        """R1->J1->J2 must produce no warnings."""
        warnings = validate_network(_make_r1_j1_j2())
        assert warnings == [], f"Expected no warnings, got: {warnings}"

    def test_island_warning(self):
        """J2->J3 island (no supply) must produce a supply-free warning."""
        net = Network()
        net.add_node("reservoir", id="R1")
        net.add_node("junction",  id="J1")
        net.add_node("junction",  id="J2")
        net.add_node("junction",  id="J3")
        net.add_pipe("R1", "J1", id="P1")
        net.add_pipe("J2", "J3", id="P2")
        warnings = validate_network(net)
        assert any("not connected to any supply" in w for w in warnings), (
            f"Expected supply-free warning; got: {warnings}"
        )

    def test_isolated_node_warning(self):
        """An isolated node (J2 with no pipes) must produce an isolated warning."""
        net = Network()
        net.add_node("reservoir", id="R1")
        net.add_node("junction",  id="J1")
        net.add_node("junction",  id="J2")  # isolated
        net.add_pipe("R1", "J1", id="P1")
        warnings = validate_network(net)
        assert any("J2" in w and "isolated" in w.lower() for w in warnings), (
            f"Expected isolated-node warning for J2; got: {warnings}"
        )

    def test_pump_cycle_warning(self):
        """J1->J2 (pump), J2->J1 (pump) + R1->J1 must produce a pump cycle warning."""
        net = Network()
        net.add_node("reservoir", id="R1")
        net.add_node("junction",  id="J1")
        net.add_node("junction",  id="J2")
        net.add_pipe("R1", "J1", id="P1")
        net.add_pipe("J1", "J2", id="P2", has_pump=True)
        net.add_pipe("J2", "J1", id="P3", has_pump=True)
        warnings = validate_network(net)
        assert any("pump cycle" in w.lower() for w in warnings), (
            f"Expected pump-cycle warning; got: {warnings}"
        )
        # Pump IDs must appear in the warning
        pump_warn = [w for w in warnings if "pump cycle" in w.lower()][0]
        assert "P2" in pump_warn or "P3" in pump_warn, (
            f"Pump IDs should appear in warning: {pump_warn}"
        )

    def test_pump_series_between_supplies_warning(self):
        """R1 -pump-> J1 -pump-> R2 must be flagged."""
        net = Network()
        net.add_node("reservoir", id="R1")
        net.add_node("junction",  id="J1")
        net.add_node("reservoir", id="R2")
        net.add_pipe("R1", "J1", id="P1", has_pump=True)
        net.add_pipe("J1", "R2", id="P2", has_pump=True)
        warnings = validate_network(net)
        pump_warns = [w for w in warnings if "pump" in w.lower()]
        assert pump_warns, (
            f"Expected a pump-related warning for series pumps between supplies; got: {warnings}"
        )


# ===========================================================================
# 6. INCIDENCE MATRIX PROPERTIES
# ===========================================================================

class TestIncidenceMatrices:

    def _make_paper_example(self) -> Network:
        """R1->J1->J2 as a representative example."""
        return _make_r1_j1_j2()

    def test_column_sums_zero(self):
        """Every column of the full incidence matrix must sum to 0."""
        net = self._make_paper_example()
        inc = incidence_matrices(net)
        B = inc["incidence"].astype(int)
        col_sums = B.sum(axis=0)
        np.testing.assert_array_equal(
            col_sums, 0,
            err_msg="Each column of the incidence matrix must sum to 0 "
                    f"(got {col_sums})"
        )

    def test_reduced_shape(self):
        """Reduced incidence must have n-1 rows."""
        net = self._make_paper_example()
        nodes = net.nodes
        inc  = incidence_matrices(net)
        assert inc["reduced"].shape[0] == len(nodes) - 1

    def test_modified_shape(self):
        """Modified incidence must delete all reservoir and tank rows."""
        net = Network()
        net.add_node("reservoir", id="R1")
        net.add_node("reservoir", id="R2")
        net.add_node("tank",      id="T1")
        net.add_node("junction",  id="J1")
        net.add_node("junction",  id="J2")
        net.add_pipe("R1", "J1", id="P1")
        net.add_pipe("R2", "J2", id="P2")
        net.add_pipe("J1", "J2", id="P3")
        net.add_pipe("T1", "J1", id="P4")
        inc = incidence_matrices(net)
        # Modified should have rows for junctions only (2)
        assert inc["modified"].shape[0] == 2

    def test_incidence_signs(self):
        """Each pipe column: -1 at start node, +1 at end node."""
        net = _make_r1_j1_j2()
        inc = incidence_matrices(net)
        B = inc["incidence"]
        node_lbl = inc["node_labels"]
        edge_lbl = inc["edge_labels"]

        for j_idx, pipe in enumerate(net.pipes):
            col = B[:, j_idx]
            start_row = node_lbl.index(pipe.start)
            end_row   = node_lbl.index(pipe.end)
            assert col[start_row] == -1, (
                f"Expected -1 at start node {pipe.start} for pipe {pipe.id}"
            )
            assert col[end_row] == +1, (
                f"Expected +1 at end node {pipe.end} for pipe {pipe.id}"
            )


# ===========================================================================
# 7. G MATRIX STRING FORMAT (regression guard)
# ===========================================================================

class TestGStrings:
    def test_non_pumped_g_strings(self):
        """G entries for non-pumped pipes must exactly match legacy format."""
        net = Network()
        net.add_node("reservoir", id="R1")
        net.add_node("junction",  id="J1")
        net.add_node("junction",  id="J2")
        net.add_pipe("R1", "J1", id="P1")
        net.add_pipe("J1", "J2", id="P2")
        mats = build_matrices(net)
        G = mats["G_matrix"]
        assert G[0] == "-C_aQ1|Q1|"
        assert G[1] == "-C_aQ2|Q2|"

    def test_pumped_g_string(self):
        """G entry for a pump must be the exact legacy string."""
        net = Network()
        net.add_node("reservoir", id="R1")
        net.add_node("junction",  id="J1")
        net.add_node("junction",  id="J2")
        net.add_pipe("R1", "J1", id="P1")
        net.add_pipe("J1", "J2", id="P2", has_pump=True)
        mats = build_matrices(net)
        G = mats["G_matrix"]
        # P2 is pump; its G row is N_np=1, so index 1
        assert G[1] == "-S_puG_pu(Q_pu)"

