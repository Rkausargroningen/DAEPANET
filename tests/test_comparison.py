"""tests/test_comparison.py – Tests for comparison module, specimen presets, and physics I/O."""

import pytest
import numpy as np
from pathlib import Path

from core.models import Network
from core.scenario import Scenario, ValveId, SwitchEvent
from core.physics import PhysicsSettings
from core.examples import (
    specimen_network,
    specimen_scenario,
    specimen_physics,
    example_paper_network,
    example_broken_network,
)
from core.io import save_project, load_project_full
from core.comparison import (
    problems,
    parameters_table,
    run_comparison,
    ComparisonResult,
    ComparisonError,
)


def test_specimen_examples_factory():
    """Verify specimen network, scenario, and physics factories in core.examples."""
    net, pos = specimen_network()
    assert len(net.nodes) == 3
    assert len(net.pipes) == 2
    assert set(pos.keys()) == {"R1", "J1", "R2"}

    p2 = [p for p in net.pipes if p.id == "P2"][0]
    assert p2.valve_start is True
    assert p2.valve_end is True

    # Test scenarios 1..5
    for k in range(1, 6):
        sc = specimen_scenario(k)
        assert sc.t_end > 0
        assert sc.title

    phys = specimen_physics()
    assert phys.friction_mode == "constant"
    assert phys.friction_factor == 0.02


def test_physics_io_persistence(tmp_path: Path):
    """Test save and load round-trip of physics settings in .wnet files."""
    net, pos = specimen_network()
    sc = specimen_scenario(1)
    phys = PhysicsSettings(friction_mode="darcy_weisbach", nu=1.05e-6, g=9.81)

    file_path = tmp_path / "project_with_physics.wnet"
    save_project(file_path, net, positions=pos, scenario=sc, physics=phys)

    loaded = load_project_full(file_path)
    assert loaded.physics is not None
    assert loaded.physics.friction_mode == "darcy_weisbach"
    assert pytest.approx(1.05e-6) == loaded.physics.nu
    assert pytest.approx(9.81) == loaded.physics.g

    # Test saving without physics defaults to None on load
    file_no_phys = tmp_path / "project_no_physics.wnet"
    save_project(file_no_phys, net, positions=pos, scenario=sc, physics=None)
    loaded_no_phys = load_project_full(file_no_phys)
    assert loaded_no_phys.physics is None

    # Test invalid physics values reject with ValueError
    bad_json = file_path.read_text(encoding="utf-8")
    bad_json_corrupt = bad_json.replace('"darcy_weisbach"', '"invalid_friction_law"')
    bad_file = tmp_path / "corrupt_physics.wnet"
    bad_file.write_text(bad_json_corrupt, encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid physics"):
        load_project_full(bad_file)


def test_comparison_problems():
    """Verify problems() returns accurate, readable blocking reasons."""
    # 1. Empty network
    empty_net = Network()
    p_empty = problems(empty_net, None)
    assert any("empty" in p.lower() for p in p_empty)

    # 2. Paper network (pumps, tanks, missing parameters)
    paper_net = example_paper_network()
    p_paper = problems(paper_net, None)
    assert any("pump" in p.lower() for p in p_paper)
    assert any("tank" in p.lower() for p in p_paper)
    assert any("missing physical parameters" in p.lower() for p in p_paper)

    # 3. Specimen network without scenario
    spec_net, _ = specimen_network()
    p_no_sc = problems(spec_net, None)
    assert any("no scenario defined" in p.lower() for p in p_no_sc)

    # 4. Specimen network with valid scenario -> ready (empty list)
    sc1 = specimen_scenario(1)
    phys = specimen_physics()
    p_ready = problems(spec_net, sc1, phys)
    assert p_ready == []

    # 5. Non-integer switch times
    v1 = ValveId("P2", "start")
    sc_bad_time = Scenario(
        "Bad Time",
        initial={v1: 1},
        events=[SwitchEvent(12.5, v1, 0)],
        t_end=50.0,
    )
    p_bad_time = problems(spec_net, sc_bad_time, phys)
    assert any("whole seconds" in p for p in p_bad_time)


def test_parameters_table():
    """Verify parameters_table generates all elements and physical values."""
    net, _ = specimen_network()
    phys = specimen_physics()
    table = parameters_table(net, phys)

    assert len(table) > 10
    headers = [row[0] for row in table]
    assert "R1" in headers
    assert "J1" in headers
    assert "R2" in headers
    assert "P1" in headers
    assert "P2" in headers
    assert "Physics" in headers


def test_run_comparison_specimen_scenario_1():
    """Test full run_comparison on specimen scenario 1."""
    net, pos = specimen_network()
    sc1 = specimen_scenario(1)
    phys = specimen_physics()

    res = run_comparison(net, sc1, phys, positions=pos)
    assert isinstance(res, ComparisonResult)
    assert res.signature
    assert not res.is_outdated(net, sc1, phys)

    # Verify equilibrium rows exist
    assert len(res.equilibrium) == 2 * 9  # 2 modes x 9 DAE states

    # Check equilibrium values for mode (0, 1) at end of interval 2 (t=120)
    final_rows = [r for r in res.equilibrium if "Mode (0, 1)" in r["interval"]]
    p1_flow_row = [r for r in final_rows if r["label"] == "Pipe_1_Flow"][0]
    assert pytest.approx(0.0, abs=1e-5) == p1_flow_row["dae"]
    assert pytest.approx(0.0, abs=1e-5) == p1_flow_row["epanet"]
    assert p1_flow_row["diff"] < 1e-5

    # Check impulses: scenario 1 has impulses at t=60s
    assert len(res.impulses) >= 3
    imp_map = {r["state"]: r["coefficient"] for r in res.impulses}
    assert pytest.approx(62.16, abs=0.5) == imp_map["Pipe_1_R"]
    assert pytest.approx(-38.85, abs=0.5) == imp_map["Pipe_2_L"]
    assert pytest.approx(62.16, abs=0.5) == imp_map["J1"]

    # Test outdated check
    net_mod = Network.from_dict(net.to_dict())
    net_mod.pipes[0].length = 450.0
    assert res.is_outdated(net_mod, sc1, phys)
