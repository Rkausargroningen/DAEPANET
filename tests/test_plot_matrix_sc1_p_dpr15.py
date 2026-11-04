"""tests/test_plot_matrix_sc1_p_dpr15.py – Scenario 1 pressure head DPR 1.5."""
import pytest
from core.examples import specimen_network, specimen_scenario, specimen_physics
from core.comparison import run_comparison
from tests.plot_matrix_base import run_matrix_subset

@pytest.fixture(scope="module")
def res1():
    net, pos = specimen_network()
    phys = specimen_physics()
    return run_comparison(net, specimen_scenario(1), phys, positions=pos)

def test_matrix_sc1_p_dpr15(res1):
    assert run_matrix_subset(res1, sc_name="sc1", dpr=1.5, mode="pressure") == 20
