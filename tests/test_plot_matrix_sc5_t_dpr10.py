"""tests/test_plot_matrix_sc5_t_dpr10.py – Scenario 5 total head DPR 1.0."""
import pytest
from core.examples import specimen_network, specimen_scenario, specimen_physics
from core.comparison import run_comparison
from tests.plot_matrix_base import run_matrix_subset

@pytest.fixture(scope="module")
def res5():
    net, pos = specimen_network()
    phys = specimen_physics()
    return run_comparison(net, specimen_scenario(5), phys, positions=pos)

def test_matrix_sc5_t_dpr10(res5):
    assert run_matrix_subset(res5, sc_name="sc5", dpr=1.0, mode="total") == 20
