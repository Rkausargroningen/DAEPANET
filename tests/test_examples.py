"""Tests for preset example networks (core/examples.py)."""

from core.examples import example_paper_network, example_broken_network
from core.matrices import build_matrices
from core.checks import numeric_pencil, is_regular, validate_network


def test_paper_network_regular_and_clean():
    net = example_paper_network()
    warnings = validate_network(net)
    assert warnings == []

    mats = build_matrices(net.to_dict())
    E_num, A_num = numeric_pencil(mats)
    assert is_regular(E_num, A_num) is True


def test_broken_network_irregular_and_has_warnings():
    net = example_broken_network()
    warnings = validate_network(net)
    assert len(warnings) >= 2

    # Verify specific warning contents
    joined_warnings = " ".join(warnings)
    assert "not connected to any supply" in joined_warnings
    assert "Pump cycle" in joined_warnings

    mats = build_matrices(net.to_dict())
    E_num, A_num = numeric_pencil(mats)
    assert is_regular(E_num, A_num) is False
